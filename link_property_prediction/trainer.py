"""Per-query-causal training + eval loop with a monotone metric head over walk bags.

The full graph (train + val + test) is ingested into Tempest once up front; causality is enforced
per query by the walk cutoff, not by ingestion order. A walk for (u, t) uses cutoff = t, which is
EXCLUSIVE: it traverses only edges with t_edge < t, so the target edge at t and any
simultaneous/future edge are never seen. TGB-Seq splits are chronological (train < val < test).

Training per batch: sample K negatives by the suite's rule, form candidates [pos | negs],
score them TWO-SIDED (walks for the source u and for every candidate v, each cut off at the query
time t_i), then cross-entropy with target 0 and a single optimizer step. E and the head train
together under the link CE with no detach.
"""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import LinkPredHead
from .negatives import NegativeSampler
from .walk_tokens import WalkTokens, build_query_walk_tokens
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    # Dataset-derived.
    num_nodes: int
    T_train: int
    train_negative_sampler: NegativeSampler

    # Embedding dimension.
    d_emb: int = 128

    # Pooler MLP width, PINNED. It does NOT move with the feature count: under the
    # old 8*n_feat rule every feature ablation silently changed pooler capacity too
    # and the two effects could not be separated.
    hidden_dim: int = 32

    # Hidden Linear->GELU stages in the pooler MLP. Each past the first inserts a
    # Linear(hidden, hidden) -> GELU. Depth and width are separate knobs so a capacity
    # change can be attributed to one of them. 1 reproduces the pre-c2a7c95 shape.
    n_layers_pooler: int = 2

    # Walks: BACKWARD only, undirected; two-sided (source and every candidate).
    num_walks_per_node: int = 10
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0    # node2vec return param (used only when a bias is TemporalNode2Vec)
    t2nv_q: float = 0.25   # node2vec in-out param

    # One constant lr for every parameter. E steps under SparseAdam, the pooler under Adam;
    # both read `lr`. Nothing is decayed: the score carries no free scale to hold down.
    lr: float = 1e-3

    # Run control. The epoch cap is high enough that patience, not the cap, ends every run:
    # a cell that prints 200 epochs is a floor, not a result.
    num_epochs: int = 200
    early_stop_patience: int = 10

    # System.
    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False


class Trainer:
    def __init__(self, config: TrainerConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        # Owns the Euclidean node embeddings and the monotone metric score.
        self.model = LinkPredHead(
            num_nodes=config.num_nodes,
            T_train=config.T_train,
            max_walk_len=config.max_walk_len,
            d_emb=int(config.d_emb),
            hidden_dim=int(config.hidden_dim),
            n_layers_pooler=int(config.n_layers_pooler),
            seed=int(config.seed),
        ).to(self.device)

        self.walk_gen = WalkGenerator(
            use_gpu=config.use_gpu_tempest,
            walk_bias=config.walk_bias,
            start_bias=config.start_bias,
            num_walks_per_node=config.num_walks_per_node,
            max_walk_len=config.max_walk_len,
            temporal_node2vec_p=config.t2nv_p,
            temporal_node2vec_q=config.t2nv_q,
            seed=int(config.seed),
        )
        self.neg_sampler_train = config.train_negative_sampler

        # Two optimisers. E's gradient is sparse (only looked-up rows), so SparseAdam updates the
        # moments of those rows alone: an absent row's state stays frozen instead of decaying.
        # The pooler gets a gradient every step, so it takes plain Adam. Two optimisers, one lr:
        # the split is forced, since SparseAdam rejects dense gradients.
        pooler = [p for n, p in self.model.named_parameters() if n != "E.weight"]
        self.opt = torch.optim.Adam(pooler, lr=float(config.lr))
        self.opt_embedding = torch.optim.SparseAdam([self.model.E.weight], lr=float(config.lr))

    # Full-graph ingestion (once, up front)

    def ingest_full_graph(self, graph: SplitData) -> None:
        """Ingest the entire graph (all splits) into Tempest in one add_edges call, once before
        train()/eval. The per-query cutoff (t_edge < t_query, EXCLUSIVE) enforces causality.
        Capacity is unbounded."""
        self.walk_gen.add_edges(graph.sources, graph.destinations, graph.timestamps, graph.edge_feat)
        print(f"  Ingested full graph into Tempest: {len(graph.sources):,} edges "
              f"(once; per-query cutoff enforces causality)")

    # Scoring — shared by train + eval

    def _long(self, array: np.ndarray) -> torch.Tensor:
        """A numpy id/time array as an int64 tensor on the model's device."""
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64)).to(self.device)

    def _walk_tokens(self, seeds: torch.Tensor, cutoffs: torch.Tensor) -> WalkTokens:
        """K backward walks per seed, each cut off (EXCLUSIVELY) at its own cutoff."""
        return build_query_walk_tokens(
            self.walk_gen, self.device, seeds, cutoffs,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

    def _score(self, batch: Batch, candidates: np.ndarray) -> torch.Tensor:
        """Logits [B, C] for `candidates` [B, C] of `batch`'s queries. Two-sided per-query walks:
        K backward walks for the source and for every candidate, each cut off at its query's t_i."""
        src = self._long(batch.src)                                          # [B]
        t_query = self._long(batch.ts)                                       # [B]
        cand = self._long(candidates)                                        # [B, C]

        # Flatten candidates [B, C] -> [B*C] query-major; each inherits its query's cutoff.
        b, c = cand.shape
        cand_cutoffs = t_query.unsqueeze(1).expand(b, c).reshape(-1)         # [B*C]

        src_tokens = self._walk_tokens(src, t_query)
        cand_tokens = self._walk_tokens(cand.reshape(-1), cand_cutoffs)
        return self.model(src_tokens, cand_tokens)

    # Per-batch training step

    def _train_step(self, batch: Batch) -> float:
        """One optimiser step; returns the batch's link CE for the epoch average."""
        neg_tgt = self.neg_sampler_train.sample(batch)                       # [B, K]
        candidates = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None], neg_tgt.astype(np.int64)], axis=1)  # [B, 1+K]

        logits = self._score(batch, candidates)                              # [B, 1+K]
        target = torch.zeros(len(batch.src), dtype=torch.long, device=self.device)  # positive first
        # link CE only; E trains end-to-end through the head (no detach).
        link_loss = F.cross_entropy(logits, target)

        self.opt.zero_grad(set_to_none=True)
        self.opt_embedding.zero_grad(set_to_none=True)
        link_loss.backward()
        self.opt.step()
        self.opt_embedding.step()

        return float(link_loss.detach())

    # Epoch-line probe

    @torch.no_grad()
    def _probe(self) -> str:
        """The epoch line's state watch: mean and max norm of E (its distance from the origin)."""
        r = self.model.E.weight.norm(dim=-1)
        return f"  r_mean={float(r.mean()):.3f}  r_max={float(r.max()):.3f}"

    # Eval — strict-causal, no_grad

    @staticmethod
    def _eval_candidates(batch: Batch, neg_tgt_list: List[np.ndarray]) -> np.ndarray:
        """[B, 1+max_K] candidates, the positive first. A positive with fewer than max_K
        negatives is right-padded with copies of itself; only its own count is ever scored."""
        max_k = max((len(neg) for neg in neg_tgt_list), default=0)
        candidates = np.tile(batch.tgt.astype(np.int64)[:, None], (1, 1 + max_k))
        for i, neg in enumerate(neg_tgt_list):
            candidates[i, 1:1 + len(neg)] = neg
        return candidates

    @torch.no_grad()
    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch]) -> float:
        """Mean of the suite's per-positive metric over `batches`."""
        self.model.eval()
        # Rewind the fixed-negative cursor so every eval pass sees the same negatives.
        # Must precede the first sample_negatives call.
        evaluator.reset()
        total, n = 0.0, 0
        for batch in batches:
            if len(batch.src) == 0:
                continue
            neg_tgt_list = evaluator.sample_negatives(batch)
            logits = self._score(batch, self._eval_candidates(batch, neg_tgt_list)).cpu().numpy()
            for i, neg in enumerate(neg_tgt_list):
                total += evaluator.score_to_metric(float(logits[i, 0]), logits[i, 1:1 + len(neg)])
            n += len(batch.src)
        return total / max(n, 1)

    # Snapshot / restore (early-stop)

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        """A CPU copy of the model's state, for restoring the best-val epoch."""
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _restore(self, snap: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snap)

    # Train loop

    def train(
        self,
        train_batches_factory,
        full_graph: SplitData,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
        # Ingest the full graph (train + val + test) into Tempest once, up front.
        self.ingest_full_graph(full_graph)

        n_epochs = self.config.num_epochs
        patience = self.config.early_stop_patience

        best_val, best_test, best_epoch = -1.0, -1.0, -1
        best_snap: Optional[Dict[str, torch.Tensor]] = None
        no_improve = 0
        per_epoch_val: List[float] = []
        new_best_test: List[float] = []

        for ep in range(1, n_epochs + 1):
            self.model.train()

            t0 = time.time()
            link_sum, n_batches = 0.0, 0
            for batch in train_batches_factory():
                link_sum += self._train_step(batch)
                n_batches += 1
            train_dt = time.time() - t0

            line = (
                f"epoch {ep}/{n_epochs}  "
                f"link={link_sum / max(n_batches, 1):.4f}  "
                f"lr={self.opt.param_groups[0]['lr']:.0e}  "
                f"train {train_dt:.1f}s"
            )

            line += self._probe()

            if val_evaluator is not None and val_batches_factory is not None:
                t1 = time.time()
                val_metric = self._eval(val_evaluator, val_batches_factory())
                per_epoch_val.append(val_metric)

                if val_metric > best_val:
                    best_val, best_epoch = val_metric, ep
                    best_snap = self._snapshot()
                    no_improve = 0
                    if test_evaluator is not None and test_batches_factory is not None:
                        best_test = self._eval(test_evaluator, test_batches_factory())
                        new_best_test.append(best_test)
                        line += f"  val {val_metric:.4f}  test {best_test:.4f} (new best)"
                    else:
                        line += f"  val {val_metric:.4f} (new best)"
                else:
                    no_improve += 1
                    line += f"  val {val_metric:.4f}  patience {no_improve}/{patience}"
                # Timed around BOTH passes. A new-best epoch also evaluates the test split,
                # and timing only the val pass printed those epochs as cheap as the rest.
                line += f"  eval {time.time() - t1:.1f}s"
            print(line)

            if patience > 0 and no_improve >= patience:
                break

        if best_snap is not None:
            self._restore(best_snap)
            print(
                f"  restored best weights from epoch {best_epoch} "
                f"(val {best_val:.4f}, test {best_test:.4f})")

        return {
            # `best_epoch` is where the restored checkpoint came from, -1 if there was no
            # validation to select one; `epochs_run` is how far the loop actually got. The
            # single old `stopped_at_epoch` field was the first but named like the second,
            # and its caller renamed it back to `best_epoch` to use it.
            "best_epoch": best_epoch,
            "epochs_run": len(per_epoch_val) if per_epoch_val else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            # Only new-best-val epochs evaluate test, so this is NOT per-epoch and is not
            # index-aligned with `per_epoch_val_mrr`.
            "new_best_test_mrr": new_best_test,
        }
