"""Training and eval loop; every walk is cut off EXCLUSIVELY at its query's time (t_edge < t)."""
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch, SplitData
from .evaluator import Evaluator
from .model import LinkPredHead, dist0
from .negatives import NegativeSampler
from .walk_tokens import WalkTokens, build_query_walk_tokens
from .walks import WalkGenerator


@dataclass
class TrainerConfig:
    num_nodes: int
    T_train: int
    train_negative_sampler: NegativeSampler

    d_emb: int = 128
    hidden_dim: int = 32
    n_layers_pooler: int = 2

    # Backward, undirected walks for the source and every candidate.
    num_walks_per_node: int = 10
    max_walk_len: int = 5
    walk_bias: str = "ExponentialWeight"
    start_bias: str = "ExponentialWeight"
    t2nv_p: float = 4.0          # TemporalNode2Vec only
    t2nv_q: float = 0.25

    lr: float = 1e-3             # shared by SparseAdam (E) and Adam (pooler)
    num_epochs: int = 200
    early_stop_patience: int = 10

    seed: int = 42
    use_gpu: bool = False
    use_gpu_tempest: bool = False
    use_recurrence_count: bool = False


class Trainer:
    def __init__(self, config: TrainerConfig, device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device(
            "cuda" if (config.use_gpu and torch.cuda.is_available()) else "cpu"
        )
        self.model = LinkPredHead(
            num_nodes=config.num_nodes,
            T_train=config.T_train,
            max_walk_len=config.max_walk_len,
            d_emb=int(config.d_emb),
            hidden_dim=int(config.hidden_dim),
            n_layers_pooler=int(config.n_layers_pooler),
            seed=int(config.seed),
            use_recurrence_count=config.use_recurrence_count,
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

        pooler = [p for n, p in self.model.named_parameters() if n != "E.weight"]
        self.opt = torch.optim.Adam(pooler, lr=float(config.lr))
        self.opt_embedding = torch.optim.SparseAdam([self.model.E.weight], lr=float(config.lr))

    def ingest_full_graph(self, graph: SplitData) -> None:
        """Ingest every split into Tempest once."""
        self.walk_gen.add_edges(graph.sources, graph.destinations, graph.timestamps, graph.edge_feat)
        print(f"  Ingested full graph into Tempest: {len(graph.sources):,} edges "
              f"(once; per-query cutoff enforces causality)")

    def _long(self, array: np.ndarray) -> torch.Tensor:
        """Numpy array -> int64 tensor on the model's device."""
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.int64)).to(self.device)

    def _walk_tokens(self, seeds: torch.Tensor, cutoffs: torch.Tensor) -> WalkTokens:
        """K backward walks per seed, each cut off at its own time."""
        return build_query_walk_tokens(
            self.walk_gen, self.device, seeds, cutoffs,
            max_walk_len=self.config.max_walk_len,
            num_walks_per_node=self.config.num_walks_per_node,
            start_bias=self.config.start_bias,
            walk_bias=self.config.walk_bias)

    def _score(self, batch: Batch, candidates: np.ndarray) -> torch.Tensor:
        """Logits [B, C]; source and candidates walk with their query's cutoff."""
        src = self._long(batch.src)                                          # [B]
        t_query = self._long(batch.ts)                                       # [B]
        cand = self._long(candidates)                                        # [B, C]

        b, c = cand.shape
        cand_cutoffs = t_query.unsqueeze(1).expand(b, c).reshape(-1)         # [B*C]

        src_tokens = self._walk_tokens(src, t_query)
        cand_tokens = self._walk_tokens(cand.reshape(-1), cand_cutoffs)
        return self.model(src_tokens, cand_tokens)

    def _train_step(self, batch: Batch) -> float:
        """One optimiser step; returns the batch loss."""
        neg_tgt = self.neg_sampler_train.sample(batch)                       # [B, K]
        candidates = np.concatenate(
            [batch.tgt.astype(np.int64)[:, None], neg_tgt.astype(np.int64)], axis=1)  # [B, 1+K]

        logits = self._score(batch, candidates)                              # [B, 1+K]
        target = torch.zeros(len(batch.src), dtype=torch.long, device=self.device)  # positive first
        link_loss = F.cross_entropy(logits, target)

        self.opt.zero_grad(set_to_none=True)
        self.opt_embedding.zero_grad(set_to_none=True)
        link_loss.backward()
        self.opt.step()
        self.opt_embedding.step()

        return float(link_loss.detach())

    @torch.no_grad()
    def _probe(self) -> str:
        """Mean and max row norm of E and the recurrence mix weight, for the epoch line."""
        r = dist0(self.model.node_embeddings)
        line = f"  r_mean={float(r.mean()):.3f}  r_max={float(r.max()):.3f}"
        if self.config.use_recurrence_count:
            line += f"  mix={float(torch.sigmoid(self.model.mix_logit)):.4f}"
        return line

    @staticmethod
    def _eval_candidates(batch: Batch, neg_tgt_list: List[np.ndarray]) -> np.ndarray:
        """[B, 1+max_K] candidates, positive first; ragged rows padded with the positive."""
        max_k = max((len(neg) for neg in neg_tgt_list), default=0)
        candidates = np.tile(batch.tgt.astype(np.int64)[:, None], (1, 1 + max_k))
        for i, neg in enumerate(neg_tgt_list):
            candidates[i, 1:1 + len(neg)] = neg
        return candidates

    @torch.no_grad()
    def _eval(self, evaluator: Evaluator, batches: Iterable[Batch]) -> float:
        """Mean of the suite's per-positive metric over `batches`."""
        self.model.eval()
        evaluator.reset()                                                    # same negatives every pass
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

    def _snapshot(self) -> Dict[str, torch.Tensor]:
        """CPU copy of the model state."""
        return {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def _restore(self, snap: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(snap)

    def train(
        self,
        train_batches_factory,
        full_graph: SplitData,
        val_evaluator: Optional[Evaluator] = None,
        val_batches_factory=None,
        test_evaluator: Optional[Evaluator] = None,
        test_batches_factory=None,
    ) -> Dict[str, Any]:
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
            "best_epoch": best_epoch,                                        # -1 if nothing selected
            "epochs_run": len(per_epoch_val) if per_epoch_val else n_epochs,
            "best_val_mrr": best_val,
            "best_test_mrr": best_test,
            "per_epoch_val_mrr": per_epoch_val,
            "new_best_test_mrr": new_best_test,                              # new-best epochs only
        }
