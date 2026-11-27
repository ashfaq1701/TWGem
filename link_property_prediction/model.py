import math
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis, floored so the gradient stays finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def dist0(x: torch.Tensor) -> torch.Tensor:
    """Euclidean distance from the origin over the last axis, floored like dist."""
    return (x ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


FEATURES = ("age", "hop", "mid", "radius")


@dataclass(frozen=True)
class Ablation:
    """Ablation switches; every default is TWGem."""
    walk_sides: str = "both"              # both | source | none: which sides pool their walks
    pooling: str = "learned"              # learned | uniform
    features: Tuple[str, ...] = FEATURES  # controller inputs, a subset of FEATURES
    raw_geo_features: bool = False        # raw distances instead of log distances
    detach: bool = True                   # controller features read from detached rows
    score: str = "log-dist"               # log-dist | dist | dot
    standardise: bool = True              # per-query standardisation of the scores

    def __post_init__(self):
        if self.walk_sides not in ("both", "source", "none"):
            raise ValueError(f"walk_sides must be both, source or none, got {self.walk_sides}")
        if self.pooling not in ("learned", "uniform"):
            raise ValueError(f"pooling must be learned or uniform, got {self.pooling}")
        if not self.features or not set(self.features) <= set(FEATURES):
            raise ValueError(f"features must be a non-empty subset of {FEATURES}")
        if self.score not in ("log-dist", "dist", "dot"):
            raise ValueError(f"score must be log-dist, dist or dot, got {self.score}")

    @classmethod
    def from_args(cls, args: dict) -> "Ablation":
        """Build from CLI args or a checkpoint's saved args; anything missing keeps the TWGem default."""
        d = cls()
        feats = args.get("controller_features")
        return cls(walk_sides=args.get("walk_sides", d.walk_sides),
                   pooling=args.get("pooling", d.pooling),
                   features=tuple(feats.split(",")) if feats else d.features,
                   raw_geo_features=args.get("raw_geo_features", d.raw_geo_features),
                   detach=not args.get("no_detach", False),
                   score=args.get("score", d.score),
                   standardise=not args.get("no_standardise", False))


class Bag(NamedTuple):
    pooled: torch.Tensor   # [Q, d]     weighted midpoint of the bag
    mid: torch.Tensor      # [Q, d]     unweighted midpoint of the bag
    points: torch.Tensor   # [Q, T, d]
    weights: torch.Tensor  # [Q, T]     pooling weights, summing to 1


class BagWeights(nn.Module):

    def __init__(self, E: nn.Embedding, T_train: int, max_walk_len: int,
                 hidden_dim: int = 32, n_layers: int = 2, ablation: Ablation = Ablation()):
        super().__init__()
        self.E = E
        self.ablation = ablation
        self.feature_idx = [FEATURES.index(f) for f in ablation.features]
        self.log_T_train = math.log1p(T_train)
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = len(self.feature_idx)
        layers = [nn.Linear(self.n_feat, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

    def _bag_nodes(self, tokens: WalkTokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token ids and a validity mask [Q, T]; padding uses the spare row, an empty bag falls back to its seed."""
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        nodes = tokens.nodes.flatten(1).masked_fill(~valid, self.E.padding_idx)
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
        return nodes, valid

    def _token_features(self, tokens: WalkTokens, fixed: torch.Tensor,
                        valid_f: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        """Per-token features [Q, T, F]: age, hop, log distance to the bag centre, log radius."""
        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        d_mid = dist(fixed, mid.unsqueeze(-2))                               # [Q, T]
        r_tok = dist0(fixed)                                                 # [Q, T]
        if not self.ablation.raw_geo_features:
            d_mid, r_tok = torch.log(d_mid), torch.log(r_tok)
        d_mid = d_mid * valid_f                                              # [Q, T]
        d0 = r_tok * valid_f                                                 # [Q, T]

        feats = torch.stack([age, pos, d_mid, d0], dim=-1)                   # [Q, T, 4]
        return feats[..., self.feature_idx]                                  # [Q, T, F]

    def forward(self, tokens: WalkTokens) -> Bag:
        """The bag's pooled point, unweighted midpoint, points and pooling weights."""
        nodes, valid = self._bag_nodes(tokens)                               # [Q, T]
        points = self.E(nodes)                                               # [Q, T, d]
        fixed = points.detach() if self.ablation.detach else points          # [Q, T, d]
        valid_f = valid.to(fixed.dtype)                                      # [Q, T]
        mid = weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        if self.ablation.pooling == "uniform":
            weights = valid_f / valid_f.sum(-1, keepdim=True)                # [Q, T]
        else:
            feats = self._token_features(tokens, fixed, valid_f, mid)        # [Q, T, F]
            logits = self.net(feats).squeeze(-1)                             # [Q, T]
            weights = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        pooled = weighted_midpoint(points, weights)                          # [Q, d]
        return Bag(pooled=pooled, mid=mid, points=points, weights=weights)


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42,
                 ablation: Ablation = Ablation()):
        super().__init__()
        self.ablation = ablation
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes + 1, self.d_emb, sparse=True,
                              padding_idx=self.num_nodes)                    # last row: padding
        nn.init.normal_(self.E.weight, std=self.d_emb ** -0.5)              # row norm ~1
        with torch.no_grad():
            self.E.weight[self.num_nodes].zero_()                            # init overwrote it

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler,
                                      ablation=ablation)

    @property
    def node_embeddings(self) -> torch.Tensor:
        """The real-node rows of E [num_nodes, d]; the last row is padding and never trains."""
        return self.E.weight[:-1]

    @staticmethod
    def num_nodes_in(state: dict) -> int:
        """Real node count behind a saved E.weight, which carries one padding row."""
        return int(state["E.weight"].shape[0]) - 1

    def mean_radius(self) -> torch.Tensor:
        """Mean row norm of E, detached."""
        r_mean = dist0(self.node_embeddings.detach()).mean()                 # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def _point(self, tokens: WalkTokens, use_walks: bool) -> torch.Tensor:
        """The pooled point of each bag [Q, d], or the node's own row when walks are ablated."""
        return self.bag_weights(tokens).pooled if use_walks else self.E(tokens.seeds)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Logits [b, c]: -log distance between pooled points, standardised per query row."""
        ab = self.ablation
        pooled_u = self._point(src_tokens, ab.walk_sides != "none")          # [b, d]
        pooled_v = self._point(cand_tokens, ab.walk_sides == "both")         # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        pooled_u = pooled_u.unsqueeze(1)                                     # [b, 1, d]
        match ab.score:
            case "log-dist":
                s = -torch.log(dist(pooled_u, pooled_v))                     # [b, c]
            case "dist":
                s = -dist(pooled_u, pooled_v)                                # [b, c]
            case "dot":
                s = (pooled_u * pooled_v).sum(-1)                            # [b, c]
        return F.layer_norm(s, s.shape[-1:]) if ab.standardise else s        # [b, c]
