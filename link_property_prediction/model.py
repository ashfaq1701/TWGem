import math
from typing import NamedTuple, Tuple

import geoopt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lorentz import LorentzManifold
from .walk_tokens import WalkTokens

_DIST_FLOOR = 1e-6  # floor under every log of a distance


class Bag(NamedTuple):
    pooled: torch.Tensor   # [Q, d]     weighted midpoint of the bag
    mid: torch.Tensor      # [Q, d]     unweighted midpoint of the bag
    points: torch.Tensor   # [Q, T, d]
    weights: torch.Tensor  # [Q, T]     pooling weights, summing to 1


class BagWeights(nn.Module):

    def __init__(self, geom: LorentzManifold, E: nn.Embedding, T_train: int, max_walk_len: int,
                 hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.geom = geom
        self.E = E
        self.log_T_train = math.log1p(T_train)
        self.max_walk_len = int(max_walk_len)
        self.hidden = int(hidden_dim)
        self.n_layers = int(n_layers)
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.n_feat = 4
        layers = [nn.Linear(self.n_feat, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()]
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.hidden, self.hidden), nn.LayerNorm(self.hidden), nn.GELU()]
        layers.append(nn.Linear(self.hidden, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _bag_nodes(tokens: WalkTokens) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token ids and validity [Q, T]; a cold query pools its own seed."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
        return nodes, valid

    def _token_features(self, tokens: WalkTokens, fixed: torch.Tensor,
                        valid_f: torch.Tensor, mid: torch.Tensor) -> torch.Tensor:
        """Per-token features [Q, T, 4]: age, hop, log distance to the bag centre, log radius."""
        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        d_tok_mid = self.geom.dist(fixed, mid.unsqueeze(-2))                 # [Q, T]
        d_mid = torch.log(d_tok_mid.clamp_min(_DIST_FLOOR)) * valid_f        # [Q, T]
        r_tok = self.geom.dist0(fixed)                                       # [Q, T]
        d0 = torch.log(r_tok.clamp_min(_DIST_FLOOR)) * valid_f               # [Q, T]

        return torch.stack([age, pos, d_mid, d0], dim=-1)                    # [Q, T, 4]

    def forward(self, tokens: WalkTokens) -> Bag:
        """The bag's pooled point, unweighted midpoint, points and pooling weights."""
        nodes, valid = self._bag_nodes(tokens)                               # [Q, T]
        points = self.E(nodes)                                               # [Q, T, d]
        fixed = points.detach()                                              # [Q, T, d]  features only
        valid_f = valid.to(fixed.dtype)                                      # [Q, T]
        mid = self.geom.weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        feats = self._token_features(tokens, fixed, valid_f, mid)            # [Q, T, 4]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        weights = log_weights.exp()                                          # [Q, T]
        pooled = self.geom.weighted_midpoint(points, weights)                # [Q, d]
        return Bag(pooled=pooled, mid=mid, points=points, weights=weights)


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        self.geom = LorentzManifold()
        torch.manual_seed(seed)

        self.E = nn.Embedding(self.num_nodes, self.d_emb, sparse=True)
        with torch.no_grad():
            init = self.geom.random(self.num_nodes, self.d_emb)              # geodesic radius ~1
        self.E.weight = geoopt.ManifoldParameter(init, manifold=self.geom)

        self.bag_weights = BagWeights(self.geom, self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

    def mean_radius(self) -> torch.Tensor:
        """Mean geodesic radius of E, detached."""
        r_mean = self.geom.dist0(self.E.weight.detach()).mean()              # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Logits [b, c]: -log distance between pooled points, standardised per query row."""
        pooled_u = self.bag_weights(src_tokens).pooled                       # [b, d]
        pooled_v = self.bag_weights(cand_tokens).pooled                      # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        d = self.geom.dist(pooled_u.unsqueeze(1), pooled_v)                  # [b, c]
        s = -torch.log(d.clamp_min(_DIST_FLOOR))                             # [b, c]
        return F.layer_norm(s, s.shape[-1:])                                 # [b, c]
