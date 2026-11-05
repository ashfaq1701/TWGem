import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .walk_tokens import WalkTokens

_SQ_DIST_FLOOR = 1e-12


def dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis; the floor keeps the gradient finite at x == y."""
    return ((x - y) ** 2).sum(-1).clamp_min(_SQ_DIST_FLOOR).sqrt()


def weighted_midpoint(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean of points x [..., T, d] under weights w [..., T] summing to 1."""
    return (w.unsqueeze(-1) * x).sum(-2)                                     # [..., d]


class BagWeights(nn.Module):

    def __init__(self, E: nn.Embedding, T_train: int, max_walk_len: int,
                 hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
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
        """The bag's token ids and validity [Q, T]. A cold query (no valid token) gets its own
        seed in slot 0, so every bag pools at least one point."""
        nodes = tokens.nodes.flatten(1).clamp_min(0).clone()                 # [Q, T]
        valid = tokens.mask.flatten(1).clone()                               # [Q, T]
        cold = ~valid.any(dim=-1)                                            # [Q]
        if bool(cold.any()):
            nodes[cold, 0] = tokens.seeds[cold]
            valid[cold, 0] = True
        return nodes, valid

    def _token_features(self, tokens: WalkTokens, fixed: torch.Tensor,
                        valid_f: torch.Tensor) -> torch.Tensor:
        """Per-token features [Q, T, 4]: age, hop, log distance to the bag's centre, log radius.
        `fixed` is the detached token points, so the features see positions only."""
        mid = weighted_midpoint(fixed, valid_f / valid_f.sum(-1, keepdim=True))  # [Q, d]

        age = tokens.ages.flatten(1).clamp_min(0).to(fixed.dtype)            # [Q, T]
        pos = tokens.positions.flatten(1).to(fixed.dtype)                    # [Q, T]
        age = torch.log1p(age) / self.log_T_train                            # [Q, T]
        pos = pos / self.max_walk_len                                        # [Q, T]

        # Mask after the log: log of a padded token's 0 is -inf, and 0 * nan poisons net's backward.
        # `dist` already floors, so its log needs no clamp of its own.
        d_mid = torch.log(dist(fixed, mid.unsqueeze(-2))) * valid_f          # [Q, T]
        r_tok = fixed.norm(dim=-1).clamp_min(_SQ_DIST_FLOOR ** 0.5)          # [Q, T]
        d0 = torch.log(r_tok) * valid_f                                      # [Q, T]

        return torch.stack([age, pos, d_mid, d0], dim=-1)                    # [Q, T, 4]

    def forward(self, tokens: WalkTokens) -> torch.Tensor:
        """The bag's pooled point: the weighted midpoint of its tokens [Q, d]."""
        nodes, valid = self._bag_nodes(tokens)                               # [Q, T]
        points = self.E(nodes)                                               # [Q, T, d]
        # Features see positions only; E's gradient flows through `points`.
        fixed = points.detach()                                              # [Q, T, d]
        valid_f = valid.to(fixed.dtype)                                      # [Q, T]

        feats = self._token_features(tokens, fixed, valid_f)                 # [Q, T, 4]
        logits = self.net(feats).squeeze(-1)                                 # [Q, T]
        log_weights = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)  # [Q, T]
        return weighted_midpoint(points, log_weights.exp())                  # [Q, d]


class LinkPredHead(nn.Module):

    def __init__(self, num_nodes: int, T_train: int, max_walk_len: int, d_emb: int,
                 hidden_dim: int = 32, n_layers_pooler: int = 2, seed: int = 42):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.T_train = int(T_train)
        self.max_walk_len = int(max_walk_len)
        self.d_emb = int(d_emb)
        torch.manual_seed(seed)

        # N(0, 1/d): row norm ~1 at any width (Flax / fairseq embedding init).
        self.E = nn.Embedding(self.num_nodes, self.d_emb, sparse=True)      # sparse grad: SparseAdam
        nn.init.normal_(self.E.weight, std=self.d_emb ** -0.5)

        self.bag_weights = BagWeights(self.E, self.T_train, self.max_walk_len,
                                      hidden_dim=hidden_dim, n_layers=n_layers_pooler)

    def mean_radius(self) -> torch.Tensor:
        """Mean norm of the table (its distance from the origin), detached. The pooler no longer
        reads it; the node-classification encoder does."""
        r_mean = self.E.weight.detach().norm(dim=-1).mean()                  # []
        return r_mean.clamp_min(torch.finfo(r_mean.dtype).tiny)

    def forward(self, src_tokens: WalkTokens, cand_tokens: WalkTokens) -> torch.Tensor:
        """Score candidates by the negated distance between the two pooled points, divided by the
        spread of that query's own scores. Standardising over the candidate axis -- the softmax's
        own axis -- leaves the score invariant to any affine change of the row's distances, so the
        table's scale cannot buy sharpness and no temperature is needed. Being a positive affine
        map within one softmax, it cannot reorder a row. Centring is inert in the loss but bounds
        every logit by sqrt(c - 1), which dividing by the spread alone does not.

        The score is the NEGATED LOG distance. Standardisation already makes either form scale
        free, so the log is not buying invariance here -- it decides how the row's fixed budget
        (mean 0, std 1) is spread across the candidates. Raw distance spends that budget on the
        far negatives: one distant candidate takes the whole spread and compresses the near
        cluster, where the ranking decision actually is. The log compresses the far tail instead
        and spends the range near the query. Monotone decreasing in d, so within a row the order
        -- and so eval MRR for a fixed model -- is unchanged.

        The standardisation is `F.layer_norm` over the candidate axis, which IS this operation:
        centre, divide by the spread. Functional and sized from the input, because the candidate
        count changes -- 1 + k_train while training, 1 + k_eval at eval, and TGB serves a
        different count per dataset. `nn.LayerNorm(c)` would both fail on that shape change and,
        with its default affine, LEAK THE LABEL: its gamma/beta are per candidate POSITION and
        the positive is always index 0 (see the target in _train_step), so it can rank by
        position alone. Verified: nn.LayerNorm(6) trained on pure noise against that target
        reaches loss 0.0176 against log 6 = 1.7918, by learning beta = [+2.8, -2.8, ...]."""
        pooled_u = self.bag_weights(src_tokens)                              # [b, d]
        pooled_v = self.bag_weights(cand_tokens)                             # [b*c, d]
        pooled_v = pooled_v.view(pooled_u.shape[0], -1, pooled_u.shape[-1])  # [b, c, d]
        s = -torch.log(dist(pooled_u.unsqueeze(1), pooled_v))                # [b, c]
        return F.layer_norm(s, s.shape[-1:])                                 # [b, c]
