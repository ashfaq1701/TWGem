"""Turn each query's backward walks into [Q, K, L] tensors."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

_TS_SENTINEL = torch.iinfo(torch.int64).max     # Tempest's seed-slot timestamp


@dataclass
class WalkTokens:
    """Each query's K walks, one [Q, K, L] tensor per field."""

    seeds: torch.Tensor                             # [Q]  the query node
    cutoffs: torch.Tensor                           # [Q]  the query time
    nodes: torch.Tensor                             # [Q, K, L]  node ids, -1 for padding
    ages: torch.Tensor                              # [Q, K, L]  how long ago each edge was; 0 at the seed, -1 for padding
    positions: torch.Tensor                         # [Q, K, L]  hops from the seed, 1 at the seed, 0 for padding
    mask: torch.Tensor                              # [Q, K, L]  True for real tokens, seed included
    edge_features: Optional[torch.Tensor] = None    # [Q, K, L, d_ef]  zero at the seed and padding; None without edge features


def build_query_walk_tokens(
    walk_gen,
    device: torch.device,
    query_seeds: torch.Tensor,
    query_cutoffs: torch.Tensor,
    *,
    max_walk_len: int,
    num_walks_per_node: int,
    start_bias: Optional[str] = None,
    walk_bias: Optional[str] = None,
) -> WalkTokens:
    """Per-query backward walks -> [Q, K, L] WalkTokens."""
    seeds_t = query_seeds.detach().to(device=device, dtype=torch.long)        # [Q]
    cutoffs_t = query_cutoffs.detach().to(device=device, dtype=torch.long)    # [Q]
    q = int(seeds_t.shape[0])

    if q == 0:
        k, length = num_walks_per_node, max_walk_len
        empty_i = torch.empty((0, k, length), dtype=torch.int64, device=device)
        empty_b = torch.empty((0, k, length), dtype=torch.bool, device=device)
        return WalkTokens(seeds_t, cutoffs_t, empty_i, empty_i, empty_i, empty_b)

    wd = walk_gen.walks_for_nodes(
        np.ascontiguousarray(seeds_t.cpu().numpy(), dtype=np.int32),
        max_walk_len=max_walk_len,
        num_walks_per_node=num_walks_per_node,
        start_bias=start_bias,
        walk_bias=walk_bias,
        cutoff_times=np.ascontiguousarray(cutoffs_t.cpu().numpy(), dtype=np.int64),
    )
    k, length = int(wd.K), int(wd.nodes.shape[1])

    # Rows [q*K, (q+1)*K) are query q's walks.
    nodes = wd.nodes.to(device=device, dtype=torch.int64).reshape(q, k, length)        # [Q, K, L]
    node_mask = nodes != -1

    # Age = cutoff - t_edge: seed 0, padding -1.
    ts = wd.timestamps.to(device=device, dtype=torch.int64).reshape(q, k, length)
    edge_ts = torch.where(ts == _TS_SENTINEL, cutoffs_t.view(q, 1, 1), ts)
    ages = cutoffs_t.view(q, 1, 1) - edge_ts
    ages = torch.where(node_mask, ages, torch.full_like(ages, -1))

    # Hop from the seed (walks are stored oldest first); padding 0.
    lens = node_mask.sum(dim=-1, keepdim=True)                                          # [Q, K, 1]
    arange = torch.arange(length, device=device).view(1, 1, length)
    positions = (lens - arange).clamp_min(0)

    # Edge features, zeroed at the seed slot and padding.
    edge_features = None
    if wd.edge_feats is not None:
        d_ef = int(wd.edge_feats.shape[-1])
        ef = wd.edge_feats.to(device=device, dtype=torch.float32).reshape(q, k, length, d_ef)
        real = (node_mask & (ages != 0)).unsqueeze(-1)
        edge_features = ef * real                                                       # [Q, K, L, d_ef]

    return WalkTokens(
        seeds_t,
        cutoffs_t,
        nodes,                              # [Q, K, L]
        ages,
        positions,
        node_mask,
        edge_features,
    )
