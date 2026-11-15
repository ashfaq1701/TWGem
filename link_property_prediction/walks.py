"""Tempest wrapper: K backward walks per seed, cut off EXCLUSIVELY at each seed's time."""
from typing import NamedTuple, Optional

import numpy as np
import torch
from tempest import Tempest


class WalkData(NamedTuple):
    nodes: torch.Tensor        # [N*K, L] int64, padding = -1
    timestamps: torch.Tensor   # [N*K, L] int64, edge (p, p+1); seed slot INT64_MAX; padding -1
    lens: torch.Tensor         # [N*K] int64
    K: int                     # walks per seed
    edge_feats: Optional[torch.Tensor] = None   # [N*K, L, d_ef]; seed slot and padding zero


class WalkGenerator:
    def __init__(
        self,
        use_gpu: bool = False,
        walk_bias: str = "ExponentialWeight",
        start_bias: str = "ExponentialWeight",
        num_walks_per_node: int = 5,
        max_walk_len: int = 20,
        timescale_bound: int = 300,
        max_time_capacity: int = -1,
        temporal_node2vec_p: float = 4.0,
        temporal_node2vec_q: float = 0.25,
        seed: Optional[int] = None,
    ):
        enable_n2v = "TemporalNode2Vec" in (walk_bias, start_bias)
        self.tempest = Tempest(
            is_directed=False,
            use_gpu=use_gpu,
            enable_weight_computation=True,
            enable_temporal_node2vec=enable_n2v,
            temporal_node2vec_p=temporal_node2vec_p,
            temporal_node2vec_q=temporal_node2vec_q,
            timescale_bound=timescale_bound,
            max_time_capacity=max_time_capacity,
            shuffle_walk_order=False,
            global_seed=(None if seed is None else int(seed)),
        )
        self.walk_bias = walk_bias
        self.start_bias = start_bias
        self.num_walks_per_node = int(num_walks_per_node)
        self.max_walk_len = int(max_walk_len)

    def add_edges(self, src: np.ndarray, tgt: np.ndarray, ts: np.ndarray,
                  edge_feat: Optional[np.ndarray] = None) -> None:
        """Ingest edges into Tempest; order is irrelevant."""
        self.tempest.add_multiple_edges(src, tgt, ts, edge_features=edge_feat)

    def get_latest_events(self, nodes: np.ndarray, cutoff_times: np.ndarray):
        """(partner, t_last) of each node's latest edge before its cutoff; -1 if none."""
        node_arr = np.ascontiguousarray(nodes, dtype=np.int32)
        cutoff_arr = np.ascontiguousarray(cutoff_times, dtype=np.int64)
        partners, t_last = self.tempest.get_latest_events_for_nodes(
            node_arr, cutoff_times=cutoff_arr, direction="Backward_In_Time")
        return np.asarray(partners, dtype=np.int64), np.asarray(t_last, dtype=np.int64)

    def get_node_recency(self, nodes: np.ndarray, cutoff_times: np.ndarray) -> np.ndarray:
        """cutoff - t_last per node (cutoff + 1 when the node has no prior edge)."""
        _, t_last = self.get_latest_events(nodes, cutoff_times)
        return np.ascontiguousarray(cutoff_times, dtype=np.int64) - t_last

    def get_node_popularity(self, nodes: np.ndarray, cutoff_times: np.ndarray) -> np.ndarray:
        """Number of edges each node has before its cutoff."""
        node_arr = np.ascontiguousarray(nodes, dtype=np.int32)
        cutoff_arr = np.ascontiguousarray(cutoff_times, dtype=np.int64)
        return np.asarray(self.tempest.get_node_participation_counts(
            node_arr, cutoff_times=cutoff_arr, direction="Backward_In_Time"), dtype=np.int64)

    def get_recurrence_count(self, us: np.ndarray, vs: np.ndarray,
                             cutoff_times: np.ndarray) -> np.ndarray:
        """Number of edges between each pair (u, v) before its cutoff."""
        u_arr = np.ascontiguousarray(us, dtype=np.int32)
        v_arr = np.ascontiguousarray(vs, dtype=np.int32)
        cutoff_arr = np.ascontiguousarray(cutoff_times, dtype=np.int64)
        return np.asarray(self.tempest.get_pair_co_occurrence_counts(
            u_arr, v_arr, cutoff_times=cutoff_arr, direction="Backward_In_Time"), dtype=np.int64)

    def walks_for_nodes(self, seeds: np.ndarray, max_walk_len: Optional[int] = None,
                        num_walks_per_node: Optional[int] = None,
                        start_bias: Optional[str] = None,
                        walk_bias: Optional[str] = None,
                        cutoff_times: Optional[np.ndarray] = None) -> WalkData:
        """K walks per seed; rows [i*K, (i+1)*K) are seed i's, seed at lens-1, padding -1."""
        mwl = self.max_walk_len if max_walk_len is None else int(max_walk_len)
        nw = self.num_walks_per_node if num_walks_per_node is None else int(num_walks_per_node)
        sb = self.start_bias if start_bias is None else start_bias
        wb = self.walk_bias if walk_bias is None else walk_bias
        seed_arr = np.ascontiguousarray(seeds, dtype=np.int32)
        cutoff_arr = None
        if cutoff_times is not None:
            cutoff_arr = np.ascontiguousarray(cutoff_times, dtype=np.int64)
            if cutoff_arr.shape[0] != seed_arr.shape[0]:
                raise ValueError(
                    "cutoff_times must have the same length as seeds "
                    f"({cutoff_arr.shape[0]} vs {seed_arr.shape[0]})")
        nodes, ts, lens, ef = self.tempest.get_random_walks_and_times_for_nodes(
            seed_nodes=seed_arr,
            max_walk_len=mwl,
            walk_bias=wb,
            initial_edge_bias=sb,
            num_walks_per_node=nw,
            walk_direction="Backward_In_Time",
            cutoff_times=cutoff_arr,
        )
        nodes_t = torch.from_numpy(np.asarray(nodes).astype(np.int64))

        # Tempest's edge feats are one column short; right-pad to L.
        ef_arr = np.asarray(ef)
        edge_feats = None
        if ef_arr.ndim == 3 and ef_arr.size > 0:
            ef_t = torch.from_numpy(np.ascontiguousarray(ef_arr, dtype=np.float32))
            pad_cols = nodes_t.shape[1] - ef_t.shape[1]
            if pad_cols > 0:
                z = torch.zeros(ef_t.shape[0], pad_cols, ef_t.shape[2], dtype=ef_t.dtype)
                ef_t = torch.cat([ef_t, z], dim=1)            # [N*K, L, d_ef]
            edge_feats = ef_t

        return WalkData(
            nodes=nodes_t,
            timestamps=torch.from_numpy(np.asarray(ts).astype(np.int64)),
            lens=torch.from_numpy(np.asarray(lens).astype(np.int64)),
            K=nw,
            edge_feats=edge_feats,
        )
