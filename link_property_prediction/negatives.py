"""Negative samplers: uniform, CRAFT's collision-free rule, and TGB's hist_rnd."""

import abc
from typing import Optional

import numpy as np

from .data import Batch, SplitData


class NegativeSampler(abc.ABC):
    """Samplers expose `sample(batch) -> [B, K]` and `rule`, a name for the draw."""

    #: Short name of the sampling rule, printed in the negatives notice.
    rule: str

    @abc.abstractmethod
    def sample(self, batch: Batch) -> np.ndarray:
        """[B, K] negative destinations for `batch`."""


class UniformNegativeSampler(NegativeSampler):
    """Random destinations from `dst_pool`."""

    rule = "uniform"

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.num_neg_per_pos = num_neg_per_pos
        self.dst_pool = np.asarray(dst_pool, dtype=np.int32)
        self.rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k={self.num_neg_per_pos}, pool={len(self.dst_pool):,})"

    def sample(self, batch: Batch) -> np.ndarray:
        """Draws with replacement from `dst_pool`; no source array is materialised."""
        B = len(batch.src)
        idx = self.rng.integers(0, len(self.dst_pool), (B, self.num_neg_per_pos))
        return self.dst_pool[idx]


class SameTimeDestinations:
    """The destinations each source links to at exactly each timestamp of `edges`."""

    def __init__(self, edges: SplitData, num_nodes: int):
        self._times = np.unique(edges.timestamps)
        self._num_nodes = int(num_nodes)
        if self._num_nodes ** 2 * len(self._times) >= np.iinfo(np.int64).max:
            raise ValueError("(source, time, destination) keys overflow int64")
        pairs, _ = self._pairs(edges.sources, edges.timestamps)
        self._keys = np.unique(pairs * self._num_nodes + edges.destinations)
        self._pair_keys, self._pair_counts = np.unique(self._keys // self._num_nodes,
                                                       return_counts=True)

    def _pairs(self, src: np.ndarray, ts: np.ndarray):
        rank = np.minimum(np.searchsorted(self._times, ts), len(self._times) - 1)
        found = self._times[rank] == ts
        return src.astype(np.int64) * len(self._times) + rank, found

    def contains(self, src: np.ndarray, ts: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """[B, K] bool: dst[i, j] is a destination of src[i] at ts[i]."""
        pairs, found = self._pairs(src, ts)
        keys = pairs[:, None] * self._num_nodes + dst
        i = np.minimum(np.searchsorted(self._keys, keys), len(self._keys) - 1)
        return found[:, None] & (self._keys[i] == keys)

    def count(self, src: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """[B] int: how many destinations src[i] links to at ts[i]."""
        pairs, found = self._pairs(src, ts)
        i = np.minimum(np.searchsorted(self._pair_keys, pairs), len(self._pair_keys) - 1)
        hit = found & (self._pair_keys[i] == pairs)
        return np.where(hit, self._pair_counts[i], 0)


def _repeats(neg: np.ndarray) -> np.ndarray:
    """[B, K] bool: True where an entry equals an earlier entry of its row."""
    order = np.argsort(neg, axis=1, kind="stable")
    ordered = np.take_along_axis(neg, order, axis=1)
    repeat_ordered = np.zeros(neg.shape, dtype=bool)
    repeat_ordered[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
    repeat = np.empty_like(repeat_ordered)
    np.put_along_axis(repeat, order, repeat_ordered, axis=1)
    return repeat


class CollisionFreeNegativeSampler(UniformNegativeSampler):
    """CRAFT's DyGLib rule: no same-timestamp destination, no repeats within a row."""

    rule = "uniform collision-free"

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        edges: SplitData,
        seed: Optional[int] = None,
    ):
        super().__init__(num_neg_per_pos, dst_pool, seed)
        num_nodes = max(self.dst_pool.max(), edges.sources.max(), edges.destinations.max()) + 1
        self.same_time = SameTimeDestinations(edges, num_nodes)

    def sample(self, batch: Batch) -> np.ndarray:
        neg = super().sample(batch)
        src, ts = batch.src, batch.ts
        distinct = len(self.dst_pool) - self.same_time.count(src, ts) >= self.num_neg_per_pos
        rows = np.arange(len(neg))
        while len(rows):
            bad = self.same_time.contains(src[rows], ts[rows], neg[rows])
            bad |= _repeats(neg[rows]) & distinct[rows, None]
            hit = bad.any(axis=1)
            rows, bad = rows[hit], bad[hit]
            redrawn = neg[rows]
            redrawn[bad] = self.dst_pool[self.rng.integers(0, len(self.dst_pool), int(bad.sum()))]
            neg[rows] = redrawn
        return neg


class PriorDestinations:
    """Each source's distinct destinations, keyed by the time it first met them."""

    def __init__(self, edges: SplitData, num_nodes: int):
        self._n = int(num_nodes)
        self._times = np.unique(edges.timestamps)
        if self._n * len(self._times) >= np.iinfo(np.int64).max:
            raise ValueError("(source, time) keys overflow int64")
        key = edges.sources.astype(np.int64) * self._n + edges.destinations
        order = np.lexsort((edges.timestamps, key))
        key_s, ts_s = key[order], edges.timestamps[order]
        first = np.ones(key_s.shape, dtype=bool)
        np.not_equal(key_s[1:], key_s[:-1], out=first[1:])              # first time src met dst
        src_u = key_s[first] // self._n
        rank_u = np.searchsorted(self._times, ts_s[first])
        regroup = np.lexsort((rank_u, src_u))
        self._dst = (key_s[first] % self._n)[regroup].astype(np.int32)  # [U] by (src, time)
        self._ckey = (src_u * len(self._times) + rank_u)[regroup]       # [U] ascending
        self._start = np.searchsorted(src_u[regroup], np.arange(self._n + 1))

    def count(self, src: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """[B] int: distinct destinations src[i] met strictly before ts[i]."""
        rank = np.searchsorted(self._times, ts)                         # strict: t_first < ts
        pos = np.searchsorted(self._ckey, src.astype(np.int64) * len(self._times) + rank)
        return pos - self._start[src]

    def at(self, src: np.ndarray, i: np.ndarray) -> np.ndarray:
        """[B] the i-th oldest prior destination of src, for i < count(src, ts)."""
        return self._dst[self._start[src] + i]


class HistRndNegativeSampler(UniformNegativeSampler):
    """TGB's rule: each slot is one of the source's prior destinations with prob `ratio`."""

    rule = "hist_rnd"

    def __init__(
        self,
        num_neg_per_pos: int,
        dst_pool: np.ndarray,
        edges: SplitData,
        ratio: float,
        seed: Optional[int] = None,
    ):
        super().__init__(num_neg_per_pos, dst_pool, seed)
        num_nodes = max(self.dst_pool.max(), edges.sources.max(), edges.destinations.max()) + 1
        self.prior = PriorDestinations(edges, num_nodes)
        self.ratio = float(ratio)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}(k={self.num_neg_per_pos}, "
                f"pool={len(self.dst_pool):,}, ratio={self.ratio:.4f})")

    def sample(self, batch: Batch) -> np.ndarray:
        """Per-slot Bernoulli, so the composition holds at any k_train."""
        neg = super().sample(batch)                                     # [B, K] uniform
        have = self.prior.count(batch.src, batch.ts)                    # [B]
        take = (self.rng.random(neg.shape) < self.ratio) & (have > 0)[:, None]
        r, c = np.nonzero(take)
        if len(r):
            pick = (self.rng.random(len(r)) * have[r]).astype(np.int64)
            drawn = self.prior.at(batch.src[r], pick)
            keep = drawn != batch.tgt[r]                                 # never the query's own positive
            neg[r[keep], c[keep]] = drawn[keep]
        return neg
