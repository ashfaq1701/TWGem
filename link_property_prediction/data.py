"""Suite-agnostic containers (SplitData, Batch, Loaded) and the chronological batch iterator."""

from typing import Iterator, NamedTuple, Optional

import numpy as np


class SplitData(NamedTuple):
    sources: np.ndarray         # [E] int64
    destinations: np.ndarray    # [E] int64
    timestamps: np.ndarray      # [E] int64
    edge_feat: Optional[np.ndarray]   # [E, d_edge] float32 or None

    def mask(self, keep: np.ndarray) -> "SplitData":
        """The sub-split selected by `keep` (cast to bool)."""
        m = np.asarray(keep, dtype=bool)
        return SplitData(
            sources=self.sources[m],
            destinations=self.destinations[m],
            timestamps=self.timestamps[m],
            edge_feat=self.edge_feat[m] if self.edge_feat is not None else None,
        )


class Batch(NamedTuple):
    """One contiguous chunk of a `SplitData`, under the query vocabulary the scorer uses."""
    src: np.ndarray             # [B] int64
    tgt: np.ndarray             # [B] int64 — the positive destination
    ts: np.ndarray              # [B] int64 — each query's cutoff
    edge_feat: Optional[np.ndarray]   # [B, d_edge] float32 or None


class Loaded(NamedTuple):
    train: SplitData
    val: SplitData
    test: SplitData
    dataset: object             # live suite dataset handle (negatives + eval)
    name: str                   # dataset name (for the Evaluator)
    max_node_count: int

    @property
    def T_train(self) -> int:
        """Length of the training period: latest minus earliest train-edge timestamp."""
        ts = self.train.timestamps
        return int(ts.max() - ts.min())


def node_count(*id_arrays: np.ndarray) -> int:
    """Embedding-table size: largest raw id plus one."""
    return int(max(a.max() for a in id_arrays)) + 1


def concat_splits(*splits: SplitData) -> SplitData:
    """Concatenate splits for a one-shot ingest; edge_feat only if every split has it."""
    src = np.concatenate([s.sources for s in splits])
    dst = np.concatenate([s.destinations for s in splits])
    ts = np.concatenate([s.timestamps for s in splits])
    efs = [s.edge_feat for s in splits]
    ef = np.concatenate(efs) if all(e is not None for e in efs) else None
    return SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=ef)


def create_batches(split: SplitData, batch_size: int) -> Iterator[Batch]:
    """Consecutive fixed-size chronological batches; the last partial batch is kept."""
    n = int(split.sources.shape[0])
    for start in range(0, n, batch_size):
        yield _slice(split, start, min(start + batch_size, n))


def _slice(split: SplitData, start: int, end: int) -> Batch:
    ef = split.edge_feat[start:end] if split.edge_feat is not None else None
    return Batch(
        src=split.sources[start:end],
        tgt=split.destinations[start:end],
        ts=split.timestamps[start:end],
        edge_feat=ef,
    )
