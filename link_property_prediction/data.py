"""Suite-agnostic data containers (`SplitData`, `Batch`, `Loaded`) and the fixed-size
chronological batch iterator. Suite modules do the loading."""

from typing import Iterator, NamedTuple, Optional

import numpy as np


class SplitData(NamedTuple):
    sources: np.ndarray         # [E] int64
    destinations: np.ndarray    # [E] int64
    timestamps: np.ndarray      # [E] int64
    edge_feat: Optional[np.ndarray]   # [E, d_edge] float32 or None

    def mask(self, keep: np.ndarray) -> "SplitData":
        """The sub-split `keep` selects, as every loader needs: build one `SplitData` over
        all of a dataset's edges, then carve the three splits out of it. `keep` is cast to
        bool, since a suite's shipped masks need not already be, and `edge_feat` stays None
        when the dataset has none -- the branch each loader used to carry itself."""
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
    """Embedding-table size for ids that index it directly: one row per id from 0, so the
    largest id plus one. Loaders pass their raw source and destination arrays -- before the
    split masks, so an id held out of training still gets its row."""
    return int(max(a.max() for a in id_arrays)) + 1


def concat_splits(*splits: SplitData) -> SplitData:
    """Concatenate splits into ONE SplitData for a one-shot ingest. edge_feat is
    concatenated only when every split has it (else None)."""
    src = np.concatenate([s.sources for s in splits])
    dst = np.concatenate([s.destinations for s in splits])
    ts = np.concatenate([s.timestamps for s in splits])
    efs = [s.edge_feat for s in splits]
    ef = np.concatenate(efs) if all(e is not None for e in efs) else None
    return SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=ef)


def create_batches(split: SplitData, batch_size: int) -> Iterator[Batch]:
    """Fixed-size chronological batches (train and eval): consecutive
    `batch_size` chunks over the time-sorted stream. The final partial batch is kept
    (drop_last=False). Batch boundaries carry no causal meaning: both the link-prediction
    trainer and the node-classification encoder ingest the full graph once and give every
    query its own EXCLUSIVE cutoff (t_edge < t), so same-timestamp edges never see each
    other wherever the boundary falls."""
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
