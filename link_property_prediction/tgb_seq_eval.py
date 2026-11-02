"""TGB-Seq (`tgb_seq.LinkPred`) — native load, negatives, and MRR metric.

VAL negatives are built once by `DataSuite.eval_negatives` (uniform) at `k_eval` over the
full dataset's candidates (fixed & seeded); TEST uses the shipped `test_ns`
verbatim. Timestamps are cast to monotone int64 for Tempest's per-query cutoff.
"""
import numpy as np

from .data import Loaded, SplitData
from .evaluator import DataSuite, Evaluator, FixedNegativeEvaluator


def _to_int64_time(time: np.ndarray) -> np.ndarray:
    """Monotone int64 timestamps for Tempest cutoffs. Integral floats cast
    directly; non-integral inputs fall back to an order-preserving dense rank."""
    t = np.asarray(time)
    if np.isfinite(t).all() and np.array_equal(t, np.floor(t)):
        return t.astype(np.int64)
    order = np.unique(t)                                  # sorted unique values
    return np.searchsorted(order, t).astype(np.int64)     # dense rank, monotone


def load_tgb_seq(name: str, root: str = "datasets") -> Loaded:
    """Native TGB-Seq load → suite-agnostic `Loaded`. Splits from the CSV's
    `split` column (0=train, 1=val, 2=test); live `TGBSeqLoader` kept on
    `Loaded.dataset` so the test evaluator can read its shipped negatives."""
    import os

    from tgb_seq.LinkPred.dataloader import TGBSeqLoader

    ds = TGBSeqLoader(name=name, root=root)
    # Self-heal a half-downloaded dataset: if either the edgelist CSV or test_ns is missing, re-download.
    if not (os.path.exists(ds._edgelist_path) and os.path.exists(ds._test_ns_path)):
        ds._download()
        ds._load_file()

    src = np.asarray(ds.src_node_ids, dtype=np.int64)
    dst = np.asarray(ds.dst_node_ids, dtype=np.int64)
    ts = _to_int64_time(ds.node_interact_times)

    edge_feat = ds.edge_features
    edge_feat = np.asarray(edge_feat, dtype=np.float32) if edge_feat is not None else None

    def _split(mask: np.ndarray) -> SplitData:
        m = np.asarray(mask, dtype=bool)
        ef = edge_feat[m] if edge_feat is not None else None
        return SplitData(sources=src[m], destinations=dst[m], timestamps=ts[m], edge_feat=ef)

    return Loaded(
        train=_split(ds.train_mask),
        val=_split(ds.val_mask),
        test=_split(ds.test_mask),
        dataset=ds,
        name=name,
        max_node_count=int(max(src.max(), dst.max())) + 1,
    )


class TGBSeqSuite(DataSuite):
    """TGB-Seq adapter. VAL negatives are built once from our sampler at
    `k_eval`; TEST uses TGB-Seq's shipped `test_ns` verbatim."""

    def _load(self) -> Loaded:
        return load_tgb_seq(self.name, self.root)

    def make_evaluator(self, split_mode: str) -> Evaluator:
        loaded = self.load()
        if split_mode == "val":
            neg = self.eval_negatives(loaded.val, self.seed, tag="[tgb-seq] val")
            return FixedNegativeEvaluator(neg_dst=neg)
        if split_mode == "test":
            neg = loaded.dataset.negative_samples          # shipped test_ns.npy [N_test, K]
            if neg is None:
                raise FileNotFoundError(
                    f"TGB-Seq test negatives (test_ns) not found for {self.name!r}; "
                    "they normally download with the dataset — re-run the loader to fetch them.")
            neg = np.asarray(neg, dtype=np.int32)
            n_test = len(loaded.test.sources)
            if neg.shape[0] != n_test:
                raise ValueError(
                    f"test_ns rows ({neg.shape[0]}) != test positives ({n_test}); "
                    "the shipped negatives are not row-aligned to the test split.")
            print(f"  [tgb-seq] test negatives: shipped test_ns ({neg.shape[1]} per positive)")
            return FixedNegativeEvaluator(neg_dst=neg)
        raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
