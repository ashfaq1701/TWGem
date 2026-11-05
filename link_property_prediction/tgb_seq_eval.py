"""TGB-Seq: native load; val negatives sampled by us, test uses the shipped test_ns."""
import numpy as np

from .data import Loaded, SplitData, node_count
from .evaluator import DataSuite, Evaluator, FixedNegativesEvaluator


def _to_int64_time(time: np.ndarray) -> np.ndarray:
    """Monotone int64 timestamps; non-integral inputs fall back to a dense rank."""
    t = np.asarray(time)
    if np.isfinite(t).all() and np.array_equal(t, np.floor(t)):
        return t.astype(np.int64)
    order = np.unique(t)                                  # sorted unique values
    return np.searchsorted(order, t).astype(np.int64)     # dense rank, monotone


def load_tgb_seq(name: str, root: str = "datasets") -> Loaded:
    """Native TGB-Seq load -> Loaded; the loader is kept for its shipped negatives."""
    import os

    from tgb_seq.LinkPred.dataloader import TGBSeqLoader

    ds = TGBSeqLoader(name=name, root=root)
    # Re-download if the edgelist or test_ns is missing.
    if not (os.path.exists(ds._edgelist_path) and os.path.exists(ds._test_ns_path)):
        ds._download()
        ds._load_file()

    src = np.asarray(ds.src_node_ids, dtype=np.int64)
    dst = np.asarray(ds.dst_node_ids, dtype=np.int64)
    ts = _to_int64_time(ds.node_interact_times)

    edge_feat = ds.edge_features
    edge_feat = np.asarray(edge_feat, dtype=np.float32) if edge_feat is not None else None

    full = SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=edge_feat)

    return Loaded(
        train=full.mask(ds.train_mask),
        val=full.mask(ds.val_mask),
        test=full.mask(ds.test_mask),
        dataset=ds,
        name=name,
        max_node_count=node_count(src, dst),
    )


class TGBSeqSuite(DataSuite):
    """TGB-Seq adapter."""

    def _load(self) -> Loaded:
        return load_tgb_seq(self.name, self.root)

    def _val_evaluator(self) -> Evaluator:
        neg = self.eval_negatives(self.load().val, self.seed, tag="[tgb-seq] val")
        return FixedNegativesEvaluator(neg_dst=neg)

    def _test_evaluator(self) -> Evaluator:
        loaded = self.load()
        neg = loaded.dataset.negative_samples              # shipped test_ns.npy [N_test, K]
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
        return FixedNegativesEvaluator(neg_dst=neg)
