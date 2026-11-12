"""TGB: native load, TGB's own val/test negatives (variable K) and its official metric."""
import re
import zipfile
from typing import List

import numpy as np

from .data import Batch, Loaded, SplitData, node_count
from .evaluator import DataSuite, Evaluator
from .negatives import HistRndNegativeSampler, NegativeSampler


def _strip_version_suffix(name: str) -> str:
    """Drop a trailing '-v<digits>' to get the TGB registry key."""
    return re.sub(r"-v\d+$", "", name)


def _missing_dataset_error(tgb_name: str, data_dir: str, detail: str) -> FileNotFoundError:
    """Error naming where a dataset's files belong and what the download fetches."""
    from tgb.utils.info import DATA_URL_DICT

    return FileNotFoundError(
        f"TGB dataset {tgb_name!r} is not usable ({detail}).\n"
        f"Its files (edgelist + val/test negatives) belong in:\n"
        f"  {data_dir}\n"
        f"The automatic download fetches {DATA_URL_DICT.get(tgb_name, '(no URL registered)')}; "
        f"if that host no longer serves the data, upgrade py-tgb for a newer URL or extract "
        f"the dataset there by hand. Pass --data-root to use a different directory.")


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB dataset into `root`; a bad download zip is deleted and reported."""
    import os
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.utils.info import PROJ_DIR

    tgb_name = _strip_version_suffix(name)
    tgb_root = os.path.relpath(root, PROJ_DIR) if os.path.isabs(root) else root
    data_dir = os.path.normpath(os.path.join(PROJ_DIR + tgb_root, tgb_name.replace("-", "_")))
    try:
        dataset = LinkPropPredDataset(name=tgb_name, root=tgb_root, preprocess=True)
    except zipfile.BadZipFile as err:
        stub = os.path.join(data_dir, tgb_name + ".zip")
        if os.path.exists(stub):
            os.remove(stub)
        raise _missing_dataset_error(tgb_name, data_dir, "download was not a zip") from err
    missing = [dataset.meta_dict[k] for k in ("val_ns", "test_ns")
               if not os.path.exists(dataset.meta_dict[k])]
    if missing:
        raise _missing_dataset_error(tgb_name, data_dir, "missing " + ", ".join(missing))
    full = dataset.full_data
    sources = np.asarray(full["sources"], dtype=np.int64)
    destinations = np.asarray(full["destinations"], dtype=np.int64)
    timestamps = np.asarray(full["timestamps"], dtype=np.int64)
    edge_feat = full.get("edge_feat", None)
    if edge_feat is not None:
        edge_feat = np.asarray(edge_feat, dtype=np.float32)

    full = SplitData(sources=sources, destinations=destinations, timestamps=timestamps,
                     edge_feat=edge_feat)

    return Loaded(
        train=full.mask(dataset.train_mask),
        val=full.mask(dataset.val_mask),
        test=full.mask(dataset.test_mask),
        dataset=dataset,
        name=tgb_name,
        max_node_count=node_count(sources, destinations),
    )


class TGBEvaluator(Evaluator):
    """TGB's per-positive negatives (keyed on src, tgt, ts) and official metric."""

    def __init__(self, dataset: object, split_mode: str,
                 tgb_dataset_name: str, eval_metric: str):
        from tgb.linkproppred.evaluate import Evaluator as TGBOfficialEvaluator

        self._split_mode = split_mode
        self._sampler = dataset.negative_sampler
        self.eval_metric = eval_metric
        self._tgb_eval = TGBOfficialEvaluator(name=tgb_dataset_name)

    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        """`neg_tgt_list` — one negative-destination array per positive, K may vary."""
        neg_dst_list = self._sampler.query_batch(
            batch.src, batch.tgt, batch.ts, split_mode=self._split_mode,
        )
        return [np.asarray(d, dtype=np.int32) for d in neg_dst_list]

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        res = self._tgb_eval.eval({
            "y_pred_pos": np.asarray([pos_score], dtype=np.float64),
            "y_pred_neg": np.asarray(neg_scores, dtype=np.float64),
            "eval_metric": [self.eval_metric],
        })
        return float(res[self.eval_metric])


class TGBSuite(DataSuite):
    """TGB adapter; TGB's own negatives for both splits, so --k-eval is ignored."""

    def _load(self) -> Loaded:
        return load_tgb(self.name, self.root)

    def negative_sampler(self, pool: np.ndarray, k: int, seed: int,
                         edges: SplitData) -> NegativeSampler:
        """`--train-hist-ratio` 0 keeps the uniform draw; above 0 mirrors TGB's hist_rnd."""
        if self.train_hist_ratio <= 0.0:
            return super().negative_sampler(pool, k, seed, edges)
        return HistRndNegativeSampler(num_neg_per_pos=k, dst_pool=pool, edges=edges,
                                      ratio=self.train_hist_ratio, seed=seed)

    def _val_evaluator(self) -> Evaluator:
        self.load().dataset.load_val_ns()
        return self._evaluator("val")

    def _test_evaluator(self) -> Evaluator:
        self.load().dataset.load_test_ns()
        return self._evaluator("test")

    def _evaluator(self, split_mode: str) -> Evaluator:
        """Shared evaluator once the split's negative file is loaded."""
        loaded = self.load()
        print(f"  [tgb] {split_mode} negatives: TGB's pre-generated set "
              f"(--k-eval={self.k_eval} ignored)")
        return TGBEvaluator(
            dataset=loaded.dataset,
            split_mode=split_mode,
            tgb_dataset_name=loaded.name,
            eval_metric=str(loaded.dataset.eval_metric),
        )
