"""TGB (`tgb.linkproppred`) — native load, negatives, and metric.

Restores the suite dropped at `ec6e448`, adapted to the `Evaluator` contract as it
stands today. Two deliberate differences from the 2026-08 version:

  - `sample_negatives` returns ONLY the negative destinations, matching the ABC.
    The old version returned `(neg_src_list, neg_tgt_list)` where every element of
    the first was `np.full(len(arr), src)` -- redundant, because
    `UniformNegativeSampler.sample` documents that "the positive's source is
    implicit: every caller pairs the negatives with batch.src itself". Widening the
    ABC back to a pair would force TGB-Seq to return a dummy second element and
    push `if suite == ...` into the trainer.
  - `TGBNegativeSampler` is gone. It existed only to subclass `NegativeSampler`,
    whose `sample` is typed `-> np.ndarray` of shape [B, K]; TGB's negatives are
    variable-K lists and never satisfied that. `Evaluator.sample_negatives` is
    already the variable-K interface, so the query folds directly into the
    evaluator -- the same shape as every other suite's evaluator.

VAL negatives are OURS and TEST negatives are TGB's, which is TGB-Seq's arrangement:
val is a fixed `[N_pos, k_eval]` array from `DataSuite.eval_negatives`, read by
`FixedNegativeEvaluator`; test is TGB's shipped per-positive set from `query_batch`,
keyed on (src, tgt, ts) and therefore order-independent, which is why `reset()` stays
the inherited no-op on the test evaluator. TGB's own `val_ns` is no longer read at all.

Two consequences. TGB val MRRs are NOT comparable with any run before this change --
different negatives mean different numbers and different checkpoint selections -- and
TGB now has a val->test protocol gap like TGB-Seq's, where before both splits shared
TGB's protocol and the gap was absent. Test stays on TGB's official negatives and
official metric, so test numbers remain leaderboard-comparable.

Never pool the suites' numbers: they differ in exactly the axis MRR is sensitive to.
"""
import re
import zipfile
from typing import List

import numpy as np

from .data import Batch, Loaded, SplitData
from .evaluator import DataSuite, Evaluator, FixedNegativeEvaluator


def _strip_version_suffix(name: str) -> str:
    """Drop a trailing '-v<digits>' to map to the TGB registry key
    ('tgbl-review-v2' -> 'tgbl-review'); the suffixed form raises inside TGB."""
    return re.sub(r"-v\d+$", "", name)


def _missing_dataset_error(tgb_name: str, data_dir: str, detail: str) -> FileNotFoundError:
    """One message for every way a TGB dataset can end up unusable: where its files
    belong, and what the automatic download fetches."""
    from tgb.utils.info import DATA_URL_DICT

    return FileNotFoundError(
        f"TGB dataset {tgb_name!r} is not usable ({detail}).\n"
        f"Its files (edgelist + test negatives) belong in:\n"
        f"  {data_dir}\n"
        f"The automatic download fetches {DATA_URL_DICT.get(tgb_name, '(no URL registered)')}; "
        f"if that host no longer serves the data, upgrade py-tgb for a newer URL or extract "
        f"the dataset there by hand. Pass --data-root to use a different directory.")


def load_tgb(name: str, root: str = "datasets") -> Loaded:
    """Load a TGB link-property-prediction dataset. `-vN` suffixes are stripped
    before the call — TGB's registry uses bare names.

    `root` is translated because TGB does `root = PROJ_DIR + root` (literal string
    concatenation, dataset.py:75), so its `root` is ALWAYS relative to the tgb
    install dir and an absolute path yields e.g.
    `.../site-packages/tgb//mnt/.../datasets/tgbl_review/` -- a real 2.1 GB download
    inside the venv, which is how this was found. Passing the relative path from
    PROJ_DIR makes `PROJ_DIR + root` resolve back to the directory the caller asked
    for, so `--data-root` is honoured rather than silently ignored. A RELATIVE root
    (the default "datasets") therefore lands inside the tgb install dir.

    A missing dataset is downloaded by TGB itself. TGB writes whatever the URL returns
    to `<name>.zip` without checking it: py-tgb 2.2.0's host returned a `NoSuchBucket`
    XML page, TGB crashed on it with `BadZipFile` and left the stub behind. So a bad zip
    is deleted here and turned into an error naming the dataset directory, and the test
    negative file is checked too, since TGB only checks the edgelist. `val_ns` is not
    checked or loaded: val negatives are ours (`TGBSuite.make_evaluator`).
    """
    import os
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.utils.info import PROJ_DIR

    tgb_name = _strip_version_suffix(name)
    tgb_root = os.path.relpath(root, PROJ_DIR) if os.path.isabs(root) else root
    # TGB's own resolution (dataset.py:75-81), reproduced only for the error message.
    data_dir = os.path.normpath(os.path.join(PROJ_DIR + tgb_root, tgb_name.replace("-", "_")))
    try:
        dataset = LinkPropPredDataset(name=tgb_name, root=tgb_root, preprocess=True)
    except zipfile.BadZipFile as err:
        stub = os.path.join(data_dir, tgb_name + ".zip")
        if os.path.exists(stub):
            os.remove(stub)                       # else the next run trips on it too
        raise _missing_dataset_error(tgb_name, data_dir, "download was not a zip") from err
    test_ns = dataset.meta_dict["test_ns"]
    if not os.path.exists(test_ns):
        raise _missing_dataset_error(tgb_name, data_dir, f"missing {test_ns}")
    full = dataset.full_data
    sources = np.asarray(full["sources"], dtype=np.int64)
    destinations = np.asarray(full["destinations"], dtype=np.int64)
    timestamps = np.asarray(full["timestamps"], dtype=np.int64)
    edge_feat = full.get("edge_feat", None)
    if edge_feat is not None:
        edge_feat = np.asarray(edge_feat, dtype=np.float32)

    train_mask = np.asarray(dataset.train_mask, dtype=bool)
    val_mask = np.asarray(dataset.val_mask, dtype=bool)
    test_mask = np.asarray(dataset.test_mask, dtype=bool)

    def _apply(mask: np.ndarray) -> SplitData:
        ef = edge_feat[mask] if edge_feat is not None else None
        return SplitData(
            sources=sources[mask],
            destinations=destinations[mask],
            timestamps=timestamps[mask],
            edge_feat=ef,
        )

    return Loaded(
        train=_apply(train_mask),
        val=_apply(val_mask),
        test=_apply(test_mask),
        dataset=dataset,
        # TGB-canonical (suffix-stripped) name — the Evaluator passes it back to TGB.
        name=tgb_name,
        max_node_count=int(max(sources.max(), destinations.max())) + 1,
    )


class TGBTestEvaluator(Evaluator):
    """TGB's native TEST evaluation: its shipped per-positive negatives from
    `dataset.negative_sampler.query_batch`, plus the official
    `tgb.linkproppred.evaluate.Evaluator` metric. This is the leaderboard protocol and
    the only place TGB's negatives are used -- val negatives are ours.

    Content-addressed by (src, tgt, ts), so it is order-independent and `reset()` stays
    the inherited no-op. K varies per positive.
    """

    def __init__(self, dataset: object, tgb_dataset_name: str, eval_metric: str):
        from tgb.linkproppred.evaluate import Evaluator as TGBOfficialEvaluator

        self._sampler = dataset.negative_sampler
        self._eval_metric = eval_metric
        self._tgb_eval = TGBOfficialEvaluator(name=tgb_dataset_name)

    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        """`neg_tgt_list` — one negative-destination array per positive, K may vary."""
        neg_dst_list = self._sampler.query_batch(
            batch.src, batch.tgt, batch.ts, split_mode="test",
        )
        return [np.asarray(d, dtype=np.int32) for d in neg_dst_list]

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        res = self._tgb_eval.eval({
            "y_pred_pos": np.asarray([pos_score], dtype=np.float64),
            "y_pred_neg": np.asarray(neg_scores, dtype=np.float64),
            "eval_metric": [self._eval_metric],
        })
        return float(res[self._eval_metric])


class TGBSuite(DataSuite):
    """TGB adapter. VAL negatives are built once from our sampler at `k_eval`; TEST uses
    TGB's shipped negatives verbatim under TGB's official metric."""

    def _load(self) -> Loaded:
        return load_tgb(self.name, self.root)

    def make_evaluator(self, split_mode: str) -> Evaluator:
        loaded = self.load()
        if split_mode == "val":
            neg = self.eval_negatives(loaded.val, self.seed, tag="[tgb] val")
            return FixedNegativeEvaluator(neg_dst=neg)
        if split_mode == "test":
            # TGB requires the split's negative-sampler file loaded before querying;
            # cached on disk after the first call.
            loaded.dataset.load_test_ns()
            print("  [tgb] test negatives: TGB's shipped test_ns under TGB's official metric")
            return TGBTestEvaluator(
                dataset=loaded.dataset,
                tgb_dataset_name=loaded.name,
                # Off the live handle, not off Loaded: `eval_metric` is TGB-specific and
                # Loaded is the vocabulary ALL suites share. Keeping it out is what lets
                # a suite carry its own metadata without widening the common type.
                eval_metric=str(loaded.dataset.eval_metric),
            )
        raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")
