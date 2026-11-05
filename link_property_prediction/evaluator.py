"""Evaluator and DataSuite interfaces, and `make_suite`, the only suite dispatch."""
import abc
from functools import cached_property
from typing import List, Optional

import numpy as np

from .data import Batch, Loaded, SplitData
from .negatives import NegativeSampler, UniformNegativeSampler


class Evaluator(abc.ABC):
    """Supplies per-positive negatives and the suite's native per-positive metric."""

    @abc.abstractmethod
    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        """`neg_tgt_list` — one negative-destination array per positive (K may vary)."""

    @abc.abstractmethod
    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        """The suite's native metric for one positive scored against its negatives."""

    def reset(self) -> None:
        """Called before every eval pass; rewinds fixed-negative cursors."""
        return None


class FixedNegativesEvaluator(Evaluator):
    """Fixed [N_pos, K] negatives walked by a cursor, scored by TGB-Seq's MRR."""

    def __init__(self, neg_dst: np.ndarray):
        from tgb_seq.LinkPred.evaluator import Evaluator as TGBSeqOfficialEvaluator

        self._neg_dst = np.asarray(neg_dst, dtype=np.int32)   # [N_pos, K]
        self._cursor = 0
        self._eval = TGBSeqOfficialEvaluator()

    def reset(self) -> None:
        self._cursor = 0

    def sample_negatives(self, batch: Batch) -> List[np.ndarray]:
        b = len(batch.src)
        rows = self._neg_dst[self._cursor:self._cursor + b]
        self._cursor += b
        return [rows[i] for i in range(b)]

    def score_to_metric(self, pos_score: float, neg_scores: np.ndarray) -> float:
        pos = np.asarray([pos_score], dtype=np.float64)
        neg = np.asarray(neg_scores, dtype=np.float64).reshape(1, -1)
        return float(self._eval.eval(pos, neg)[0])            # mrr_list[0]


class DataSuite(abc.ABC):
    """Benchmark adapter: native load and native evaluators."""

    def __init__(self, name: str, root: str, is_bipartite: bool, k_eval: int, seed: int):
        self.name = name
        self.root = root
        self.is_bipartite = bool(is_bipartite)
        self.k_eval = int(k_eval)
        self.seed = seed
        self._loaded: Optional[Loaded] = None

    def load(self) -> Loaded:
        """Native load, cached so downstream calls reuse one copy."""
        if self._loaded is None:
            self._loaded = self._load()
        return self._loaded

    @abc.abstractmethod
    def _load(self) -> Loaded:
        """Do the suite's native load and return a `Loaded`."""

    def make_evaluator(self, split_mode: str) -> Evaluator:
        """Evaluator for split_mode 'val' or 'test'."""
        if split_mode == "val":
            return self._val_evaluator()
        if split_mode == "test":
            return self._test_evaluator()
        raise ValueError(f"split_mode must be 'val' or 'test', got {split_mode!r}")

    @abc.abstractmethod
    def _val_evaluator(self) -> Evaluator:
        """Native evaluator over the val split's negatives."""

    @abc.abstractmethod
    def _test_evaluator(self) -> Evaluator:
        """Native evaluator over the test split's negatives."""

    @property
    def candidate_kind(self) -> str:
        """What a negative can be: destinations on a bipartite graph, any node otherwise."""
        return "destinations" if self.is_bipartite else "nodes"

    def _candidates(self, *splits) -> np.ndarray:
        """Unique candidates: destinations if bipartite, else sources and destinations."""
        ids = [s.destinations for s in splits]
        if not self.is_bipartite:
            ids += [s.sources for s in splits]
        return np.unique(np.concatenate(ids)).astype(np.int32)

    @cached_property
    def train_negative_pool(self) -> np.ndarray:
        """Training negatives come from the train split only."""
        return self._candidates(self.load().train)

    @cached_property
    def eval_negative_pool(self) -> np.ndarray:
        """Eval negatives come from the full dataset."""
        loaded = self.load()
        return self._candidates(loaded.train, loaded.val, loaded.test)

    def negative_sampler(self, pool: np.ndarray, k: int, seed: int,
                         edges: SplitData) -> NegativeSampler:
        """The suite's sampling rule for negatives of `edges` drawn from `pool`."""
        return UniformNegativeSampler(num_neg_per_pos=k, dst_pool=pool, seed=seed)

    def train_negative_sampler(self, k_train: int) -> NegativeSampler:
        return self.negative_sampler(self.train_negative_pool, k_train, self.seed,
                                     self.load().train)

    def eval_negatives(self, split: SplitData, seed: int, tag: str) -> np.ndarray:
        """Fixed [N_pos, k_eval] negatives for `split`."""
        pool = self.eval_negative_pool
        sampler = self.negative_sampler(pool, self.k_eval, seed, split)
        print(f"  {tag} negatives: {self.k_eval} {sampler.rule} per positive over "
              f"{len(pool):,} full-dataset candidates")
        whole = Batch(src=split.sources, tgt=split.destinations,
                      ts=split.timestamps, edge_feat=None)
        return sampler.sample(whole).astype(np.int32)


#: Valid `--data-suite` values; the train script uses them as its choices.
SUITES = ("tgb-seq", "tgb", "dyglib")


def make_suite(data_suite: str, **kwargs) -> DataSuite:
    """Map `--data-suite` to its suite; imported lazily."""
    if data_suite == "tgb-seq":
        from .tgb_seq_eval import TGBSeqSuite
        return TGBSeqSuite(**kwargs)
    if data_suite == "tgb":
        from .tgb_eval import TGBSuite
        return TGBSuite(**kwargs)
    if data_suite == "dyglib":
        from .dyglib_eval import DyGLibSuite
        return DyGLibSuite(**kwargs)
    raise ValueError(
        f"unknown --data-suite {data_suite!r} (expected one of {sorted(SUITES)})")
