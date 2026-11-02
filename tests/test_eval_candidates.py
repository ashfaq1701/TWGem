"""Trainer._eval_candidates: the positive first, each row's own negatives, ragged K padded."""
import numpy as np

from link_property_prediction.data import Batch
from link_property_prediction.trainer import Trainer


def _batch(tgt):
    tgt = np.asarray(tgt, dtype=np.int64)
    zeros = np.zeros_like(tgt)
    return Batch(src=zeros, tgt=tgt, ts=zeros, edge_feat=None)


def test_ragged_negatives_are_padded_with_the_positive():
    batch = _batch([7, 8, 9])
    negs = [np.array([1, 2, 3], dtype=np.int32), np.array([4], dtype=np.int32),
            np.array([], dtype=np.int32)]
    cand = Trainer._eval_candidates(batch, negs)
    assert cand.dtype == np.int64
    np.testing.assert_array_equal(cand, [[7, 1, 2, 3], [8, 4, 8, 8], [9, 9, 9, 9]])


def test_fixed_k_rows_are_positive_then_negatives():
    batch = _batch([5, 6])
    negs = [np.array([1, 2], dtype=np.int32), np.array([3, 4], dtype=np.int32)]
    np.testing.assert_array_equal(Trainer._eval_candidates(batch, negs), [[5, 1, 2], [6, 3, 4]])
