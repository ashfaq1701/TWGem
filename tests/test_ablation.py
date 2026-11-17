import dataclasses

import pytest
import torch

from link_property_prediction.model import Ablation, LinkPredHead
from link_property_prediction.walk_tokens import WalkTokens

ARMS = {
    "A1": dict(walk_sides="none"), "A16": dict(walk_sides="source"),
    "A2": dict(pooling="uniform"), "A3+4": dict(features=("mid", "radius")),
    "A7": dict(features=("age", "hop")), "A5": dict(features=("age", "hop", "radius")),
    "A6": dict(features=("age", "hop", "mid")), "A8": dict(raw_geo_features=True),
    "A9": dict(detach=False), "A10": dict(standardise=False),
    "A13": dict(score="dist"), "A14": dict(score="dot"),
}


def _tokens(q, seed, k=3, length=4):
    g = torch.Generator().manual_seed(seed)
    nodes = torch.randint(0, 40, (q, k, length), generator=g)
    mask = torch.rand(q, k, length, generator=g) > 0.3
    mask[:, :, 0] = True
    nodes = torch.where(mask, nodes, torch.full_like(nodes, -1))
    ages = torch.where(mask, torch.randint(0, 50, (q, k, length), generator=g),
                       torch.full_like(nodes, -1))
    pos = torch.where(mask, torch.arange(1, length + 1).expand(q, k, length),
                      torch.zeros_like(nodes))
    return WalkTokens(seeds=nodes[:, 0, 0].clone(), cutoffs=torch.full((q,), 100),
                      nodes=nodes, ages=ages, positions=pos, mask=mask)


def _model(**kw):
    return LinkPredHead(num_nodes=50, T_train=100, max_walk_len=4, d_emb=16, seed=5,
                        ablation=Ablation(**kw))


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_arm_runs(arm):
    logits = _model(**ARMS[arm])(_tokens(6, 0), _tokens(66, 1))
    logits.sum().backward()
    assert logits.shape == (6, 11) and torch.isfinite(logits).all()


def test_no_walks_reads_only_the_seed_rows():
    model, src, cand = _model(walk_sides="none"), _tokens(6, 0), _tokens(66, 1)
    other = dataclasses.replace(_tokens(6, 7), seeds=src.seeds)              # new walks, same seeds
    assert torch.equal(model(src, cand), model(other, cand))


def test_feature_subset_sets_controller_width():
    assert _model(features=("age", "hop")).bag_weights.n_feat == 2
    assert _model().bag_weights.n_feat == 4


def test_no_detach_changes_only_the_gradient():
    src, cand = _tokens(6, 0), _tokens(66, 1)
    a, b = _model(), _model(detach=False)
    la, lb = a(src, cand), b(src, cand)
    assert torch.equal(la, lb)
    la.sum().backward()
    lb.sum().backward()
    assert not torch.equal(a.E.weight.grad.to_dense(), b.E.weight.grad.to_dense())


def test_invalid_switch_is_rejected():
    with pytest.raises(ValueError):
        Ablation(features=("age", "colour"))
