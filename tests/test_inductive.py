import torch

from link_property_prediction.model import LinkPredHead
from link_property_prediction.walk_tokens import WalkTokens


def _tokens(nodes):
    nodes = torch.tensor(nodes)                                              # [Q, K, L], -1 = padding
    mask = nodes >= 0
    q, k, length = nodes.shape
    ages = torch.where(mask, torch.arange(length).expand(q, k, length), torch.full_like(nodes, -1))
    pos = torch.where(mask, torch.arange(1, length + 1).expand(q, k, length), torch.zeros_like(nodes))
    return WalkTokens(seeds=nodes[:, 0, 0].clone(), cutoffs=torch.full((q,), 100), nodes=nodes,
                      ages=ages, positions=pos, mask=mask)


def _model():
    return LinkPredHead(num_nodes=50, T_train=100, max_walk_len=3, d_emb=8, seed=5)


def test_marking_every_node_seen_changes_nothing():
    a, b = _model(), _model()
    b.bag_weights.mark_seen(torch.arange(50))
    src = _tokens([[[3, 4, 5], [3, 6, -1]]])
    cand = _tokens([[[7, 8, -1], [7, -1, -1]], [[9, 3, 4], [9, 10, 11]]])
    assert torch.equal(a(src, cand), b(src, cand))


def test_unseen_token_is_dropped_from_the_bag():
    model = _model()
    model.bag_weights.mark_seen(torch.tensor([3, 4, 5, 6]))
    with_unseen = model.bag_weights(_tokens([[[3, 4, 40], [3, 6, -1]]])).pooled
    without = model.bag_weights(_tokens([[[3, 4, -1], [3, 6, -1]]])).pooled
    assert torch.allclose(with_unseen, without)


def test_fully_unseen_node_keeps_its_own_row():
    model = _model()
    model.bag_weights.mark_seen(torch.tensor([1, 2, 3]))
    pooled = model.bag_weights(_tokens([[[40, 41, -1]]])).pooled
    assert torch.equal(pooled[0], model.E.weight[40])
