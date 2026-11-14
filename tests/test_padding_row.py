import torch
import torch.nn.functional as F

from link_property_prediction.model import LinkPredHead
from link_property_prediction.walk_tokens import WalkTokens


def _tokens(ids):
    """WalkTokens from a [Q, K, L] id array; -1 is padding, the seed sits at the last hop."""
    nodes = torch.tensor(ids)
    mask = nodes >= 0
    Q, K, L = nodes.shape
    ages = torch.where(mask, torch.arange(L).flip(0).expand(Q, K, L), torch.full_like(nodes, -1))
    pos = torch.where(mask, torch.arange(L, 0, -1).expand(Q, K, L), torch.zeros_like(nodes))
    return WalkTokens(seeds=nodes[:, 0, -1].clone(), cutoffs=torch.full((Q,), 100),
                      nodes=nodes, ages=ages, positions=pos, mask=mask)


def _step(model, opt, ctx, cands, seed_id=7):
    src = _tokens([[[-1, ctx, seed_id]]])
    cand = _tokens([[[-1, -1, c]] for c in cands])
    opt.zero_grad()
    F.cross_entropy(model(src, cand), torch.zeros(1, dtype=torch.long)).backward()
    touched = sorted(set(model.E.weight.grad.coalesce().indices()[0].tolist()))
    opt.step()
    return touched


def test_padding_never_enters_the_sparse_gradient():
    """A padded slot must not put its row in E's gradient: SparseAdam steps every listed row."""
    model = LinkPredHead(num_nodes=50, T_train=100, max_walk_len=3, d_emb=8, seed=5)
    opt = torch.optim.SparseAdam([model.E.weight], lr=0.1)

    assert _step(model, opt, 0, [20, 21]) == [0, 7, 20, 21]      # node 0 is a real bag member
    before = model.E.weight.detach().clone()

    assert _step(model, opt, 3, [20, 21]) == [3, 7, 20, 21]      # node 0 is in no bag now
    assert torch.equal(model.E.weight[0], before[0])             # so it must not move
    assert torch.equal(model.E.weight[5], before[5])             # nor any untouched row
    assert torch.equal(model.E.weight[model.E.padding_idx],
                       torch.zeros(model.d_emb))                 # sentinel stays zero
