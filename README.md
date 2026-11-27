# TWGem

**Temporal Walk Geometric Embedding.** A backward temporal random walk from a query node is
treated as a sample of that node's causal multi-hop neighbourhood; the walk's tokens are pooled
to a single point and scored by a metric distance against candidate destinations. Geometric in
the sense TransE is: the score is a distance in the embedding space, here Euclidean. Two arms
share one encoder: **link prediction** (TGB-Seq, TGB, DyGLib; MRR) trains the embedding table
and the walk-bag pooler, and **dynamic node classification** (DyGLib Wikipedia / Reddit; ROC-AUC)
freezes a link-prediction checkpoint and trains a small classifier on features of its walks.

## Features

- **Temporal random walks as supervision, not as sequences.** The temporal-walk family is
  dominated by skip-gram: CTDNE and its descendants (T-EDGE, STWalk) read a walk as a window of
  co-occurring nodes and optimise a word2vec-style objective. Here a walk is a *neighbourhood
  sample* instead — pooled to a single representation and trained end to end on the ranking loss
  the benchmark actually scores.
- **Node identity is kept.** Causal-anonymous-walk methods (CAW-N) deliberately erase it,
  encoding only how often a node recurs across positions. This model keeps per-node parameters,
  which is what makes the geometry — and the second arm — possible.
- **A stored, stateful, trained embedding table.** Node state persists in parameters rather than
  being recomputed per query, so node-centric predictive tasks and dynamic-state methods can read
  it directly. The node-classification arm is exactly that: a frozen checkpoint plus a ~2.4k-param
  head beats every published number on DyGLib Wikipedia and Reddit.
- **Powered by a GPU-accelerated streaming walk engine.** Walks come from
  [Tempest](https://pypi.org/project/tempest-rw/) (Salehin, Parisis & Berthouze, ICPP '26):
  forward or backward walks under uniform, linear, exponential or temporal-node2vec bias, with a
  sliding window that evicts old edges as new ones arrive.
- **Strictly causal by construction.** Causality is enforced per query, not per split: a walk for
  `(u, t)` may only use edges strictly older than `t`, so no evaluation query can see its own
  future.
- **Evaluated on the modern benchmarks, on their own terms.** TGB-Seq (ICLR 2025), TGB and DyGLib,
  each with its shipped negative-sampling protocol and metric rather than a re-rolled one.

## Running

```
# link prediction (saves the frozen encoder for node classification)
scripts/train_link_property_prediction.py --data-suite dyglib --dataset wikipedia \
  --is-bipartite --k-train 5 --k-eval 5 --seed 42 --use-gpu --use-gpu-tempest \
  --save-checkpoint <ckpt.pt>

# node classification on that checkpoint
scripts/train_node_classification.py --dataset wikipedia --checkpoint <ckpt.pt> \
  --seed 42 --use-gpu --use-gpu-tempest
```

Per-dataset config is `default_conf_variations.txt`. Dataset, protocol and result notes are in
`CLAUDE.md`; the node-classification write-up is `reports/node_classification_2026-10-08.md`.
