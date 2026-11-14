# Euclidean geometry: the paper runs

Two arms, three seeds each (5, 42, 1), one log per cell. Archive layout:

    euclidean/<arm>/<dataset>/seed_<S>.log

Only runs that closed `rc=0` are copied here, from `logs/` once finished. Record BOTH
test@val-checkpoint and max test: on this suite val flickers walk the checkpointed number
down, so the printed `best_test_mrr` alone is not like-for-like across arms.

## default -- TWGem

Master `78bbf8f`. Every knob is the argparse default except the per-dataset flags in
`default_conf_variations.txt`, reproduced verbatim on each log's command line.
Defaults: d_emb 128, wpn 10, mwl 5, k_train 10, k_eval 100, lr 1e-3, batch 1000,
eval_batch 1000, epochs 200, patience 10, hidden 32, n_layers_pooler 2.

18 datasets x 3 seeds, 39 Slurm jobs `11689843`-`11689881` (DyGLib runs its six in one job).

| suite | datasets |
|---|---|
| TGB-Seq | GoogleLocal, YouTube, Flickr, Patent, ML-20M, Taobao, Yelp, WikiLink |
| TGB | tgbl-review, tgbl-coin, tgbl-comment, tgbl-flight |
| DyGLib | wikipedia, reddit, mooc, lastfm, uci, Flights |

### Closed cells (2026-11-14 21:39)

| dataset | seed | test@val-ckpt | max test | best val | sel/ran |
|---|---|---|---|---|---|
| GoogleLocal | 5 | 0.6802 | 0.6802 | 0.7082 | 161/171 |
| GoogleLocal | 42 | 0.6960 | 0.6964 | 0.7196 | 199/200 |
| YouTube | 5 | 0.6274 | 0.6274 | 0.7019 | 14/24 |
| YouTube | 42 | 0.6146 | 0.6201 | 0.7014 | 15/25 |
| Flickr | 5 | 0.6655 | 0.6655 | 0.7087 | 37/47 |
| Flickr | 42 | 0.6621 | 0.6622 | 0.7074 | 54/64 |
| Patent | 5 | 0.0417 | 0.0417 | 0.0793 | 1/11 |
| Patent | 42 | 0.0418 | 0.0418 | 0.0748 | 1/11 |
| ML-20M | 5 | 0.2855 | 0.2855 | 0.3458 | 1/11 |
| ML-20M | 42 | 0.2834 | 0.2834 | 0.3456 | 1/11 |
| Taobao | 5 | 0.6440 | 0.6450 | 0.6677 | 24/34 |
| Taobao | 42 | 0.6436 | 0.6452 | 0.6685 | 30/40 |
| Yelp | 5 | 0.6849 | 0.6854 | 0.7230 | 40/50 |
| Yelp | 42 | 0.6856 | 0.6869 | 0.7247 | 49/59 |
| tgbl-review | 5 | 0.3017 | 0.3040 | 0.2503 | 3/13 |
| tgbl-review | 42 | 0.3010 | 0.3013 | 0.2504 | 3/13 |
| tgbl-coin | 5 | 0.7502 | 0.7502 | 0.7404 | 14/24 |
| wikipedia | 5 | 0.8784 | 0.8800 | 0.9001 | 13/23 |
| reddit | 5 | 0.8618 | 0.8632 | 0.8764 | 11/21 |
| mooc | 5 | 0.3757 | 0.3757 | 0.3840 | 13/23 |
| lastfm | 5 | 0.4495 | 0.4549 | 0.4490 | 7/17 |
| uci | 5 | 0.7123 | 0.7123 | 0.6700 | 5/15 |
| Flights | 5 | 0.8042 | 0.8042 | 0.8182 | 3/13 |

Open: WikiLink, tgbl-comment, tgbl-flight at seed 5; WikiLink, tgbl-coin, tgbl-comment,
tgbl-flight and the DyGLib six at seed 42; all 13 cells at seed 1.

Taobao and Yelp replicate to 0.0004 and 0.0007 across the two seeds; tgbl-review selects at
ep3 in both. GoogleLocal seed 42 selected at ep199 of 200, so that cell is an epoch-cap floor.

## recurrence -- TWGem-R

Branch `feature/recurrence-mix`, pin `016b821` (master merged in). One change against the
default arm: the scorer mixes the geometry logit with a recurrence logit at a learned weight,

    logits = (1 - m) * layer_norm(-log d) + m * layer_norm(log1p(pair count before t))
    m = sigmoid(mix_logit), one scalar, init 0 so m = 0.5

where the pair count is Tempest's `get_pair_co_occurrence_counts(u, v, cutoff, Backward_In_Time)`.
That call does not advance Tempest's RNG, so at a fixed seed this arm samples bit-identical
walks to the default arm -- a one-variable comparison. `m` is logged as `mix=` per epoch.
Off (no `--use-recurrence-count`) the parameter does not exist and master is reproduced.
`--k-train 10` is passed explicitly: the branch forks before the commit that made 10 the default.

8 datasets x 3 seeds, 9 Slurm jobs. These are the eight cells CRAFT reports CRAFT-R on
(CRAFT + a raw repeat-time feature); the remaining ten default-arm datasets have no
recurrence arm.

| cell | jobs (seed 5 / 42 / 1) |
|---|---|
| tgbl-coin | 11695524 / 11696376 / 11696378 |
| tgbl-flight | 11696255 / 11696377 / 11696379 |
| DyGLib six | 11696356 / 11696357 / 11696358 |

No cell has closed. tempest_rw 1.0.7 is required: 1.0.6 has no
`get_pair_co_occurrence_counts`.

## Not in this tree

- **k=5 side runs** (`logs/k5_side`, jobs `11690735`/`11690736`): a k_train probe against the
  closed k10 Yelp and Taobao cells, not part of either arm. Not archived, not reported.
- **tgbl-wiki**: no arm here. It exists only in `experiment_logs/craft_reproduction`.
- **Patent**: run and archived at both arms' default, but CRAFT does not report it, so it
  carries no comparison and stays out of the paper tables.
