# Euclidean geometry: the paper runs

Two arms at three seeds each (5, 42, 1) plus a k_train probe, one log per cell.
Archive layout:

    euclidean/<arm>/<dataset>/seed_<S>.log

Only runs that closed `rc=0` are copied here, from `logs/` once finished. Record BOTH
test@val-checkpoint and max test: on this suite val flickers walk the checkpointed number
down, so the printed `best_test_mrr` alone is not like-for-like across arms.

## default -- TWGem

Master `78bbf8f`. Every knob is the argparse default except the per-dataset flags in
`default_conf_variations.txt`, reproduced verbatim on each log's command line.
Defaults: d_emb 128, wpn 10, mwl 5, k_train 10, k_eval 100, lr 1e-3, batch 1000,
eval_batch 1000, epochs 200, patience 10, hidden 32, n_layers_pooler 2.

18 datasets x 3 seeds, 39 Slurm jobs (DyGLib runs its six in one job). Ids
`11689843`-`11689881`, except eight small cells moved off RTX onto A40 on 2026-11-14 and
resubmitted as `11696628`-`11696635`: seed 42 DyGLib, and seed 1 tgbl-review, YouTube,
Patent, ML-20M, DyGLib, GoogleLocal, Flickr. All six RTX GPUs were held by this sweep's own
WikiLink / tgbl-comment / tgbl-flight cells, which cost 1.1-1.7 days each, and equal job
priority runs by job id, so the cheap cells sat behind them. A40 is 3-5x slower per epoch
than RTX here, which these cells can absorb and the long ones cannot. Per-epoch times are
therefore NOT comparable between an A40 cell and an RTX one. `logs/paper_euclidean/MANIFEST.tsv`
carries the live job id per cell and is what `scripts/pe_status.sh` reads.

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

## k5_jobs -- the k_train probe

Code identical to the default arm (`78bbf8f`, clean tree, conf line read from the file); one
flag differs, `--k-train 5` against the arm's 10. Evaluation is untouched, so these compare
directly with the default arm's Yelp and Taobao cells. Seed 5 only: jobs `11690736` (Taobao)
and `11690735` (Yelp), logs in `logs/k5_side/seed_5`.

Live at 2026-11-14 23:00, both still running, so both numbers are floors:

| cell | k5 (live) | at ep | k10 seed 5 | k10 seed 42 |
|---|---|---|---|---|
| Taobao | 0.6510 | 33 | 0.6440 | 0.6436 |
| Yelp | 0.6849 | 74 | 0.6849 | 0.6856 |

Taobao k5 leads by 0.0070 and its best val (0.6699) is above both k10 cells' (0.6677, 0.6685),
so the gain is not only in the val->test gap. Against that it carries `r_max` 12.0-12.4 while
the two k10 Taobao seeds disagree 11.80 vs 4.29, so Taobao's radius is unstable at both k and
the margin may sit inside that. Yelp k5 is level with one k10 seed and 0.0007 under the other.
Each run has broken out of 8/10 or 9/10 patience twice, so neither is near converged.

Reporting Taobao at k=5 and the rest at k=10 would be a per-dataset hyperparameter. That
decision is open and these directories exist to hold the evidence either way, not to presume it.

## Not in this tree

- **tgbl-wiki**: no arm here. It exists only in `experiment_logs/craft_reproduction`.
- **Patent**: run and archived at both arms' default, but CRAFT does not report it, so it
  carries no comparison and stays out of the paper tables.
