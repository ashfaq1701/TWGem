# Euclidean geometry: the paper runs

Three arms, three seeds each (5, 42, 1), one log per cell. Archive layout:

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
WikiLink / tgbl-comment / tgbl-flight cells, which cost 1.1-2.0 days each, and equal job
priority runs by job id, so the cheap cells sat behind them. A40 is 3-7x slower per epoch
than RTX here, which these cells can absorb and the long ones cannot. Per-epoch times are
therefore NOT comparable between an A40 cell and an RTX one. `logs/paper_euclidean/MANIFEST.tsv`
carries the live job id per cell and is what `scripts/pe_status.sh` reads.

| suite | datasets |
|---|---|
| TGB-Seq | GoogleLocal, YouTube, Flickr, Patent, ML-20M, Taobao, Yelp, WikiLink |
| TGB | tgbl-review, tgbl-coin, tgbl-comment, tgbl-flight |
| DyGLib | wikipedia, reddit, mooc, lastfm, uci, Flights |

### Closed cells (2026-11-15 19:00)

| dataset | seed | test@val-ckpt | max test | best val | sel/ran |
|---|---|---|---|---|---|
| GoogleLocal | 5 | 0.6802 | 0.6802 | 0.7082 | 161/171 |
| GoogleLocal | 42 | 0.6960 | 0.6964 | 0.7196 | 199/200 |
| GoogleLocal | 1 | 0.6886 | 0.6886 | 0.7155 | 197/200 |
| YouTube | 5 | 0.6274 | 0.6274 | 0.7019 | 14/24 |
| YouTube | 42 | 0.6146 | 0.6201 | 0.7014 | 15/25 |
| YouTube | 1 | 0.6241 | 0.6285 | 0.7020 | 13/23 |
| Flickr | 5 | 0.6655 | 0.6655 | 0.7087 | 37/47 |
| Flickr | 42 | 0.6621 | 0.6622 | 0.7074 | 54/64 |
| Patent | 5 | 0.0417 | 0.0417 | 0.0793 | 1/11 |
| Patent | 42 | 0.0418 | 0.0418 | 0.0748 | 1/11 |
| Patent | 1 | 0.0387 | 0.0387 | 0.0755 | 1/11 |
| ML-20M | 5 | 0.2855 | 0.2855 | 0.3458 | 1/11 |
| ML-20M | 42 | 0.2834 | 0.2834 | 0.3456 | 1/11 |
| ML-20M | 1 | 0.2830 | 0.2830 | 0.3447 | 1/11 |
| Taobao | 5 | 0.6440 | 0.6450 | 0.6677 | 24/34 |
| Taobao | 42 | 0.6436 | 0.6452 | 0.6685 | 30/40 |
| Taobao | 1 | 0.6477 | 0.6483 | 0.6708 | 26/36 |
| Yelp | 5 | 0.6849 | 0.6854 | 0.7230 | 40/50 |
| Yelp | 42 | 0.6856 | 0.6869 | 0.7247 | 49/59 |
| WikiLink | 5 | 0.8207 | 0.8207 | 0.8398 | 136/146 |
| tgbl-review | 5 | 0.3017 | 0.3040 | 0.2503 | 3/13 |
| tgbl-review | 42 | 0.3010 | 0.3013 | 0.2504 | 3/13 |
| tgbl-review | 1 | 0.3024 | 0.3024 | 0.2482 | 2/12 |
| tgbl-coin | 5 | 0.7502 | 0.7502 | 0.7404 | 14/24 |
| tgbl-coin | 42 | 0.7485 | 0.7500 | 0.7387 | 17/27 |
| tgbl-comment | 5 | 0.7592 | 0.7592 | 0.7222 | 17/27 |
| tgbl-comment | 42 | 0.7592 | 0.7598 | 0.7216 | 11/21 |
| tgbl-flight | 5 | 0.8126 | 0.8126 | 0.8344 | 2/12 |
| wikipedia | 5 | 0.8784 | 0.8800 | 0.9001 | 13/23 |
| wikipedia | 42 | 0.8803 | 0.8811 | 0.9018 | 11/21 |
| wikipedia | 1 | 0.8792 | 0.8792 | 0.9008 | 8/18 |
| reddit | 5 | 0.8618 | 0.8632 | 0.8764 | 11/21 |
| reddit | 1 | 0.8637 | 0.8656 | 0.8769 | 13/23 |
| mooc | 5 | 0.3757 | 0.3757 | 0.3840 | 13/23 |
| mooc | 1 | 0.3754 | 0.3763 | 0.3822 | 13/23 |
| lastfm | 5 | 0.4495 | 0.4549 | 0.4490 | 7/17 |
| lastfm | 1 | 0.4519 | 0.4605 | 0.4488 | 4/14 |
| uci | 5 | 0.7123 | 0.7123 | 0.6700 | 5/15 |
| uci | 1 | 0.6920 | 0.6965 | 0.6660 | 5/15 |
| Flights | 5 | 0.8042 | 0.8042 | 0.8182 | 3/13 |
| Flights | 1 | 0.8035 | 0.8036 | 0.8177 | 4/14 |

41 of 54 cells. Open: Flickr and Yelp at seed 1; WikiLink at seeds 42 and 1; tgbl-flight at
seeds 42 and 1; tgbl-coin and tgbl-comment at seed 1; the DyGLib five at seed 42.

**tgbl-comment seed 5 exited `rc=135` and the cell is valid anyway.** All 27 epochs ran,
patience fired at 10/10, the checkpoint was restored and the final block printed; the process
then took SIGBUS (`128 + 7`) during interpreter teardown, where the 44.3M-edge Tempest ingest
and the CUDA context both release. Slurm recorded the job COMPLETED 0:0 and MaxRSS was 31.8 GB
against a 64 GB request, so this is not a memory failure. The run is therefore reported, and the
`rc=0`-only archive rule is waived for this one log.

Replication across seeds: tgbl-comment 0.0000, Taobao 0.0041, tgbl-review 0.0014, tgbl-coin
0.0017, Yelp 0.0007, wikipedia 0.0019, reddit 0.0019, mooc 0.0003, lastfm 0.0024, Flights
0.0007, Patent 0.0031, ML-20M 0.0025, YouTube 0.0128, GoogleLocal 0.0158, **uci 0.0203**. uci
is the loosest cell in the arm and the smallest dataset in it (110-310 s a run). tgbl-review
selects at ep2-3 and Patent and ML-20M at ep1 in all three seeds.

**GoogleLocal is an epoch-cap floor at two of three seeds** -- sel 199/200 and 197/200, both
still improving when the 200-epoch cap stopped them. WikiLink seed 5 is not: it selected at
ep136 of 146 and patience fired, but its last 45 epochs bought +0.0015 test while `r_mean` grew
a near-constant +0.0385/epoch from 6.49 to 8.52, so most of that run was radius inflation
rather than learning.

## recurrence -- TWGem-R

Branch `feature/recurrence-mix`, pin `016b821` (master merged in). One change against the
default arm: the scorer mixes the geometry logit with a recurrence logit at a learned weight,

    logits = (1 - m) * layer_norm(-log d) + m * layer_norm(log1p(pair count before t))
    m = sigmoid(mix_logit), one scalar, init 0 so m = 0.5

where the pair count is Tempest's `get_pair_co_occurrence_counts(u, v, cutoff, Backward_In_Time)`.
That call does not advance Tempest's RNG, so at a fixed seed this arm samples bit-identical
walks to the default arm -- a one-variable comparison. `m` is logged as `mix=` per epoch.
Off (no `--use-recurrence-count`) the parameter does not exist and master is reproduced; the
flag is on master since `18bc6f8`, so the default arm's pin stays comparable with this one.
`--k-train 10` is passed explicitly: the branch forks before the commit that made 10 the default.

8 datasets x 3 seeds, 9 Slurm jobs. These are the eight cells CRAFT reports CRAFT-R on
(CRAFT + a raw repeat-time feature); the remaining ten default-arm datasets have no
recurrence arm.

| cell | jobs (seed 5 / 42 / 1) |
|---|---|
| tgbl-coin | 11695524 / 11699061 / 11699062 |
| tgbl-flight | 11696255 / 11696377 / 11696379 |
| DyGLib six | 11696356 / 11698293 / 11696358 |

tempest_rw 1.0.7 is required: 1.0.6 has no `get_pair_co_occurrence_counts`.

### Closed cells (2026-11-15 19:00)

`mix` is the gate's final logged value, `default` the same cell in the default arm at the same
seed. Walks are bit-identical between the two arms at a fixed seed, so each delta is the
recurrence term alone.

| dataset | seed | test@val-ckpt | max test | best val | sel/ran | mix | default | delta |
|---|---|---|---|---|---|---|---|---|
| reddit | 5 | 0.8937 | 0.8937 | 0.8928 | 2/12 | 0.0666 | 0.8618 | +0.0319 |
| reddit | 42 | 0.8936 | 0.8936 | 0.8924 | 2/12 | 0.0655 | pending | -- |
| reddit | 1 | 0.8941 | 0.8941 | 0.8929 | 2/12 | 0.0666 | 0.8637 | +0.0304 |
| wikipedia | 5 | 0.8920 | 0.8920 | 0.9067 | 8/18 | 0.1571 | 0.8784 | +0.0136 |
| wikipedia | 42 | 0.8938 | 0.8938 | 0.9079 | 7/17 | 0.1688 | 0.8803 | +0.0135 |
| wikipedia | 1 | 0.8915 | 0.8915 | 0.9078 | 8/18 | 0.1582 | 0.8792 | +0.0123 |
| uci | 5 | 0.7616 | 0.7656 | 0.7076 | 10/20 | 0.3295 | 0.7123 | +0.0493 |
| uci | 42 | 0.7541 | 0.7541 | 0.7064 | 10/20 | 0.3352 | pending | -- |
| uci | 1 | 0.7549 | 0.7549 | 0.7062 | 7/17 | 0.3649 | 0.6920 | +0.0629 |
| lastfm | 5 | 0.4956 | 0.5010 | 0.4754 | 2/12 | 0.0247 | 0.4495 | +0.0461 |
| lastfm | 42 | 0.4971 | 0.5030 | 0.4748 | 2/12 | 0.0249 | pending | -- |
| lastfm | 1 | 0.4951 | 0.5004 | 0.4743 | 2/12 | 0.0244 | 0.4519 | +0.0432 |
| Flights | 5 | 0.8354 | 0.8387 | 0.8400 | 2/12 | 0.0041 | 0.8042 | +0.0312 |
| Flights | 42 | 0.8361 | 0.8393 | 0.8401 | 2/12 | 0.0042 | pending | -- |
| Flights | 1 | 0.8362 | 0.8390 | 0.8406 | 2/12 | 0.0041 | 0.8035 | +0.0327 |
| mooc | 5 | 0.3830 | 0.3830 | 0.3915 | 7/17 | 0.0466 | 0.3757 | +0.0073 |
| mooc | 42 | 0.3815 | 0.3817 | 0.3882 | 9/19 | 0.0369 | pending | -- |
| mooc | 1 | 0.3812 | 0.3843 | 0.3915 | 10/20 | 0.0326 | 0.3754 | +0.0058 |
| tgbl-coin | 5 | 0.7732 | 0.7732 | 0.7566 | 5/15 | 0.0821 | 0.7502 | +0.0230 |

19 of 24 cells. **The DyGLib block is complete: 6 datasets x 3 seeds, 18 of 18.** Fourteen
deltas are measurable and **all fourteen are positive**, +0.4032 summed (+0.2024 at seed 5,
+0.1873 at seed 1, +0.0135 on the one seed-42 cell whose twin has closed). The five seed-42
deltas wait on the default arm's relaunched DyGLib job.

**The arm replicates tightly and `mix` is a property of the dataset, not the draw.** Five of
six DyGLib cells agree within 0.0023 across three seeds (reddit 0.0005, Flights 0.0008, mooc
0.0018, lastfm 0.0020, wikipedia 0.0023), uci spreads 0.0075 -- and `mix` lands on the same
per-dataset value to three or four decimals: reddit 0.0655-0.0666, Flights 0.0041-0.0042,
lastfm 0.0244-0.0249, wikipedia 0.157-0.169, mooc 0.033-0.047, uci 0.330-0.365.

Against CRAFT's own Table 3 the arm takes wikipedia (+0.95 over CRAFT-R 88.25 and +0.51 over
DyGFormer 88.69, the best number in that column), uci (+1.05 over 75.11, +1.55 over DyGFormer
76.61), reddit (+0.04 over 89.33) and Flights (+0.38 over 83.16, though DyGFormer 84.13 is
still ahead); lastfm (-5.65) and mooc (-24.02) stay well short. tgbl-coin 0.7732 is +2.63 over
CRAFT reproduced with corrected test negatives (74.69) and +1.75 over GraphMixer 75.57, but
-1.14 against SGNN-HN 78.46.

**Every DyGLib cell settles `mix` far below the 0.5 init** -- 0.004 to 0.36, i.e. the gate
keeps mostly geometry and uses the recurrence count as a small correction, and still gains
0.6 to 6.3 points. tgbl-coin runs the other way, rising to 0.68 at ep1 before decaying to
0.08, so the gate is dataset-adaptive rather than a constant that could be folded into the
scorer.

Open: tgbl-flight at all three seeds, tgbl-coin at seeds 42 and 1.

### Hardware moves in this arm

**The artemis-a40-10 hang.** `11696357` (this arm's DyGLib seed 42) and `11696628` (the default
arm's DyGLib seed 42) both landed on `artemis-a40-10` and hung at the Tempest ingest with zero
epoch lines, for 12 h 15 m and 9 h 52 m, on wikipedia -- a 157k-edge graph that takes about two
minutes an epoch on A40. The node was at cpu=128/128 allocated, CPULoad 106. Both were cancelled
on 2026-11-15 and resubmitted as `11698293` and `11698294` with `--exclude=artemis-a40-10`;
`11698293` then closed all five remaining seed-42 cells in 2 h 18 m on `artemis-a40-11`. The two
zero-epoch logs are parked under `logs/<arm>/aborted/` and are not archived here.

**tgbl-coin seeds 42 and 1 moved to A40** on 2026-11-15 (`11696376`/`11696378` cancelled, neither
having started, resubmitted as `11699061`/`11699062`), to clear two of the thirteen cells
queued for RTX. Seed 5 ran this cell entirely on `artemis-a40-14` and closed `rc=0`, so the
move has a same-cell precedent. Measured there: train 3,151 s and eval 1,276 s an epoch, or
2,707 s of eval on a new-best epoch when test runs too -- **4,427 s ordinary / 5,858 s
new-best**, 20.5 h for the 15-epoch cell. The default arm's coin on RTX is 480 s train and
152 s eval, so the gap on this cell is about 7x, which conflates the card with the arm since
default coin has never run on A40 and this arm's coin has never run on RTX. **tgbl-flight stays
off A40 at every seed**: at 7,110 s an epoch on RTX it would cost more there than the rest of
the sweep combined.

## k5_jobs -- TWGem at k_train 5

Code identical to the default arm (`78bbf8f`, clean tree, conf line read from the file); one
flag differs, `--k-train 5` against the arm's 10. Evaluation is untouched, so these compare
cell for cell with the default arm's Yelp and Taobao.

2 datasets x 3 seeds, 6 jobs. Seed 5 ran on `scripts/run_k5s.sh`; seeds 42 and 1 on
`scripts/run_k5b.sh`, which differs only in header text (per-seed baseline path, and it no
longer calls itself a side run). Logs in `logs/k5_side/seed_<S>`, tracked by
`logs/k5_side/MANIFEST.tsv`; `scripts/pe_status.sh logs/k5_side` prints the arm.

| cell | seed 5 | seed 42 | seed 1 |
|---|---|---|---|
| Yelp | `11690735` | `11696684` | `11696686` |
| Taobao | `11690736` | `11696685` | `11696687` |

Seed 5 ran on A40, seeds 42 and 1 on RTX like the default arm. Per-epoch times therefore do
not compare between seed 5 and the other two, nor between seed 5 and any default-arm cell;
the MRRs do.

### Closed cells (2026-11-15 02:49), seed 5

| cell | k5 test | best val | sel/ran | k10 s5 | k10 s42 | k10 s1 | delta vs best k10 |
|---|---|---|---|---|---|---|---|
| Taobao | 0.6514 | 0.6718 | 38/48 | 0.6440 | 0.6436 | 0.6477 | **+0.0037** |
| Yelp | 0.6862 | 0.7226 | 80/90 | 0.6849 | 0.6856 | live | **+0.0006** |

**The margin narrowed when the third k10 seed closed.** Taobao k10 seed 1 came in at 0.6477,
above both earlier k10 seeds, so k5 leads the best k10 cell by 0.0037 rather than the 0.0074
it led the first two by, and on val by 0.0010 (0.6718 against 0.6708) rather than clearly
above every k10 val. Taobao k10 also carries `r_max` 11.80 / 4.29 / 18.97 across its three
seeds -- a factor of four -- and the k5 margin sits inside that spread. Yelp's +0.0006 is
smaller than its own val deficit (k5 val 0.7226 against k10 0.7230 and 0.7247), so it lives
entirely in the val->test gap and is not evidence on its own.

Both cells closed on clean plateaus rather than escapes. Taobao held a 0.0021 val band for the
ten epochs after its ep38 checkpoint with train loss moving 0.0004/epoch, and its `r_max`
**contracted** 12.41 -> 9.28 over ep31-48, the first fall that cell's radius has shown. Yelp
held 0.7208-0.7219 over its last four epochs against a 0.7226 best.

Reporting Taobao at k=5 and the rest at k=10 would be a per-dataset hyperparameter, so the
arm is carried at full seed coverage rather than as a one-seed probe, and the decision is made
on three seeds against three. Seeds 42 and 1 are pending on RTX.

## Not in this tree

- **k5 beyond Yelp and Taobao**: no other dataset has a k5 cell. The arm exists because those
  two are the sweep's two largest deficits against CRAFT, not as a general k sweep.
- **tgbl-wiki**: no arm here. It exists only in `experiment_logs/craft_reproduction`.
- **Patent**: run and archived at both arms' default, but CRAFT does not report it, so it
  carries no comparison and stays out of the paper tables.
