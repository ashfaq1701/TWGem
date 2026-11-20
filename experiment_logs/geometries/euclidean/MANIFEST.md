# Euclidean geometry: the paper runs

Two arms, three seeds each (5, 42, 1), one log per cell. Archive layout:

    euclidean/<arm>/<dataset>/seed_<S>.log

Only runs that closed `rc=0` are copied here, from `logs/` once finished. Record BOTH
test@val-checkpoint and max test: on this suite val flickers walk the checkpointed number
down, so the printed `best_test_mrr` alone is not like-for-like across arms.

Populated 2026-11-15, completed 2026-11-19 at **78 of 78 cells** -- default 54, recurrence 24.
**Both arms are closed.** Every dataset ran all three seeds to `rc=0`, bar the one waived
`rc=135` below.

**The six DyGLib cells in both arms, and ML-20M in the default arm, run
`--num-walks-per-node 30`.** The DyGLib six were re-run at wpn 30 on 2026-11-18/19 and ML-20M on
2026-11-19/20; in both cases these logs replaced the wpn 10 ones, which are one revision back in
git history. `default_conf_variations.txt` carries wpn 30 on those seven lines, so both arm
definitions below still hold literally. The eleven remaining datasets stay at the argparse
default wpn 10.

## default -- TWGem

Master `78bbf8f`, **`5711eee` for the six DyGLib cells** and **`953af61` for ML-20M**. The only
difference among those pins inside `link_property_prediction/` is the epoch line's timing split
(`train_time`/`val_time`/`test_time` instead of `train`/`eval`), so the model and the data path
are identical and a wpn 10 cell compares with a wpn 30 one on one variable. Every knob is the
argparse default except the per-dataset flags in `default_conf_variations.txt`, reproduced
verbatim on each log's command line.
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

**Taobao runs `--k-train 5`**, which `default_conf_variations.txt` now carries on its line, so it
is a documented per-dataset flag like `--is-bipartite` and not an undocumented departure from the
arm. It is the only dataset in the suite whose conf line sets a k. Measured against k_train 10 at
the same three seeds: +0.0074, +0.0082 and +0.0028, mean 0.6512 against 0.6451, and it replicates
tighter as well (spread 0.0013 against 0.0041). The three Taobao logs here are therefore the k5
sweep's, jobs `11690736` / `11738156` / `11738157` from `logs/k5_side/seed_<S>/Taobao.log`: their
headers read `experiment=k5_side` and `k_train=5`, name `logs/paper_euclidean` as their baseline,
and come from `scripts/run_k5s.sh` / `run_k5b.sh` / `run_k5t2.sh` rather than `run_pe.sh`. Code,
pin and evaluation are identical to the rest of the arm. Seeds 42 and 1 were moved to A40 on
2026-11-17 (`11696685`/`11696687` cancelled unstarted, resubmitted on
`--partition=gpu --gres=gpu:A40:1`) after sitting on `QOSMaxGRESPerUser`; seed 5 had already run
this cell on `artemis-a40-02`, so all three are A40 and their per-epoch times do not compare with
an RTX cell's.

**The six DyGLib datasets run `--num-walks-per-node 30`**, carried on their conf lines the same
way, so they too are documented per-dataset flags rather than a departure from the arm. Measured
against wpn 10 at the same three seeds, as three-seed mean of test@val-ckpt: mooc 0.4038 against
0.3733 (**+0.0305**), lastfm 0.4784 against 0.4516 (**+0.0268**), uci 0.7079 against 0.7009
(+0.0071), wikipedia 0.8806 against 0.8793 (+0.0013), reddit 0.8637 against 0.8630 (+0.0007),
Flights 0.8048 against 0.8043 (+0.0005). **It never loses a cell and the gain is concentrated in
the two weakest datasets**: mooc and lastfm carry 86% of the +0.0669 summed over the six, and the
other four move less than their own three-seed spread. The gain orders inversely with baseline
MRR across all six, which is a regularity over six cells and not a tested mechanism. Cost is
2.1-3.0x the wpn 10 wall clock on the same node and dataset. Jobs `11763778`-`11763780`, one seed
each, from `scripts/run_dwa30.sh`; their headers read `experiment=dyg_wpn30` and `wpn=30`.

### Closed cells (2026-11-15 19:00, Taobao 2026-11-18, DyGLib at wpn 30 2026-11-19)

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
| Flickr | 1 | 0.6621 | 0.6633 | 0.7069 | 45/55 |
| Patent | 5 | 0.0417 | 0.0417 | 0.0793 | 1/11 |
| Patent | 42 | 0.0418 | 0.0418 | 0.0748 | 1/11 |
| Patent | 1 | 0.0387 | 0.0387 | 0.0755 | 1/11 |
| ML-20M | 5 | 0.3022 | 0.3022 | 0.3737 | 2/12 |
| ML-20M | 42 | 0.3015 | 0.3015 | 0.3731 | 2/12 |
| ML-20M | 1 | 0.3023 | 0.3023 | 0.3737 | 2/12 |
| Taobao | 5 | 0.6514 | 0.6514 | 0.6718 | 38/48 |
| Taobao | 42 | 0.6518 | 0.6518 | 0.6723 | 36/46 |
| Taobao | 1 | 0.6505 | 0.6505 | 0.6713 | 38/48 |
| Yelp | 5 | 0.6849 | 0.6854 | 0.7230 | 40/50 |
| Yelp | 42 | 0.6856 | 0.6869 | 0.7247 | 49/59 |
| Yelp | 1 | 0.6906 | 0.6906 | 0.7288 | 83/93 |
| WikiLink | 5 | 0.8207 | 0.8207 | 0.8398 | 136/146 |
| WikiLink | 42 | 0.8198 | 0.8198 | 0.8387 | 103/113 |
| WikiLink | 1 | 0.8191 | 0.8191 | 0.8382 | 84/94 |
| tgbl-review | 5 | 0.3017 | 0.3040 | 0.2503 | 3/13 |
| tgbl-review | 42 | 0.3010 | 0.3013 | 0.2504 | 3/13 |
| tgbl-review | 1 | 0.3024 | 0.3024 | 0.2482 | 2/12 |
| tgbl-coin | 5 | 0.7502 | 0.7502 | 0.7404 | 14/24 |
| tgbl-coin | 42 | 0.7485 | 0.7500 | 0.7387 | 17/27 |
| tgbl-coin | 1 | 0.7517 | 0.7517 | 0.7423 | 14/24 |
| tgbl-comment | 5 | 0.7592 | 0.7592 | 0.7222 | 17/27 |
| tgbl-comment | 42 | 0.7592 | 0.7598 | 0.7216 | 11/21 |
| tgbl-comment | 1 | 0.7631 | 0.7631 | 0.7239 | 18/28 |
| tgbl-flight | 5 | 0.8126 | 0.8126 | 0.8344 | 2/12 |
| tgbl-flight | 42 | 0.8133 | 0.8133 | 0.8347 | 2/12 |
| tgbl-flight | 1 | 0.8136 | 0.8136 | 0.8351 | 2/12 |
| wikipedia | 5 | 0.8794 | 0.8808 | 0.9009 | 13/23 |
| wikipedia | 42 | 0.8815 | 0.8817 | 0.9026 | 8/18 |
| wikipedia | 1 | 0.8808 | 0.8808 | 0.9012 | 8/18 |
| reddit | 5 | 0.8629 | 0.8644 | 0.8769 | 11/21 |
| reddit | 42 | 0.8643 | 0.8654 | 0.8769 | 12/22 |
| reddit | 1 | 0.8639 | 0.8659 | 0.8777 | 14/24 |
| mooc | 5 | 0.4007 | 0.4007 | 0.4137 | 11/21 |
| mooc | 42 | 0.4050 | 0.4081 | 0.4153 | 10/20 |
| mooc | 1 | 0.4058 | 0.4070 | 0.4160 | 11/21 |
| lastfm | 5 | 0.4766 | 0.4766 | 0.4740 | 3/13 |
| lastfm | 42 | 0.4805 | 0.4844 | 0.4728 | 3/13 |
| lastfm | 1 | 0.4781 | 0.4821 | 0.4759 | 4/14 |
| uci | 5 | 0.7138 | 0.7188 | 0.6773 | 6/16 |
| uci | 42 | 0.7082 | 0.7141 | 0.6733 | 8/18 |
| uci | 1 | 0.7018 | 0.7018 | 0.6767 | 7/17 |
| Flights | 5 | 0.8052 | 0.8052 | 0.8187 | 3/13 |
| Flights | 42 | 0.8053 | 0.8053 | 0.8183 | 3/13 |
| Flights | 1 | 0.8039 | 0.8040 | 0.8180 | 4/14 |

**54 of 54 cells -- the default arm is closed.** All 18 datasets ran all three seeds to
`rc=0`, bar the one waived `rc=135` below.

**tgbl-comment seed 5 exited `rc=135` and the cell is valid anyway.** All 27 epochs ran,
patience fired at 10/10, the checkpoint was restored and the final block printed; the process
then took SIGBUS (`128 + 7`) during interpreter teardown, where the 44.3M-edge Tempest ingest
and the CUDA context both release. Slurm recorded the job COMPLETED 0:0 and MaxRSS was 31.8 GB
against a 64 GB request, so this is not a memory failure. The run is therefore reported, and the
`rc=0`-only archive rule is waived for this one log.

Replication across all three seeds, as max-min of test MRR, ascending: ML-20M 0.0008,
tgbl-flight 0.0010, Taobao 0.0013, Flights 0.0014, reddit 0.0014, tgbl-review 0.0014,
WikiLink 0.0016, wikipedia 0.0021, Patent 0.0031, tgbl-coin 0.0032, Flickr 0.0034, lastfm 0.0039,
tgbl-comment 0.0039, mooc 0.0051, Yelp 0.0057, **uci 0.0120**, YouTube 0.0128, GoogleLocal
0.0158. **Fifteen of eighteen datasets replicate inside 0.0070 and the median spread is 0.0032.**
uci is the loosest DyGLib cell and the smallest dataset in the arm (110-905 s a run);
GoogleLocal's 0.0158 is an artefact of the epoch cap, two of its three seeds having been stopped
at 197-199 of 200 while still improving. tgbl-review selects at ep2-3, Patent at ep1 and ML-20M at
ep2 in all three seeds. **wpn 30 did not cost replication**: ML-20M tightened 0.0025 -> 0.0008
and is now the arm's tightest cell; five of the six DyGLib spreads
tightened (uci 0.0203 -> 0.0120, mooc 0.0069 -> 0.0051, reddit 0.0019 -> 0.0014, Flights
0.0016 -> 0.0014, lastfm 0.0040 -> 0.0039) and wikipedia moved 0.0019 -> 0.0021.

**GoogleLocal is an epoch-cap floor at two of three seeds** -- sel 199/200 and 197/200, both
still improving when the 200-epoch cap stopped them. WikiLink seed 5 is not: it selected at
ep136 of 146 and patience fired, but its last 45 epochs bought +0.0015 test while `r_mean` grew
a near-constant +0.0385/epoch from 6.49 to 8.52, so most of that run was radius inflation
rather than learning.

## recurrence -- TWGem-R

Branch `feature/recurrence-mix`, pin `016b821` (master merged in) for tgbl-coin and tgbl-flight;
**the six DyGLib cells ran at `7884d4e`**, the tip of `feature/twgem-r-do-not-remove`, which is
`016b821`'s lineage plus a per-phase timing cherry-pick that does not touch the model. The code
is preserved on that branch and fixed by the annotated tag **`twgem-r-results`** at `f793d56`,
which is master carrying the channel plus this archive as it stood on 2026-11-15; the wpn 30
logs post-date the tag, so the tag fixes the code to reproduce with and not this log set. Cite
the tag: a branch can move or be deleted, the tag cannot. **`ba17bda` removed the channel from
master** -- these logs stay here, the code does not.

One change against the default arm: the scorer mixes the geometry logit with a recurrence
logit at a learned weight,

    logits = (1 - m) * layer_norm(-log d) + m * layer_norm(log1p(pair count before t))
    m = sigmoid(mix_logit), one scalar, init 0 so m = 0.5

where the pair count is Tempest's `get_pair_co_occurrence_counts(u, v, cutoff, Backward_In_Time)`.
That call does not advance Tempest's RNG, so at a fixed seed this arm samples bit-identical
walks to the default arm -- a one-variable comparison. `m` is logged as `mix=` per epoch.
Off (no `--use-recurrence-count`) the parameter does not exist and master is reproduced. The
flag was on master from `18bc6f8` until `ba17bda` removed it, and both arms' pins predate the
removal, so the default arm's pin stays comparable with this one. **Master's tip reproduces
the default arm but can no longer run this one**: with the flag off the code is what it was,
so nothing about the default arm changed.
`--k-train 10` is passed explicitly: the branch forks before the commit that made 10 the default.

8 datasets x 3 seeds, 9 Slurm jobs. These are the eight cells CRAFT reports CRAFT-R on
(CRAFT + a raw repeat-time feature); the remaining ten default-arm datasets have no
recurrence arm.

| cell | jobs (seed 5 / 42 / 1) |
|---|---|
| tgbl-coin | 11695524 / 11699061 / 11699062 |
| tgbl-flight | 11696255 / 11696377 / 11696379 |
| DyGLib six, wpn 30 | 11763781 / 11763782 / 11763783 |

The DyGLib six first ran at wpn 10 on `11696356` / `11698293` / `11696358`; those logs are one
revision back in git history. The wpn 30 re-run is from `scripts/run_rwa30.sh` and its logs read
`experiment=rcm_dyg_wpn30`. `016b821` and `7884d4e` differ only by the same epoch-line timing
cherry-pick that separates the default arm's two pins, so wpn 10 and wpn 30 compare on one
variable in this arm too.

tempest_rw 1.0.7 is required to reproduce this arm: 1.0.6 has no
`get_pair_co_occurrence_counts`. Master's `requirements.txt` went back to `>=1.0.6` at
`ba17bda`, so reproduce from `feature/twgem-r-do-not-remove`, not from master.

### Closed cells (2026-11-15 19:00, DyGLib at wpn 30 and tgbl-flight 2026-11-19)

`mix` is the gate's final logged value, `default` the same cell in the default arm at the same
seed. Walks are bit-identical between the two arms at a fixed seed, so each delta is the
recurrence term alone.

| dataset | seed | test@val-ckpt | max test | best val | sel/ran | mix | default | delta |
|---|---|---|---|---|---|---|---|---|
| reddit | 5 | 0.8935 | 0.8936 | 0.8929 | 2/12 | 0.0649 | 0.8629 | +0.0306 |
| reddit | 42 | 0.8935 | 0.8935 | 0.8929 | 2/12 | 0.0640 | 0.8643 | +0.0292 |
| reddit | 1 | 0.8940 | 0.8940 | 0.8930 | 2/12 | 0.0653 | 0.8639 | +0.0301 |
| wikipedia | 5 | 0.8910 | 0.8923 | 0.9068 | 12/22 | 0.1242 | 0.8794 | +0.0116 |
| wikipedia | 42 | 0.8933 | 0.8942 | 0.9085 | 8/18 | 0.1584 | 0.8815 | +0.0118 |
| wikipedia | 1 | 0.8914 | 0.8917 | 0.9078 | 8/18 | 0.1576 | 0.8808 | +0.0106 |
| uci | 5 | 0.7633 | 0.7636 | 0.7106 | 12/22 | 0.3081 | 0.7138 | +0.0495 |
| uci | 42 | 0.7548 | 0.7608 | 0.7075 | 17/27 | 0.2708 | 0.7082 | +0.0466 |
| uci | 1 | 0.7557 | 0.7557 | 0.7099 | 7/17 | 0.3602 | 0.7018 | +0.0539 |
| lastfm | 5 | 0.5116 | 0.5147 | 0.4981 | 2/12 | 0.0148 | 0.4766 | +0.0350 |
| lastfm | 42 | 0.5111 | 0.5190 | 0.4951 | 2/12 | 0.0148 | 0.4805 | +0.0306 |
| lastfm | 1 | 0.5087 | 0.5168 | 0.4965 | 2/12 | 0.0147 | 0.4781 | +0.0306 |
| Flights | 5 | 0.8418 | 0.8418 | 0.8406 | 1/11 | 0.0060 | 0.8052 | +0.0366 |
| Flights | 42 | 0.8406 | 0.8406 | 0.8393 | 1/11 | 0.0058 | 0.8053 | +0.0353 |
| Flights | 1 | 0.8413 | 0.8413 | 0.8403 | 1/11 | 0.0059 | 0.8039 | +0.0374 |
| mooc | 5 | 0.4021 | 0.4050 | 0.4171 | 7/17 | 0.0496 | 0.4007 | +0.0014 |
| mooc | 42 | 0.4064 | 0.4064 | 0.4191 | 11/21 | 0.0301 | 0.4050 | +0.0014 |
| mooc | 1 | 0.4042 | 0.4080 | 0.4217 | 21/31 | 0.0092 | 0.4058 | **-0.0016** |
| tgbl-coin | 5 | 0.7732 | 0.7732 | 0.7566 | 5/15 | 0.0821 | 0.7502 | +0.0230 |
| tgbl-coin | 42 | 0.7723 | 0.7723 | 0.7556 | 6/16 | 0.1243 | 0.7485 | +0.0238 |
| tgbl-coin | 1 | 0.7714 | 0.7714 | 0.7549 | 5/15 | 0.0337 | 0.7517 | +0.0197 |
| tgbl-flight | 5 | 0.8664 | 0.8664 | 0.8762 | 1/11 | 0.1722 | 0.8126 | +0.0538 |
| tgbl-flight | 42 | 0.8655 | 0.8655 | 0.8758 | 1/11 | 0.1764 | 0.8133 | +0.0522 |
| tgbl-flight | 1 | 0.8659 | 0.8659 | 0.8758 | 1/11 | 0.1761 | 0.8136 | +0.0523 |

**24 of 24 cells -- the recurrence arm is closed.** Summed over all 24 the arm is **+0.7054**
ahead of the default arm and the total is even across seeds: +0.2415 at seed 5, +0.2309 at
seed 42, +0.2330 at seed 1, eight cells each.

**23 of 24 deltas are positive and one is not: mooc seed 1 at -0.0016.** That is the first cell
this arm has lost anywhere, and it arrived with wpn 30 -- at wpn 10 mooc paid +0.0073 / +0.0127 /
+0.0058. The dataset mean is now +0.0004, i.e. recurrence and walk breadth are substitutes on
mooc and breadth has taken the whole of what the gate was supplying. lastfm shows the same
compression (+0.0443 at wpn 10, +0.0321 here), and so, mildly, do wikipedia (+0.0131 -> +0.0113)
and uci (+0.0560 -> +0.0500). Only Flights moved the other way (+0.0316 -> +0.0364).

**The arm replicates tightly.** Five of six DyGLib cells agree within 0.0043 across three seeds
(reddit 0.0005, Flights 0.0012, wikipedia 0.0023, lastfm 0.0029, mooc 0.0043) and uci spreads
0.0085; tgbl-flight is the tightest cell in either arm at 0.0009.

**`mix` is a property of the dataset on three cells and not on the other three.** reddit
0.0640-0.0653, Flights 0.0058-0.0060 and lastfm 0.0147-0.0148 land on the same value to three
or four decimals, as every DyGLib cell did at wpn 10. wikipedia (0.124-0.158), uci (0.271-0.360)
and above all mooc (0.0092-0.0496, a 5x range) do not. mooc is also the cell whose seeds
disagree on the sign of the delta, and its seed 1 ran 31 epochs against 17 and 21 -- so where
the gate's settled value is unstable, so is the gain.

Against CRAFT's own Table 3, on the three-seed mean of test@val-ckpt in MRR %, the arm takes
wikipedia 89.19 (**+0.94** over CRAFT-R 88.25, **+0.50** over DyGFormer 88.69, the best number
in that column), reddit 89.37 (**+0.04** over 89.33), Flights 84.12 (**+0.96** over 83.16, and
now level with DyGFormer 84.13 at -0.01, where wpn 10 was -0.54) and uci 75.79 (**+0.68** over
75.11, but **-0.82** against DyGFormer 76.61). lastfm 51.05 and mooc 40.42 stay well short, by
roughly 4 and 22 points on the same basis. **Re-check every CRAFT and DyGFormer figure against
the papers before these go in a table**: the baselines here are the ones recorded in this file
earlier, the deltas above are recomputed from them, and lastfm's and mooc's baselines are
derived from previously recorded deltas rather than read fresh. tgbl-coin 0.7732 is +2.63 over
CRAFT reproduced with corrected test negatives (74.69) and +1.75 over GraphMixer 75.57, but
-1.14 against SGNN-HN 78.46.

**Every DyGLib cell settles `mix` far below the 0.5 init** -- 0.006 to 0.36, i.e. the gate
keeps mostly geometry and uses the recurrence count as a small correction, and the per-cell
deltas run from -0.2 to +5.4 points. tgbl-coin runs the other way, rising to 0.68 at ep1 before
decaying to 0.08, so the gate is dataset-adaptive rather than a constant that could be folded
into the scorer.

**tgbl-coin replicates to 0.0018 across three seeds** -- 0.7732 / 0.7723 / 0.7714, tighter
than the default arm's own 0.0032 on the same cell -- while `mix` spreads 0.0337 to 0.1243,
an order of magnitude wider than any DyGLib cell. So the gain is stable even where the gate's
settled value is not, which argues the recurrence channel matters less by its final weight than
by having been available during the first few epochs.

**tgbl-flight is the arm's largest and cleanest gain: +0.0527 on the mean** (0.8659 against
0.8132), the three seeds inside 0.0009 on both sides. All three select at **epoch 1 of 11** and
then decline for ten straight epochs while `mix` falls 0.64 -> 0.17 and `r_mean` grows
near-linearly 1.13 -> 2.8, so the whole of the gain is in the first pass, before the geometry
inflates -- the same one-epoch pattern the TGB notes in CLAUDE.md describe for this suite. The
three cells cost 206,835 s, 221,728 s and 252,574 s of wall clock on RTX (2.4-2.9 days each),
the most expensive cells in either arm.

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

## Not in this tree

- **a separate k5 arm**: there is none any more. Taobao's k5 cells became the default arm's
  Taobao when `--k-train 5` went into `default_conf_variations.txt`, and the Yelp k5 cells were
  deleted. Yelp's measured k5 numbers, for the record: 0.6862 / 0.6851 / 0.6801 at seeds 5 / 42 /
  1 against k10's 0.6849 / 0.6856 / 0.6906, i.e. +0.0013, -0.0005 and -0.0105, mean 0.6838 against
  0.6870. k5 does not help Yelp, nearly all of the loss is seed 1 (selected ep25, stopped ep35
  against 90 and 121 epochs elsewhere, best val 0.7133 against 0.7222/0.7226), and the other two
  seeds are a tie. No dataset other than Yelp and Taobao ever had a k5 cell.
- **tgbl-wiki**: no arm here. It exists only in `experiment_logs/craft_reproduction`.
- **Patent**: run and archived at both arms' default, but CRAFT does not report it, so it
  carries no comparison and stays out of the paper tables.
