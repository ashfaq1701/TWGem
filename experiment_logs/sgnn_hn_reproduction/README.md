# SGNN-HN reproduction with corrected TGB test negatives

Sibling of `craft_reproduction/`, same purpose and same archive convention: CRAFT's published
TGB numbers were measured against test negatives whose node ids are offset by one, and this
directory holds our re-runs under the corrected loader.

**Why SGNN-HN needs its own directory.** CRAFT reports SGNN-HN on all four TGB datasets
(tgbl-review 35.62, tgbl-comment 55.24 in Table 2; tgbl-coin 78.46, tgbl-flight 89.03 in
Table 3), and those numbers go through the same broken path. `evaluate_models_utils.py:147`
-- inside `evaluate_model_link_prediction_multi_negs`, the function that returns the reported
`mrr` -- is an `elif model_name in ['SASRec', 'SGNNHN']` branch that concatenates the shifted positive
with the unshifted negatives (`items = torch.cat([pos_item.unsqueeze(1), neg_item], dim=1)`)
and hands them to `SGNNHN.predict`, which looks up `self.item_embedding(test_item)` directly.
The fix is one line in the shared loader (`utils/DataLoader.py`, `test_ns = test_ns + 1`), so
it repairs SGNN-HN with no model-side change.

This matters for our own table: SGNN-HN's uncorrected tgbl-review 35.62 and tgbl-coin 78.46
both beat TWGem (30.2 and 75.0), so quoting them as published would put two false losses in
front of us.

## Code

| | |
|---|---|
| repo | `ashfaq1701/CRAFT` fork at `/mnt/nfs2/inf/ms2420/CRAFT_fork` |
| branch / commit | `fix/tgb-test-negative-shift` @ `e77bf60` |
| worktree runs use | `/mnt/nfs2/inf/ms2420/craft_runs/wt-shift` (detached at the same commit) |
| python | `/mnt/nfs2/inf/ms2420/CRAFT_fork/venv/bin/python` |

`torch_scatter` and `torch_sparse` are absent from that venv; torch_geometric 2.6.1 falls back to
pure PyTorch and `SGNNHN` instantiates and runs without them.

## The configuration is RECONSTRUCTED, not recorded -- say so when reporting

Nothing in the repo or the paper records SGNN-HN's hyperparameters. Three independent checks:

- `load_link_prediction_best_configs()` has no `SGNNHN` branch and falls through to
  `raise ValueError`, so `--load_best_configs` cannot be used for this model.
- the README documents CRAFT only, with no SGNN-HN command.
- Table 6 is titled "Optimal hyperparameters ... **for CRAFT**".

What the paper does fix across every method is the batch size (200 on tgbl-review), the
learning rate (1e-4), the embedding size (128 for large datasets) and 3 runs.

| flag | value | provenance |
|---|---|---|
| `--batch_size` | 200 | paper, fixed across all methods (Table 6 `B`) |
| `--embedding_size` | 128 | paper, 128 for large datasets |
| `--learning_rate` | 1e-4 | paper, fixed across all methods |
| `--output_dim` | 128 | **forced by the code**, see below |
| `--num_neighbors` | 120 | **ASSUMPTION** -- CRAFT's Table 6 tgbl-review row, reused as SGNN-HN's `max_seq_length` |
| `--loss` | BPR | **forced by the code** -- BCE crashes in eval, see below |
| `--dropout` `--step` `--scale` | 0.1 / 1 / 12 | argparse defaults |
| `--num_epochs` `--patience` | 200 / 15 | argparse defaults, same as our CRAFT runs |
| `--num_runs` `--seed` | 3 / 1 | seeds 1,2,3, matching `craft_reproduction/s_{1,2,3}` |

### Three code facts that shape the command

**`--output_dim` must equal `--embedding_size` or training crashes.** For every model except
CRAFT, `model = nn.Sequential(backbone, MergeLayer(input_dim1=args.output_dim,
input_dim2=args.output_dim, ...))` and the training loop applies `model[1]`. At the default
`output_dim=172` with `embedding_size=128`, `fc1 = Linear(344, 172)` receives 256 features and
raises.

**`--loss` is chosen outside the model, and only BPR runs.** `SGNNHN` stores `loss_type` and
never reads it; the loss comes from `get_loss_fn(args)`, which accepts only `BPR` or `BCE`. BCE
is the argparse default and the first job (`11758651`) died on it after 902 s:
`Loss.cu:90: Assertion input_val >= zero && input_val <= one failed` at
`evaluate_models_utils.py:362`. The cause is that training sigmoids its scores through
`MergeLayer` while the eval branch calls `model[0].predict` with no sigmoid and then
`loss_func(pos, neg)` -- the `BPRLoss(pos_score, neg_score)` signature -- so `BCELoss` receives
out-of-range inputs and the negatives as its target. BPR is therefore the only loss this fork
admits for SGNN-HN, not a tuning choice, and it is also what CRAFT used for itself. The failed
run is kept outside git as `craft_runs/out/sgnnhn_review/tgbl-review.bce-crash.log`.

**`--scale` cannot change the metric.** `predict` returns `(seq_output . item_emb) * self.scale`,
a uniform positive factor, so it leaves the ranking and therefore MRR untouched.

**`--is_bipartite` must be omitted, never passed.** It is declared `type=bool`, so
`--is_bipartite False` evaluates to `True`. tgbl-review is not in the hardcoded bipartite list
at `train_link_prediction.py:318`, so omitting it gives `is_bipartite=False` and matches our
CRAFT run on this cell exactly.

### Train/eval scoring mismatch in CRAFT's SGNN-HN integration

Training optimises through `MergeLayer` -- an MLP on the concatenated source and destination
embeddings -- while evaluation scores with a dot product in `SGNNHN.predict` and ignores
`MergeLayer` entirely. So the head that is trained is not the head that is tested. This is
CRAFT's integration, not something we introduced.

**The original does not do this.** Upstream commit `81590a7` states the file is "adopted from
RecBole-GNN". There, `loss_type` is `'BPR'` or `'CE'` and **both** training paths score with the
same mean-centred, L2-normalised dot product times `scale` that `predict` and
`full_sort_predict` use, so training and ranking share one functional form; the CE path takes
cross-entropy over the whole item catalogue, which is how the SGNN-HN paper trains. This fork
has no CE path at all, and its BPR path routes through `MergeLayer`. Both differences are in
CRAFT's integration layer, not in our command line, and together they are the most likely
explanation of the number below -- say so when reporting it.

## The command

Run from `/mnt/nfs2/inf/ms2420/craft_runs/wt-shift`:

```
/mnt/nfs2/inf/ms2420/CRAFT_fork/venv/bin/python train_link_prediction.py \
  --dataset_name tgbl-review --model_name SGNNHN --gpu 0 --dataset_path ./data/ \
  --batch_size 200 --embedding_size 128 --output_dim 128 --num_neighbors 120 \
  --dropout 0.1 --step 1 --scale 12 --loss BPR --learning_rate 0.0001 \
  --num_epochs 200 --patience 15 --num_runs 3 --seed 1 --shuffle --version sgnnhn
```

Driver: `/mnt/nfs2/inf/ms2420/craft_runs/sgnnhn_review_a40_3.sbatch` (A40, partition `gpu`,
4 CPUs, 16 G). The memory figure is measured, not guessed: a review-only CRAFT job peaked at
`MaxRSS 3.74 GB` (job `11592561`), and an earlier 96 G request could not be scheduled on any
A40 node because the only two with a free card had 20.6 G and 7.2 G unallocated.

## Layout

One file per seed per dataset, same as `craft_reproduction/`:

```
sgnn_hn_reproduction/s_{1,2,3}/tgbl-review.log
```

`--num_runs 3` writes all three seeds into one stdout file, so it is split per seed on the way
in, exactly as `craft_runs/out/s23/flight.log` was split into `s_2` and `s_3`. Each archived
file opens with a header giving the dataset and seed, the branch and commit, the worktree, the
slurm lane and its completion marker, CRAFT's own per-run logger path, the full command line,
and this run's test metrics block; then the body, with tqdm redraws collapsed by
`sed 's/.*\r//'`. Verify the cleaning preserved every line by comparing
`grep -ac 'INFO:root:'` on the raw source against `grep -ac '^INFO:root:'` on the cleaned file.

Only `rc=0` runs are archived.

## Status: closed 2026-11-18, three seeds

Job `11760185` ran all three seeds in one process on `artemis-a40-00`, COMPLETED `0:0` in
14:56:53 (53,812 s), and `s_1`, `s_2`, `s_3` hold one split log each.

| seed | test MRR | test AP | test ROC-AUC | epochs | best val AP | at ep | cost |
|---|---|---|---|---|---|---|---|
| 1 | 0.05128 | 0.54112 | 0.54522 | 26 | 0.55620 | 11 | 21,503 s |
| 2 | 0.04947 | 0.51554 | 0.52538 | 24 | 0.50019 | 9 | 19,629 s |
| 3 | 0.04535 | 0.49219 | 0.48934 | 16 | 0.48937 | **1** | 12,495 s |
| **mean** | **0.04870 +- 0.00304** | 0.51628 +- 0.02447 | 0.51998 +- 0.02833 | | | | |

**This is a collapse, not a weak result, and it must not be reported as a failed reproduction
of 35.62.** 0.0487 is where uniform-random scoring lands on this cell: tgbl-review serves ~100
negatives per positive, and random MRR there is about 0.052. Test AP 0.516 and ROC-AUC 0.520
against one random negative say the same thing from the other side. The run is a baseline at
a RECONSTRUCTED config, and two things in CRAFT's integration are the prime suspects before any
hyperparameter is: the train/eval scoring-form mismatch and the missing CE path, both above.
Seed 3 shows the degenerate version -- best val at epoch 1, then sixteen epochs downhill.

What a defensible SGNN-HN baseline would need next, in order: the CE objective the original
trains with, or at minimum BPR on the same dot product the metric reads; then a session length
chosen for this dataset rather than borrowed from CRAFT. Neither was run.

References for the cell:

| | tgbl-review MRR (%) |
|---|---|
| CRAFT paper, SGNN-HN (uncorrected) | 35.62 +- 0.53 |
| CRAFT paper, CRAFT (uncorrected) | 41.77 +- 0.06 |
| our CRAFT, corrected, 3 seeds | **22.71 +- 0.23** |
| our SGNN-HN, corrected, 3 seeds | **4.87 +- 0.30** (see the caveat above) |

No independent third-party rerun of SGNN-HN exists for this dataset: MambaTG
(arXiv 2611.07569) reruns tgbl-coin 66.25, tgbl-flight 81.86 and tgbl-comment 40.38 against
CRAFT's 78.46 / 89.03 / 55.24, but does not report tgbl-review.

**Settled 2026-11-18: the arm is needed.** The paper's §5.1 says "The results for all baselines
on the TGB-Seq and TGB datasets are directly taken from previous studies [40, 41]", which taken
literally would mean SGNN-HN's four TGB numbers never touched the broken loader. Both sources
were checked directly:

- **[40], TGB-Seq (arXiv 2502.02975).** SGNN-HN appears only in its Table 3, on ML-20M, Taobao,
  Yelp, GoogleLocal, wikipedia and reddit. No `tgbl-*` dataset. It also gives no SGNN-HN
  hyperparameters; its only fixed settings are "a learning rate of 0.0001 across all methods and
  datasets" and batch 200/400.
- **[41], Le Yu, "An Empirical Evaluation of Temporal Graph Benchmark" (arXiv 2307.12510).**
  Its eleven methods are JODIE, DyRep, TGAT, TGN, CAWN, EdgeBank, TCL, GraphMixer, DyGFormer,
  Persistent Forecast and Moving Average. SGNN-HN does not appear anywhere in the paper.

So SGNN-HN's tgbl-review/coin/comment/flight numbers exist in no prior study and were produced
by the CRAFT authors with this code, i.e. through the unshifted `test_ns`. The §5.1 sentence is
loose, as the DyGLib rows already suggested. Our own corrected number is not yet a usable
replacement for them -- see the Status caveat -- but the premise of the arm holds.
