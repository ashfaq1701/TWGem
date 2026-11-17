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
| `--loss` | BCE | argparse default; **unmeasured fork**, see below |
| `--dropout` `--step` `--scale` | 0.1 / 1 / 12 | argparse defaults |
| `--num_epochs` `--patience` | 200 / 15 | argparse defaults, same as our CRAFT runs |
| `--num_runs` `--seed` | 3 / 1 | seeds 1,2,3, matching `craft_reproduction/s_{1,2,3}` |

### Three code facts that shape the command

**`--output_dim` must equal `--embedding_size` or training crashes.** For every model except
CRAFT, `model = nn.Sequential(backbone, MergeLayer(input_dim1=args.output_dim,
input_dim2=args.output_dim, ...))` and the training loop applies `model[1]`. At the default
`output_dim=172` with `embedding_size=128`, `fc1 = Linear(344, 172)` receives 256 features and
raises.

**`--loss` is chosen outside the model.** `SGNNHN` stores `loss_type` and never reads it; the
loss comes from `get_loss_fn(args)`, which accepts only `BPR` or `BCE`. BCE is the default and
is what we ran first. BPR is what CRAFT used for itself and is the stronger ranking loss, so if
we publish a corrected SGNN-HN number we should run both and report the better one, or a
reviewer will say we under-tuned the baseline.

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
CRAFT's integration, not something we introduced, and it is worth a sentence if the number
comes out low.

## The command

Run from `/mnt/nfs2/inf/ms2420/craft_runs/wt-shift`:

```
/mnt/nfs2/inf/ms2420/CRAFT_fork/venv/bin/python train_link_prediction.py \
  --dataset_name tgbl-review --model_name SGNNHN --gpu 0 --dataset_path ./data/ \
  --batch_size 200 --embedding_size 128 --output_dim 128 --num_neighbors 120 \
  --dropout 0.1 --step 1 --scale 12 --loss BCE --learning_rate 0.0001 \
  --num_epochs 200 --patience 15 --num_runs 3 --seed 1 --shuffle --version sgnnhn
```

Driver: `/mnt/nfs2/inf/ms2420/craft_runs/sgnnhn_review_a40_2.sbatch` (A40, partition `gpu`,
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

## Status

`s_1`, `s_2` and `s_3` are empty. Job `11758651` is queued on A40 for all three seeds; CRAFT
cost 3,016 s a seed on RTX for this cell, so expect a few hours once it starts.

References for the cell once it lands:

| | tgbl-review MRR (%) |
|---|---|
| CRAFT paper, SGNN-HN (uncorrected) | 35.62 +- 0.53 |
| CRAFT paper, CRAFT (uncorrected) | 41.77 +- 0.06 |
| our CRAFT, corrected, 3 seeds | **22.71 +- 0.23** |
| our SGNN-HN, corrected | pending |

No independent third-party rerun of SGNN-HN exists for this dataset: MambaTG
(arXiv 2611.07569) reruns tgbl-coin 66.25, tgbl-flight 81.86 and tgbl-comment 40.38 against
CRAFT's 78.46 / 89.03 / 55.24, but does not report tgbl-review.

**Open question that decides whether this arm is needed at all.** The paper's own §5.1 says
"The results for all baselines on the TGB-Seq and TGB datasets are directly taken from previous
studies [40, 41]", where [40] is the TGB-Seq paper and [41] is Le Yu's DyGLib-TGB report. Taken
literally, SGNN-HN's four TGB numbers were not produced by the broken loader and nothing needs
re-running. The counter-evidence is that Table 3 also reports SGNN-HN on the six DyGLib
datasets, and SGNN-HN appears in neither [41] nor DyGLib [42], so the authors demonstrably ran
it themselves somewhere and that blanket sentence is loose. That is an argument from absence
and has not been checked against [40] or [41] directly; neither is on disk. Confirm that
neither reports SGNN-HN on tgbl-review/coin/comment/flight before writing any claim about it.
