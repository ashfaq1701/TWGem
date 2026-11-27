# Community analysis

Reproduces "What the table encodes" and its appendix: the communities of each training graph,
whether TWGem's trained table keeps them together, and, on GoogleLocal, whether it recovers
where businesses are.

Datasets: GoogleLocal, Flickr, YouTube. Everything is seeded 42. Communities are Leiden under
modularity at resolution 3.0, on the training split only; nodes with fewer than 5 training
edges are left unlabelled (`-1`).

## Inputs

- `TWGEM_DATA_ROOT`: the TGB-Seq datasets, as used for training (default `datasets/`).
- `TWGEM_EMBEDDING_DIR`: the exported best-validation tables,
  `<Dataset>_seed42_demb128_ep*.npy` (default `logs/embeddings/`).

Extra packages: `igraph`, `leidenalg`, `reverse_geocoder`, plus `tgb_seq` for `locations.py`.

## Run, in order

```
cd analysis/communities
for d in GoogleLocal Flickr YouTube; do python communities.py $d; done
python locations.py
for d in GoogleLocal Flickr YouTube; do python table_tests.py $d; done
python table_geography.py
python tsne.py
cd ../.. && python plots/fig_communities.py
```

| script | what it computes | writes (in `out/<dataset>/`) |
|---|---|---|
| `communities.py` | Leiden communities, modularity against a degree-preserving null, stability over seeds 42 to 46 | `community.npy`, `degree.npy`, `side.npy` (GoogleLocal), `communities.json` |
| `locations.py` | GoogleLocal business GPS and regions from the raw data, checked against the training data; AMI of communities against region | `latlon.npy`, `region.npy`, `regions.json`, `locations.json` |
| `table_tests.py` | Test 1: 10 nearest rows in the same community. Test 2: the same after removing training partners. Test 3: AMI of Leiden on the table's 10-NN graph. Controls: initial table, permuted labels | `table_tests.json` |
| `table_geography.py` | Same-region share of the 10 nearest businesses; location error of the median of their positions | `table_geography.json` |
| `tsne.py` | t-SNE of the trained table, six largest US communities | `tsne.npz` |

`common.py` holds the paths, constants and shared helpers. `locations.py` downloads about
1.7 GB of raw GoogleLocal data into `$TWGEM_DATA_ROOT/../googlelocal_raw/`.
