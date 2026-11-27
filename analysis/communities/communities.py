"""Find the communities of a training graph and check they are real and stable.

Usage: python analysis/communities/communities.py <dataset>
"""

import json
import random
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score

import common as C

STABILITY_SEEDS = [42, 43, 44, 45, 46]


def null_modularity(g) -> float:
    """Modularity of Leiden on a degree-preserving rewiring of the same graph."""
    random.seed(C.SEED)
    h = g.copy()
    h.rewire(n=10 * h.ecount())
    h.es["weight"] = g.es["weight"]
    return h.modularity(C.leiden(h).tolist(), weights="weight")


def main(dataset: str) -> None:
    g, src, dst, n = C.train_graph(dataset)
    degree = np.asarray(g.strength(weights="weight"), dtype=np.int64)
    core = degree >= C.MIN_DEG

    runs = [C.leiden(g, seed) for seed in STABILITY_SEEDS]
    community = runs[0].copy()
    community[~core] = -1
    stability = [adjusted_rand_score(a[core], b[core])
                 for i, a in enumerate(runs) for b in runs[i + 1:]]

    out = C.out_dir(dataset)
    np.save(out / "community.npy", community)
    np.save(out / "degree.npy", degree)
    if dataset == "GoogleLocal":
        side = np.full(n, -1, dtype=np.int64)
        side[np.unique(dst)] = 1
        side[np.unique(src)] = 0
        np.save(out / "side.npy", side)

    result = {"nodes": n, "core_nodes": int(core.sum()),
              "core_communities": int(len(np.unique(community[core]))),
              "modularity": g.modularity(runs[0].tolist(), weights="weight"),
              "null_modularity": null_modularity(g),
              "stability_ari": float(np.mean(stability))}
    print(json.dumps(result, indent=2))
    (out / "communities.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
