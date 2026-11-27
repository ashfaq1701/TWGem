"""Does the trained table keep communities together? Compared with the untrained table and shuffled labels.

Usage: python analysis/communities/table_tests.py <dataset>
"""

import json
import sys

import igraph as ig
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

import common as C


def same_label(neighbours: np.ndarray, queries: np.ndarray, lab: np.ndarray) -> float:
    """Share of each query's neighbours that carry the query's own label."""
    return float((lab[neighbours] == lab[queries][:, None]).mean())


def nearest(bank: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Test 1: the K nearest rows, the query itself excluded."""
    return C.knn(bank, queries, C.K + 1)[:, 1:]


def nearest_non_partners(bank, queries, core, partners) -> np.ndarray:
    """Test 2: the K nearest rows after removing the query and its training partners."""
    width = min(len(bank), C.K + 1 + max(len(partners[core[q]]) for q in queries))
    found = C.knn(bank, queries, width)
    out = np.empty((len(queries), C.K), dtype=np.int64)
    for i, q in enumerate(queries):
        banned = np.append(partners[core[q]], q)
        out[i] = found[i][~np.isin(found[i], banned)][:C.K]
    return out


def recovered(bank: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Test 3: Leiden on the table's own 10-NN graph."""
    nn = C.knn(bank, np.arange(len(bank)), C.K + 1)[:, 1:]
    rows = np.repeat(np.arange(len(bank)), C.K)
    edges = np.unique(np.sort(np.stack([rows, nn.ravel()], 1), axis=1), axis=0)
    return C.leiden(ig.Graph(n=len(bank), edges=edges.tolist()), weighted=False)


def main(dataset: str) -> None:
    lab = C.labels(dataset)
    _, src, dst, n = C.train_graph(dataset)
    core = np.flatnonzero(lab["community"] >= 0)
    comm = lab["community"][core]

    # each core node's training partners, by position in `core` (-1 if not core)
    pos = np.full(n, -1, dtype=np.int64)
    pos[core] = np.arange(len(core))
    a, b = np.concatenate([src, dst]), np.concatenate([dst, src])
    order = np.argsort(a, kind="stable")
    split = np.cumsum(np.bincount(a, minlength=n))[:-1]
    partners = [pos[p][pos[p] >= 0] for p in np.split(b[order], split)]

    gen = np.random.default_rng(C.SEED)
    queries = gen.choice(len(core), size=min(C.N_SAMPLE, len(core)), replace=False)
    permuted = comm[gen.permutation(len(core))]
    trained = C.trained_table(dataset)[core]
    initial = C.initial_table(n)[core]

    result = {"chance": C.chance(comm), "test1": {}, "test2": {}, "test3": {}}
    for arm, bank in (("trained", trained), ("initial", initial)):
        nn1 = nearest(bank, queries)
        nn2 = nearest_non_partners(bank, queries, core, partners)
        found = recovered(bank, comm)
        result["test1"][arm] = same_label(nn1, queries, comm)
        result["test2"][arm] = same_label(nn2, queries, comm)
        result["test3"][arm] = adjusted_mutual_info_score(comm, found)
        if arm == "trained":
            result["test1"]["permuted"] = same_label(nn1, queries, permuted)
            result["test2"]["permuted"] = same_label(nn2, queries, permuted)
            result["test3"]["permuted"] = adjusted_mutual_info_score(permuted, found)
            deg = lab["degree"][core][queries]
            tercile = np.digitize(deg, np.quantile(deg, [1 / 3, 2 / 3]))
            result["test2"]["trained_by_degree_tercile"] = [
                same_label(nn2[tercile == t], queries[tercile == t], comm) for t in range(3)]

    print(json.dumps(result, indent=2))
    (C.out_dir(dataset) / "table_tests.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
