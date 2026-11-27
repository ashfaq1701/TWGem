"""Shared paths, data loaders and helpers for the community analysis."""

import os
import pathlib
import sys

import igraph as ig
import leidenalg as la
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.environ.get("TWGEM_DATA_ROOT", ROOT / "datasets"))
TABLES = pathlib.Path(os.environ.get("TWGEM_EMBEDDING_DIR", ROOT / "logs" / "embeddings"))
RAW = DATA_ROOT.parent / "googlelocal_raw"
OUT = pathlib.Path(__file__).resolve().parent / "out"

DATASETS = ["GoogleLocal", "Flickr", "YouTube"]
SEED = 42
RESOLUTION = 3.0
MIN_DEG = 5
D_EMB = 128
K = 10
N_SAMPLE = 20_000

sys.path.insert(0, str(ROOT))


def out_dir(dataset: str) -> pathlib.Path:
    d = OUT / dataset.lower()
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_split(dataset: str):
    """Training edges exactly as TWGem trains on them, and the node count."""
    from link_property_prediction.tgb_seq_eval import load_tgb_seq

    data = load_tgb_seq(dataset, root=str(DATA_ROOT))
    return data.train, int(data.max_node_count)


def train_graph(dataset: str):
    """Training edges as one undirected graph on all nodes, repeated pairs merged into a weight."""
    train, n = train_split(dataset)
    src, dst = np.asarray(train.sources), np.asarray(train.destinations)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keys, counts = np.unique(lo.astype(np.int64) * n + hi, return_counts=True)
    g = ig.Graph(n=n, edges=np.stack([keys // n, keys % n], 1).tolist(), directed=False)
    g.es["weight"] = counts.tolist()
    return g, src, dst, n


def leiden(g: ig.Graph, seed: int = SEED, weighted: bool = True) -> np.ndarray:
    """Leiden communities under modularity at RESOLUTION."""
    part = la.find_partition(g, la.RBConfigurationVertexPartition,
                             weights="weight" if weighted else None,
                             resolution_parameter=RESOLUTION, seed=seed, n_iterations=5)
    return np.asarray(part.membership, dtype=np.int64)


def labels(dataset: str) -> dict:
    """The per-node arrays written by communities.py and locations.py."""
    d = out_dir(dataset)
    return {p.stem: np.load(p) for p in d.glob("*.npy")}


def trained_table(dataset: str) -> np.ndarray:
    """The exported best-validation table at seed 42, row i = node i."""
    (path,) = TABLES.glob(f"{dataset}_seed{SEED}_demb{D_EMB}_ep*.npy")
    return np.load(path)


def initial_table(n: int) -> np.ndarray:
    """The table training starts from, rebuilt with the model's own constructor at seed 42."""
    from link_property_prediction.model import LinkPredHead

    head = LinkPredHead(num_nodes=n, T_train=1, max_walk_len=5, d_emb=D_EMB,
                        hidden_dim=32, n_layers_pooler=2, seed=SEED)
    return head.E.weight.detach().numpy()[:n]


def knn(bank: np.ndarray, queries: np.ndarray, k: int, chunk: int = 512) -> np.ndarray:
    """Exact Euclidean k nearest rows of `bank` for each query row index, nearest first."""
    X = torch.from_numpy(np.ascontiguousarray(bank, dtype=np.float32))
    sq = (X * X).sum(1)
    out = np.empty((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = torch.from_numpy(queries[s:s + chunk].astype(np.int64))
        d2 = sq[q].unsqueeze(1) - 2.0 * (X[q] @ X.T) + sq.unsqueeze(0)
        out[s:s + chunk] = d2.topk(k, dim=1, largest=False).indices.numpy()
    return out


def chance(lab: np.ndarray) -> float:
    """Probability that two nodes drawn at random share a label."""
    _, counts = np.unique(lab, return_counts=True)
    p = counts / counts.sum()
    return float((p * p).sum())


def haversine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between two [n, 2] arrays of (lat, lon) degrees."""
    la1, lo1, la2, lo2 = map(np.radians, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
