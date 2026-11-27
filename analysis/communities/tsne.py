"""GoogleLocal: a 2-D t-SNE map of the trained table for the six largest US communities.

Usage: python analysis/communities/tsne.py (after locations.py)
"""

import numpy as np
from sklearn.manifold import TSNE

import common as C

DATASET = "GoogleLocal"
US_BOX = (-125.0, -66.5, 24.0, 49.5)
N_COMMUNITIES = 6
PER_COMMUNITY = 1000


def largest_us_communities(community: np.ndarray, latlon: np.ndarray) -> np.ndarray:
    """The communities with the most businesses inside the continental US, largest first."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    us = (community >= 0) & (lon > US_BOX[0]) & (lon < US_BOX[1]) \
        & (lat > US_BOX[2]) & (lat < US_BOX[3])
    ids, counts = np.unique(community[us], return_counts=True)
    return ids[np.argsort(counts)[::-1][:N_COMMUNITIES]]


def main() -> None:
    lab = C.labels(DATASET)
    community = lab["community"]
    shown = largest_us_communities(community, lab["latlon"])

    gen = np.random.default_rng(C.SEED)
    nodes = np.concatenate([
        gen.choice(np.flatnonzero(community == c),
                   size=min(PER_COMMUNITY, int((community == c).sum())), replace=False)
        for c in shown])
    rows = C.trained_table(DATASET)[nodes].astype(np.float64)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=C.SEED,
              max_iter=1000).fit_transform(rows)

    np.savez(C.out_dir(DATASET) / "tsne.npz", xy=xy.astype(np.float32),
             labels=community[nodes], communities=shown)
    print(f"communities {shown.tolist()}, {len(nodes)} nodes")


if __name__ == "__main__":
    main()
