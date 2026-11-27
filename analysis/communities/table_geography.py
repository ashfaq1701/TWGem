"""GoogleLocal: are a business's nearest rows in the table also near it on the map?

Usage: python analysis/communities/table_geography.py (after locations.py)
"""

import json

import numpy as np

import common as C

DATASET = "GoogleLocal"


def nearest(bank: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return C.knn(bank, queries, C.K + 1)[:, 1:]


def same(lab: np.ndarray, neighbours: np.ndarray, queries: np.ndarray) -> float:
    """Share of each query's neighbours that carry the query's own label."""
    return float((lab[neighbours] == lab[queries][:, None]).mean())


def location_error(neighbours: np.ndarray, queries: np.ndarray, coords: np.ndarray) -> dict:
    """Estimate each location as the median of its neighbours' locations; error in km."""
    err = C.haversine(np.median(coords[neighbours], axis=1), coords[queries])
    return {"median_km": float(np.median(err)), "p90_km": float(np.quantile(err, 0.9)),
            "within_100km": float((err < 100).mean())}


def main() -> None:
    lab = C.labels(DATASET)
    n = len(lab["community"])
    trained, initial = C.trained_table(DATASET), C.initial_table(n)
    gen = np.random.default_rng(C.SEED)
    result = {}

    # region agreement: core businesses with a region
    biz = np.flatnonzero((lab["community"] >= 0) & (lab["side"] == 1) & (lab["region"] >= 0))
    region = lab["region"][biz]
    queries = gen.choice(len(biz), size=min(C.N_SAMPLE, len(biz)), replace=False)
    nn = nearest(trained[biz], queries)
    result["same_region"] = {"chance": C.chance(region),
                             "trained": same(region, nn, queries),
                             "initial": same(region, nearest(initial[biz], queries), queries),
                             "permuted": same(region[gen.permutation(len(biz))], nn, queries)}

    # location: every business with GPS
    biz = np.flatnonzero((lab["side"] == 1) & np.isfinite(lab["latlon"][:, 0]))
    coords = lab["latlon"][biz].astype(np.float64)
    queries = gen.choice(len(biz), size=min(C.N_SAMPLE, len(biz)), replace=False)
    nn = nearest(trained[biz], queries)
    result["location"] = {
        "trained": location_error(nn, queries, coords),
        "initial": location_error(nearest(initial[biz], queries), queries, coords),
        "permuted": location_error(nn, queries, coords[gen.permutation(len(biz))])}

    print(json.dumps(result, indent=2))
    (C.out_dir(DATASET) / "table_geography.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
