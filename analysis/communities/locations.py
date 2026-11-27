"""Recover where each GoogleLocal business is, matched to our node ids.

Usage: python analysis/communities/locations.py (after communities.py GoogleLocal)
"""

import ast
import gzip
import json
import re
import subprocess
import tarfile
import types

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

import common as C

DATASET = "GoogleLocal"
MIRROR = "https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal"
FILES = ["reviews.clean.json.gz", "places.clean.json.gz"]


def fetch() -> None:
    """Download the two source files and pack them as the archive TGB-Seq's loader expects."""
    members = C.RAW / "members"
    members.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if not (members / name).exists():
            subprocess.run(["curl", "-sSL", "-o", str(members / name), f"{MIRROR}/{name}"],
                           check=True)
    tgz = C.RAW / DATASET / "googlelocal.tar.gz"
    tgz.parent.mkdir(parents=True, exist_ok=True)
    if not tgz.exists():
        with tarfile.open(tgz, "w:gz", compresslevel=1) as tar:
            for name in FILES:
                tar.add(members / name, arcname=f"googlelocal/{name}")


def preprocess() -> pd.DataFrame:
    """Rerun TGB-Seq's preprocessing to get each node's original id.

    Its source has f-strings broken across lines, so we rejoin them before running it.
    """
    named = C.RAW / DATASET / f"{DATASET}_named.csv"
    if not named.exists():
        import tgb_seq.datasets as pkg

        path = pkg.__path__[0] + "/preprocess.py"
        source = re.sub(r'(f"[^"\n]*\{)\s*\n\s*', r"\1", open(path).read())
        module = types.ModuleType("tgb_seq_preprocess")
        exec(compile(source, path, "exec"), module.__dict__)
        module.TGBSeqPreprocessor(DATASET, root=str(C.RAW))
    return pd.read_csv(named, dtype={"ori_src": "str", "ori_dst": "str"})


def check(named: pd.DataFrame) -> None:
    """Make sure the reprocessed edges are exactly the ones we train on (order aside)."""
    ours = pd.read_csv(C.DATA_ROOT / DATASET / f"{DATASET}.csv")
    cols = ["src", "dst", "time", "split"]
    rows = [df[cols].sort_values(cols).to_numpy() for df in (ours, named)]
    assert rows[0].shape == rows[1].shape and (rows[0] == rows[1]).all(), \
        "reprocessed GoogleLocal does not match the training data"


def place_gps(wanted: set) -> dict:
    """gPlusPlaceId -> (lat, lon) for the businesses in the training split."""
    gps = {}
    with gzip.open(C.RAW / "members" / "places.clean.json.gz", "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rec = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            ll = rec.get("gps") or [None, None]
            if rec["gPlusPlaceId"] in wanted and None not in ll[:2]:
                gps[rec["gPlusPlaceId"]] = (float(ll[0]), float(ll[1]))
    return gps


def regions_of(coords: np.ndarray) -> list:
    """Country code, or US-<state> for US points, by offline reverse geocoding."""
    import reverse_geocoder as rg

    hits = rg.search([tuple(c) for c in coords], mode=1)
    return [f"US-{h['admin1']}" if h["cc"] == "US" else h["cc"] for h in hits]


def main() -> None:
    fetch()
    named = preprocess()
    check(named)
    train = named[named["split"] == 0]
    lab = C.labels(DATASET)
    n = len(lab["community"])

    biz = train[["dst", "ori_dst"]].drop_duplicates()
    gps = place_gps(set(biz["ori_dst"]))
    biz = biz[biz["ori_dst"].isin(gps)]
    ids, coords = biz["dst"].to_numpy(), np.array([gps[p] for p in biz["ori_dst"]])

    latlon = np.full((n, 2), np.nan, dtype=np.float32)
    latlon[ids] = coords
    names = regions_of(coords)
    codes = {r: i for i, r in enumerate(sorted(set(names)))}
    region = np.full(n, -1, dtype=np.int64)
    region[ids] = [codes[r] for r in names]

    # a user takes the region of most of their training reviews
    votes = pd.DataFrame({"src": train["src"].to_numpy(), "r": region[train["dst"].to_numpy()]})
    votes = votes[votes["r"] >= 0]
    winner = votes.groupby("src")["r"].agg(lambda s: s.value_counts().idxmax())
    region[winner.index.to_numpy()] = winner.to_numpy()

    out = C.out_dir(DATASET)
    np.save(out / "latlon.npy", latlon)
    np.save(out / "region.npy", region)
    (out / "regions.json").write_text(json.dumps({i: r for r, i in codes.items()}, indent=2))

    sel = (lab["community"] >= 0) & (region >= 0)
    result = {"businesses_with_gps": len(ids), "regions": len(codes),
              "community_vs_region_ami": adjusted_mutual_info_score(lab["community"][sel],
                                                                    region[sel])}
    print(json.dumps(result, indent=2))
    (out / "locations.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
