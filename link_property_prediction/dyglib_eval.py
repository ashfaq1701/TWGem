"""DyGLib datasets: Zenodo download, quantile split, CRAFT's collision-free negatives."""
import hashlib
import os
import shutil
import subprocess
import zipfile
from typing import Dict, NamedTuple, Tuple

import numpy as np

from .data import Loaded, SplitData, node_count
from .evaluator import DataSuite, Evaluator, FixedNegativesEvaluator
from .negatives import CollisionFreeNegativeSampler, NegativeSampler

ZENODO_RECORD = "https://zenodo.org/api/records/7213796"
VAL_QUANTILE, TEST_QUANTILE = 0.70, 0.85


def _paths(name: str, root: str) -> Tuple[str, str, str]:
    """(data_dir, ml_<name>.csv, ml_<name>.npy) under `root`."""
    data_dir = os.path.join(root, name)
    return data_dir, os.path.join(data_dir, f"ml_{name}.csv"), os.path.join(data_dir, f"ml_{name}.npy")


def _fetch(name: str, root: str) -> None:
    """Download and md5-check `<name>.zip`, extracting only ml_<name>.csv/.npy."""
    import requests

    data_dir, csv_path, npy_path = _paths(name, root)
    if os.path.exists(csv_path) and os.path.exists(npy_path):
        return

    record = requests.get(ZENODO_RECORD, timeout=60)
    record.raise_for_status()
    files = {f["key"]: f for f in record.json()["files"]}
    entry = files.get(f"{name}.zip")
    if entry is None:
        available = sorted(k[:-4] for k in files if k.endswith(".zip"))
        raise ValueError(f"DyGLib dataset {name!r} is not in the Zenodo record; "
                         f"available: {available}")

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip.part")
    print(f"  [dyglib] downloading {name}.zip ({entry['size'] / 1e6:.0f} MB) from Zenodo ...")
    md5 = hashlib.md5()
    with requests.get(entry["links"]["self"], stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                md5.update(chunk)

    expected = entry["checksum"].split(":", 1)[1]
    if md5.hexdigest() != expected:
        os.remove(zip_path)
        raise IOError(f"DyGLib {name}.zip failed its md5 check "
                      f"(got {md5.hexdigest()}, record says {expected}); removed, retry.")

    wanted = {f"{name}/ml_{name}.csv": csv_path, f"{name}/ml_{name}.npy": npy_path}
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for member, dest in wanted.items():
                with zf.open(member) as packed, open(dest, "wb") as out:
                    shutil.copyfileobj(packed, out)
    except NotImplementedError:
        # Deflate64 (reddit.zip) needs Info-ZIP.
        subprocess.run(["unzip", "-o", "-j", zip_path, *wanted, "-d", data_dir], check=True,
                       stdout=subprocess.DEVNULL)
    os.remove(zip_path)
    print(f"  [dyglib] extracted ml_{name}.csv / .npy into {data_dir}")


def _integral_time_scale(name: str, ts: np.ndarray) -> int:
    """Smallest power of ten making every timestamp integral."""
    for k in range(7):
        scaled = ts * 10 ** k
        if np.abs(scaled - np.round(scaled)).max() < 1e-3:
            return 10 ** k
    raise ValueError(f"{name}: timestamps are not integral at any scale up to 1e6")


class DyGLibDataset(NamedTuple):
    """`Loaded.dataset` for this suite: what DyGLib carries beyond the shared splits."""
    labels: Dict[str, np.ndarray]       # "train" / "val" / "test" -> [E_split] float32
    time_scale: int                     # int64 timestamps = shipped float ts * time_scale


def load_dyglib(name: str, root: str = "datasets") -> Loaded:
    """Native DyGLib load -> Loaded; labels ride on Loaded.dataset."""
    import pandas as pd

    _fetch(name, root)
    _, csv_path, npy_path = _paths(name, root)
    df = pd.read_csv(csv_path)
    edge_feat_table = np.load(npy_path)

    ts_float = df.ts.to_numpy(dtype=np.float64)
    time_scale = _integral_time_scale(name, ts_float)
    if not np.all(np.diff(ts_float) >= 0):
        raise ValueError(f"{name}: edges are not in time order")

    src = df.u.to_numpy(dtype=np.int64)
    dst = df.i.to_numpy(dtype=np.int64)
    ts = np.round(ts_float * time_scale).astype(np.int64)
    edge_feat = edge_feat_table[df.idx.to_numpy()].astype(np.float32)
    labels = df.label.to_numpy(dtype=np.float32)

    # DyGLib's split: quantiles of the float timestamps.
    val_time, test_time = np.quantile(ts_float, [VAL_QUANTILE, TEST_QUANTILE])
    masks = {
        "train": ts_float <= val_time,
        "val": (ts_float > val_time) & (ts_float <= test_time),
        "test": ts_float > test_time,
    }

    full = SplitData(sources=src, destinations=dst, timestamps=ts, edge_feat=edge_feat)

    return Loaded(
        train=full.mask(masks["train"]),
        val=full.mask(masks["val"]),
        test=full.mask(masks["test"]),
        dataset=DyGLibDataset(labels={k: labels[m] for k, m in masks.items()},
                              time_scale=time_scale),
        name=name,
        max_node_count=node_count(src, dst),
    )


class DyGLibSuite(DataSuite):
    """DyGLib adapter: fixed collision-free val/test negatives, scored by MRR."""

    def _load(self) -> Loaded:
        return load_dyglib(self.name, self.root)

    def _val_evaluator(self) -> Evaluator:
        neg = self.eval_negatives(self.load().val, self.seed, tag="[dyglib] val")
        return FixedNegativesEvaluator(neg_dst=neg)

    def _test_evaluator(self) -> Evaluator:
        # One seed apart from val.
        neg = self.eval_negatives(self.load().test, self.seed + 1, tag="[dyglib] test")
        return FixedNegativesEvaluator(neg_dst=neg)

    def negative_sampler(self, pool: np.ndarray, k: int, seed: int,
                         edges: SplitData) -> NegativeSampler:
        return CollisionFreeNegativeSampler(num_neg_per_pos=k, dst_pool=pool, edges=edges, seed=seed)
