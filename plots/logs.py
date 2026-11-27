"""Readers for the archived training logs."""

import pathlib
import re
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_LOGS = ROOT / "experiment_logs" / "geometries" / "euclidean" / "default"
ABLATION_LOGS = ROOT / "experiment_logs" / "ablation"
TIMING_LOGS = ROOT / "experiment_logs" / "timing"

EPOCH_LINE = re.compile(
    r"^epoch (\d+)/\d+\s+link=([\d.]+).*?r_mean=([\d.]+)\s+r_max=([\d.]+)\s+val ([\d.]+)", re.M)

TRAIN_TIME = re.compile(r"^(?:epoch \d+/\d+|INFO:root:Epoch: \d+,).*?train_time ([\d.]+)s", re.M)
TEST_TIME = re.compile(r"test_time(?:_multi_negs)? ([\d.]+)s")


def read_epochs(path: pathlib.Path) -> Dict[str, List[float]]:
    """Per-epoch training loss, mean and max radius and validation MRR."""
    rows = EPOCH_LINE.findall(path.read_text())
    epoch, loss, r_mean, r_max, val = (list(map(float, c)) for c in zip(*rows))
    return {"epoch": [int(e) for e in epoch], "loss": loss,
            "r_mean": r_mean, "r_max": r_max, "val": val}


def default_log(dataset: str, seed: int) -> pathlib.Path:
    """Archived default-arm log of one dataset and seed."""
    return DEFAULT_LOGS / dataset / f"seed_{seed}.log"


def ablation_log(dataset: str, arm: str) -> pathlib.Path:
    """Archived log of one ablation arm on one dataset."""
    return ABLATION_LOGS / dataset / f"{arm}.log"


def timing_log(method: str, dataset: str) -> pathlib.Path:
    """Archived 3-epoch runtime log of one method on one dataset."""
    return TIMING_LOGS / method / f"{dataset}.log"


def read_timing(path: pathlib.Path) -> Dict[str, float]:
    """Training seconds per epoch (mean of epochs 2 and 3) and the last test pass in seconds."""
    text = path.read_text()
    train = [float(t) for t in TRAIN_TIME.findall(text)]
    return {"train": sum(train[1:3]) / 2, "test": float(TEST_TIME.findall(text)[-1])}


SENSITIVITY_LOGS = ROOT / "experiment_logs" / "sensitivity"
BEST_TEST = re.compile(r"best_test_mrr:\s+([\d.]+)")


def sensitivity_log(dataset: str, cell: str) -> pathlib.Path:
    """Archived sensitivity cell, one setting changed from the default run."""
    return SENSITIVITY_LOGS / dataset / f"{cell}.log"


def best_test(path: pathlib.Path) -> float:
    """Test MRR at the best-validation epoch, from the run's final summary."""
    return float(BEST_TEST.search(path.read_text()).group(1))
