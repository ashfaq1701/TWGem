"""Validation MRR and training loss over epochs, one figure per dataset.

Usage: python plots/fig_training_curves.py (UCI comes from its longer patience-50 run)
"""

import pathlib

from matplotlib.ticker import MaxNLocator, MultipleLocator

import logs
import tmlr_style as style

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"

SEED = 42
DATASETS = {"wikilink": "WikiLink", "googlelocal": "GoogleLocal", "flickr": "Flickr", "uci": "UCI"}
EXTENDED = {"uci": 50}                        # datasets drawn from a longer run, and its patience
PATIENCE = 10

C_VAL = "#1F4E9C"
C_LOSS = "#9CA3AF"
C_MARK = "#6B7280"
C_SHADE = "#EEF0F3"


def span(runs: dict, key: str) -> tuple:
    """One shared axis range for a quantity across all runs, with a small margin."""
    lo = min(min(r[key]) for r in runs.values())
    hi = max(max(r[key]) for r in runs.values())
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def read(dataset: str) -> dict:
    if dataset in EXTENDED:
        return logs.read_epochs(logs.ROOT / "experiment_logs" / "extended" / f"{dataset}_seed{SEED}.log")
    return logs.read_epochs(logs.default_log(dataset, SEED))


def plot(run: dict, dataset: str, title: str, val_lim: tuple, loss_lim: tuple) -> None:
    epoch, val, loss = run["epoch"], run["val"], run["loss"]
    best = epoch[val.index(max(val))]

    fig, ax = style.figure_quarter()
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1)       # validation drawn above loss and shading
    ax.patch.set_visible(False)

    ax2.axvspan(best, best + EXTENDED.get(dataset, PATIENCE), color=C_SHADE, lw=0, zorder=0)
    ax2.plot(epoch, loss, color=C_LOSS, lw=0.9)
    ax2.set_ylim(loss_lim)
    ax2.yaxis.set_major_locator(MultipleLocator(0.5))
    ax2.tick_params(axis="y", colors=C_LOSS)
    ax2.spines["right"].set_visible(True)

    ax.axvline(best, color=C_MARK, lw=0.8, ls=":")
    ax.plot(epoch, val, color=C_VAL)
    ax.set_ylim(val_lim)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", colors=C_VAL)
    ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    ax.set_xlabel("Epoch")
    ax.set_title(title)

    fig.savefig(OUT / f"training_curves_{dataset}.pdf")


if __name__ == "__main__":
    style.setup_style()
    OUT.mkdir(exist_ok=True)
    runs = {d: read(d) for d in DATASETS}
    val_lim, loss_lim = span(runs, "val"), span(runs, "loss")
    for dataset, title in DATASETS.items():
        plot(runs[dataset], dataset, title, val_lim, loss_lim)
