"""How test MRR moves when one setting changes, on Flickr and GoogleLocal.

Usage: python plots/fig_sensitivity.py
"""

import pathlib

import logs
import tmlr_style as style

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"

SEED = 42
DATASETS = {"flickr": "#1F4E9C", "googlelocal": "#C2410C"}
SWEEPS = {
    "len": ("Walk length $L$", [2, 3, 5, 10, 20], 5),
    "wpn": ("Walks per node $K$", [3, 5, 10, 20, 30, 50], 10),
    "neg": ("Training negatives $k$", [5, 10, 20], 10),
    "d": ("Width $d$", [32, 64, 128, 256], 128),
}


def mrr(dataset: str, cell: str) -> float:
    path = logs.default_log(dataset, SEED) if cell is None else logs.sensitivity_log(dataset, cell)
    return 100 * logs.best_test(path)


def plot_sweep(name: str, label: str, grid: list, default: int) -> None:
    fig, ax = style.figure_third()
    for dataset, colour in DATASETS.items():
        ys = [mrr(dataset, None if v == default else f"{name}{v}") for v in grid]
        ax.plot(grid, ys, color=colour, marker="o", markersize=3)
        ax.plot(default, ys[grid.index(default)], color=colour, marker="o", markersize=6,
                markerfacecolor="white", markeredgewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks(grid, [str(v) for v in grid])
    ax.minorticks_off()
    ax.set_xlabel(label)
    fig.savefig(OUT / f"sensitivity_{name}.pdf")


def plot_bias() -> None:
    fig, ax = style.figure_third()
    for i, (dataset, colour) in enumerate(DATASETS.items()):
        ax.bar([i - 0.18, i + 0.18], [mrr(dataset, None), mrr(dataset, "unibias")], 0.34,
               color=[colour, "white"], edgecolor=colour, linewidth=1.2)
    ax.set_xticks([0, 1], ["Flickr", "GoogleLocal"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("Walk bias")
    fig.savefig(OUT / "sensitivity_bias.pdf")


if __name__ == "__main__":
    style.setup_style()
    OUT.mkdir(exist_ok=True)
    for name, (label, grid, default) in SWEEPS.items():
        plot_sweep(name, label, grid, default)
    plot_bias()
