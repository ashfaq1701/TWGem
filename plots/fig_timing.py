"""Training and test time of TWGem, CRAFT and JODIE.

Usage: python plots/fig_timing.py
"""

import pathlib

import logs
import tmlr_style as style

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"

DATASETS = ["Flights", "Yelp", "WikiLink"]
METHODS = {"TWGem": "#1F4E9C", "CRAFT": "#9CA3AF", "JODIE": "#D1D5DB"}
WIDTH = 0.26


def plot(quantity: str) -> None:
    times = {m: [logs.read_timing(logs.timing_log(m, d))[quantity] / 60 for d in DATASETS]
             for m in METHODS}

    fig, ax = style.figure_half()
    offset = (len(METHODS) - 1) / 2
    for j, m in enumerate(METHODS):
        xs = [i + (j - offset) * WIDTH for i in range(len(DATASETS))]
        ax.bar(xs, times[m], WIDTH, color=METHODS[m])
    for i in range(len(DATASETS)):
        x = i - offset * WIDTH
        ax.text(x, times["TWGem"][i], f"{times['CRAFT'][i] / times['TWGem'][i]:.1f}x",
                ha="center", va="bottom", fontsize=8, color=METHODS["TWGem"])
    ax.set_xticks(range(len(DATASETS)), DATASETS)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Minutes")

    fig.savefig(OUT / f"timing_{quantity}.pdf")


if __name__ == "__main__":
    style.setup_style()
    OUT.mkdir(exist_ok=True)
    plot("train")
    plot("test")
