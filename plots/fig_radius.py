"""How the embedding radius grows during training, with and without standardisation.

Usage: python plots/fig_radius.py
"""

import pathlib

from matplotlib.ticker import LogLocator, NullFormatter, NullLocator, StrMethodFormatter

import logs
import tmlr_style as style

OUT = pathlib.Path(__file__).resolve().parent.parent / "figures"

SEED = 42
DATASETS = {"flickr": "Flickr", "googlelocal": "GoogleLocal"}

C_DEFAULT = "#1F4E9C"
C_NOSTD = "#C2410C"


def plot(dataset: str, title: str) -> None:
    default = logs.read_epochs(logs.default_log(dataset, SEED))
    nostd = logs.read_epochs(logs.ablation_log(dataset, "nostd"))

    fig, ax = style.figure_third()
    for run, colour in ((default, C_DEFAULT), (nostd, C_NOSTD)):
        ax.plot(run["epoch"], run["r_max"], color=colour)
        ax.plot(run["epoch"], run["r_mean"], color=colour, ls="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Epoch")
    ax.set_title(title)

    fig.savefig(OUT / f"radius_{dataset}.pdf")


if __name__ == "__main__":
    style.setup_style()
    OUT.mkdir(exist_ok=True)
    for dataset, title in DATASETS.items():
        plot(dataset, title)
