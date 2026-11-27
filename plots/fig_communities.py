"""GoogleLocal communities on the US map, next to the trained table's t-SNE.

Usage: python plots/fig_communities.py (after the analysis in analysis/communities/)
"""

import pathlib

import numpy as np

import tmlr_style as style

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "analysis" / "communities" / "out" / "googlelocal"
OUT = ROOT / "figures"

US_BOX = (-125.0, -66.5, 24.0, 49.5)
HEIGHT = 0.24 * style.TEXT_WIDTH                     # quarter width, shared by both panels
GREY = "#CBD0D6"
PALETTE = ["#1F4E9C", "#D55E00", "#7C3AED", "#E69F00", "#009E73", "#CC79A7"]


def draw_map(community, latlon, colours) -> None:
    biz = np.isfinite(latlon[:, 0]) & (community >= 0)
    lat, lon, lab = latlon[biz, 0], latlon[biz, 1], community[biz]
    rest = ~np.isin(lab, list(colours))

    fig, ax = style.figure_half(HEIGHT / (0.49 * style.TEXT_WIDTH))
    ax.scatter(lon[rest], lat[rest], s=0.3, c=GREY, linewidths=0, rasterized=True)
    for c, colour in colours.items():
        m = lab == c
        ax.scatter(lon[m], lat[m], s=1.2, c=colour, linewidths=0, rasterized=True)
    ax.set_xlim(US_BOX[:2])
    ax.set_ylim(US_BOX[2:])
    ax.set_aspect(1 / np.cos(np.radians(37)))          # equal distances at mid-US latitude
    ax.axis("off")
    fig.savefig(OUT / "communities_map.pdf")


def draw_tsne(xy, labels, colours) -> None:
    fig, ax = style.figure_quarter(1.0)
    for c, colour in colours.items():
        m = labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=1.6, c=colour, alpha=0.6, linewidths=0,
                   rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(OUT / "communities_tsne.pdf")


if __name__ == "__main__":
    style.setup_style()
    OUT.mkdir(exist_ok=True)
    tsne = np.load(DATA / "tsne.npz")
    colours = dict(zip(tsne["communities"].tolist(), PALETTE))
    draw_map(np.load(DATA / "community.npy"), np.load(DATA / "latlon.npy"), colours)
    draw_tsne(tsne["xy"], tsne["labels"], colours)
