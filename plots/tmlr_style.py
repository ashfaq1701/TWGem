"""Shared figure style for the paper.

Each figure is drawn at the width it is printed at, so the fonts come out at their true size.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH = 6.5                       # inches, the TMLR text width
GOLDEN = 0.618                         # default height-to-width ratio


def setup_style() -> None:
    """Sans-serif Computer Modern text, sized for the printed page."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["cmss10", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,           # cmss10 has no minus glyph
        "axes.formatter.use_mathtext": False,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,  # fixed figure size, labels fit inside
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "pdf.fonttype": 42,                    # embed TrueType, editable text
        "ps.fonttype": 42,
    })


def figure_quarter(aspect: float = 0.85):
    """Figure for one quarter of the text width (0.24\\linewidth)."""
    width = 0.24 * TEXT_WIDTH
    return plt.subplots(figsize=(width, aspect * width))


def figure_third(aspect: float = 0.8):
    """Figure for one third of the text width (0.32\\linewidth)."""
    width = 0.32 * TEXT_WIDTH
    return plt.subplots(figsize=(width, aspect * width))


def figure_half(aspect: float = GOLDEN):
    """Figure for half of the text width (0.49\\linewidth)."""
    width = 0.49 * TEXT_WIDTH
    return plt.subplots(figsize=(width, aspect * width))


def figure_full(aspect: float = 0.35):
    """Figure for the full text width (\\linewidth)."""
    return plt.subplots(figsize=(TEXT_WIDTH, aspect * TEXT_WIDTH))
