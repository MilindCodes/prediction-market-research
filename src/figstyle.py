"""Shared publication settings for the §4 figures.

The §4 figures were written straight to 130-dpi PNG, which is below what a
journal will accept and rasterises text that then blurs when the figure is
scaled to column width.  Everything here funnels through `save_figure`, which
writes a vector PDF (what LaTeX should \\includegraphics) alongside a 300-dpi
PNG for quick viewing.

Font sizes are set in points against a fixed figure width, so a figure scaled
to a 3.4in column keeps its labels legible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Publication defaults ------------------------------------------------------
FIG_DPI: int = 300
FIG_FORMATS: tuple[str, ...] = ("pdf", "png")   # PDF first: the LaTeX target

RC_PUBLICATION: dict = {
    "figure.dpi":        110,
    "savefig.dpi":       FIG_DPI,
    "savefig.bbox":      "tight",
    "pdf.fonttype":      42,      # embed TrueType, not Type-3 — required by
    "ps.fonttype":       42,      # most journals' PDF checkers
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   10,
    "figure.titlesize":  13,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "lines.linewidth":   1.4,
}


def apply_publication_style() -> None:
    """Install the publication rcParams process-wide."""
    mpl.rcParams.update(RC_PUBLICATION)


def save_figure(
    out_base: str | Path,
    dpi: int = FIG_DPI,
    formats: tuple[str, ...] = FIG_FORMATS,
    close: bool = True,
) -> list[Path]:
    """Save the current figure once per format at publication settings.

    Parameters
    ----------
    out_base : str | Path
        Output path.  Any extension is stripped and replaced per format, so
        both ``fig.png`` and ``fig`` produce ``fig.pdf`` and ``fig.png``.
    dpi : int
        Raster resolution.  Ignored by vector formats.
    formats : tuple[str, ...]
        Extensions to write, without the dot.
    close : bool
        Close the figure afterwards (the callers all do).

    Returns
    -------
    list[Path]
        Paths written, in `formats` order.
    """
    base = Path(out_base).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for ext in formats:
        out = base.with_suffix(f".{ext}")
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        written.append(out)
    if close:
        plt.close()
    return written
