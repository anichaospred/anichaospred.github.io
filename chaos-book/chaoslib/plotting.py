"""The book's figure design system: one palette, two styling helpers.

Lifted verbatim (colours and layout values) from the Lorenz 63 notebook already
already published at aneeshcs.com/chaos, so the two look like the same
publication. The violet accent is this book's identity, distinct
from the GFD textbook's sky-blue.

Semantic, not decorative: each colour means one thing across every chapter, so a
reader who has learned that rose is "the perturbed run" never has to relearn it.
Import these rather than choosing colours per figure.

Two figure kinds, deliberately:

* **Plotly** (:func:`style2d`, :func:`style3d`) -- interactive panels, for chapters
  where hovering over a curve or reading a value off it is part of the point.
* **matplotlib** (:func:`mpl_panels`, :func:`finish_mpl`) -- *static* figures, for
  phase-space projections. A rotatable 3-D scene looks impressive and is usually
  worse than three flat projections: the reader has to find a good viewpoint before
  they can see anything, the projection they end up with is unrepeatable, and it costs
  far more in the browser. Where the shape of an attractor is the message, a fixed
  x-z, x-y, x(t) row says it better.

The same semantic colours serve both.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "C_CONTEXT",
    "C_TRUTH",
    "C_PERT",
    "C_SPREAD",
    "C_MEAN",
    "C_FIXED",
    "C_SAT",
    "C_START",
    "C_OBS",
    "C_BG",
    "C_ANALYSIS",
    "SCENE_BG",
    "FONT",
    "TIME_SCALE",
    "MPL_DIVERGING",
    "MPL_SEQUENTIAL",
    "style2d",
    "style3d",
    "MPL_RC",
    "mpl_panels",
    "mpl_grid",
    "finish_mpl",
    "mpl_colour",
]

# ---- semantic palette (violet accent: this book's identity) ----
C_CONTEXT = "rgba(150,150,165,0.16)"  # faint reference attractor, drawn behind
C_TRUTH = "#3730a3"  # indigo    -- truth / control run
C_PERT = "#e11d48"  # rose      -- perturbed forecast
C_SPREAD = "#7c3aed"  # violet    -- error / spread curves
C_MEAN = "#0f766e"  # teal      -- ensemble mean
C_FIXED = "#f59e0b"  # amber     -- fixed points
C_SAT = "#b91c1c"  # firebrick -- saturation level
C_START = "#10b981"  # emerald   -- start marker

# Data assimilation adds three roles. They are distinguished from the forecast
# colours above because a DA figure routinely shows all six at once, and the
# reader has to be able to tell an observation from a perturbed forecast at a
# glance.
C_OBS = "#ea580c"  # orange    -- observations (noisy, discrete)
C_BG = "#64748b"  # slate     -- background / free forecast (the thing DA improves on)
C_ANALYSIS = "#0891b2"  # cyan      -- analysis (the DA estimate)
SCENE_BG = "rgba(250,249,255,0.92)"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
TIME_SCALE = "Plasma"  # colour = time along a trajectory

# Colour *maps*, for fields rather than lines. The choice is semantic and fixed
# here so that no figure picks its own: a field that takes both signs is drawn
# with a diverging map about zero, so that the sign is legible and zero is not a
# colour; a non-negative field gets a perceptually uniform sequential map.
# Getting this backwards -- a sequential map on a signed field -- hides the sign
# and invents a gradient across zero that the data does not have.
MPL_DIVERGING = "RdBu_r"  # signed fields: deviations, tendencies, errors
MPL_SEQUENTIAL = "plasma"  # non-negative fields, and colour-as-time


def style2d(fig: Any, height: int = 460, title: str | None = None) -> Any:
    """Apply the house 2-D layout to a Plotly figure, in place. Returns ``fig``.

    Light ground with low-contrast violet-tinted gridlines: the data should be
    the only saturated thing in the frame.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=12, color="#211d33"),
        margin=dict(l=64, r=24, t=54 if title else 24, b=52),
        legend=dict(
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#e6e1f2",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title, x=0.5, xanchor="center", font=dict(size=13)
            )
        )
    fig.update_xaxes(gridcolor="#ece8f6", zerolinecolor="#ddd6ee")
    fig.update_yaxes(gridcolor="#ece8f6", zerolinecolor="#ddd6ee")
    return fig


def style3d(
    fig: Any,
    height: int = 500,
    title: str | None = None,
    eye: tuple[float, float, float] = (1.5, 1.1, 0.8),
    axis_titles: tuple[str, str, str] = ("X", "Y", "Z"),
) -> Any:
    """Apply the house 3-D scene style to a Plotly figure, in place.

    The default camera ``eye`` is chosen to show both lobes of the Lorenz
    attractor without the classic degenerate head-on view. ``axis_titles`` is
    parameterised because not every 3-D figure in the book is in
    :math:`(X,Y,Z)`.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="white",
        showlegend=True,
        font=dict(family=FONT, size=12, color="#211d33"),
        margin=dict(l=0, r=0, t=52 if title else 0, b=0),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.86)",
            bordercolor="#e6e1f2",
            borderwidth=1,
            font=dict(size=10),
        ),
    )
    if title:
        fig.update_layout(
            title=dict(
                text=title, x=0.5, xanchor="center", font=dict(size=13)
            )
        )
    fig.update_scenes(
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        zaxis_title=axis_titles[2],
        bgcolor=SCENE_BG,
        xaxis=dict(
            gridcolor="#e5e0f2", backgroundcolor=SCENE_BG, showbackground=True
        ),
        yaxis=dict(
            gridcolor="#e5e0f2", backgroundcolor=SCENE_BG, showbackground=True
        ),
        zaxis=dict(
            gridcolor="#e5e0f2", backgroundcolor=SCENE_BG, showbackground=True
        ),
        camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])),
    )
    return fig


# --------------------------------------------------------------------------
# Static (matplotlib) figures
# --------------------------------------------------------------------------
#: House rcParams for static figures. Kept small and explicit rather than a
#: stylesheet file, so a chapter can see exactly what it is getting.
MPL_RC: dict[str, object] = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": "#c9c2de",
    "axes.labelcolor": "#211d33",
    "axes.titlecolor": "#211d33",
    "axes.grid": True,
    "grid.color": "#ece8f6",
    "grid.linewidth": 0.7,
    "xtick.color": "#6b6580",
    "ytick.color": "#6b6580",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#e6e1f2",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def mpl_colour(colour: str) -> str | tuple[float, float, float, float]:
    """Translate a palette entry into something matplotlib accepts.

    The palette is written in CSS, because Plotly consumes it directly. Two entries
    (:data:`C_CONTEXT` and :data:`SCENE_BG`) use the ``rgba(r,g,b,a)`` form, which
    matplotlib rejects outright -- it wants 0-1 float tuples or hex. Hex entries pass
    through unchanged.

    Always route a palette colour through this function in a matplotlib figure.
    Passing ``C_CONTEXT`` straight to ``ax.plot`` raises, and it raises at *render*
    time, which in a notebook means the cell fails rather than the import.
    """
    if not isinstance(colour, str):
        return colour
    text = colour.strip()
    if text.startswith("#"):
        return text
    if text.startswith(("rgba(", "rgb(")):
        inner = text[text.index("(") + 1 : text.rindex(")")]
        parts = [float(v) for v in inner.split(",")]
        r, g, b = (v / 255.0 for v in parts[:3])
        alpha = parts[3] if len(parts) > 3 else 1.0
        return (r, g, b, alpha)
    return text  # a named colour; matplotlib knows its own names


def mpl_panels(
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    titles: tuple[str, ...] | None = None,
    height: float = 3.6,
):
    """A row of matplotlib panels with the book's static-figure styling.

    Returns ``(fig, axes)`` with ``axes`` always a 1-D array, so a caller does not
    have to special-case ``ncols == 1``.

    ``figsize`` defaults to a width that scales with the panel count, which keeps
    the aspect ratio of a phase-space projection roughly square whether the figure
    has two panels or four.

    The house rcParams are applied through a context-free ``rcParams.update`` on the
    figure's own axes rather than globally, so importing chaoslib never mutates a
    reader's matplotlib state.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (3.7 * ncols, height)
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(1, ncols, figsize=figsize, squeeze=False)
    axes = axes[0]
    for ax in axes:
        _apply_axes_style(ax)
    if titles:
        for ax, title in zip(axes, titles):
            ax.set_title(title)
    return fig, axes


def mpl_grid(
    nrows: int = 2,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
    titles: tuple[str, ...] | None = None,
    panel: tuple[float, float] = (3.7, 3.3),
):
    """A grid of styled matplotlib panels. Returns ``(fig, axes_flat)``.

    The companion to :func:`mpl_panels` for figures that need two rows -- a pair of
    phase-space projections above a pair of diagnostics, say. ``axes_flat`` is
    row-major and always 1-D, so panels can be unpacked positionally.

    ``figsize`` defaults to ``panel`` scaled by the grid shape, which keeps each
    panel roughly square regardless of the layout.
    """
    import matplotlib.pyplot as plt

    if figsize is None:
        figsize = (panel[0] * ncols, panel[1] * nrows)
    with plt.rc_context(MPL_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat:
        _apply_axes_style(ax)
    if titles:
        for ax, title in zip(flat, titles):
            ax.set_title(title)
    return fig, flat


def _apply_axes_style(ax) -> None:
    """Apply the house look to one axes object.

    Done per-axes rather than via a global stylesheet because marimo notebooks may
    create figures from several cells, and a global rcParams change is exactly the
    sort of hidden state that makes a notebook's output depend on execution order.
    """
    ax.set_facecolor("white")
    ax.grid(True, color="#ece8f6", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c2de")
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors="#6b6580", labelsize=8, length=3)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_color("#6b6580")


def finish_mpl(fig, suptitle: str | None = None):
    """Tighten the layout and optionally add a title. Returns ``fig``.

    Call this last: marimo renders whatever the cell's final expression is, so the
    idiom in a chapter is ``finish_mpl(fig)`` as the closing line.
    """
    # marimo saves at `fig.dpi` (marimo/_output/mpl.py), so setting it here fixes
    # the exported pixel size regardless of what the renderer would otherwise pick.
    # Worth knowing: marimo's own version also affects this -- an unpinned notebook
    # exported under 0.24.0 produced PNGs at twice the resolution and ~3x the bytes
    # of the same notebook under the pinned 0.23.9. Pinning marimo is what makes
    # figure output reproducible; this line just makes it explicit.
    fig.set_dpi(130)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, color="#211d33")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        fig.tight_layout()
    return fig
