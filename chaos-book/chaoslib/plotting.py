"""The book's figure design system: one palette, two styling helpers.

Lifted verbatim (colours and layout values) from the Lorenz 63 notebook already
already published at aneeshcs.com/chaos, so the two look like the same
publication. The violet accent is this book's identity, distinct
from the GFD textbook's sky-blue.

Semantic, not decorative: each colour means one thing across every chapter, so a
reader who has learned that rose is "the perturbed run" never has to relearn it.
Import these rather than choosing colours per figure.
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
    "SCENE_BG",
    "FONT",
    "TIME_SCALE",
    "style2d",
    "style3d",
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
SCENE_BG = "rgba(250,249,255,0.92)"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
TIME_SCALE = "Plasma"  # colour = time along a trajectory


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
