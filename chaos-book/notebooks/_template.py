# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "plotly",
# ]
# ///
"""Chapter NN -- <Title>

Template for a new chapter. Copy to `chNN_slug.py` and fill in.

The PEP 723 header above is load-bearing: `marimo export html-wasm --sandbox`
builds the export environment from it. Pin marimo EXACTLY (its version
determines the Pyodide build every reader's browser gets) but leave
numpy/scipy/plotly unpinned, so micropip resolves whatever build the bundled
Pyodide actually ships. Pinning those too breaks the in-browser install.

To edit:  marimo edit notebooks/chNN_slug.py
To export: make nb-one NB=chNN_slug
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter NN -- <Title>")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Chapter NN · <Title>

        **The forecasting question.** <One or two sentences. Start from something a
        forecaster or climate scientist would actually want to know, then let the
        dynamical-systems machinery arrive as the way to answer it -- this book is
        weather/climate-first throughout.>

        **What you will be able to do afterwards.** <The concrete new capability.>

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The equations

        Use the SAME symbols as `NOTATION.md`. The equation printed here must be
        the equation `chaoslib` actually steps -- no silent rescaling.

        $$\dot X = \sigma (Y - X), \qquad
          \dot Y = X(\rho - Z) - Y, \qquad
          \dot Z = XY - \beta Z$$
        """
    )
    return


@app.cell
async def _(mo):
    # --- dependencies: import vetted primitives, never re-implement numerics ---
    import sys

    if sys.platform == "emscripten":
        # Browser (Pyodide/WASM): install the chaoslib wheel that `make notebooks`
        # ships in the export's shared public/ folder. Every chapter exports into
        # ONE directory, so this resolves to /nb/public/ and the wheel is stored
        # once for the whole book.
        import micropip

        await micropip.install(
            str(
                mo.notebook_location()
                / "public"
                / "chaoslib-0.1.0-py3-none-any.whl"
            )
        )
    else:
        # Local (marimo edit / pytest): import straight from the repo.
        sys.path.insert(0, str(mo.notebook_dir().parent))

    import numpy as np
    import plotly.graph_objects as go

    from chaoslib import integrate, lyapunov, plotting, systems

    return go, integrate, lyapunov, np, plotting, systems


@app.cell(hide_code=True)
def _(mo):
    # --- controls: the 1-3 parameters this chapter is actually about ---
    rho = mo.ui.slider(0.5, 220.0, value=28.0, step=0.5, label="ρ (Rayleigh number)")
    lead = mo.ui.slider(1, 30, value=10, label="lead time (MTU)")
    mo.hstack([rho, lead], justify="start", gap=2)
    return lead, rho


@app.cell
def _(go, integrate, np, plotting, rho, systems):
    _grid = integrate.trajectory_grid(t_final=40.0, dt=0.01)
    _traj = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid, rho=rho.value
    )

    _fig = go.Figure()
    _fig.add_scatter3d(
        x=_traj[:, 0],
        y=_traj[:, 1],
        z=_traj[:, 2],
        mode="lines",
        line=dict(width=2, color=plotting.C_TRUTH),
        name="trajectory",
    )
    plotting.style3d(_fig, title="Chapter NN · the attractor")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Try this

        <2-4 exploratory prompts. Ask the reader to *find a transition*, not to
        admire a picture.>

        ## What you should have seen

        <State the expected result plainly, so a reader who saw something else
        knows to look again.>

        ## Further reading

        - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, ch. N
        - Kalnay (2003), *Atmospheric Modeling, Data Assimilation and
          Predictability*, §N
        """
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
