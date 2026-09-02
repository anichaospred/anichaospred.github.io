# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 11 -- Lorenz 96: a many-variable atmosphere analogue.

Forty variables on a ring: the smallest system in the book that behaves like a
*field*, and the first whose attractor grows with the size of the domain.

Part IV of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
Figures are static matplotlib, matching chapters 5-7.

To edit:   marimo edit notebooks/ch11_lorenz96.py
To export: make nb-one NB=ch11_lorenz96
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 11: Lorenz 96")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
@app.cell
async def imports():
    import marimo as mo

    import sys

    if sys.platform == "emscripten":
        import micropip

        await micropip.install(
            str(
                mo.notebook_location()
                / "public"
                / "chaoslib-0.1.0-py3-none-any.whl"
            )
        )
    else:
        sys.path.insert(0, str(mo.notebook_dir().parent))

    import numpy as np

    from chaoslib import integrate, lyapunov, plotting, spatial, systems

    C_CONTEXT = plotting.mpl_colour(plotting.C_CONTEXT)
    C_TRUTH = plotting.C_TRUTH
    C_PERT = plotting.C_PERT
    C_SPREAD = plotting.C_SPREAD
    C_MEAN = plotting.C_MEAN
    C_FIXED = plotting.C_FIXED
    C_SAT = plotting.C_SAT
    C_START = plotting.C_START
    C_OBS = plotting.C_OBS
    C_BG = plotting.C_BG
    MPL_DIVERGING = plotting.MPL_DIVERGING
    mpl_panels = plotting.mpl_panels
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    DT = 0.01
    DAYS_PER_TU = 5.0  # the conventional reading; Section 5 discusses it
    LAMBDA1_L63 = 0.9056  # chapter 7, pinned in chaoslib's tests

    return (
        C_BG,
        C_CONTEXT,
        C_FIXED,
        C_MEAN,
        C_OBS,
        C_PERT,
        C_SAT,
        C_SPREAD,
        C_START,
        C_TRUTH,
        DAYS_PER_TU,
        DT,
        LAMBDA1_L63,
        MPL_DIVERGING,
        finish_mpl,
        integrate,
        lyapunov,
        mo,
        mpl_grid,
        mpl_panels,
        np,
        spatial,
        systems,
    )


# ---------------------------------------------------------------------------
# Precomputed Lyapunov sweeps
# ---------------------------------------------------------------------------
@app.cell
def sweep_data():
    # From scripts/generate_ch11_data.py: full spectra at dt=0.01, T=300,
    # transient 40.
    #
    # A full Lorenz 96 spectrum is the most expensive object in this book. The
    # Benettin algorithm carries N tangent vectors alongside the trajectory and
    # re-orthonormalises every step, and the cost is dominated by evaluating the
    # N x N Jacobian T/dt times -- 4.8 s natively at N=40, so roughly 45 s under
    # Pyodide, for ONE parameter value. Restricting to the leading few exponents
    # saves almost nothing (0.62 s against 0.83 s at T=50), because the
    # trajectory and its Jacobian are needed either way.
    #
    # So the two sweeps below are precomputed, and the chapter's live figures are
    # the ones that are genuinely cheap: a Hovmoller diagram is a plain
    # integration (330 ms for 60 time units at N=40) and the diagnostics on top
    # of it are FFTs (2 ms).
    F_GRID = (
        0.5, 0.8, 0.9, 1.2, 1.6, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75,
        5, 5.5, 6, 7, 8, 10, 12, 16, 20,
    )
    F_LAMBDA1 = (
        -0.4459, -0.1106, 0.0012, -0.0072, -0.0077, -0.0037,
        -0.0118, -0.0027, 0.0140, 0.0063, 0.0234, 0.0904, 0.2071,
        0.4384, 0.6935, 0.9888, 1.2974, 1.6679, 2.3285, 2.8250,
        3.8481, 4.7149,
    )
    F_NPOS = (
        0, 0, 2, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 8, 10, 11, 13, 13,
        14, 14, 15, 16,
    )
    F_HKS = (
        0.000, 0.000, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000,
        0.016, 0.010, 0.044, 0.173, 0.489, 1.580, 3.326, 5.011,
        7.293, 10.206, 14.509, 19.047, 26.105, 33.339,
    )
    F_DKY = (
        0.00, 0.00, 2.19, 0.00, 0.00, 0.00, 0.00, 0.00, 3.38, 2.60,
        4.77, 8.07, 11.36, 16.31, 20.01, 22.56, 24.72, 27.07, 29.32,
        31.03, 32.74, 33.88,
    )
    SPECTRUM_N40_F8 = (
        1.6679, 1.4569, 1.3332, 1.1547, 1.0238, 0.8737, 0.7659,
        0.5922, 0.4901, 0.3852, 0.2833, 0.1589, 0.0197, -0.0070,
        -0.0676, -0.1846, -0.3311, -0.4224, -0.5272, -0.6761,
        -0.7811, -0.9250, -1.0141, -1.1104, -1.2492, -1.3355,
        -1.4670, -1.6044, -1.7455, -1.9140, -2.1037, -2.3247,
        -2.6099, -2.8200, -3.2636, -3.7730, -3.9899, -4.3805,
        -4.6321, -4.9457,
    )
    N_GRID = (
        12, 16, 20, 24, 30, 36, 40, 48, 60, 80,
    )
    N_LAMBDA1 = (
        1.4750, 1.5075, 1.5649, 1.5931, 1.6733, 1.7388, 1.6679,
        1.7520, 1.6717, 1.7622,
    )
    N_NPOS = (
        4, 5, 6, 7, 9, 12, 13, 16, 19, 27,
    )
    N_HKS = (
        2.876, 3.821, 4.907, 6.008, 7.655, 9.187, 10.206, 12.518,
        15.051, 20.186,
    )
    N_DKY = (
        8.25, 10.77, 13.43, 16.11, 20.28, 24.39, 27.07, 32.67,
        40.35, 54.06,
    )
    HKS_DENSITY = 0.25561   # intercept -0.114
    DKY_DENSITY = 0.67535   # intercept +0.018
    NPOS_DENSITY = 0.33997   # intercept -0.643
    SHAPE_N16 = (
        1.5075, 1.0902, 0.6913, 0.4416, 0.0903, -0.0051, -0.2029,
        -0.5737, -0.7792, -1.1402, -1.4606, -1.7433, -2.2531,
        -2.8460, -4.1709, -4.6458,
    )
    SHAPE_N40 = (
        1.6679, 1.4569, 1.3332, 1.1547, 1.0238, 0.8737, 0.7659,
        0.5922, 0.4901, 0.3852, 0.2833, 0.1589, 0.0197, -0.0070,
        -0.0676, -0.1846, -0.3311, -0.4224, -0.5272, -0.6761,
        -0.7811, -0.9250, -1.0141, -1.1104, -1.2492, -1.3355,
        -1.4670, -1.6044, -1.7455, -1.9140, -2.1037, -2.3247,
        -2.6099, -2.8200, -3.2636, -3.7730, -3.9899, -4.3805,
        -4.6321, -4.9457,
    )
    SHAPE_N80 = (
        1.7622, 1.6068, 1.5534, 1.4169, 1.3138, 1.2580, 1.1402,
        1.1051, 0.9938, 0.9613, 0.8987, 0.8277, 0.7816, 0.6999,
        0.6535, 0.5812, 0.5088, 0.4507, 0.4195, 0.3130, 0.2957,
        0.2326, 0.2033, 0.1215, 0.0511, 0.0337, 0.0021, -0.0163,
        -0.0700, -0.1331, -0.1684, -0.2489, -0.3048, -0.3620,
        -0.4122, -0.4514, -0.5230, -0.5768, -0.6043, -0.6885,
        -0.7336, -0.7899, -0.8319, -0.9121, -0.9775, -1.0070,
        -1.0725, -1.1366, -1.1877, -1.2484, -1.3260, -1.3680,
        -1.4375, -1.5068, -1.5702, -1.6196, -1.6766, -1.7887,
        -1.8815, -1.9414, -2.0572, -2.1208, -2.2553, -2.3258,
        -2.5131, -2.6228, -2.7844, -2.9637, -3.1616, -3.4091,
        -3.4858, -3.8179, -3.9802, -4.1404, -4.3137, -4.4654,
        -4.5775, -4.7053, -4.8514, -5.0618,
    )
    return (
        DKY_DENSITY,
        F_DKY,
        F_GRID,
        F_HKS,
        F_LAMBDA1,
        F_NPOS,
        HKS_DENSITY,
        NPOS_DENSITY,
        N_DKY,
        N_GRID,
        N_HKS,
        N_LAMBDA1,
        N_NPOS,
        SHAPE_N16,
        SHAPE_N40,
        SHAPE_N80,
        SPECTRUM_N40_F8,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 11 · Lorenz 96: A Many-Variable Atmosphere Analogue

    **Part IV — Many scales, many degrees of freedom.**

    **The forecasting question.** Everything in Part III was measured on a
    three-variable system. A forecast model has $10^9$. Which of those
    conclusions survive the change of scale, and which were artefacts of working
    in three dimensions?

    The question is not rhetorical, and chapter 16 already found one casualty: in
    Lorenz 63 the gradient of a forecast metric points almost exactly along the
    fastest-growing direction, and in a forty-variable system the two are nearly
    orthogonal. Something has to sit between three variables and a general
    circulation model.

    Lorenz's 1996 model is that something:

    $$\frac{dx_k}{dt} = \bigl(x_{k+1} - x_{k-2}\bigr)x_{k-1} - x_k + F,
      \qquad k = 1 \ldots N \quad \text{(cyclic)}.$$

    Read it as a latitude circle. The quadratic terms conserve
    $\sum_k x_k^2$ and stand in for advection, $-x_k$ is dissipation, and $F$ is
    the forcing that keeps the system alive. Nothing about it is derived from the
    equations of motion — it is a caricature, chosen because it is the smallest
    thing that has the *structural* features that matter.

    **What those features are, and why three variables cannot have them.**

    | | Lorenz 63 | Lorenz 96 ($N=40$, $F=8$) |
    |---|---|---|
    | positive exponents | 1 | 13 |
    | Kaplan–Yorke dimension | 2.06 | 27.1 |
    | KS entropy (nats/time unit) | 0.905 | 10.2 |
    | does an error have a *wavelength*? | no | yes |
    | does a structure *propagate*? | no | yes |
    | does the attractor grow with the domain? | no domain to grow | yes, linearly |

    The last row is the one this chapter is really about. In Lorenz 63 there is
    no space, so there is no question about the scale of an error and no way for
    the system to be *large*. Lorenz 96 has a size, and almost everything about
    its attractor is proportional to that size — which turns out to be the
    property that makes a small model informative about a big one.

    ---

    ## What is here

    | Section | The question |
    |---|---|
    | 1 | What does the flow look like? Waves, and their scale |
    | 2 | Where does that scale come from? Linear theory, in closed form |
    | 3 | The Lyapunov spectrum, and what the forcing does to it |
    | 4 | Extensivity: what happens when the domain grows |
    | 5 | Why this is the standard testbed, and what the time unit means |
    """
    )
    return


# ===========================================================================
# 1. The flow
# ===========================================================================
@app.cell(hide_code=True)
def s1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · Waves on a ring

    Start it from the uniform state $x_k = F$ — an **exact** fixed point for
    every $F$ and $N$, since $(x_{k+1}-x_{k-2})x_{k-1} = (F-F)F = 0$ and
    $-F + F = 0$ — with one site nudged by $0.01$.

    The Hovmöller diagram below plots the deviation from the mean, site on the
    horizontal axis and time increasing upward. What to look for: the tilt of the
    stripes is a **propagation speed**, their horizontal spacing is a
    **wavelength**, and their finite vertical extent is the **lifetime** of a
    structure. None of those three quantities exists in a three-variable system.
    """
    )
    return


@app.cell(hide_code=True)
def s1_controls(mo):
    forcing_slider = mo.ui.slider(
        start=0.5, stop=16.0, step=0.5, value=8.0,
        label="forcing F", show_value=True,
    )
    sites_slider = mo.ui.slider(
        start=12, stop=80, step=4, value=40,
        label="sites N", show_value=True,
    )
    span_slider = mo.ui.slider(
        start=10.0, stop=60.0, step=5.0, value=25.0,
        label="time shown (time units)", show_value=True,
    )
    return forcing_slider, sites_slider, span_slider


@app.cell(hide_code=True)
def s1_figure(
    C_FIXED,
    C_TRUTH,
    DT,
    MPL_DIVERGING,
    finish_mpl,
    forcing_slider,
    integrate,
    mo,
    mpl_panels,
    np,
    sites_slider,
    span_slider,
    spatial,
    systems,
):
    _n = int(sites_slider.value)
    _forcing = float(forcing_slider.value)
    _span = float(span_slider.value)

    _x0 = systems.lorenz96_uniform_state(_forcing, _n)
    _x0[_n // 2] += 0.01
    # Spin-up long enough to leave the uniform state and settle onto whatever
    # the attractor is; then keep `span` time units for the diagram.
    _traj = integrate.rk4(
        systems.lorenz96, _x0,
        integrate.trajectory_grid(_span + 40.0, DT), forcing=_forcing,
    )
    _field = _traj[int(40.0 / DT):]
    _deviation = _field - _field.mean()

    _fig, _ax = mpl_panels(
        3,
        titles=("Hovmöller: deviation from the mean",
                "A snapshot", "Power in space"),
        height=3.9,
    )
    _limit = float(np.abs(_deviation).max())
    _im = _ax[0].imshow(
        _deviation, origin="lower", aspect="auto", cmap=MPL_DIVERGING,
        vmin=-_limit, vmax=_limit,
        extent=(0.5, _n + 0.5, 0.0, _span),
    )
    _ax[0].set_xlabel("site k")
    _ax[0].set_ylabel("time (time units)")
    _fig.colorbar(_im, ax=_ax[0], fraction=0.046, pad=0.03)

    _ax[1].plot(np.arange(1, _n + 1), _field[-1], marker="o", markersize=3,
                color=C_TRUTH, linewidth=1.4)
    _ax[1].axhline(_forcing, color=C_FIXED, linewidth=1.2, linestyle="--",
                   label=f"$x_k = F$ = {_forcing:g}")
    _ax[1].set_xlabel("site k")
    _ax[1].set_ylabel("$x_k$")
    _ax[1].legend(loc="upper right", fontsize=7, framealpha=0.9)

    _m, _power = spatial.spatial_power_spectrum(_field)
    _ax[2].semilogy(_m[1:], np.maximum(_power[1:], 1e-14), marker="o",
                    markersize=3.4, color=C_TRUTH, linewidth=1.3)
    _crit_forcing, _m_star = systems.lorenz96_critical_forcing(_n)
    if _m_star <= _m[-1]:
        _ax[2].axvline(_m_star, color=C_FIXED, linewidth=1.2, linestyle="--",
                       label=f"linear theory selects m = {_m_star}")
        _ax[2].legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    _ax[2].set_xlabel("wavenumber m")
    _ax[2].set_ylabel("mean power")
    finish_mpl(_fig, suptitle=f"Lorenz 96, N = {_n}, F = {_forcing:g}")

    # Diagnostics, all measured from the field above.
    _dominant = spatial.dominant_wavenumber(_field)
    _speed = spatial.phase_speed(_field, _dominant, DT)
    _length = spatial.correlation_length(_field)
    _range = float(_field.max() - _field.min())
    _quiet = _range < 1e-6

    if _quiet:
        _verdict = (
            f"""**The flow is the uniform state.** The field is constant to
{_range:.1e} across all {_n} sites: at F = {_forcing:g} the forcing is too weak
to sustain waves, and the perturbation has decayed. Section 2 gives the exact
forcing at which that stops being true — for N = {_n} it is
**{_crit_forcing:.4f}**."""
        )
    else:
        _verdict = (
            f"""| measured from the field above | |
|---|---|
| dominant wavenumber | **{_dominant}** ({_n / _dominant:.2f} sites per wave) |
| phase speed of that mode | **{_speed:+.2f}** sites per time unit |
| correlation length | {_length:.2f} sites |
| field range | {_field.min():.2f} to {_field.max():.2f} |

The sign of the phase speed is the propagation direction: negative means the
pattern moves toward **decreasing** $k$. Lorenz took $k$ to increase eastward,
which makes these waves westward-propagating."""
        )

    mo.vstack([
        mo.hstack([forcing_slider, sites_slider, span_slider], justify="start"),
        _fig,
        mo.md(_verdict),
    ])
    return


# ===========================================================================
# 2. Linear theory
# ===========================================================================
@app.cell(hide_code=True)
def s2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · Where the wavelength comes from

    The scale in Section 1 is not arbitrary, and it does not need a simulation to
    find. Linearise about the uniform state $x_k = F$. The Jacobian row is
    $-F$ at $k-2$, $0$ at $k-1$, $-1$ at $k$, $+F$ at $k+1$ — the same four
    numbers in every row, so the matrix is **circulant** and the Fourier modes
    $e^{i\theta k}$, $\theta = 2\pi m/N$, diagonalise it exactly:

    $$\sigma(\theta) = -1 + F\left(e^{i\theta} - e^{-2i\theta}\right).$$

    This is not an approximation. The $N$ values it returns *are* the $N$
    eigenvalues of the Jacobian, to machine precision, and a test in `chaoslib`
    asserts it. Two things follow immediately.

    **The threshold.** Taking real parts,
    $\operatorname{Re}\sigma = -1 + F(\cos\theta - \cos 2\theta)$. Writing
    $u = \cos\theta$ the bracket is $1 + u - 2u^2$, maximised at $u = 1/4$ with
    value $9/8$. So a long chain becomes unstable at $F = 8/9$, and a finite one
    at

    $$F_{\rm crit} = \Bigl[\max_m\bigl(\cos\theta_m - \cos2\theta_m\bigr)\Bigr]^{-1},$$

    the maximum over the integer wavenumbers the ring admits. At $N = 40$ the
    best available mode is $m = 8$, giving $F_{\rm crit} = 2/\sqrt5 = 0.8944$
    exactly.

    **The preferred scale.** The instability selects the wavenumber nearest
    $\theta = \arccos(1/4)$, which is where Section 1's wavelength comes from —
    approximately, and the discrepancy is itself informative.
    """
    )
    return


@app.cell(hide_code=True)
def s2_figure(
    C_FIXED,
    C_PERT,
    C_SAT,
    C_TRUTH,
    DT,
    finish_mpl,
    forcing_slider,
    integrate,
    mo,
    mpl_panels,
    np,
    sites_slider,
    spatial,
    systems,
):
    _n = int(sites_slider.value)
    _forcing = float(forcing_slider.value)
    _m = np.arange(_n // 2 + 1)
    _sigma = systems.lorenz96_dispersion(_m, _n, _forcing)
    _crit_forcing, _m_star = systems.lorenz96_critical_forcing(_n)

    # The nonlinear flow, for comparison with what linear theory predicts.
    _x0 = systems.lorenz96_uniform_state(_forcing, _n)
    _x0[_n // 2] += 0.01
    _field = integrate.rk4(
        systems.lorenz96, _x0, integrate.trajectory_grid(70.0, DT),
        forcing=_forcing,
    )[int(40.0 / DT):]
    _alive = float(_field.max() - _field.min()) > 1e-6
    _dominant = spatial.dominant_wavenumber(_field) if _alive else None

    _fig, _ax = mpl_panels(
        2,
        titles=("Growth rate of each wavenumber", "Phase speed"),
        height=3.5,
    )
    _ax[0].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    _ax[0].plot(_m, _sigma.real, marker="o", markersize=3.4, color=C_TRUTH,
                linewidth=1.4, label=r"$\mathrm{Re}\,\sigma$ (linear theory)")
    _ax[0].axvline(_m_star, color=C_FIXED, linewidth=1.2, linestyle="--",
                   label=f"most unstable: m = {_m_star}")
    if _dominant is not None:
        _ax[0].axvline(_dominant, color=C_PERT, linewidth=1.2, linestyle=":",
                       label=f"nonlinear peak: m = {_dominant}")
    _ax[0].set_xlabel("wavenumber m")
    _ax[0].set_ylabel("growth rate (per time unit)")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    # Phase speed c = -omega/theta; m = 0 does not propagate.
    _theta = 2.0 * np.pi * _m[1:] / _n
    _ax[1].axhline(0.0, color="#94a3b8", linewidth=0.9)
    _ax[1].plot(_m[1:], -_sigma.imag[1:] / _theta, marker="o", markersize=3.4,
                color=C_TRUTH, linewidth=1.4, label="linear theory")
    if _dominant is not None:
        _measured = spatial.phase_speed(_field, _dominant, DT)
        _ax[1].plot([_dominant], [_measured], marker="*", markersize=15,
                    color=C_PERT, zorder=6,
                    label=f"measured at m = {_dominant}")
    _ax[1].set_xlabel("wavenumber m")
    _ax[1].set_ylabel("phase speed (sites / time unit)")
    _ax[1].legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"Linear theory about $x_k = F$, N = {_n}, F = {_forcing:g}")

    _fastest = int(np.argmax(_sigma.real))
    if _dominant is None:
        _reading = f"""At F = {_forcing:g} every mode has
$\\mathrm{{Re}}\\,\\sigma < 0$ — the largest is
{_sigma.real.max():+.4f} at m = {_fastest} — so the uniform state is stable and
there is no nonlinear flow to compare against. Raise F above
**{_crit_forcing:.4f}**."""
    else:
        _measured = spatial.phase_speed(_field, _dominant, DT)
        _linear_speed = -_sigma.imag[_dominant] / (2.0 * np.pi * _dominant / _n)
        _reading = f"""| | linear theory | nonlinear flow |
|---|---|---|
| preferred wavenumber | {_m_star} | **{_dominant}** |
| phase speed there | {_linear_speed:+.2f} sites/TU | **{_measured:+.2f}** sites/TU |
| growth rate of m = {_m_star} | {_sigma.real[_m_star]:+.3f} /TU | — |

**Linear theory gets the scale nearly right and the speed badly wrong**, and
both halves of that are worth taking seriously. The wavelength is set at the
moment the uniform state destabilises, and the nonlinear flow keeps it, shifted
by one wavenumber. The phase speed is not: the measured
{_measured:+.2f} sites/TU is about
{abs(_linear_speed / _measured):.1f}× slower than the
{_linear_speed:+.2f} predicted. The prediction is made about a state the system
is nowhere near — waves of finite amplitude propagate on a flow they have
themselves modified, which is exactly the regime linearisation about $x_k = F$
does not describe.

This is the same lesson as chapter 15's window of validity, arriving from a
different direction: the linearisation is quantitatively reliable for the
instability that creates a structure, and not for the structure's later life."""

    mo.vstack([_fig, mo.md(_reading)])
    return


# ===========================================================================
# 3. The Lyapunov spectrum
# ===========================================================================
@app.cell(hide_code=True)
def s3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · Forty exponents

    Chapter 7 measured Lorenz 63's three exponents and used the exact identity
    $\sum_i\lambda_i = \operatorname{tr}\mathbf{J} = -(\sigma+1+\beta)$ as a
    check. Lorenz 96 has the same identity in an even simpler form: every
    diagonal entry of its Jacobian is $-1$, from the $-x_k$ dissipation, so

    $$\sum_{i=1}^{N}\lambda_i = \operatorname{tr}\mathbf{J} = -N$$

    for **every** $F$, every $N$ and every trajectory. It is the strongest
    available check on a forty-dimensional measurement, and the residual below is
    a real diagnostic rather than a formality.

    These spectra are **precomputed** — see the note in the data cell. One
    spectrum at $N = 40$ costs about 45 seconds in the browser, which no slider
    can survive.
    """
    )
    return


@app.cell(hide_code=True)
def s3_controls(mo):
    spectrum_forcing = mo.ui.dropdown(
        options={
            "F = 0.5  (below threshold)": "0.5",
            "F = 0.9  (just above threshold)": "0.9",
            "F = 2.5  (periodic)": "2.5",
            "F = 4.5  (chaos beginning)": "4.5",
            "F = 5.0": "5.0",
            "F = 6.0": "6.0",
            "F = 8.0  (the standard case)": "8.0",
            "F = 12.0": "12.0",
            "F = 20.0": "20.0",
        },
        value="F = 8.0  (the standard case)",
        label="forcing (precomputed grid)",
    )
    return (spectrum_forcing,)


@app.cell(hide_code=True)
def s3_figure(
    C_CONTEXT,
    C_FIXED,
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    DAYS_PER_TU,
    F_DKY,
    F_GRID,
    F_HKS,
    F_LAMBDA1,
    F_NPOS,
    LAMBDA1_L63,
    SPECTRUM_N40_F8,
    finish_mpl,
    lyapunov,
    mo,
    mpl_panels,
    np,
    spectrum_forcing,
    systems,
):
    _f_sel = float(str(spectrum_forcing.value))
    _fg = np.asarray(F_GRID, dtype=float)
    _idx = int(np.argmin(np.abs(_fg - _f_sel)))

    _spectrum = np.asarray(SPECTRUM_N40_F8, dtype=float)
    _crit_forcing, _ = systems.lorenz96_critical_forcing(40)

    _fig, _ax = mpl_panels(
        3,
        titles=("The spectrum at N = 40, F = 8",
                "Leading exponent vs forcing",
                "Active degrees of freedom"),
        height=3.6,
    )
    _i = np.arange(1, _spectrum.size + 1)
    _ax[0].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    _ax[0].fill_between(_i, 0.0, _spectrum, where=_spectrum > 0,
                        color=C_SPREAD, alpha=0.25)
    _ax[0].plot(_i, _spectrum, marker="o", markersize=3.2, color=C_TRUTH,
                linewidth=1.3)
    _ax[0].set_xlabel("index i")
    _ax[0].set_ylabel(r"$\lambda_i$ (per time unit)")

    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    _ax[1].axhline(LAMBDA1_L63, color=C_MEAN, linewidth=1.2,
                   linestyle=":", label=f"Lorenz 63: {LAMBDA1_L63}")
    _ax[1].plot(_fg, np.asarray(F_LAMBDA1), marker="o", markersize=3.6,
                color=C_TRUTH, linewidth=1.4, label=r"$\lambda_1$")
    _ax[1].axvline(_crit_forcing, color=C_FIXED, linewidth=1.2, linestyle="--",
                   label=f"$F_{{crit}}$ = {_crit_forcing:.3f}")
    _ax[1].plot([_fg[_idx]], [F_LAMBDA1[_idx]], marker="*", markersize=15,
                color=C_PERT, zorder=6, label="selected")
    _ax[1].set_xscale("log")
    _ax[1].set_xlabel("forcing F")
    _ax[1].set_ylabel(r"$\lambda_1$ (per time unit)")
    _ax[1].legend(loc="upper left", fontsize=6.5, framealpha=0.9)

    # Mask the non-chaotic parameters rather than plotting a number already
    # declared meaningless: kaplan_yorke_dimension returns 0 whenever the
    # leading exponent measures negative, which for a limit cycle (true
    # lambda_1 = 0) is the sign of round-off, not a dimension. Drawing those
    # zeros puts a confident flat line where there is no measurement.
    _chaotic = np.asarray(F_LAMBDA1, dtype=float) > 0.02
    _npos_m = np.where(_chaotic, np.asarray(F_NPOS, dtype=float), np.nan)
    _dky_m = np.where(_chaotic, np.asarray(F_DKY, dtype=float), np.nan)
    _first_chaotic = float(_fg[_chaotic][0])
    _ax[2].axvspan(_fg[0], _first_chaotic, color="#e8e6f0", zorder=0)
    _ax[2].annotate(
        "not chaotic:\n$D_{KY}$ undefined",
        (float(np.sqrt(_fg[0] * _first_chaotic)), 22.0),
        ha="center", va="center", fontsize=6.5, color="#6b7280",
    )
    _ax[2].plot(_fg, _npos_m, marker="o", markersize=3.6,
                color=C_SPREAD, linewidth=1.4, label="positive exponents")
    _ax[2].plot(_fg, _dky_m, marker="s", markersize=3.6,
                color=C_TRUTH, linewidth=1.4, label=r"$D_{KY}$")
    _ax[2].axhline(40, color="#9ca3af", linewidth=1.2, linestyle="--",
                   label="N = 40")
    _ax[2].set_ylim(0, 44)
    _ax[2].set_xscale("log")
    _ax[2].set_xlabel("forcing F")
    _ax[2].set_ylabel("count / dimension")
    _ax[2].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    _lam1 = float(F_LAMBDA1[_idx])
    _doubling = np.log(2.0) / _lam1 if _lam1 > 1e-3 else float("nan")
    _l63_doubling = np.log(2.0) / LAMBDA1_L63
    _residual = abs(_spectrum.sum() + 40.0)

    if _f_sel < _crit_forcing:
        _regime = "the **uniform state**, which is still stable"
    elif _lam1 < 0.02:
        _regime = (
            "**periodic or quasi-periodic** — $\\lambda_1$ is zero to within "
            "measurement error, which is the signature of a limit cycle, not of "
            "a fixed point"
        )
    else:
        _regime = "**chaotic**"

    mo.vstack([
        mo.hstack([spectrum_forcing], justify="start"),
        _fig,
        mo.md(
            f"""
At F = {_fg[_idx]:g} the attractor is {_regime}, with
$\\lambda_1$ = **{_lam1:+.4f}** per time unit,
**{F_NPOS[_idx]}** positive exponents,
$h_{{KS}}$ = **{F_HKS[_idx]:.3f}** nats per time unit and
$D_{{KY}}$ = **{F_DKY[_idx]:.2f}**.

**The trace identity, as a check.** Summing the forty exponents at F = 8 gives
{_spectrum.sum():.6f} against the exact $-40$ — a residual of
{_residual:.1e}. That residual is *not* a convergence error: it is independent
of the averaging time and falls by a factor of 16 for each halving of $\\Delta t$,
which identifies it as RK4 truncation in the tangent propagator. Both facts are
asserted by tests in `chaoslib`. Across the whole sweep the residual grows from
$2\\times10^{{-8}}$ at F = 0.5 to $7\\times10^{{-5}}$ at F = 20, tracking the
size of the Jacobian — so the identity doubles as a resolution diagnostic.
"""
        ),
        mo.callout(
            mo.md(
                f"""### Three things the spectrum says that a three-variable model cannot

**The zero exponent is not visible.** Every autonomous flow has one exponent
that is exactly zero, along the trajectory. In Lorenz 63 it stands out —
$\\lambda_2$ measures $+0.006$ against neighbours at $+0.905$ and $-14.57$. Here
the spectrum near zero reads
{', '.join(f'{v:+.3f}' for v in _spectrum[11:15])}: the mean spacing is about
0.05, so the neutral direction cannot be picked out by inspection at all. A
dense spectrum is what a large system looks like, and it means the *count* of
positive exponents is a measurement with an uncertainty, not an integer read off
a list.

**Chaos does not begin where the waves do.** The uniform state destabilises at
$F_{{crit}}$ = {_crit_forcing:.4f}, but from there up to about $F = 4.25$ the
leading exponent stays within $\\pm 0.025$ of zero — a limit cycle or a torus,
periodic waves with no error growth at all. $\\lambda_1$ only becomes robustly
positive near **F = 4.5**. There are two thresholds, and they are far apart.
The third panel leaves that whole band blank on purpose: the Kaplan–Yorke
formula returns 0 whenever the leading exponent measures negative, so for a
limit cycle — where the true value is exactly zero and its measured sign is
round-off — it reports a dimension of 0 rather than declining to answer.

**Error growth is faster than in Lorenz 63, and by the right amount.** At
F = 8, $\\lambda_1$ = {_lam1:.3f} per time unit gives a doubling time of
{_doubling:.3f} time units — **{_doubling * DAYS_PER_TU:.2f} days** under the
conventional reading of 1 time unit as {DAYS_PER_TU:g} days. Lorenz 63 under its
own 5-days-per-unit convention gives {_l63_doubling * DAYS_PER_TU:.2f} days.
Section 5 returns to which of those is the accident."""
            ),
            kind="info",
        ),
    ])
    return


# ===========================================================================
# 4. Extensivity
# ===========================================================================
@app.cell(hide_code=True)
def s4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · What happens when the domain grows

    Here is the property that makes Lorenz 96 more than a bigger Lorenz 63, and
    it is worth stating as a prediction before looking at the figure.

    Suppose the ring is long enough to hold many waves. Then a structure at site
    5 does not know or care how far away site 400 is: the dynamics are **local**,
    and the correlation length measured in Section 1 is a couple of sites.
    Doubling $N$ therefore does not create a faster instability — it creates
    *more independent copies* of the same instability. So:

    - the fastest growth rate $\lambda_1$ should be **independent** of $N$;
    - the *shape* of the spectrum, plotted against $i/N$, should be
      **independent** of $N$;
    - the total entropy production $h_{KS} = \sum_{\lambda_i>0}\lambda_i$ and the
      attractor dimension $D_{KY}$ should be **proportional** to $N$.

    Quantities of the first two kinds are called **intensive** and of the last
    kind **extensive**, by direct analogy with temperature and volume in
    thermodynamics. A system whose entropy and dimension are extensive is
    exhibiting *spatiotemporal* chaos rather than merely low-dimensional chaos,
    and Lorenz 63 cannot be either, because it has no $N$ to vary.
    """
    )
    return


@app.cell(hide_code=True)
def s4_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    DKY_DENSITY,
    HKS_DENSITY,
    NPOS_DENSITY,
    N_DKY,
    N_GRID,
    N_HKS,
    N_LAMBDA1,
    N_NPOS,
    SHAPE_N16,
    SHAPE_N40,
    SHAPE_N80,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _ns = np.asarray(N_GRID, dtype=float)
    _lam1 = np.asarray(N_LAMBDA1, dtype=float)
    _hks = np.asarray(N_HKS, dtype=float)
    _dky = np.asarray(N_DKY, dtype=float)

    _fig, _ax = mpl_panels(
        3,
        titles=("Intensive: the leading exponent",
                "Intensive: the shape of the spectrum",
                "Extensive: entropy and dimension"),
        height=3.6,
    )
    _ax[0].plot(_ns, _lam1, marker="o", markersize=4.5, color=C_TRUTH,
                linewidth=1.5)
    _ax[0].axhline(float(_lam1[_ns >= 30].mean()), color=C_SAT, linewidth=1.2,
                   linestyle="--",
                   label=f"mean for N ≥ 30: {_lam1[_ns >= 30].mean():.3f}")
    _ax[0].set_xlabel("sites N")
    _ax[0].set_ylabel(r"$\lambda_1$ (per time unit)")
    _ax[0].set_ylim(0.0, 2.2)
    _ax[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    for _label, _values, _colour in (
        ("N = 16", SHAPE_N16, C_MEAN),
        ("N = 40", SHAPE_N40, C_TRUTH),
        ("N = 80", SHAPE_N80, C_PERT),
    ):
        _s = np.asarray(_values, dtype=float)
        _ax[1].plot((np.arange(_s.size) + 0.5) / _s.size, _s, marker="o",
                    markersize=2.6, linewidth=1.3, color=_colour, label=_label)
    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    _ax[1].set_xlabel("index fraction  i / N")
    _ax[1].set_ylabel(r"$\lambda_i$ (per time unit)")
    _ax[1].set_ylim(-5.2, 2.2)
    _ax[1].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _ax[2].plot(_ns, _dky, marker="o", markersize=4.5, color=C_TRUTH,
                linewidth=1.5, label=r"$D_{KY}$")
    _ax[2].plot(_ns, DKY_DENSITY * _ns, color=C_TRUTH, linewidth=1.1,
                linestyle="--", label=f"{DKY_DENSITY:.3f} N")
    _ax[2].plot(_ns, _hks, marker="s", markersize=4.5, color=C_SPREAD,
                linewidth=1.5, label=r"$h_{KS}$")
    _ax[2].plot(_ns, HKS_DENSITY * _ns, color=C_SPREAD, linewidth=1.1,
                linestyle="--", label=f"{HKS_DENSITY:.3f} N")
    _ax[2].set_xlabel("sites N")
    _ax[2].set_ylabel("dimension / nats per time unit")
    _ax[2].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle="Lorenz 96 at F = 8, domain size varied")

    _large = _ns >= 30
    _rows = "\n".join(
        f"| {int(n)} | {l:.3f} | {int(p)} | {h:.2f} | {h / n:.4f} | {d:.2f} | {d / n:.4f} |"
        for n, l, p, h, d in zip(_ns, _lam1, N_NPOS, _hks, _dky)
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
| N | $\\lambda_1$ | positive | $h_{{KS}}$ | $h_{{KS}}/N$ | $D_{{KY}}$ | $D_{{KY}}/N$ |
|---|---|---|---|---|---|---|
{_rows}

**All three predictions hold.** Over a factor of {_ns[-1] / _ns[0]:.1f} in
domain size:

- $\\lambda_1$ is flat for $N \\ge 30$, mean
  **{_lam1[_large].mean():.3f}** with a spread of
  {_lam1[_large].max() - _lam1[_large].min():.3f}. Below that it is suppressed —
  {_lam1[0]:.3f} at N = 12 — and the reason is visible in Section 1: the
  preferred wavelength is about 4.4 sites, so a 12-site ring holds fewer than
  three waves and the instability is cramped by its own periodicity. Extensivity
  is a large-domain statement, and "large" means large *compared with the
  correlation length*.
- the spectra collapse onto one curve under $i \\to i/N$, which is the same
  statement made shape-wise rather than through a single number.
- $D_{{KY}} = {DKY_DENSITY:.3f}\\,N$ and
  $h_{{KS}} = {HKS_DENSITY:.3f}\\,N$, both fitted through essentially the
  origin ($D_{{KY}}$ intercept $+0.02$). $D_{{KY}}/N$ varies by only
  {100 * (max(_dky / _ns) - min(_dky / _ns)) / np.mean(_dky / _ns):.1f}% across
  the whole range, including the cramped small-$N$ cases. The number of positive
  exponents follows the same law, {NPOS_DENSITY:.3f} per site.
"""
        ),
        mo.callout(
            mo.md(
                f"""### The consequence for forecasting

**A large domain is not a small domain scaled up, and it cannot be studied by
shrinking it.** Whatever its length, a Lorenz 96 ring puts its attractor in
about two-thirds of the available dimensions and has about **one third** of its
directions actively growing. Double the domain and you double the number of
independent error-growing structures, double the rate at which the system
generates information, and double the dimension of the manifold an analysis has
to be pinned to — while the *fastest* growth rate, and hence the predictability
horizon of a single structure, does not budge.

That combination is why an operational ensemble has the size it does. The lead
time you can achieve is set by $\\lambda_1$, an intensive quantity, so it is a
property of the dynamics and no amount of computing changes it. The number of
directions the ensemble must span is set by $D_{{KY}} \\approx
{DKY_DENSITY:.2f}N$, an extensive one — so it grows with the size of the
domain. At {NPOS_DENSITY:.2f} growing directions per variable, a model with
$10^7$ variables would have of order $3\\times10^6$ of them. Fifty members is
not few because someone was frugal; it is a rounding error against the number
of directions that need sampling, and every ensemble method in Part V is a
strategy for living with that ratio. Chapter 17 designs
around that, and chapter 19's localisation is precisely an exploitation of the
locality that makes the system extensive in the first place."""
            ),
            kind="warn",
        ),
    ])
    return


# ===========================================================================
# 5. Why this model
# ===========================================================================
@app.cell(hide_code=True)
def s5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · The standard testbed, and what its time unit means

    Lorenz 96 became the default system for testing data assimilation and
    ensemble methods, and it is worth being clear about which of its properties
    earn that and which are conveniences.

    **What it genuinely has.** Locality, so that spatial covariance structure and
    localisation are meaningful. Extensivity, so that a method's cost and skill
    can be studied as functions of system size. A dense Lyapunov spectrum, so
    that the number of growing directions can exceed the number of ensemble
    members you can afford — the actual operational predicament, which a
    three-variable system cannot reproduce at any ensemble size. And a state
    dimension small enough that the exact answer is computable: chapter 19 checks
    the EnKF against the Kalman filter, which needs a $40\times40$ covariance
    rather than a $10^7 \times 10^7$ one.

    **What is a convention.** The reading of one time unit as five days rests on
    the dissipation term: $-x_k$ gives an $e$-folding decay time of one time
    unit, and five days is the conventional figure for dissipative damping in the
    atmosphere *[citation needed: Lorenz (1996); Lorenz and Emanuel (1998)]*. It
    is calibrated to *dissipation*, and everything else follows from it rather
    than being fitted.

    That distinction is what makes the doubling time worth quoting. Section 3
    measured $\lambda_1 = 1.668$ per time unit, so errors double in
    $\ln 2/\lambda_1 = 0.416$ time units — **2.08 days** under a convention
    fixed by an entirely different consideration. The observed doubling time for
    synoptic-scale forecast errors is about two days
    *[citation needed: Palmer and Hagedorn (2006)]*. Lorenz 63 under its own
    five-days-per-unit convention gives 3.83 days, roughly twice too slow, which
    is the discrepancy chapter 6 flags when it converts model time units to days.

    **What is not right.** The measured phase speed of about $-2.1$ sites per
    time unit is, at 40 sites to a latitude circle and five days to the unit,
    about $3.8°$ of longitude per day westward. Mid-latitude synoptic systems
    travel eastward several times faster than that. The model gets error growth
    and spatial structure into the right range and does not pretend to get
    advection right; nothing in Part V depends on the phase speed.

    Where this goes: **chapter 12** uses a multiscale extension of this model to
    ask whether the error cascade sets a finite predictability limit;
    **chapters 18–20** run variational and ensemble assimilation on exactly this
    system; **chapter 19**'s localisation and **chapter 17**'s ensemble sizing
    both rest on the extensivity measured in Section 4.
    """
    )
    return


# ===========================================================================
# 6. Closing
# ===========================================================================
@app.cell(hide_code=True)
def closing(mo):
    mo.md(
        r"""
    ---
    ## Try this

    1. **Find the threshold, then find the other one.** Set $N = 40$ and lower
       $F$ in Section 1 until the waves disappear. Compare where that happens
       with Section 2's exact $F_{\rm crit} = 2/\sqrt5$. Then use Section 3 to
       find where $\lambda_1$ becomes positive. Why are these two thresholds
       different, and what is the attractor in between?
    2. **Break extensivity on purpose.** Section 4 shows $\lambda_1$ suppressed
       at $N = 12$. Use Section 1 to measure the correlation length at $N = 12$
       and at $N = 40$, and explain the suppression in terms of how many
       correlation lengths fit on the ring.
    3. **Predict the preferred wavenumber at another $N$.** Section 2's
       $m^*$ is the integer nearest $N\arccos(1/4)/2\pi$. Compute it for
       $N = 60$ by hand, then check it against the figure — and against where
       the nonlinear flow actually puts its peak.
    4. **Watch the linearisation fail.** Compare the measured and predicted
       phase speeds at $F = 2$, $F = 8$ and $F = 16$. Does the linear estimate
       get better or worse as the forcing grows, and why would you expect that
       before measuring it?
    5. **Count what an ensemble would have to span.** At $F = 8$ there are 13
       positive exponents in 40 dimensions. If a forecast centre could afford
       50 members for a model with $10^7$ variables, and Lorenz 96's ratios held,
       how many of the growing directions would go unsampled? This is chapter
       17's problem, in one line of arithmetic.

    ## What you should have seen

    Forty variables on a ring produce westward-propagating waves with a
    preferred wavelength of about 4.4 sites, and that scale is set by a linear
    instability of the uniform state $x_k = F$ whose eigenvalues are available in
    closed form: $\sigma(\theta) = -1 + F(e^{i\theta} - e^{-2i\theta})$, exact to
    machine precision. The uniform state loses stability at
    $F_{\rm crit} = 2/\sqrt5 = 0.8944$ for $N = 40$, but $\lambda_1$ does not
    become positive until $F \approx 4.5$ — periodic waves first, chaos later.

    At $F = 8$ the spectrum has 13 positive exponents,
    $h_{KS} = 10.2$ nats per time unit and $D_{KY} = 27.1$, against Lorenz 63's
    1, 0.905 and 2.06. The forty exponents sum to $-40$ to within
    $2.5\times10^{-5}$, a residual that is RK4 truncation rather than
    non-convergence — it is independent of the averaging time and falls as
    $\Delta t^4$.

    And the property that matters most: the model is **extensive**. $\lambda_1$
    is independent of $N$ above about 30 sites, the spectrum collapses under
    $i \to i/N$, and $D_{KY} = 0.675\,N$ and $h_{KS} = 0.256\,N$ through the
    origin. Doubling the domain doubles the number of growing directions and the
    rate of information loss, and leaves the fastest growth rate — and so the
    predictability horizon — untouched.

    ## Further reading

    - Lorenz, E. N. (1996). Predictability: a problem partly solved. *Proceedings
      of the ECMWF Seminar on Predictability*, vol. 1, 1–18.
    - Lorenz, E. N. and Emanuel, K. A. (1998). Optimal sites for supplementary
      weather observations. *Journal of the Atmospheric Sciences*, **55**,
      399–414 — where the model is first used for an observing-network question.
    - Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and
      Predictability*, §5.5 *[citation needed: confirm section]*.
    - Grassberger, P. (1989). Information content and predictability of lumped
      and distributed dynamical systems *[citation needed]* — on extensivity of
      the Lyapunov spectrum in spatially extended systems.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate* *[citation needed: chapter and pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
