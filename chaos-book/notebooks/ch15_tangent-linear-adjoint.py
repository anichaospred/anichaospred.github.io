# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 15 -- Tangent linear and adjoint models.

Linearisation, the adjoint identity, the window of validity, and the two tests
that tell you whether an adjoint is correct.

Part V of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
Figures are static matplotlib, matching chapters 6 and 7.

To edit:   marimo edit notebooks/ch15_tangent-linear-adjoint.py
To export: make nb-one NB=ch15_tangent-linear-adjoint
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 15: Tangent Linear and Adjoint Models")


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
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from chaoslib import adjoint, integrate, lyapunov, plotting, systems

    SIGMA0, RHO0, BETA0 = 10.0, 28.0, 8.0 / 3.0
    L63_TRACE = -(SIGMA0 + 1.0 + BETA0)
    DT = 0.005  # tangent and nonlinear models share this step

    C_CONTEXT = plotting.mpl_colour(plotting.C_CONTEXT)
    C_TRUTH = plotting.C_TRUTH
    C_PERT = plotting.C_PERT
    C_SPREAD = plotting.C_SPREAD
    C_MEAN = plotting.C_MEAN
    C_FIXED = plotting.C_FIXED
    C_SAT = plotting.C_SAT
    C_START = plotting.C_START
    mpl_panels = plotting.mpl_panels
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    return (
        BETA0, C_CONTEXT, C_FIXED, C_MEAN, C_PERT, C_SAT, C_SPREAD, C_START,
        C_TRUTH, DT, L63_TRACE, LogNorm, RHO0, SIGMA0, adjoint, finish_mpl,
        integrate, lyapunov, mo, mpl_grid, mpl_panels, np, plotting, plt, systems,
    )


@app.cell
def helpers(BETA0, DT, RHO0, SIGMA0, adjoint, integrate, np, systems):
    def propagate(x0, tau, dt=DT):
        """Nonlinear model: integrate Lorenz 63 from x0 for tau, return the end state."""
        if tau <= 0.0:
            return np.asarray(x0, dtype=float)
        n = max(2, int(round(tau / dt)) + 1)
        return integrate.rk4(
            systems.lorenz63, np.asarray(x0, dtype=float),
            np.linspace(0.0, tau, n),
            sigma=SIGMA0, rho=RHO0, beta=BETA0,
        )[-1]

    def propagator(x0, tau, dt=DT):
        """The tangent linear propagator M(x0, tau)."""
        return adjoint.tangent_linear_propagator(
            systems.lorenz63, systems.lorenz63_jacobian,
            np.asarray(x0, dtype=float), tau, dt=dt,
            sigma=SIGMA0, rho=RHO0, beta=BETA0,
        )

    def forecast_metric(x0, tau, dt=DT):
        """A scalar a forecaster might care about: z at lead time tau.

        In Lorenz 63, z measures how far the vertical temperature profile departs
        from linear -- the depth of the overturning. Any scalar would do; the
        point is that the adjoint delivers its gradient for one model run's worth
        of extra work, whatever it is.
        """
        return float(propagate(x0, tau, dt)[2])

    def metric_gradient(x0, tau, dt=DT):
        """dJ/dx0 for J = z(tau), by one adjoint application.

        dJ/dx_tau is the constant vector e_z, and the adjoint carries it back to
        the start of the window: dJ/dx0 = M^T e_z. One matrix-vector product --
        no matter how many components x0 has.
        """
        return propagator(x0, tau, dt).T @ np.array([0.0, 0.0, 1.0])

    return forecast_metric, metric_gradient, propagate, propagator


@app.cell
def validity_data():
    # Section 2, precomputed: relative error of the linear prediction over a
    # grid of lead times and amplitudes, and the lead time at which it first
    # exceeds 10%. Costs ~7 s natively (~40 s in Pyodide) and has no
    # knob, so it is computed once by scripts/generate_ch15_data.py.
    VALIDITY_TAUS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
    VALIDITY_ERROR = {
        1e-06: (1.1200e-07, 4.7258e-07, 1.0165e-06, 1.2327e-06, 1.3469e-06, 2.1994e-05, 4.7037e-05, 1.3701e-03, 1.2225e-02, 4.2267e-02),
        0.0001: (1.7202e-05, 6.7593e-05, 1.2295e-04, 1.3156e-04, 1.3869e-04, 2.2153e-03, 4.7274e-03, 1.3672e-01, 1.3887e+00, 6.0762e+00),
        0.01: (1.7219e-03, 6.7911e-03, 1.2323e-02, 1.3070e-02, 1.3849e-02, 2.2991e-01, 9.0866e-01, 8.9450e+00, 1.0501e+01, 1.5629e+02),
        0.1: (1.7371e-02, 7.0974e-02, 1.2615e-01, 1.2281e-01, 1.3556e-01, 2.8454e+00, 4.4896e+00, 7.6121e+01, 1.1752e+02, 1.5446e+03),
    }
    VALIDITY_CROSSING = {
        1e-06: 16.5,
        1e-05: 14.0,
        0.0001: 12.0,
        0.001: 9.5,
        0.01: 6.5,
        0.1: 2.5,
    }

    return VALIDITY_CROSSING, VALIDITY_ERROR, VALIDITY_TAUS


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 15 · Tangent Linear and Adjoint Models

    **Part V — The machinery of prediction.**

    **The forecasting question.** Monday's forecast for Thursday was badly wrong. Which
    part of Monday's initial state was responsible — and by how much would each part
    have had to change to fix it?

    That is a question about a **gradient**: the derivative of one forecast quantity
    with respect to every component of the initial state. The obvious way to get it is
    to perturb each component in turn and re-run the model. For Lorenz 63 that is four
    runs and perfectly reasonable. For an operational model with $10^8$ state variables
    it is a hundred million runs, and it is never going to happen.

    The **adjoint** delivers the whole gradient in one extra model-sized integration,
    independent of the state dimension. That single fact is why 4D-Var exists, why
    singular vectors are computable, and why targeted observing is a real technique
    rather than a thought experiment. This chapter builds the machinery, finds the
    limits of it, and — at some length, because it matters more than anything else here
    — shows how you know it is correct.
    """
    )
    return


# ===========================================================================
# 1. The tangent linear model
# ===========================================================================
@app.cell(hide_code=True)
def sec1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · Linearise the model

    Write the nonlinear model as a map $\mathcal{M}$ taking a state forward by $\tau$.
    For a small perturbation $\delta x$ of the initial state, Taylor gives

    $$\mathcal{M}(x_0 + \delta x) - \mathcal{M}(x_0)
      = \mathbf{M}(x_0,\tau)\,\delta x + O(\|\delta x\|^2),$$

    where $\mathbf{M}(x_0,\tau) = \partial\mathcal{M}/\partial x$ is the **tangent
    linear propagator**. It obeys the linear equation

    $$\frac{d}{dt}\,\delta x = \mathbf{J}(x(t))\,\delta x, \qquad
      \mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j},$$

    integrated along the nonlinear trajectory. Two consequences worth stating plainly:

    - **$\mathbf{M}$ depends on the state.** It is built along a particular trajectory,
      so it is different on different days. That is the same flow-dependence chapter 7
      measured as finite-time exponents, and it is why $\mathbf{B}$ being static is
      3D-Var's central limitation (chapter 18).
    - **$\mathbf{M}$ is linear.** Once you have it, the effect of *any* small
      perturbation is a matrix-vector product. No further model runs.

    ### Linearise the discrete model, not the continuous one

    A detail that is easy to get wrong and hard to notice. The model you actually run
    is the *discretised* one — RK4, in this book. Its derivative is not the same object
    as the discretisation of the continuous $\mathbf{J}$: the tangent must be stepped
    through the **same RK4 stages**, with $\mathbf{J}$ evaluated at each intermediate
    state,

    $$\mathbf{K}_1 = \mathbf{J}(x)\mathbf{V},\quad
      \mathbf{K}_2 = \mathbf{J}(x + \tfrac{h}{2}k_1)(\mathbf{V} + \tfrac{h}{2}\mathbf{K}_1),
      \ \dots$$

    Freezing $\mathbf{J}$ at the start of each step instead — the obvious
    simplification — leaves an $O(h)$ inconsistency between the tangent and nonlinear
    models. Section 5 shows what that looks like when you test for it, and why no
    amount of reducing the perturbation size reveals it.
    """
    )
    return


@app.cell(hide_code=True)
def tlm_controls(mo):
    tau_sl = mo.ui.slider(
        start=0.25, stop=4.0, step=0.25, value=1.0,
        label="lead time τ (MTU)", show_value=True,
    )
    amp_sl = mo.ui.slider(
        start=-9, stop=-1, step=0.5, value=-3,
        label="log₁₀ perturbation amplitude (marker below)", show_value=True,
    )
    return amp_sl, tau_sl


@app.cell
def validation_figure(
    C_FIXED, C_PERT, C_SAT, C_TRUTH, adjoint, amp_sl, finish_mpl, mo, mpl_panels,
    np, propagator, systems, tau_sl,
):
    _tau = float(tau_sl.value)
    _amp_marked = 10.0 ** float(amp_sl.value)

    _amps = np.logspace(-9.0, -1.0, 17)
    _, _err = adjoint.tangent_linear_error(
        systems.lorenz63, systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]), _tau, _amps, dt=0.005,
    )

    # Fit the slope on the clean part, above round-off. A correct TLM gives 1.
    _clean = _amps >= 1e-5
    _slope = float(np.polyfit(np.log10(_amps[_clean]), np.log10(_err[_clean]), 1)[0])

    _M = propagator(np.array([1.0, 1.0, 20.0]), _tau)
    _residual = adjoint.adjoint_identity_residual(_M, n_trials=64)

    _fig, _ax = mpl_panels(
        2,
        titles=("The validation curve", "The adjoint identity"),
        height=3.6,
    )

    # ---- (a) the defining test ----
    _ax[0].loglog(_amps, _err, marker="o", markersize=4, color=C_TRUTH,
                  linewidth=1.4, label="measured")
    _ref = _err[_clean][0] * (_amps / _amps[_clean][0])
    _ax[0].loglog(_amps, _ref, color=C_SAT, linewidth=1.2, linestyle="--",
                  label="slope 1 (correct TLM)")
    _idx = int(np.argmin(np.abs(_amps - _amp_marked)))
    _ax[0].plot([_amps[_idx]], [_err[_idx]], marker="o", markersize=10,
                color=C_PERT, markeredgecolor="white", markeredgewidth=1.0,
                linestyle="none", zorder=6, label="slider amplitude")
    _ax[0].set_xlabel("perturbation amplitude α")
    _ax[0].set_ylabel("relative discrepancy vs nonlinear")
    _ax[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    _ax[0].annotate(
        f"fitted slope {_slope:.3f}",
        (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8,
        color="#211d33",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e6e1f2",
                  boxstyle="round,pad=0.3"),
    )

    # ---- (b) the identity, over random vector pairs ----
    _rng = np.random.default_rng(3)
    _lhs, _rhs = [], []
    for _ in range(60):
        _u = _rng.normal(size=3)
        _v = _rng.normal(size=3)
        _lhs.append(float((_M @ _u) @ _v))
        _rhs.append(float(_u @ (_M.T @ _v)))
    _lhs = np.asarray(_lhs)
    _rhs = np.asarray(_rhs)
    _lim = float(max(np.abs(_lhs).max(), np.abs(_rhs).max())) * 1.1
    _ax[1].plot([-_lim, _lim], [-_lim, _lim], color=C_SAT, linewidth=1.2,
                linestyle="--", label="exact equality")
    _ax[1].plot(_lhs, _rhs, marker="o", markersize=4, color=C_FIXED,
                linestyle="none", alpha=0.8, label="random (u, v) pairs")
    _ax[1].set_xlabel(r"$\langle \mathbf{M}u,\ v\rangle$")
    _ax[1].set_ylabel(r"$\langle u,\ \mathbf{M}^{\top}v\rangle$")
    _ax[1].legend(loc="upper left", fontsize=7, framealpha=0.9)
    _ax[1].annotate(
        f"max relative\nviolation {_residual:.1e}",
        (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=8,
        color="#211d33",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e6e1f2",
                  boxstyle="round,pad=0.3"),
    )
    finish_mpl(_fig, suptitle=f"τ = {_tau:g} MTU")

    _verdict = (
        "consistent with a correct tangent linear model"
        if 0.95 <= _slope <= 1.05 else
        "**not** slope 1 — the tangent and nonlinear models disagree"
    )
    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([tau_sl, amp_sl], gap="3rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; τ = {_tau:g} MTU  \n"
                f"Validation slope over α ∈ [10⁻⁵, 10⁻¹]: **{_slope:.3f}** — {_verdict}  \n"
                f"At α = {_amps[_idx]:.1e} the linear prediction misses by "
                f"**{_err[_idx]:.2%}**  \n"
                f"Adjoint identity: worst relative violation **{_residual:.1e}** over 64 "
                f"random pairs  \n"
                f"‖M‖ = **{np.linalg.norm(_M):.3g}** — the propagator itself is large, "
                f"which is the whole problem"
            ),
            kind="success" if 0.95 <= _slope <= 1.05 else "danger",
        ),
        mo.md(
            r"""**Why slope 1 is the test.** The term the linear model throws away is
            $O(\|\delta x\|^2)$. Divide by the signal, which is $O(\|\delta x\|)$, and
            the *relative* discrepancy is $O(\|\delta x\|)$ — a straight line of slope 1
            on log–log axes. Halve the perturbation and the error halves.

            The curve flattens at the smallest amplitudes because there the discrepancy
            has fallen below floating-point round-off in the nonlinear difference
            itself. That floor is expected and harmless. **A floor at large amplitudes
            is not** — it means a discrepancy that does not shrink with the perturbation,
            which is not a truncation error at all but a wrong derivative. Section 5."""
        ),
    ])
    return


# ===========================================================================
# 2. How far can you trust it?
# ===========================================================================
@app.cell(hide_code=True)
def sec2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · The window of validity

    Slope 1 says the linearisation is *correct*. It does not say it is *useful*. The
    neglected term is $O(\|\delta x\|^2)$, and $\|\delta x\|$ grows like
    $e^{\lambda_1 t}$ — so however small the perturbation starts, the quadratic term
    eventually catches up. The tangent linear model has a shelf life.

    The figure below measures it: the discrepancy between the linear prediction and the
    true nonlinear difference, over a grid of lead times and initial amplitudes, and
    the lead time at which it first exceeds 10 %.
    """
    )
    return


@app.cell
def validity_figure(
    C_SAT, C_TRUTH, VALIDITY_CROSSING, VALIDITY_ERROR, VALIDITY_TAUS, finish_mpl,
    mo, mpl_panels, np,
):
    _fig, _ax = mpl_panels(
        2,
        titles=("Linear prediction error vs lead time", "How long the linearisation lasts"),
        height=3.6,
    )

    _shades = ("#c4b5fd", "#a78bfa", "#7c3aed", "#4c1d95")
    for _i, (_amp, _errs) in enumerate(VALIDITY_ERROR.items()):
        _ax[0].loglog(VALIDITY_TAUS, _errs, marker="o", markersize=4,
                      color=_shades[_i], linewidth=1.3, label=f"δ₀ = {_amp:.0e}")
    _ax[0].axhline(0.10, color=C_SAT, linewidth=1.2, linestyle="--")
    _ax[0].annotate("10 % — the linearisation is done", (0.03, 0.10),
                    xycoords=("axes fraction", "data"), fontsize=6.5,
                    color="#b91c1c", va="bottom")
    _ax[0].set_xlabel("lead time τ (MTU)")
    _ax[0].set_ylabel("relative error of the linear prediction")
    _ax[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    _amps = np.array(list(VALIDITY_CROSSING))
    _taus = np.array([VALIDITY_CROSSING[a] for a in VALIDITY_CROSSING])
    _x = -np.log(_amps)
    _slope, _icpt = np.polyfit(_x, _taus, 1)
    _ax[1].plot(_x, _taus, marker="o", markersize=8, color=C_TRUTH,
                linestyle="none", label="measured")
    _ax[1].plot(_x, _slope * _x + _icpt, color=C_SAT, linewidth=1.3,
                linestyle="--", label=f"slope {_slope:.2f} MTU per e-fold")
    _ax[1].set_xlabel("−ln δ₀")
    _ax[1].set_ylabel("τ at which the linear model fails (MTU)")
    _ax[1].legend(loc="upper left", fontsize=7, framealpha=0.9)
    finish_mpl(_fig)

    # The book's one value for lambda_1, computed in chapter 7 and pinned in
    # chaoslib's test suite. Recomputing it here would cost about eight seconds in
    # the browser to reproduce a number that is already settled -- and if the two
    # chapters ever disagreed, that would be a bug rather than a feature.
    _lam = 0.9056
    _rows = "\n".join(
        f"| {a:.0e} | {VALIDITY_CROSSING[a]:.1f} |" for a in VALIDITY_CROSSING
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
| initial perturbation δ₀ | linear model valid to (MTU) |
|---|---|
{_rows}

**Measured slope {_slope:.2f} MTU per e-fold of δ₀, against $1/\\lambda_1$ =
{1.0 / _lam:.2f}**, from the leading Lyapunov exponent
$\\lambda_1$ = {_lam:.4f} MTU⁻¹ measured in chapter 7.

This is the logarithmic law a third time. Chapter 7 found it in the growth rate,
chapter 20 in the forecast horizon, and here it sets how long a linearisation lasts:

$$\\tau_{{\\text{{valid}}}} \\approx \\frac{{1}}{{\\lambda_1}}\\ln\\frac{{\\delta_c}}{{\\delta_0}}.$$

All three are the same statement — errors grow exponentially, so every quantity that
depends on their size moves logarithmically. **A hundredfold smaller perturbation buys
about five extra MTU of linearity, not a hundredfold longer window.**

The practical consequence is the one 4D-Var lives with: the assimilation window has to
be short enough that the linearisation holds across it. Chapter 18's windows are around
0.8 MTU for exactly this reason, and extending them is not a matter of computing power.
"""
        ),
    ])
    return


# ===========================================================================
# 3. The adjoint
# ===========================================================================
@app.cell(hide_code=True)
def sec3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · The adjoint, and why it is the whole game

    The **adjoint** $\mathbf{M}^{*}$ is defined by one identity: for all $u$ and $v$,

    $$\langle \mathbf{M}u,\ v\rangle = \langle u,\ \mathbf{M}^{*}v\rangle .$$

    Under the ordinary Euclidean inner product $\mathbf{M}^{*} = \mathbf{M}^{\top}$,
    which makes it look like a triviality. It is not — the *identity* is the definition,
    and under a weighted inner product (an energy norm, say, with
    $\langle u,v\rangle_E = u^{\top}\mathbf{E}v$) the adjoint acquires the weights:
    $\mathbf{M}^{*} = \mathbf{E}^{-1}\mathbf{M}^{\top}\mathbf{E}$. Operational adjoints
    are written for weighted norms, and this is where they go wrong.

    ### The gradient, for free

    Take any scalar forecast metric $J$ evaluated at the end of the window. The chain
    rule gives

    $$\frac{\partial J}{\partial x_0}
      = \mathbf{M}^{\top}\frac{\partial J}{\partial x_\tau}.$$

    Read the right-hand side carefully. $\partial J/\partial x_\tau$ is one vector, known
    analytically. Applying $\mathbf{M}^{\top}$ to it means integrating the adjoint
    equations *backwards* once. The result is the derivative of $J$ with respect to
    **every** component of the initial state.

    | State dimension $n$ | Finite differences | Adjoint |
    |---|---|---|
    | 3 (Lorenz 63) | 4 model runs | 2 |
    | 40 (Lorenz 96) | 41 | 2 |
    | $10^6$ | $10^6 + 1$ | 2 |
    | $10^8$ (operational NWP) | $10^8 + 1$ | 2 |

    The adjoint column does not depend on $n$. That is the entire reason variational
    assimilation is possible at operational size, and it is worth more than any other
    single idea in Part V.

    The price is that somebody has to write and maintain adjoint code for every line of
    the forecast model — historically an enormous undertaking, and the reason the EnKF
    (chapter 19) spread to coupled systems first. Automatic differentiation and learned
    models are changing that arithmetic; chapter 29 picks it up.
    """
    )
    return


@app.cell
def gradient_check(
    DT, forecast_metric, metric_gradient, mo, np,
):
    _x0 = np.array([1.0, 1.0, 20.0])
    _rows = []
    for _tau in (0.5, 1.0, 2.0):
        _g = metric_gradient(_x0, _tau)
        _eps = 1e-6
        _gfd = np.array([
            (forecast_metric(_x0 + _eps * _e, _tau)
             - forecast_metric(_x0 - _eps * _e, _tau)) / (2.0 * _eps)
            for _e in np.eye(3)
        ])
        _rel = float(np.linalg.norm(_gfd - _g) / np.linalg.norm(_g))
        _rows.append(
            f"| {_tau:.1f} | ({', '.join(f'{v:+.4f}' for v in _g)}) | "
            f"({', '.join(f'{v:+.4f}' for v in _gfd)}) | {_rel:.1e} |"
        )
    mo.md(
        "**The gradient of $J = z(\\tau)$, by adjoint and by central differences.**\n\n"
        "| τ (MTU) | adjoint $\\mathbf{M}^{\\top}e_z$ | central differences | relative difference |\n"
        "|---|---|---|---|\n"
        + "\n".join(_rows)
        + "\n\nAgreement to one part in $10^{8}$–$10^{9}$. For three variables the "
        "finite-difference column costs six extra model runs and is perfectly "
        "practical; the point of the table is that the two agree, so that the adjoint "
        "can be trusted where finite differences are impossible."
    )
    return


# ===========================================================================
# 4. Sensitivity
# ===========================================================================
@app.cell(hide_code=True)
def sec4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · Which initial states matter

    With the gradient in hand, the opening question has an answer. $\partial J/\partial
    x_0$ points in the direction in which the initial state most strongly influences the
    forecast, and its magnitude says how strongly.

    Compute it at many points on the attractor and a pattern appears: sensitivity is not
    uniform. Some initial states barely affect the forecast a lead time later; others
    dominate it.
    """
    )
    return


@app.cell(hide_code=True)
def sens_controls(mo):
    sens_tau = mo.ui.slider(
        start=0.25, stop=3.0, step=0.25, value=1.0,
        label="lead time τ for the metric (MTU)", show_value=True,
    )
    return (sens_tau,)


@app.cell
def sensitivity_figure(
    BETA0, C_CONTEXT, C_TRUTH, LogNorm, RHO0, SIGMA0, finish_mpl, integrate,
    metric_gradient, mo, mpl_panels, np, sens_tau, systems,
):
    _tau = float(sens_tau.value)
    _grid = integrate.trajectory_grid(t_final=120.0, dt=0.01)
    _traj = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid,
        sigma=SIGMA0, rho=RHO0, beta=BETA0,
    )
    _states = _traj[2000::150]
    _grads = np.array([metric_gradient(_x, _tau) for _x in _states])
    _mag = np.linalg.norm(_grads, axis=1)

    _fig, _ax = mpl_panels(
        2,
        titles=(
            f"Sensitivity of z(τ = {_tau:g}) to the initial state",
            "Distribution of |∂J/∂x₀|",
        ),
        height=3.7,
    )
    _ax[0].plot(_traj[2000:, 0], _traj[2000:, 2], color=C_CONTEXT,
                linewidth=0.35, zorder=1)
    _sc = _ax[0].scatter(
        _states[:, 0], _states[:, 2], c=_mag, cmap="plasma",
        # Log colour scale: the sensitivities span orders of magnitude, and on a
        # linear scale every point but the few most sensitive would look identical.
        norm=LogNorm(vmin=max(_mag.min(), 1e-3), vmax=_mag.max()),
        s=28, edgecolors="white", linewidths=0.4, zorder=3,
    )
    _cb = _fig.colorbar(_sc, ax=_ax[0], fraction=0.046, pad=0.03)
    _cb.set_label("|∂J/∂x₀|", fontsize=7.5)
    _cb.ax.tick_params(labelsize=7)
    _ax[0].set_xlabel("x")
    _ax[0].set_ylabel("z")

    _ax[1].hist(np.log10(np.maximum(_mag, 1e-6)), bins=18, color=C_TRUTH, alpha=0.78)
    _ax[1].set_xlabel("log₁₀ |∂J/∂x₀|")
    _ax[1].set_ylabel("count")
    finish_mpl(_fig)

    _ratio = float(_mag.max() / max(_mag.min(), 1e-12))
    _hi = _states[int(_mag.argmax())]
    _lo = _states[int(_mag.argmin())]
    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([sens_tau], justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; τ = {_tau:g} MTU, "
                f"{_states.shape[0]} initial states  \n"
                f"|∂J/∂x₀| ranges from **{_mag.min():.2f}** to **{_mag.max():.1f}** — a "
                f"factor of **{_ratio:.0f}** across one attractor  \n"
                f"Most sensitive near (x, z) = ({_hi[0]:+.1f}, {_hi[2]:.1f}); least near "
                f"({_lo[0]:+.1f}, {_lo[2]:.1f})  \n"
                f"Median **{np.median(_mag):.2f}**"
            ),
            kind="warn",
        ),
        mo.md(
            """The histogram is plotted in log₁₀ because a linear axis cannot hold the
            range. Two or three orders of magnitude separate the most and least
            sensitive initial states — for the *same* system, the *same* metric and the
            *same* lead time. Only the situation differs.

            This is the quantitative basis of **targeted observing**: if an extra
            observation can be placed anywhere, place it where the forecast is most
            sensitive to being wrong. Operational campaigns have flown aircraft into
            regions chosen exactly this way *[citation needed]*.

            Two cautions before reading too much into a sensitivity map. It is a
            **local, linear** statement, so Section 2's validity window applies: at
            τ = 3 MTU the map is already describing a linearisation that is starting to
            fail for anything but the smallest perturbations. And large *sensitivity*
            is not the same as large *growth* — chapter 16 asks the related but distinct
            question of which perturbation grows most, which is a singular vector
            rather than a gradient."""
        ),
    ])
    return


# ===========================================================================
# 5. How you know it is right
# ===========================================================================
@app.cell(hide_code=True)
def sec5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · How you know an adjoint is correct

    An adjoint is a second implementation of a model, written backwards, that produces
    a quantity you cannot check by eye. A wrong one does not crash. It returns a
    plausible gradient, the minimiser in chapter 18 walks slowly downhill anyway, and
    the result is a system that works badly for reasons nobody can locate. Adjoint bugs
    are notorious for surviving for years.

    There are exactly two tests worth running, and between them they catch almost
    everything.

    ### Test 1 — the adjoint identity

    $$\langle \mathbf{M}u,\ v\rangle = \langle u,\ \mathbf{M}^{\top}v\rangle
      \quad\text{for random } u,\ v.$$

    This must hold to **machine precision** — $10^{-14}$ or so, as the right-hand panel
    of Section 1 shows — because it is an algebraic identity, not an approximation.
    Anything larger means the adjoint is not the adjoint of the tangent linear model you
    are actually running. Crucially it tests $\mathbf{M}^{\top}$ against $\mathbf{M}$
    and says **nothing** about whether $\mathbf{M}$ itself is right.

    ### Test 2 — the finite-difference check

    That is what Section 1's left panel is for. The relative discrepancy between the
    linear prediction and the true nonlinear difference must fall **linearly** with the
    perturbation amplitude. Slope 1, down to round-off.

    You need both. Test 1 catches a transposed index or a missing weight matrix; Test 2
    catches a wrong Jacobian. An adjoint that passes only Test 1 can be the perfect
    adjoint of the wrong tangent linear model.

    ### Three bugs these caught, in this book

    Not hypothetical — all three were live in `chaoslib` during writing, and each was
    found by one of the tests above rather than by inspection.

    **A floor that no smaller perturbation removed.** The first tangent linear model
    linearised the *continuous* flow and stepped it with $\mathbf{J}$ frozen over each
    RK4 step. Test 2 gave a **constant 4.6 % relative error, independent of amplitude**
    — a flat line where there should have been slope 1. Because the error did not shrink
    with $\alpha$, it was not truncation: it was an $O(h)$ mismatch between the tangent
    and the discrete map. Fixed by stepping the tangent through the same RK4 stages
    (Section 1).

    **A zero-length window that advanced anyway.** The propagator computed
    `n_steps = max(1, round(tau/dt))`, so a window of $\tau = 0$ took one step of $h$
    instead of returning the identity. In cycling 4D-Var an observation sits at the
    *start* of the window — the normal case — so this corrupted essentially every
    gradient, by about 6.8 %. Found only when chapter 20's 4D-Var underperformed for no
    visible reason and its gradient was checked directly against central differences.

    **A propagator covering the wrong interval.** The same expression made
    $\mathbf{M}$ correspond to `n_steps * dt` rather than to $\tau$ whenever $\tau$ was
    not an exact multiple of the step. Caught by the identity
    $\det\mathbf{M} = e^{\tau\,\operatorname{tr}\mathbf{J}}$, which pins the interval
    independently of the trajectory.

    That last one points at a third kind of test worth looking for in any system you
    work with: an **exact identity the propagator must satisfy**. For Lorenz 63 the
    divergence of the flow is constant, so Liouville's theorem gives

    $$\det \mathbf{M}(x_0,\tau) = e^{\tau\,\operatorname{tr}\mathbf{J}}
      = e^{-(\sigma+1+\beta)\tau}$$

    exactly, for every trajectory and every $\tau$. It is the same kind of check as the
    trace identity in chapter 7, and it is verified live below.
    """
    )
    return


@app.cell
def determinant_check(BETA0, L63_TRACE, SIGMA0, mo, np, propagator):
    _rows = []
    _rng = np.random.default_rng(11)
    for _tau in (0.25, 0.5, 1.0, 2.0):
        _M = propagator(np.array([1.0, 1.0, 20.0]), _tau)
        _det = float(np.linalg.det(_M))
        _exact = float(np.exp(L63_TRACE * _tau))
        _rel = abs(_det - _exact) / _exact
        _rows.append(f"| {_tau:.2f} | {_det:.6e} | {_exact:.6e} | {_rel:.1e} |")
    mo.md(
        "**Liouville's theorem, checked live.**\n\n"
        "| τ (MTU) | det M | $e^{\\tau\\,\\mathrm{tr}\\mathbf{J}}$ | relative error |\n"
        "|---|---|---|---|\n" + "\n".join(_rows)
        + "\n\nThe agreement is limited only by the RK4 truncation of the tangent "
        "integration, and it tightens as the step shrinks — it is a statement about "
        "the interval the propagator covers, which is exactly the property the third "
        "bug above violated."
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

    1. **Watch the slope survive.** Move τ from 0.25 to 4 MTU. The validation curve
       shifts upward — the linear model is less accurate at longer lead — but the slope
       stays at 1. Accuracy and correctness are different properties, and only one of
       them is a bug.
    2. **Find the round-off floor.** At τ = 0.25, at what amplitude does the curve stop
       falling? Now reason about why testing an adjoint with an amplitude of $10^{-12}$
       would tell you nothing.
    3. **Break the window.** Set the sensitivity lead time to 3 MTU and compare the map
       with the one at 0.5. Is the pattern the same? Section 2 says the linearisation is
       already suspect at 3 MTU for δ₀ larger than about $10^{-2}$ — does the map look
       like something you would still act on?
    4. **Predict before you look.** Using $\tau_{\text{valid}} \approx
       \lambda_1^{-1}\ln(\delta_c/\delta_0)$ and $\lambda_1 = 0.906$, estimate the
       validity window for $\delta_0 = 10^{-8}$. Then check it against the trend in the
       right-hand panel of Section 2.

    ## What you should have seen

    The tangent linear model reproduces the nonlinear difference with an error that
    falls **linearly** in the perturbation amplitude, down to round-off, and its adjoint
    satisfies $\langle \mathbf{M}u,v\rangle = \langle u,\mathbf{M}^{\top}v\rangle$ to
    about $10^{-14}$. Those two facts, together, are what "the adjoint is correct" means.

    The linearisation is trustworthy only over a finite window that grows like
    $\lambda_1^{-1}\ln(1/\delta_0)$ — the logarithmic law once again, now setting how
    long 4D-Var's assimilation window can be.

    And one adjoint integration yields the derivative of a forecast metric with respect
    to every component of the initial state, at a cost independent of the state
    dimension. Everything in chapters 16, 18 and 20 is built on that.

    ## Further reading

    - Errico, R. M. (1997). What is an adjoint model? *Bulletin of the American
      Meteorological Society*, **78**, 2577–2591 — the clearest introduction there is.
    - Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*,
      §6.2–6.3.
    - Giering, R. and Kaminski, T. (1998). Recipes for adjoint code construction.
      *ACM TOMS*, **24**, 437–474 *[citation needed: pages]*.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 5 *[citation needed: pages]*.

    ---

    *System: Lorenz (1963), σ = 10, ρ = 28, β = 8/3.*
    *Tangent and nonlinear models share an RK4 step of dt = 0.005; the tangent is
    stepped through the same stages as the nonlinear map.*
    *Validity-window grid precomputed by `scripts/generate_ch15_data.py`.*
    *Time unit: 1 MTU read as ≈ 5 days — a loose convention, not a calibration; see
    chapter 6, Section 4.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
