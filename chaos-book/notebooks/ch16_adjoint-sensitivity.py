# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 16 -- Adjoint sensitivity and optimal perturbations.

Singular vectors, how they differ from both the gradient and the Lyapunov
vectors, why they depend on the norm you choose, and what that means for
targeted observing.

Part V of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
Figures are static matplotlib, matching chapters 6, 7 and 15.

To edit:   marimo edit notebooks/ch16_adjoint-sensitivity.py
To export: make nb-one NB=ch16_adjoint-sensitivity
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 16: Adjoint Sensitivity and Singular Vectors")


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

    from chaoslib import adjoint, integrate, plotting, systems

    SIGMA0, RHO0, BETA0 = 10.0, 28.0, 8.0 / 3.0
    DT = 0.005
    LAMBDA1 = 0.9056  # chapter 7, pinned in chaoslib's tests

    C_CONTEXT = plotting.mpl_colour(plotting.C_CONTEXT)
    C_TRUTH = plotting.C_TRUTH
    C_PERT = plotting.C_PERT
    C_SPREAD = plotting.C_SPREAD
    C_MEAN = plotting.C_MEAN
    C_FIXED = plotting.C_FIXED
    C_SAT = plotting.C_SAT
    C_START = plotting.C_START
    C_OBS = plotting.C_OBS
    mpl_panels = plotting.mpl_panels
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    return (
        BETA0, C_CONTEXT, C_FIXED, C_MEAN, C_OBS, C_PERT, C_SAT, C_SPREAD,
        C_START, C_TRUTH, DT, LAMBDA1, RHO0, SIGMA0, adjoint, finish_mpl,
        integrate, mo, mpl_grid, mpl_panels, np, plotting, plt, systems,
    )


@app.cell
def helpers(BETA0, DT, RHO0, SIGMA0, adjoint, integrate, np, systems):
    def l63_propagator(x0, tau, dt=DT):
        return adjoint.tangent_linear_propagator(
            systems.lorenz63, systems.lorenz63_jacobian,
            np.asarray(x0, dtype=float), tau, dt=dt,
            sigma=SIGMA0, rho=RHO0, beta=BETA0,
        )

    def l96_propagator(x0, tau, dt=0.01, forcing=8.0):
        return adjoint.tangent_linear_propagator(
            systems.lorenz96, systems.lorenz96_jacobian,
            np.asarray(x0, dtype=float), tau, dt=dt, forcing=forcing,
        )

    def angle_between(a, b):
        """Angle in degrees between two directions, ignoring sign."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        cos = abs(float(a @ b)) / (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.degrees(np.arccos(min(1.0, cos))))

    # A reference L63 attractor and a base state on it, shared by the figures.
    _grid = integrate.trajectory_grid(t_final=120.0, dt=0.01)
    attractor = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid,
        sigma=SIGMA0, rho=RHO0, beta=BETA0,
    )[2000:]
    base_state = attractor[0].copy()

    # A spun-up Lorenz 96 state, for the high-dimensional comparison.
    _x96 = 8.0 * np.ones(40)
    _x96[19] += 0.01
    l96_state = integrate.rk4(
        systems.lorenz96, _x96, integrate.trajectory_grid(30.0, 0.01), forcing=8.0
    )[-1]

    return (
        angle_between, attractor, base_state, l63_propagator, l96_propagator,
        l96_state,
    )


@app.cell
def amplification_data():
    # Section 2, precomputed: the leading singular value of the L63 propagator,
    # geometric-mean over 33 decorrelated base points on the attractor,
    # with the range across those points. Averaging is what makes the curve
    # monotonic -- at a single base point sigma_1(tau) is not. Costs ~10 s
    # natively (~60 s in Pyodide) and has no knob, so it is computed once by
    # scripts/generate_ch16_data.py.
    AMP_TAUS = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    AMP_GEOMEAN = (2.4668, 4.1387, 6.4604, 12.6322, 32.2683, 93.5071, 373.6599, 3231.8706)
    AMP_MIN = (1.0044, 1.1973, 1.2045, 2.6799, 4.1918, 4.9203, 9.7729, 307.3166)
    AMP_MAX = (14.0467, 109.1267, 697.8335, 237.7141, 440.8853, 6547.3964, 7098.2340, 169533.8043)
    AMP_N_POINTS = 33

    return AMP_GEOMEAN, AMP_MAX, AMP_MIN, AMP_N_POINTS, AMP_TAUS


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 16 · Adjoint Sensitivity and Optimal Perturbations

    **Part V — The machinery of prediction.**

    **The forecasting question.** You have one extra observation to place, anywhere in
    the domain, and it will be taken now. Where should it go?

    Chapter 15 answered a nearby question — *what does this forecast quantity depend on?*
    — with the gradient $\partial J/\partial x_0$. That is not quite what an observing
    plan needs. The gradient says where an error would matter most; it does not say where
    an error is most likely to **grow**. Those are different questions with different
    answers, and this chapter is about the difference.

    | | Question | Object |
    |---|---|---|
    | **Sensitivity** | What does *this metric* depend on? | the gradient $\mathbf{M}^{\top}\partial J/\partial x_\tau$ |
    | **Optimal growth** | Which perturbation grows most, in *this norm*? | the leading singular vector of $\mathbf{M}$ |
    | **Asymptotic growth** | Which direction grows in the long run? | the leading Lyapunov vector (chapter 7) |

    All three come out of the same propagator. The rest of the chapter separates them,
    and finds that the answer to the middle question depends on something that looks like
    a technicality and is not: the norm.
    """
    )
    return


# ===========================================================================
# 1. Singular vectors
# ===========================================================================
@app.cell(hide_code=True)
def sec1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · The perturbation that grows most

    Over a finite window the propagator $\mathbf{M}(x_0,\tau)$ maps a unit sphere of
    perturbations onto an ellipsoid. Its singular value decomposition
    $\mathbf{M} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^{\top}$ names the axes:

    - $v_1$, the leading **right** singular vector, is the initial perturbation that
      grows most over the window — the **leading singular vector**;
    - $u_1$, the leading **left** singular vector, is what it grows *into*;
    - $\sigma_1$ is the amplification: $\|\mathbf{M}v_1\| = \sigma_1\|v_1\|$, and no
      other direction does better.

    Computing them needs $\mathbf{M}$, and $\mathbf{M}$ is never formed explicitly at
    operational size — a $10^8 \times 10^8$ matrix is not an object. Instead $\sigma_1$
    and $v_1$ are found by iterative methods (Lanczos) that need only the *action* of
    $\mathbf{M}$ and $\mathbf{M}^{\top}$ on a vector — one tangent linear integration
    and one adjoint integration each. That is the whole reason chapter 15 came first.

    Move the optimisation window below and watch the ellipsoid change.
    """
    )
    return


@app.cell(hide_code=True)
def sv_controls(mo):
    tau_sl = mo.ui.slider(
        start=0.25, stop=8.0, step=0.25, value=1.0,
        label="optimisation window τ (MTU)", show_value=True,
    )
    scale_sl = mo.ui.slider(
        start=1, stop=12, step=1, value=6,
        label="arrow scale (display only)", show_value=True,
    )
    return scale_sl, tau_sl


@app.cell
def sigma_curve_data(adjoint, base_state, l63_propagator, np):
    # sigma_1 against the optimisation window at the chapter's base state.
    # Independent of every slider, so it is computed once.
    _tt = np.linspace(0.25, 8.0, 24)
    _s1 = np.array([
        float(adjoint.singular_vectors(l63_propagator(base_state, float(t)), 1)[0][0])
        for t in _tt
    ])
    sigma_curve = (_tt, _s1)
    return (sigma_curve,)


@app.cell
def sv_figure(
    C_CONTEXT, C_MEAN, C_PERT, C_SAT, C_SPREAD, C_START, C_TRUTH, LAMBDA1,
    adjoint, attractor, base_state, finish_mpl, integrate, l63_propagator, mo,
    mpl_panels, np, scale_sl, sigma_curve, systems, tau_sl,
):
    _tau = float(tau_sl.value)
    _scale = float(scale_sl.value)
    _M = l63_propagator(base_state, _tau)
    _sigma, _initial, _final = adjoint.singular_vectors(_M, 3)

    # Where the base state ends up after tau, so the evolved vector is drawn there.
    _n = max(2, int(round(_tau / 0.005)) + 1)
    _end = integrate.rk4(
        systems.lorenz63, base_state, np.linspace(0.0, _tau, _n)
    )[-1]

    _v1 = _initial[:, 0] / np.linalg.norm(_initial[:, 0])
    _u1 = _final[:, 0] / np.linalg.norm(_final[:, 0])
    _achieved = float(np.linalg.norm(_M @ _v1))

    # At a SINGLE base point sigma_1 can fall below exp(lambda_1 tau) for a long
    # window: the asymptotic rate is a long-time average, and this particular
    # stretch of trajectory may be quieter than average. Section 2 averages over
    # the attractor, where the inequality does hold at every tau.
    _ratio = float(_sigma[0] / np.exp(LAMBDA1 * _tau))
    _ratio_str = (
        f"{_ratio:.1f}× larger" if _ratio >= 1.05
        else (f"{1.0 / _ratio:.1f}× *smaller* — this window is quieter than average"
              if _ratio <= 0.95 else "about the same")
    )

    _fig, _ax = mpl_panels(
        3,
        titles=(
            "Where it starts, and what it becomes",
            "The singular values",
            "Optimal vs asymptotic growth",
        ),
        height=3.7,
    )

    # ---- (a) the vectors, on the x-z projection ----
    _ax[0].plot(attractor[:, 0], attractor[:, 2], color=C_CONTEXT,
                linewidth=0.35, zorder=1)
    _ax[0].annotate(
        "", xy=(base_state[0] + _scale * _v1[0], base_state[2] + _scale * _v1[2]),
        xytext=(base_state[0], base_state[2]),
        arrowprops=dict(arrowstyle="-|>", color=C_PERT, linewidth=2.0),
        zorder=5,
    )
    _ax[0].annotate(
        "", xy=(_end[0] + _scale * _u1[0], _end[2] + _scale * _u1[2]),
        xytext=(_end[0], _end[2]),
        arrowprops=dict(arrowstyle="-|>", color=C_MEAN, linewidth=2.0),
        zorder=5,
    )
    _ax[0].plot([base_state[0]], [base_state[2]], marker="o", markersize=7,
                color=C_START, markeredgecolor="white", markeredgewidth=0.9,
                linestyle="none", zorder=6, label="base state x₀")
    _ax[0].plot([_end[0]], [_end[2]], marker="s", markersize=7, color=C_TRUTH,
                markeredgecolor="white", markeredgewidth=0.9, linestyle="none",
                zorder=6, label=f"x(τ = {_tau:g})")
    _ax[0].plot([], [], color=C_PERT, linewidth=2.0, label="v₁ (optimal initial)")
    _ax[0].plot([], [], color=C_MEAN, linewidth=2.0, label="u₁ (what it becomes)")
    _ax[0].set_xlabel("x")
    _ax[0].set_ylabel("z")
    _ax[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)

    # ---- (b) the singular values ----
    _pos = np.arange(_sigma.size)
    _ax[1].bar(_pos, _sigma, color=[C_TRUTH, C_MEAN, C_SPREAD][: _sigma.size],
               alpha=0.85, width=0.6)
    _ax[1].set_yscale("log")
    _ax[1].set_xticks(_pos)
    _ax[1].set_xticklabels([f"σ{i + 1}" for i in _pos])
    _ax[1].set_ylabel("amplification")
    for _i, _v in enumerate(_sigma):
        _ax[1].annotate(f"{_v:.3g}", (_i, _v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7.5)
    _ax[1].annotate(
        f"σ₁/σ₂ = {_sigma[0] / _sigma[1]:.1f}",
        (0.5, 0.04), xycoords="axes fraction", ha="center", fontsize=7.5,
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e6e1f2",
                  boxstyle="round,pad=0.25"),
    )

    # ---- (c) sigma_1 against the Lyapunov estimate, at this base point ----
    # The curve itself does not depend on the slider, so it lives in its own cell
    # and marimo does not recompute it on every drag -- worth about eight seconds
    # of browser time per slider move.
    _tt, _s1 = sigma_curve
    _ax[2].semilogy(_tt, _s1, color=C_TRUTH, linewidth=1.5,
                    label="σ₁ (this base point)")
    _ax[2].semilogy(_tt, np.exp(LAMBDA1 * _tt), color=C_SAT, linewidth=1.3,
                    linestyle="--", label=r"$e^{\lambda_1\tau}$")
    _ax[2].axvline(_tau, color=C_PERT, linewidth=1.2, linestyle=":")
    _ax[2].set_xlabel("optimisation window τ (MTU)")
    _ax[2].set_ylabel("amplification")
    _ax[2].legend(loc="upper left", fontsize=7, framealpha=0.9)

    finish_mpl(_fig, suptitle=f"Leading singular vector at one point on the attractor, τ = {_tau:g} MTU")

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([tau_sl, scale_sl], gap="3rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; τ = {_tau:g} MTU  \n"
                f"σ = **{', '.join(f'{v:.4g}' for v in _sigma)}** &nbsp;·&nbsp; "
                f"σ₁/σ₂ = **{_sigma[0] / _sigma[1]:.1f}**  \n"
                f"v₁ = ({', '.join(f'{v:+.3f}' for v in _v1)}) → "
                f"u₁ = ({', '.join(f'{v:+.3f}' for v in _u1)})  \n"
                f"Achieved ‖Mv₁‖ = **{_achieved:.4f}** against σ₁ = "
                f"**{_sigma[0]:.4f}** — the optimum is attained, not merely bounded  \n"
                f"Lyapunov estimate $e^{{\\lambda_1\\tau}}$ = "
                f"**{np.exp(LAMBDA1 * _tau):.3f}** — optimal growth is "
                f"**{_ratio_str}**"
            ),
            kind="warn",
        ),
        mo.md(
            r"""The third panel is deliberately drawn at a *single* base point, and it
            behaves in two ways worth noticing.

            It is **not monotonic**: σ₁ can fall as the window lengthens, because a
            longer window may include a stretch in which the flow contracts. That is a
            real property of a non-autonomous linear system, not a numerical artefact.

            And beyond about τ = 3 MTU it drops **below** $e^{\lambda_1\tau}$ at this
            particular point. That is not a contradiction: $\lambda_1$ is a long-time
            average over the whole attractor, and one specific six-MTU stretch of
            trajectory can be quieter than average. What Section 2 shows is that the
            inequality does hold once you average over base points — which is the only
            level at which the statement "optimal growth exceeds the Lyapunov estimate"
            is actually true."""
        ),
    ])
    return


# ===========================================================================
# 2. Optimal growth vs asymptotic growth
# ===========================================================================
@app.cell(hide_code=True)
def sec2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · Why not just use the Lyapunov vector?

    Chapter 7 already found the direction that grows fastest **in the long run**: the
    leading Lyapunov vector, at rate $\lambda_1 = 0.906$ MTU⁻¹. If that is the
    fastest-growing direction, why compute anything else?

    Because operational forecasting does not run in the long run. Over a *finite*
    window a non-normal system amplifies some directions far more than
    $e^{\lambda_1\tau}$, and the leading singular vector finds them. The figure below
    averages over the attractor — necessary, as Section 1's third panel showed — and
    the gap is systematic rather than incidental.
    """
    )
    return


@app.cell
def amplification_figure(
    AMP_GEOMEAN, AMP_MAX, AMP_MIN, AMP_N_POINTS, AMP_TAUS, C_CONTEXT, C_SAT,
    C_TRUTH, LAMBDA1, finish_mpl, mo, mpl_panels, np,
):
    _t = np.asarray(AMP_TAUS)
    _geo = np.asarray(AMP_GEOMEAN)
    _lo = np.asarray(AMP_MIN)
    _hi = np.asarray(AMP_MAX)
    _lyap = np.exp(LAMBDA1 * _t)

    _fig, _ax = mpl_panels(
        2,
        titles=("Optimal amplification vs the Lyapunov estimate",
                "As a growth rate"),
        height=3.6,
    )
    _ax[0].fill_between(_t, _lo, _hi, color=C_TRUTH, alpha=0.12,
                        label="range across base points")
    _ax[0].loglog(_t, _geo, marker="o", markersize=4.5, color=C_TRUTH,
                  linewidth=1.6, label="σ₁ (geometric mean)")
    _ax[0].loglog(_t, _lyap, color=C_SAT, linewidth=1.4, linestyle="--",
                  label=r"$e^{\lambda_1\tau}$")
    # Explicit ticks at the tau values computed: log-scale minor ticks collide into
    # an unreadable smear at this panel width.
    _ax[0].set_xticks(list(_t))
    _ax[0].set_xticklabels([f"{v:g}" for v in _t])
    _ax[0].minorticks_off()
    _ax[0].set_xlabel("optimisation window τ (MTU)")
    _ax[0].set_ylabel("amplification")
    _ax[0].legend(loc="upper left", fontsize=7, framealpha=0.9)

    _rate = np.log(_geo) / _t
    _ax[1].plot(_t, _rate, marker="o", markersize=4.5, color=C_TRUTH,
                linewidth=1.6, label="ln σ₁ / τ")
    _ax[1].axhline(LAMBDA1, color=C_SAT, linewidth=1.4, linestyle="--",
                   label=f"λ₁ = {LAMBDA1}")
    _ax[1].set_xlabel("optimisation window τ (MTU)")
    _ax[1].set_ylabel("effective growth rate (MTU⁻¹)")
    _ax[1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"Averaged over {AMP_N_POINTS} base points on the attractor")

    _ratio = _geo / _lyap
    _rows = "\n".join(
        f"| {t:g} | {g:.1f} | {l:.2f} | {r:.1f}× | {ra:.2f} |"
        for t, g, l, r, ra in zip(_t, _geo, _lyap, _ratio, _rate)
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
| τ (MTU) | σ₁ (geo-mean) | $e^{{\\lambda_1\\tau}}$ | ratio | ln σ₁/τ |
|---|---|---|---|---|
{_rows}

**Optimal growth beats the Lyapunov estimate by a factor of
{_ratio.min():.1f}–{_ratio.max():.1f} at every window length tested**, and the
effective rate $\\ln\\sigma_1/\\tau$ falls from {_rate[0]:.2f} towards λ₁ = {LAMBDA1}
as the window lengthens — {_rate[-1]:.2f} MTU⁻¹ at τ = {_t[-1]:g}. The two agree only
asymptotically, and "asymptotically" here means far beyond any useful forecast range.

That gap is **non-normality**: $\\mathbf{{M}}$ is not symmetric, so its singular
vectors are not its eigenvectors, and over a finite window the transient amplification
of a well-chosen direction exceeds the asymptotic rate of any direction. Chapter 7 saw
the same thing from the other side — the finite-time exponents there ran about 3×
above λ₁ at τ = 0.5, and roughly half of that elevation was exactly this effect.

**This is why operational centres perturb along singular vectors.** An ensemble built
from random perturbations wastes most of its members on directions that will not grow;
one built from the leading singular vectors spans the subspace where error actually
develops over the forecast range that matters. The shaded band is the other half of the
argument: at a given τ the amplification varies by orders of magnitude across the
attractor, so *which* singular vectors matter is a question that has to be answered
afresh every day.
"""
        ),
    ])
    return


# ===========================================================================
# 3. Sensitivity is not growth
# ===========================================================================
@app.cell(hide_code=True)
def sec3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · Sensitivity is not growth

    Chapter 15 left this open deliberately. Both objects come from the same propagator:

    $$\underbrace{\frac{\partial J}{\partial x_0} = \mathbf{M}^{\top}\frac{\partial J}{\partial x_\tau}}_{\text{sensitivity of one metric}}
      \qquad\text{versus}\qquad
      \underbrace{v_1 : \ \|\mathbf{M}v_1\| = \sigma_1}_{\text{fastest-growing perturbation}}$$

    Are they the same direction? The answer is instructive precisely because it is
    "sometimes", and the mechanism is visible in the algebra. Expand the gradient in the
    singular basis:

    $$\mathbf{M}^{\top}\frac{\partial J}{\partial x_\tau}
      = \sum_i \sigma_i \left(u_i \cdot \frac{\partial J}{\partial x_\tau}\right) v_i .$$

    The gradient is a $\sigma_i$-weighted combination of the singular vectors, and two
    things decide whether $v_1$ dominates it:

    1. **How dominant $\sigma_1$ is.** If $\sigma_1 \gg \sigma_2$, the first term swamps
       the rest.
    2. **Whether the metric "sees" $u_1$.** The weight on $v_1$ is proportional to
       $u_1 \cdot \partial J/\partial x_\tau$. If the metric is orthogonal to the evolved
       leading singular vector, that term vanishes entirely.

    Both conditions are testable, and the two systems in this book fall on opposite
    sides of them.
    """
    )
    return


@app.cell(hide_code=True)
def l96_controls(mo):
    # A separate window for the Lorenz 96 comparison. Over this slider's grid the
    # angle(grad J, v1) runs 88.6, 88.8, 68.6, 24.6, 64.8, 84.1, 85.2, 85.9 degrees --
    # near-orthogonal except for one dip at tau = 1.0, which is exactly where Section
    # 1's shared slider sits by default. Hence a separate control defaulting to 0.5.
    l96_tau_sl = mo.ui.slider(
        start=0.25, stop=2.0, step=0.25, value=0.5,
        label="Lorenz 96 window τ (MTU)", show_value=True,
    )
    return (l96_tau_sl,)


@app.cell
def sensitivity_vs_growth(
    C_MEAN, C_OBS, C_PERT, C_TRUTH, adjoint, angle_between, base_state, finish_mpl,
    l63_propagator, l96_propagator, l96_state, l96_tau_sl, mo, mpl_panels, np,
    tau_sl,
):
    _tau = float(tau_sl.value)

    # ---- Lorenz 63: three natural metrics ----
    _M = l63_propagator(base_state, _tau)
    _sigma, _initial, _final = adjoint.singular_vectors(_M, 3)
    _v1 = _initial[:, 0]
    _u1 = _final[:, 0]

    _l63_rows = []
    for _name, _dJ in (("z(τ)", np.array([0.0, 0.0, 1.0])),
                       ("x(τ)", np.array([1.0, 0.0, 0.0])),
                       ("y(τ)", np.array([0.0, 1.0, 0.0]))):
        _g = _M.T @ _dJ
        _l63_rows.append((_name, abs(float(_dJ @ _u1)), angle_between(_g, _v1)))
    # A metric deliberately orthogonal to u1: the extreme case of condition 2.
    _dJ_orth = np.linalg.svd(_u1.reshape(1, -1))[2][1]
    _g_orth = _M.T @ _dJ_orth
    _l63_rows.append(("⊥ u₁", abs(float(_dJ_orth @ _u1)), angle_between(_g_orth, _v1)))

    # ---- Lorenz 96: a LOCAL metric in 40 dimensions ----
    _tau96 = float(l96_tau_sl.value)
    _M96 = l96_propagator(l96_state, _tau96, dt=0.01)
    _s96, _i96, _f96 = adjoint.singular_vectors(_M96, 2)
    _v1_96 = _i96[:, 0]
    _u1_96 = _f96[:, 0]
    _e20 = np.zeros(40)
    _e20[19] = 1.0
    _g96 = _M96.T @ _e20
    _angle96 = angle_between(_g96, _v1_96)
    _overlap96 = abs(float(_e20 @ _u1_96))

    _fig, _ax = mpl_panels(
        2,
        titles=(
            "Lorenz 96: v₁ and ∇J at this window",
            "…and both are localised, differently",
        ),
        height=3.6,
    )
    _sites = np.arange(1, 41)
    _ax[0].plot(_sites, _v1_96 / np.linalg.norm(_v1_96), marker="o", markersize=3,
                color=C_PERT, linewidth=1.4, label="v₁ (fastest-growing)")
    _ax[0].plot(_sites, _g96 / np.linalg.norm(_g96), marker="s", markersize=3,
                color=C_TRUTH, linewidth=1.4, label="∂J/∂x₀ for J = x₂₀(τ)")
    _ax[0].axvline(20, color=C_OBS, linewidth=1.1, linestyle=":")
    _ax[0].annotate("site 20\n(the metric)", (20, 0.97),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=6.5, color="#c2410c")
    _ax[0].axhline(0.0, color="#c9c2de", linewidth=0.8)
    _ax[0].set_xlabel("site k")
    _ax[0].set_ylabel("normalised amplitude")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _ax[1].plot(_sites, (_v1_96 / np.linalg.norm(_v1_96)) ** 2, marker="o",
                markersize=3, color=C_PERT, linewidth=1.4, label="v₁")
    _ax[1].plot(_sites, (_u1_96 / np.linalg.norm(_u1_96)) ** 2, marker="^",
                markersize=3, color=C_MEAN, linewidth=1.4, label="u₁ (evolved)")
    _ax[1].plot(_sites, (_g96 / np.linalg.norm(_g96)) ** 2, marker="s",
                markersize=3, color=C_TRUTH, linewidth=1.4, label="∂J/∂x₀")
    _ax[1].set_xlabel("site k")
    _ax[1].set_ylabel("share of the variance")
    _ax[1].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"Lorenz 96, τ = {_tau96:g} MTU")

    def _support(v, frac=0.9):
        _p = np.sort(np.asarray(v, dtype=float) ** 2)[::-1]
        return int(np.searchsorted(np.cumsum(_p) / _p.sum(), frac)) + 1

    _table = "\n".join(
        f"| {n} | {o:.3f} | {a:.1f}° |" for n, o, a in _l63_rows
    )
    mo.vstack([
        mo.md(
            f"""### Lorenz 63 — three dimensions, one dominant direction

At τ = {_tau:g} the singular values are
{', '.join(f'{v:.3g}' for v in _sigma)}, so σ₁/σ₂ = **{_sigma[0] / _sigma[1]:.1f}**.

| metric $J$ | $|u_1\\cdot\\partial J/\\partial x_\\tau|$ | angle(∇J, v₁) |
|---|---|---|
{_table}

For every *natural* metric the gradient lies close to $v_1$ — condition 1 is satisfied
and the leading term dominates. The last row is the exception constructed on purpose: a
metric orthogonal to $u_1$ kills the $v_1$ term exactly, and the gradient comes out at
**90°** to the fastest-growing perturbation. In a strongly rank-1 system the two
questions usually have the same answer, and that is a property of the system, not a
general truth."""
        ),
        mo.md("### ⚙️ Lorenz 96 window"),
        mo.hstack([l96_tau_sl], justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"""### Lorenz 96 — forty dimensions, no dominant direction

At τ = {_tau96:g}: σ₁ = **{_s96[0]:.2f}**, σ₁/σ₂ = **{_s96[0] / _s96[1]:.2f}** —
nothing like dominant. The local metric $J = x_{{20}}(\\tau)$ overlaps the evolved
leading singular vector by only **{_overlap96:.3f}**, so both of Section 3's conditions
fail, and

**angle(∇J, v₁) = {_angle96:.1f}°.**

The perturbation that grows fastest and the perturbation this forecast quantity depends
on are close to orthogonal. Asking "where will error grow?" and "what does my forecast
depend on?" genuinely returns different places.

**And the angle is not a monotonic function of the window.** Across this slider's
range it reads 88.6°, 88.8°, 68.6°, **24.6°**, 64.8°, 84.1°, 85.2°, 85.9° — one dip,
centred on τ = 1, where the overlap jumps to 0.44 before falling back to 0.05 by
τ = 1.5. Meanwhile σ₁/σ₂ stays between 1.1 and 2.9 throughout, so condition 1 never
recovers, and the angle is governed almost entirely by the overlap — that is, by
whether the fastest-growing structure happens to land on site 20 within this particular
window. **Near-orthogonality is the generic case here; agreement is the coincidence.**
Note that τ = 1 is precisely where Section 1's slider sits by default, which is why
this section has its own.

Both structures are also **localised**: 90 % of the variance of v₁ sits on
{_support(_v1_96)} of 40 sites, of ∂J/∂x₀ on {_support(_g96)}, and of the evolved u₁ on
{_support(_u1_96)}. Perturbations do not grow everywhere; they grow in particular
places, and those places move."""
            ),
            kind="danger",
        ),
        mo.md(
            """**Which one does an observing plan want?** Both, and the operational
            answer has changed over time. Adjoint *sensitivity* fields point at what a
            given forecast depends on and are cheap — one adjoint run per metric.
            Singular vectors point at where errors will grow regardless of metric and
            cost an iterative eigensolve. Field campaigns have used both to choose
            flight tracks *[citation needed]*, and ensemble-based sensitivity has since
            displaced much of the adjoint machinery because it needs no adjoint code at
            all — chapter 19's ensemble is already there."""
        ),
    ])
    return


# ===========================================================================
# 4. The norm
# ===========================================================================
@app.cell(hide_code=True)
def sec4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · "Fastest-growing" is not well defined until you choose a norm

    This looks like a technicality and is not. "Which perturbation grows most" presumes
    a way of measuring size, and singular vectors are the answer to

    $$\max_{v}\ \frac{\|\mathbf{M}v\|_E}{\|v\|_E},
      \qquad \|v\|_E^2 = v^{\top}\mathbf{E}v ,$$

    which depends on $\mathbf{E}$. Change the norm and both the amplification and the
    optimal direction change. There is no norm-free notion of the fastest-growing
    perturbation.

    That is not a mathematical curiosity. Operational singular vectors are computed in a
    **total-energy** norm, and the choice was argued over for years, because a norm that
    weights small scales heavily produces optimal perturbations concentrated at small
    scales — which then grow impressively and matter little for the forecast anyone
    cares about. Choosing $\mathbf{E}$ is choosing what "an important error" means.

    `chaoslib.adjoint.singular_vectors` takes a `weight` argument for this reason.
    Below, the same propagator under three different norms.
    """
    )
    return


@app.cell
def norm_figure(
    C_MEAN, C_PERT, C_TRUTH, adjoint, angle_between, base_state, finish_mpl,
    l63_propagator, mo, mpl_panels, np, tau_sl,
):
    _tau = float(tau_sl.value)
    _M = l63_propagator(base_state, _tau)

    _norms = (
        ("Euclidean", np.array([1.0, 1.0, 1.0]), C_TRUTH),
        ("z weighted ×25", np.array([1.0, 1.0, 25.0]), C_MEAN),
        ("x weighted ×25", np.array([25.0, 1.0, 1.0]), C_PERT),
    )
    _results = []
    for _name, _w, _col in _norms:
        _s, _v, _u = adjoint.singular_vectors(_M, 1, weight=_w)
        _dir = _v[:, 0] / np.linalg.norm(_v[:, 0])
        _results.append((_name, _w, _col, float(_s[0]), _dir))

    _euclid_dir = _results[0][4]

    _fig, _ax = mpl_panels(
        2,
        titles=("The optimal direction depends on the norm",
                "…and so does the amplification"),
        height=3.5,
    )
    _comp = np.arange(3)
    _width = 0.26
    for _i, (_name, _w, _col, _s1, _dir) in enumerate(_results):
        _ax[0].bar(_comp + (_i - 1) * _width, _dir, width=_width, color=_col,
                   alpha=0.85, label=_name)
    _ax[0].axhline(0.0, color="#8b8299", linewidth=0.9)
    _ax[0].set_xticks(_comp)
    _ax[0].set_xticklabels(["v₁ · x̂", "v₁ · ŷ", "v₁ · ẑ"])
    _ax[0].set_ylabel("component of the optimal direction")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _names = [r[0] for r in _results]
    _amps = [r[3] for r in _results]
    _ax[1].bar(_names, _amps, color=[r[2] for r in _results], alpha=0.85, width=0.55)
    _ax[1].set_ylabel("σ₁ in that norm")
    _ax[1].tick_params(axis="x", labelrotation=12)
    for _i, _v in enumerate(_amps):
        _ax[1].annotate(f"{_v:.2f}", (_i, _v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7.5)
    finish_mpl(_fig, suptitle=f"Same propagator, three norms, τ = {_tau:g} MTU")

    _rows = "\n".join(
        f"| {n} | {s:.3f} | ({', '.join(f'{c:+.3f}' for c in d)}) | "
        f"{angle_between(d, _euclid_dir):.1f}° |"
        for n, _w, _c, s, d in _results
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
| norm | σ₁ | optimal direction v₁ | angle to the Euclidean answer |
|---|---|---|---|
{_rows}

The amplifications are **not comparable across rows** — each is measured in its own
norm, so a larger number does not mean more growth in any absolute sense. What is
comparable is the *direction*, and it moves: weighting one component by 25 rotates the
optimal perturbation by tens of degrees.

So a statement like "the leading singular vector points here" is incomplete without the
norm, in the same way that "the error is 0.3" is incomplete without units. When a paper
reports singular vectors, the norm is part of the result.
"""
        ),
    ])
    return


# ===========================================================================
# 5. Convergence to the Lyapunov vector
# ===========================================================================
@app.cell(hide_code=True)
def sec5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · The long-window limit

    One loose end. If singular vectors are the finite-window answer and Lyapunov vectors
    the asymptotic one, the two should meet as $\tau \to \infty$ — and they do. The
    leading singular vector settles onto a fixed direction, which is (up to sign) the
    leading backward Lyapunov vector at that point.

    The figure measures the settling directly: the angle between the leading singular
    vector computed at $\tau$ and the one computed at the next-larger $\tau$.
    """
    )
    return


@app.cell
def convergence_figure(
    C_SAT, C_TRUTH, adjoint, angle_between, base_state, finish_mpl,
    l63_propagator, mo, mpl_panels, np,
):
    _taus = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    _dirs = []
    for _t in _taus:
        _s, _v, _u = adjoint.singular_vectors(l63_propagator(base_state, float(_t)), 1)
        _dirs.append(_v[:, 0] / np.linalg.norm(_v[:, 0]))
    _angles = [angle_between(_dirs[i], _dirs[i + 1]) for i in range(len(_dirs) - 1)]

    _fig, _ax = mpl_panels(
        2,
        titles=("The leading singular vector settles",
                "Its components"),
        height=3.4,
    )
    _ax[0].semilogy(_taus[1:], np.maximum(_angles, 1e-3), marker="o", markersize=5,
                    color=C_TRUTH, linewidth=1.5)
    _ax[0].axhline(1.0, color=C_SAT, linewidth=1.2, linestyle="--")
    _ax[0].annotate("1°", (0.02, 1.0), xycoords=("axes fraction", "data"),
                    fontsize=7, color="#b91c1c", va="bottom")
    _ax[0].set_xlabel("optimisation window τ (MTU)")
    _ax[0].set_ylabel("angle to the next-larger τ (degrees)")

    _arr = np.array([_d * np.sign(_d[0]) for _d in _dirs])  # fix the sign for display
    for _i, _lab in enumerate(("x̂", "ŷ", "ẑ")):
        _ax[1].plot(_taus, _arr[:, _i], marker="o", markersize=4, linewidth=1.4,
                    label=f"v₁ · {_lab}")
    _ax[1].set_xlabel("optimisation window τ (MTU)")
    _ax[1].set_ylabel("component (sign fixed for display)")
    _ax[1].legend(loc="center right", fontsize=7, framealpha=0.9)
    finish_mpl(_fig)

    _rows = "\n".join(
        f"| {t:g} → {tn:g} | {a:.2f}° |"
        for t, tn, a in zip(_taus[:-1], _taus[1:], _angles)
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
| τ interval | angle between successive leading SVs |
|---|---|
{_rows}

By τ ≈ 4 MTU the direction has stopped moving to within a fraction of a degree: the
singular vector has converged onto the Lyapunov direction at this point. Note that this
is convergence of the **direction** while the **amplification** is still far above
$e^{{\\lambda_1\\tau}}$ — the two converge at very different rates, and the one that
matters for ensemble design is the slower.

The non-monotonic step near τ = 1 is not noise. Section 1's third panel shows σ₁
dipping around the same window, and when the leading and second singular values come
close the leading direction is poorly determined and can swap. It is the same reason
condition 1 of Section 3 matters.
"""
        ),
    ])
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

    1. **Watch the ellipsoid collapse.** Take τ from 0.25 to 8 MTU in Section 1 and
       watch σ₁/σ₂. At long windows the propagator becomes effectively rank one — every
       perturbation ends up pointing the same way. What does that imply about the useful
       *number* of singular vectors in an ensemble, as a function of lead time?
    2. **Make sensitivity and growth disagree in Lorenz 63.** Section 3's last row
       constructs a metric orthogonal to $u_1$ and gets 90°. Is such a metric anything a
       forecaster would care about? (Its gradient magnitude is also tiny — check it.
       That is the point: the metrics we care about are the ones that see $u_1$.)
    3. **Choose a norm badly.** In Section 4, weight one component by 25 and read the
       optimal direction. Now argue which norm you would want if the forecast product
       were a warning threshold on $z$.
    4. **Predict the L96 angle — and fail.** Section 3 reports the angle between ∇J
       and $v_1$ in Lorenz 96. Before moving τ, predict how it changes as the window
       grows. Now step through the whole range. The angle is *not* monotonic and
       σ₁/σ₂ barely moves; what actually controls it is the overlap
       $|\partial J/\partial x_\tau \cdot u_1|$, which depends on where the
       fastest-growing structure happens to land. What does that unpredictability imply
       about designing an observing network from singular vectors alone?

    ## What you should have seen

    The leading singular vector achieves an amplification $\sigma_1$ that exceeds
    $e^{\lambda_1\tau}$ by a factor of 1.6–2.6 at every window tested, because
    $\mathbf{M}$ is non-normal — and that gap, not the asymptotic rate, is what an
    ensemble built for a five-day forecast has to span.

    "Fastest-growing" is meaningless without a norm; changing the norm rotates the
    answer by tens of degrees.

    And sensitivity is not growth. In Lorenz 63, with one strongly dominant singular
    value, the gradient of any natural metric lies within a few degrees of $v_1$ — which
    makes the two look interchangeable. In Lorenz 96 they are nearly **orthogonal**. The
    distinction is real; it is just invisible in a three-variable model, which is a
    useful reminder about what low-order models can and cannot show you.

    ## Further reading

    - Buizza, R. and Palmer, T. N. (1995). The singular-vector structure of the
      atmospheric global circulation. *Journal of the Atmospheric Sciences*, **52**,
      1434–1456.
    - Palmer, T. N., Gelaro, R., Barkmeijer, J. and Buizza, R. (1998). Singular vectors,
      metrics, and adaptive observations. *Journal of the Atmospheric Sciences*, **55**,
      633–653 — the norm question, argued properly.
    - Errico, R. M. (1997). What is an adjoint model? *BAMS*, **78**, 2577–2591.
    - Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*,
      §6.4.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
      ch. 5 *[citation needed: pages]*.

    ---

    *Systems: Lorenz (1963), σ = 10, ρ = 28, β = 8/3; Lorenz (1996), N = 40, F = 8.*
    *Propagators: RK4 tangent stepped through the same stages as the nonlinear map,
    dt = 0.005 (L63) and 0.01 (L96).*
    *Amplification sweep precomputed by `scripts/generate_ch16_data.py`.*
    *λ₁ = 0.9056 MTU⁻¹ from chapter 7, pinned in chaoslib's tests.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
