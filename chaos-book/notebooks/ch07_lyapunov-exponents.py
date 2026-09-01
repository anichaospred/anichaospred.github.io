# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 7 -- Lyapunov exponents and doubling times.

The full spectrum by the Benettin algorithm, the exact trace identity as a live
check, global versus finite-time exponents, and where the measurement stops
being trustworthy.

Part III of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
Figures are static matplotlib, matching chapter 6.

To edit:   marimo edit notebooks/ch07_lyapunov-exponents.py
To export: make nb-one NB=ch07_lyapunov-exponents
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 7: Lyapunov Exponents")


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
    import matplotlib.pyplot as plt

    from chaoslib import integrate, lyapunov, plotting, systems

    SIGMA0, RHO0, BETA0 = 10.0, 28.0, 8.0 / 3.0
    L63_TRACE = -(SIGMA0 + 1.0 + BETA0)  # exact divergence of the Lorenz 63 flow

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
        BETA0,
        C_CONTEXT,
        C_FIXED,
        C_MEAN,
        C_PERT,
        C_SAT,
        C_SPREAD,
        C_START,
        C_TRUTH,
        L63_TRACE,
        RHO0,
        SIGMA0,
        finish_mpl,
        integrate,
        lyapunov,
        mo,
        mpl_grid,
        mpl_panels,
        np,
        plotting,
        plt,
        systems,
    )


# ---------------------------------------------------------------------------
# Precomputed lambda_1(rho), embedded rather than computed live
# ---------------------------------------------------------------------------
@app.cell
def rho_sweep_data(np):
    # Each rho needs its own Benettin run; a usable sweep costs about 80 seconds
    # in Pyodide, well past this chapter's budget -- and it is the same curve for
    # every reader, so there is nothing to gain from recomputing it. Generated
    # once at higher accuracy than a live run could afford by
    # scripts/generate_rho_sweep.py (Benettin, dt = 0.01, T = 200, transient = 30).
    # The chapter computes the *full spectrum* live at the reader's chosen rho and
    # marks that point on this curve.
    RHO_GRID = (
        0.5, 1.35357, 2.20714, 3.06071, 3.91429, 4.76786, 5.62143, 6.475,
        7.32857, 8.18214, 9.03571, 9.88929, 10.7429, 11.5964, 12.45, 13.3036,
        14.1571, 15.0107, 15.8643, 16.7179, 17.5714, 18.425, 19.2786, 20.1321,
        20.9857, 21.8393, 22.6929, 23.5464, 24.4, 25.2536, 26.1071, 26.9607,
        27.8143, 28.6679, 29.5214, 30.375, 31.2286, 32.0821, 32.9357, 33.7893,
        34.6429, 35.4964, 36.35, 37.2036, 38.0571, 38.9107, 39.7643, 40.6179,
        41.4714, 42.325, 43.1786, 44.0321, 44.8857, 45.7393, 46.5929, 47.4464,
        48.3, 49.1536, 50.0071, 50.8607, 51.7143, 52.5679, 53.4214, 54.275,
        55.1286, 55.9821, 56.8357, 57.6893, 58.5429, 59.3964, 60.25, 61.1036,
        61.9571, 62.8107, 63.6643, 64.5179, 65.3714, 66.225, 67.0786, 67.9321,
        68.7857, 69.6393, 70.4929, 71.3464, 72.2, 73.0536, 73.9071, 74.7607,
        75.6143, 76.4679, 77.3214, 78.175, 79.0286, 79.8821, 80.7357, 81.5893,
        82.4429, 83.2964, 84.15, 85.0036, 85.8571, 86.7107, 87.5643, 88.4179,
        89.2714, 90.125, 90.9786, 91.8321, 92.6857, 93.5393, 94.3929, 95.2464,
        96.1, 96.9536, 97.8071, 98.6607, 99.5143, 100.368, 101.221, 102.075,
        102.929, 103.782, 104.636, 105.489, 106.343, 107.196, 108.05, 108.904,
        109.757, 110.611, 111.464, 112.318, 113.171, 114.025, 114.879, 115.732,
        116.586, 117.439, 118.293, 119.146, 120,
    )
    LAMBDA1_GRID = (
        -0.4882, -1.2982, -1.1971, -1.1079, -1.0289, -0.9531, -0.8855, -0.8228,
        -0.7632, -0.7078, -0.6570, -0.6069, -0.5591, -0.5148, -0.4719, -0.4314,
        -0.3913, -0.3545, -0.3174, -0.2831, -0.2484, -0.2157, -0.1849, -0.1537,
        -0.1087, -0.0888, -0.0468, 0.7472, 0.8252, 0.8521, 0.8545, 0.8902,
        0.9125, 0.9157, 0.9370, 0.9647, 0.9855, 0.9727, 1.0182, 1.0369,
        1.0341, 1.0739, 1.0621, 1.1114, 1.1040, 1.1425, 1.1426, 1.1593,
        1.1442, 1.1556, 1.1873, 1.2059, 1.2220, 1.2389, 1.2585, 1.2829,
        1.2606, 1.2231, 1.3361, 1.2754, 1.2874, 1.3117, 1.2670, 1.3512,
        1.3164, 1.3258, 1.3840, 1.3342, 1.3463, 1.4628, 1.3906, 1.3783,
        1.4520, 1.4359, 1.4191, 1.3837, 1.4900, 1.4748, 1.4602, 1.3759,
        1.3933, 1.2703, 1.4530, 1.2273, 1.4920, 1.4889, 1.5249, 1.4546,
        1.4542, 1.5148, 1.5365, 1.5867, 1.5296, 1.5641, 1.5952, 1.5220,
        1.5953, 1.4274, 1.5974, 1.5629, 1.5025, 1.5201, 1.6324, 1.3117,
        1.3957, 1.4536, 1.6542, 0.4062, -0.0072, 1.6833, 1.7430, 1.6363,
        1.7036, 1.6088, 1.5472, 1.3818, 0.0817, -0.0058, 1.3720, 1.7008,
        1.5247, 1.5389, 1.5899, 1.5355, 1.7236, 1.6399, 1.5407, 1.5942,
        1.4982, 0.0094, 1.6962, 1.7248, 1.4765, 1.5083, 1.6246, 1.5457,
        1.5060, 1.7292, 1.6566, 1.7253, 1.6267,
    )

    rho_grid = np.asarray(RHO_GRID)
    lambda1_grid = np.asarray(LAMBDA1_GRID)

    # Below rho ~ 24 a finite-T estimate cannot be trusted: Lorenz 63 has a
    # chaotic saddle there, so a trajectory wanders chaotically for a long time
    # and then settles onto a fixed point. Section 5 measures this directly. The
    # curve is drawn but shaded in that band.
    RHO_UNRELIABLE = 24.0
    return RHO_UNRELIABLE, lambda1_grid, rho_grid


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 7 · Lyapunov Exponents and Doubling Times

    **Part III — Quantifying chaos and predictability.**

    **The forecasting question.** Chapter 6 showed two nearly identical Lorenz
    trajectories pulling apart, and fitted a growth rate to the separation. Run that
    experiment again from a different point on the attractor and you get a different
    number — 0.6 here, 1.1 there. So which is *the* rate? Is there one at all? And if
    the answer depends on where you start, what does a forecast centre mean when it
    says the atmosphere has a two-week limit?

    This chapter turns "chaotic" into a number, and is careful about which number.
    There are three, they are routinely confused, and they answer different questions:

    | | What it describes | Depends on |
    |---|---|---|
    | The **spectrum** $\lambda_1 \ge \lambda_2 \ge \lambda_3$ | the attractor | nothing — it is a property of the system |
    | The **finite-time exponent** $\lambda(x,\tau)$ | one state, one lead time | *where you are today* |
    | A **twin-trajectory fit** | one trajectory, one realisation | everything, including luck |

    Chapter 6 measured the third. This chapter computes the first, shows how widely
    the second varies, and ends with the case where all of them mislead.
    """
    )
    return


# ===========================================================================
# 1. Why one twin pair is not an answer
# ===========================================================================
@app.cell(hide_code=True)
def sec1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · Why one twin pair is not an answer

    Chapter 6's experiment: perturb a state by $\delta_0$, integrate both copies, fit a
    straight line to $\ln\|\delta(t)\|$. It is the right idea and it is the honest way
    to *see* exponential growth. As a measurement it is noisy, and the figure below
    shows how noisy.

    Each bar is one twin experiment launched from a different point on the same
    attractor, with the same $\delta_0$ and the same fitting window. If a single
    experiment measured a property of the system, they would all agree.
    """
    )
    return


@app.cell
def twin_scatter(C_PERT, C_TRUTH, finish_mpl, integrate, lyapunov, mo, mpl_panels, np, systems):
    _grid = integrate.trajectory_grid(t_final=200.0, dt=0.01)
    _long = integrate.rk4(systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid)
    _starts = _long[3000::1200]  # decorrelated points on the attractor

    _window = integrate.trajectory_grid(t_final=15.0, dt=0.005)
    _rates = []
    for _i, _start in enumerate(_starts):
        _sep, _ = lyapunov.twin_trajectory_growth(
            systems.lorenz63, _start, 1e-7, _window, seed=200 + _i
        )
        try:
            _rate, _ = lyapunov.fit_growth_rate(_window, _sep)
        except ValueError:
            continue
        _rates.append(_rate)
    twin_rates = np.asarray(_rates)

    _fig, _ax = mpl_panels(
        2,
        titles=(
            "One twin experiment each, same system",
            "Their distribution",
        ),
        height=3.4,
    )
    _idx = np.arange(twin_rates.size)
    _ax[0].bar(_idx, twin_rates, color=C_PERT, alpha=0.75, width=0.72)
    _ax[0].axhline(0.9056, color=C_TRUTH, linewidth=1.6,
                   label="λ₁ = 0.9056 (this chapter's answer)")
    _ax[0].set_xlabel("experiment")
    _ax[0].set_ylabel("fitted growth rate (MTU⁻¹)")
    _ax[0].legend(loc="lower right", fontsize=7, framealpha=0.9)

    _ax[1].hist(twin_rates, bins=8, color=C_PERT, alpha=0.75,
                orientation="horizontal")
    _ax[1].axhline(0.9056, color=C_TRUTH, linewidth=1.6)
    _ax[1].axhline(twin_rates.mean(), color=C_TRUTH, linewidth=1.3,
                   linestyle="--", label=f"mean = {twin_rates.mean():.2f}")
    _ax[1].set_xlabel("count")
    _ax[1].set_ylabel("fitted growth rate (MTU⁻¹)")
    _ax[1].legend(loc="lower right", fontsize=7, framealpha=0.9)
    finish_mpl(_fig)

    mo.vstack([
        _fig,
        mo.md(
            f"""**{twin_rates.size} experiments, range {twin_rates.min():.2f} to
            {twin_rates.max():.2f} MTU⁻¹, mean {twin_rates.mean():.2f}, standard
            deviation {twin_rates.std():.2f}.**

            The spread is not sloppiness — it is the physics of the next section.
            Growth on this attractor is *bursty*: a trajectory crossing the gap
            between the lobes separates far faster than one circling inside a lobe.
            A single 15-MTU experiment samples whichever happened to occur.

            The mean is close to the right answer, which is the clue: the quantity
            that describes the *system* is an average, and to compute it properly you
            average along a trajectory rather than over a handful of experiments."""
        ),
    ])
    return (twin_rates,)


# ===========================================================================
# 2. The spectrum
# ===========================================================================
@app.cell(hide_code=True)
def sec2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · The spectrum, and how it is computed

    ### What Oseledets guarantees

    Linearise the flow about a trajectory. An infinitesimal perturbation $\delta x$
    obeys $\dot{\delta x} = \mathbf{J}(x(t))\,\delta x$, and its growth over a time
    $T$ is governed by the propagator $\mathbf{M}(x_0, T)$ built in chapter 15.
    Oseledets' multiplicative ergodic theorem says that for almost every starting
    point on the attractor the limits

    $$\lambda_i = \lim_{T\to\infty}\frac{1}{T}\ln \sigma_i\!\left(\mathbf{M}(x_0,T)\right)$$

    exist and **do not depend on $x_0$**. That independence is what makes them
    properties of the system rather than of an experiment, and it is exactly what the
    twin-experiment scatter above lacks. There are $n$ of them, conventionally ordered
    $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$.

    ### Why you cannot just integrate the propagator

    The obvious approach — build $\mathbf{M}$ for large $T$ and take its singular
    values — fails numerically. Every column of $\mathbf{M}$ grows like
    $e^{\lambda_1 T}$, so by $T = 20$ the matrix has lost all information about
    every direction except the leading one, and by $T = 400$ it has overflowed.

    The **Benettin algorithm** fixes this by re-orthonormalising as it goes. Carry an
    orthonormal frame along the trajectory; after each step, factor it as
    $\mathbf{Q}\mathbf{R}$, keep $\mathbf{Q}$ as the new frame, and accumulate the
    logarithms of the diagonal of $\mathbf{R}$:

    $$\lambda_i = \lim_{T\to\infty}\frac{1}{T}\sum_n \ln\left|R^{(n)}_{ii}\right|$$

    The $\mathbf{R}$ diagonal records how much each direction stretched *before* the
    frame was renormalised, so nothing overflows and the subdominant directions stay
    resolved. `chaoslib.lyapunov.lyapunov_spectrum` does this with a QR factorisation
    at every step.

    ### The check that makes it trustworthy

    For Lorenz 63 the divergence of the flow is the same everywhere:

    $$\operatorname{tr}\mathbf{J} = -\sigma - 1 - \beta \quad\text{for every state,}$$

    and the sum of the Lyapunov exponents equals the time-averaged divergence.
    So

    $$\sum_i \lambda_i = -(\sigma + 1 + \beta) = -13.6\overline{6}$$

    **exactly**, at any integration length, independent of the trajectory. That makes
    it a genuine test of the implementation rather than of its convergence: a
    Benettin loop with a sign error or a mis-ordered QR fails it immediately, while
    the leading exponent alone would still look plausible. It is asserted in
    `chaoslib`'s test suite and displayed live below.
    """
    )
    return


@app.cell(hide_code=True)
def spectrum_controls(mo):
    rho_sl = mo.ui.slider(
        start=0.5, stop=120.0, step=0.5, value=28.0,
        label="ρ (Rayleigh number)", show_value=True,
    )
    t_final_sl = mo.ui.slider(
        start=50, stop=400, step=50, value=200,
        label="integration length T (MTU)", show_value=True,
    )
    return rho_sl, t_final_sl


@app.cell
def spectrum_figure(
    BETA0, C_FIXED, C_MEAN, C_PERT, C_SAT, C_SPREAD, C_TRUTH, L63_TRACE,
    RHO_UNRELIABLE, SIGMA0, finish_mpl, lambda1_grid, lyapunov, mo, mpl_panels,
    np, rho_grid, rho_sl, systems, t_final_sl,
):
    _rho = float(rho_sl.value)
    _T = float(t_final_sl.value)

    # ONE Benettin pass gives both the converged spectrum and the convergence
    # curve. Calling lyapunov_spectrum separately would double the cost and, being
    # a different chaotic trajectory, would not even agree with the curve's
    # endpoint to better than a few hundredths.
    _times, _running = lyapunov.lyapunov_convergence(
        systems.lorenz63,
        systems.lorenz63_jacobian,
        np.array([1.0, 1.0, 20.0]),
        dt=0.01,
        t_final=_T,
        t_transient=30.0,
        n_samples=180,
        sigma=SIGMA0,
        rho=_rho,
        beta=BETA0,
    )
    spectrum = _running[-1]
    _sum = float(spectrum.sum())
    _residual = abs(_sum - L63_TRACE)

    _lam1 = float(spectrum[0])
    _dky = lyapunov.kaplan_yorke_dimension(spectrum)
    _hks = lyapunov.ks_entropy(spectrum)
    _rho_h = systems.lorenz63_hopf_rho(SIGMA0, BETA0)

    if _lam1 > 0.05:
        _regime, _ck = "**chaotic** — a positive exponent, so nearby states separate exponentially", "danger"
    elif _lam1 > -0.02:
        _regime, _ck = "**marginal** — λ₁ ≈ 0, the signature of periodic (or quasi-periodic) motion", "warn"
    else:
        _regime, _ck = "**non-chaotic** — every direction contracts, so forecasts do not degrade", "success"

    _fig, _ax = mpl_panels(
        3,
        titles=(
            "Convergence of the running estimate",
            "The spectrum at this ρ",
            "λ₁ against ρ",
        ),
        height=3.7,
    )

    # ---- (a) convergence ----
    _cols = (C_TRUTH, C_MEAN, C_SPREAD)
    for _i in range(_running.shape[1]):
        _ax[0].plot(_times, _running[:, _i], color=_cols[_i], linewidth=1.3,
                    label=f"λ{_i + 1}")
    _ax[0].axhline(0.0, color="#c9c2de", linewidth=0.8)
    _ax[0].set_xlabel("integration length T (MTU)")
    _ax[0].set_ylabel("running estimate (MTU⁻¹)")
    _ax[0].legend(loc="center right", fontsize=7, framealpha=0.9)
    # Log x: the 1/T convergence is otherwise squeezed into the first tenth of the
    # panel. Symlog y: lambda_3 is about -14.6 while the interesting behaviour is
    # lambda_1 settling near +0.9, and on a linear axis lambda_3 flattens the other
    # two onto the zero line. Symlog keeps all three visible and expands the region
    # near zero, where lambda_2 lives.
    _ax[0].set_xscale("log")
    _ax[0].set_yscale("symlog", linthresh=0.3, linscale=0.6)
    _ax[0].set_yticks([-10.0, -1.0, 0.0, 1.0])
    _ax[0].set_yticklabels(["−10", "−1", "0", "1"])

    # ---- (b) the spectrum, and the identity ----
    _pos = np.arange(spectrum.size)
    _ax[1].bar(_pos, spectrum,
               color=[C_TRUTH if v > 0 else (C_MEAN if abs(v) < 0.05 else C_SPREAD)
                      for v in spectrum],
               alpha=0.85, width=0.6)
    _ax[1].axhline(0.0, color="#8b8299", linewidth=0.9)
    _ax[1].set_xticks(_pos)
    _ax[1].set_xticklabels([f"λ{i + 1}" for i in _pos])
    _ax[1].set_ylabel("exponent (MTU⁻¹)")
    for _i, _v in enumerate(spectrum):
        # A near-zero exponent draws no visible bar, so anchor its label to the axis
        # rather than to the (invisible) bar top, where it would float mid-panel.
        _anchor = _v if abs(_v) > 0.05 else 0.0
        _ax[1].annotate(f"{_v:+.3f}", (_i, _anchor), textcoords="offset points",
                        xytext=(0, 5 if _v >= 0 else -12), ha="center", fontsize=7,
                        color="#211d33")
    _ax[1].annotate(
        f"Σλᵢ = {_sum:+.4f}\nexact: {L63_TRACE:+.4f}\nresidual {_residual:.1e}",
        (0.5, 0.06), xycoords="axes fraction", ha="center", va="bottom",
        fontsize=7.5, color="#211d33",
        bbox=dict(facecolor="white", alpha=0.9, edgecolor="#e6e1f2",
                  boxstyle="round,pad=0.3"),
    )

    # ---- (c) the precomputed sweep, with this rho marked ----
    _ax[2].axhline(0.0, color="#8b8299", linewidth=0.9)
    _ax[2].axvspan(rho_grid[0], RHO_UNRELIABLE, color="#faf0f0", zorder=0)
    _ax[2].annotate("finite-T\nestimate\nunreliable\n(§5)", (RHO_UNRELIABLE * 0.5, 0.97),
                    xycoords=("data", "axes fraction"), ha="center", va="top",
                    fontsize=6.5, color="#9b6b6b")
    _ax[2].plot(rho_grid, lambda1_grid, color=C_TRUTH, linewidth=1.3,
                label="precomputed")
    _ax[2].axvline(_rho_h, color=C_FIXED, linewidth=1.1, linestyle=":")
    _ax[2].annotate(f"ρ_H = {_rho_h:.2f}", (_rho_h, 0.30),
                    xycoords=("data", "axes fraction"), rotation=90, fontsize=6.5,
                    color="#92600b", ha="right", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                              boxstyle="round,pad=0.15"))
    _ax[2].plot([_rho], [_lam1], marker="o", markersize=8, color=C_PERT,
                markeredgecolor="white", markeredgewidth=1.0, linestyle="none",
                zorder=6, label="computed live, this ρ")
    _ax[2].set_xlabel("ρ")
    _ax[2].set_ylabel("λ₁ (MTU⁻¹)")
    _ax[2].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    finish_mpl(_fig, suptitle=f"σ = {SIGMA0:g},  ρ = {_rho:g},  β = {BETA0:.3f},  T = {_T:g} MTU")

    _tau_lam = 1.0 / _lam1 if _lam1 > 0 else float("nan")
    _t_double = np.log(2.0) / _lam1 if _lam1 > 0 else float("nan")
    _times_str = (
        f"Lyapunov (e-folding) time 1/λ₁ = **{_tau_lam:.2f} MTU** &nbsp;·&nbsp; "
        f"doubling time ln2/λ₁ = **{_t_double:.2f} MTU**"
        if _lam1 > 0 else
        "No positive exponent, so neither an e-folding nor a doubling time is defined."
    )

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([rho_sl, t_final_sl], gap="3rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; ρ = {_rho:g}, T = {_T:g} MTU  \n"
                f"Spectrum: **{', '.join(f'{v:+.3f}' for v in spectrum)}** MTU⁻¹  \n"
                f"This trajectory is {_regime}.  \n"
                f"{_times_str}  \n"
                f"Σλᵢ = **{_sum:+.5f}** against the exact **{L63_TRACE:+.5f}** — "
                f"residual **{_residual:.1e}**  \n"
                f"Kaplan–Yorke dimension **{_dky:.3f}** &nbsp;·&nbsp; "
                f"KS entropy **{_hks:.3f}** nats MTU⁻¹"
            ),
            kind=_ck,
        ),
    ])
    return (spectrum,)


# ===========================================================================
# 3. Reading the spectrum
# ===========================================================================
@app.cell(hide_code=True)
def sec3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · What the three numbers tell you

    At the classical parameters the spectrum is approximately
    $(+0.906,\; 0,\; -14.57)$ MTU⁻¹, and each entry says something different.

    **$\lambda_1 > 0$ — the forecast problem.** Errors along the leading direction
    grow like $e^{\lambda_1 t}$. Two derived times follow, and they are *not* the
    same number:

    $$\text{Lyapunov (e-folding) time } \tau_\lambda = \frac{1}{\lambda_1} \approx 1.10\ \text{MTU},
      \qquad
      \text{doubling time } = \frac{\ln 2}{\lambda_1} \approx 0.77\ \text{MTU}.$$

    They differ by a factor of $\ln 2 \approx 0.69$, and conflating them is the single
    most common slip in this subject — chapter 6 had exactly that error in an earlier
    draft. Quote which one you mean.

    **$\lambda_2 \approx 0$ — the direction along the flow.** Displace a state
    *forward along its own trajectory* and it neither grows nor decays; it is the same
    orbit, reached slightly later. Any continuous-time system with a bounded attractor
    that is not a fixed point has such a neutral direction, so $\lambda_2 \approx 0$
    is a structural fact rather than a coincidence. It is also a useful diagnostic:
    if your computed $\lambda_2$ is not near zero, the calculation is wrong.

    **$\lambda_3 \ll 0$ — why the attractor is thin.** Contraction at
    $\approx -14.6$ MTU⁻¹ is roughly sixteen times faster than the growth. Volumes
    collapse onto the attractor almost immediately, which is why the object looks like
    a surface, and why the sum is so strongly negative.

    ### Two numbers built from the spectrum

    **Kaplan–Yorke dimension.** Add exponents until the running sum would go negative,
    then interpolate:

    $$D_{KY} = j + \frac{\sum_{i \le j}\lambda_i}{|\lambda_{j+1}|}
      = 2 + \frac{\lambda_1}{|\lambda_3|} \approx 2.06.$$

    A volume shrinks, an area grows slightly: the attractor is more than a surface and
    less than a solid. Chapter 8 measures the same dimension a completely different
    way — from a sampled trajectory, with no reference to the dynamics — and the two
    agree.

    **Kolmogorov–Sinai entropy.** By Pesin's identity, $h_{KS} = \sum_{\lambda_i>0}\lambda_i$,
    which for this system is just $\lambda_1$. Read it as an information rate: the
    system destroys about 0.9 nats — 1.3 bits — of information about its initial state
    per MTU. Whatever precision you start with is spent at that rate, and chapter 10
    develops that reading properly.

    ### The one number this chapter cannot give you

    None of this converts to days on its own. The exponent is $0.906$ **per MTU**, and
    turning that into a doubling time in days needs a convention relating MTU to
    atmospheric time — which, as chapter 6's comparison table sets out, does not
    reconcile with the atmosphere under any single choice. The *law* is what carries
    over: error grows exponentially at a rate set by the system, so predictability is
    bought logarithmically. Chapter 20 measures that consequence directly.
    """
    )
    return


# ===========================================================================
# 4. Finite-time exponents
# ===========================================================================
@app.cell(hide_code=True)
def sec4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · Today's forecast: finite-time exponents

    $\lambda_1$ describes the attractor. It does not describe *today*.

    Over a finite window $\tau$ the amplification of the fastest-growing perturbation
    at a particular state $x$ is the leading singular value of the propagator, and the
    corresponding rate

    $$\lambda(x,\tau) = \frac{1}{\tau}\ln\sigma_1\!\left(\mathbf{M}(x,\tau)\right)$$

    is the **finite-time (or local) Lyapunov exponent**. It is a property of a state
    and a lead time, it varies across the attractor, and it is what an operational
    forecaster actually cares about: not "the atmosphere has a two-week limit" but
    "*this* situation is unusually hard to predict".

    ### Two reasons the short-window numbers come out high

    At the default $\tau = 0.5$ MTU the mean local exponent is about 2.6 MTU⁻¹ —
    roughly **three times** the asymptotic $\lambda_1 = 0.91$. Every state appears to be
    growing faster than the attractor's own rate, which cannot be a statement about
    today's weather. Two distinct effects are at work, and they are worth separating:

    1. **Optimisation over direction (non-normality).** $\sigma_1(\mathbf{M})$ is the
       growth of the *fastest-growing* perturbation, not of a typical one, and in a
       non-normal system those differ sharply over short windows. Measured here: at
       $\tau = 0.5$ the optimal direction averages 2.64 MTU⁻¹ while a *random*
       direction averages 1.36 — so about half the elevation is this effect alone. It
       is the same phenomenon that makes singular vectors worth computing in chapter
       16, and averaging over more states does not reduce it.
    2. **Too short an average.** A 0.5-MTU window samples one stretch of trajectory.
       The asymptotic exponent is a time average over the whole attractor, and a short
       window has not done that averaging yet.

    Both effects vanish as $\tau$ grows, and the measured convergence is clean:

    | $\tau$ (MTU) | 0.5 | 1 | 2 | 4 | 8 |
    |---|---|---|---|---|---|
    | mean $\lambda(x,\tau)$ | 2.59 | 1.78 | 1.36 | 1.12 | 1.01 |
    | standard deviation | 2.14 | 1.01 | 0.56 | 0.32 | 0.16 |

    Mean and spread both fall towards $\lambda_1$ — Oseledets again. Move the slider and
    watch it happen. **What survives at every $\tau$ is the *ordering*:** the same
    regions of the attractor are consistently fast and the same ones consistently slow,
    and that is the flow-dependence a forecaster can act on.
    """
    )
    return


@app.cell(hide_code=True)
def ftle_controls(mo):
    tau_sl = mo.ui.slider(
        start=0.25, stop=4.0, step=0.25, value=0.5,
        label="window τ (MTU)", show_value=True,
    )
    return (tau_sl,)


@app.cell
def ftle_figure(
    BETA0, C_CONTEXT, C_SAT, C_TRUTH, RHO0, SIGMA0, finish_mpl, integrate,
    lyapunov, mo, mpl_panels, np, plt, spectrum, systems, tau_sl,
):
    _tau = float(tau_sl.value)
    _grid = integrate.trajectory_grid(t_final=150.0, dt=0.01)
    _traj = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid,
        sigma=SIGMA0, rho=RHO0, beta=BETA0,
    )
    _states = _traj[2000::130]  # decorrelated samples on the attractor
    _local = lyapunov.finite_time_exponents(
        systems.lorenz63, systems.lorenz63_jacobian, _states, tau=_tau, dt=0.01,
        sigma=SIGMA0, rho=RHO0, beta=BETA0,
    )
    _lam1 = float(spectrum[0])

    _fig, _ax = mpl_panels(
        2,
        titles=(
            f"Where growth is fast (τ = {_tau:g} MTU)",
            "Distribution of local exponents",
        ),
        height=3.7,
    )

    # ---- (a) the attractor, coloured by the local exponent ----
    _ax[0].plot(_traj[2000:, 0], _traj[2000:, 2], color=C_CONTEXT,
                linewidth=0.35, zorder=1)
    _sc = _ax[0].scatter(
        _states[:, 0], _states[:, 2], c=_local, cmap="plasma", s=26,
        edgecolors="white", linewidths=0.4, zorder=3,
    )
    _cb = _fig.colorbar(_sc, ax=_ax[0], fraction=0.046, pad=0.03)
    _cb.set_label("λ(x, τ)  (MTU⁻¹)", fontsize=7.5)
    _cb.ax.tick_params(labelsize=7)
    _ax[0].set_xlabel("x")
    _ax[0].set_ylabel("z")

    # ---- (b) the distribution ----
    _ax[1].hist(_local, bins=18, color=C_TRUTH, alpha=0.75)
    _ax[1].axvline(_lam1, color=C_SAT, linewidth=1.6,
                   label=f"λ₁ = {_lam1:.3f} (asymptotic)")
    _ax[1].axvline(float(_local.mean()), color=C_SAT, linewidth=1.3,
                   linestyle="--", label=f"mean = {_local.mean():.3f}")
    _ax[1].set_xlabel("λ(x, τ)  (MTU⁻¹)")
    _ax[1].set_ylabel("count")
    _ax[1].legend(loc="upper left", fontsize=7, framealpha=0.9)
    finish_mpl(_fig)

    _ratio = float(_local.max() / max(_local.min(), 1e-9))
    _elevation = float(_local.mean() / _lam1) if _lam1 > 0 else float("nan")
    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([tau_sl], justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; τ = {_tau:g} MTU, "
                f"{_local.size} states sampled  \n"
                f"Local exponents run from **{_local.min():.3f}** to "
                f"**{_local.max():.3f}** MTU⁻¹ — a factor of **{_ratio:.1f}** across "
                f"one attractor  \n"
                f"Mean **{_local.mean():.3f}**, standard deviation "
                f"**{_local.std():.3f}**; asymptotic λ₁ = **{_lam1:.3f}**  \n"
                f"The mean sits **{_elevation:.1f}×** above λ₁ — a window effect "
                f"rather than a statement about the weather; raise τ and it falls "
                f"towards 1×"
            ),
            kind="warn",
        ),
        mo.md(
            """Look at the left panel. The fast points cluster where the trajectory
            crosses between lobes and near the origin — the saddle region — and the
            slow ones sit deep inside a lobe. That is the mechanism behind chapter 6's
            twin-experiment scatter and behind the singular vectors of chapter 16:
            predictability is a property of the situation, and the situations differ.

            Now raise τ. The histogram both narrows *and* slides left towards λ₁, for
            the two reasons set out above. At τ = 8 MTU the mean is within about 10 % of
            the asymptotic value and the spread is a twelfth of what it was at 0.5 —
            the asymptotic exponent is what you get once you stop asking about today.

            What does *not* wash out is which regions are fast. Compare the left panel
            at τ = 0.5 and at τ = 2: the colours dim, but the bright points stay in the
            same places."""
        ),
    ])
    return


# ===========================================================================
# 5. Where the measurement fails
# ===========================================================================
@app.cell(hide_code=True)
def sec5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · Where this measurement fails

    Everything so far has treated $\lambda_1$ as something you compute and then trust.
    Here is the case where that goes wrong, and it is not exotic — it sits just below
    the parameter value this chapter has been using.

    Set the ρ slider in Section 2 to **22.7** and read off $\lambda_1$. Then move the
    integration length T. The answer changes sign.

    That is not numerical noise. Lorenz 63 has a **chaotic saddle** below the
    transition: an invariant set that is genuinely chaotic but is *not* an attractor.
    A trajectory launched nearby wanders around it — separating exponentially, looking
    chaotic in every diagnostic — and then, after a time that can be enormous, falls
    off it and settles onto one of the stable fixed points $C^\pm$. This is
    **transient chaos**, and while it lasts, a finite-$T$ Benettin estimate faithfully
    reports the saddle's positive exponent for a system whose actual long-term
    behaviour is a fixed point.

    The figure below measures it directly. It computes $\lambda_1$ at three values of
    ρ for a range of integration lengths, and tracks whether the trajectory has
    actually stopped moving.
    """
    )
    return


@app.cell
def transient_data():
    # Section 5, precomputed: lambda_1 against integration length at three
    # rho values, and whether a 900-MTU run has stopped moving. Costs ~46 s
    # natively (>4 min in Pyodide) and has no knob, so it is computed once by
    # scripts/generate_rho_sweep.py --transient.
    TRANSIENT_LENGTHS = (50.0, 100.0, 200.0, 400.0, 800.0)
    TRANSIENT_LAMBDA1 = {
        22.7: (+0.7343, +0.5895, +0.2662, +0.1010, +0.0188),
        23.6: (+0.7405, +0.8076, +0.7960, +0.7916, +0.7623),
        28.0: (+0.8249, +0.8777, +0.8964, +0.8948, +0.8987),
    }
    TRANSIENT_SETTLED_RANGE = {
        22.7: 2.41585e-13,
        23.6: 42.1747,
        28.0: 50.2886,
    }

    return TRANSIENT_LAMBDA1, TRANSIENT_LENGTHS, TRANSIENT_SETTLED_RANGE


@app.cell
def transient_chaos(
    BETA0, C_FIXED, C_PERT, C_TRUTH, SIGMA0, TRANSIENT_LAMBDA1, TRANSIENT_LENGTHS,
    TRANSIENT_SETTLED_RANGE, finish_mpl, mo, mpl_panels, systems,
):
    _rhos = tuple(TRANSIENT_LAMBDA1)
    _cols = (C_PERT, C_FIXED, C_TRUTH)

    _fig, _ax = mpl_panels(
        2,
        titles=("λ₁ against integration length", "Is the trajectory still moving?"),
        height=3.5,
    )
    for _i, _rho in enumerate(_rhos):
        _ax[0].plot(TRANSIENT_LENGTHS, TRANSIENT_LAMBDA1[_rho], marker="o",
                    markersize=5, color=_cols[_i], linewidth=1.4,
                    label=f"ρ = {_rho}")
    _ax[0].axhline(0.0, color="#8b8299", linewidth=0.9)
    _ax[0].set_xscale("log")
    # Explicit ticks at the five lengths actually computed: log-scale minor ticks
    # collide into an unreadable smear at this panel width.
    _ax[0].set_xticks(list(TRANSIENT_LENGTHS))
    _ax[0].set_xticklabels([f"{int(v)}" for v in TRANSIENT_LENGTHS])
    _ax[0].minorticks_off()
    _ax[0].set_xlabel("integration length T (MTU)")
    _ax[0].set_ylabel("λ₁ (MTU⁻¹)")
    _ax[0].legend(loc="center left", fontsize=7, framealpha=0.9)

    # Log scale: the ranges span thirteen orders of magnitude, and the whole point
    # is that one of them is indistinguishable from zero.
    _ranges = [TRANSIENT_SETTLED_RANGE[_r] for _r in _rhos]
    _ax[1].bar([f"ρ = {_r}" for _r in _rhos], _ranges,
               color=list(_cols), alpha=0.8, width=0.55, log=True)
    _ax[1].set_ylabel("state range over the last 90 MTU")
    for _i, _v in enumerate(_ranges):
        _ax[1].annotate(f"{_v:.2g}", (_i, _v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=7.5)
    finish_mpl(_fig, suptitle="Transient chaos below the transition")

    _rho_h = systems.lorenz63_hopf_rho(SIGMA0, BETA0)
    _l227 = TRANSIENT_LAMBDA1[22.7]
    mo.vstack([
        _fig,
        mo.callout(
            mo.md(
                f"""**At ρ = 22.7** the state range over the last 90 MTU of a 900-MTU
                run is **{TRANSIENT_SETTLED_RANGE[22.7]:.2g}** — the trajectory has
                stopped moving entirely and is sitting on a fixed point. Yet λ₁
                measured over T = 50, 100, 200 comes out at
                {', '.join(f'{v:+.2f}' for v in _l227[:3])}. Watch what happens as T
                grows: **{_l227[0]:+.2f} → {_l227[-1]:+.2f}**. The estimate decays
                towards zero because the chaotic stretch is a fixed length of time and
                a longer average dilutes it. Those early numbers are real measurements
                of a real chaotic saddle, and they say nothing about the system's
                long-term behaviour.

                **At ρ = 23.6** the range is **{TRANSIENT_SETTLED_RANGE[23.6]:.1f}**:
                still wandering after 900 MTU, and λ₁ holds near
                {TRANSIENT_LAMBDA1[23.6][-1]:+.2f} at every length. The transient here
                outlives any integration this chapter can afford — which is not the
                same as there being no transient.

                **At ρ = 28** the range is **{TRANSIENT_SETTLED_RANGE[28.0]:.1f}** and
                λ₁ settles to {TRANSIENT_LAMBDA1[28.0][-1]:+.2f}: a genuine attractor.

                The Hopf threshold ρ_H = {_rho_h:.2f} marks where $C^\\pm$ lose
                stability, so above it there is nowhere for a trajectory to settle.
                Below it the fixed points are stable and the chaotic set is a saddle.
                The literature places the birth of the strange attractor near
                ρ ≈ 24.06, with a chaotic saddle existing from ρ ≈ 13.93
                *[citation needed: Sparrow (1982)]*."""
            ),
            kind="danger",
        ),
        mo.md(
            r"""### What to take from this

            **A positive finite-$T$ exponent is not proof of a chaotic attractor.** It
            is evidence of exponential separation over the window you measured. Those
            are different claims, and the gap between them is exactly the gap between
            $\lim_{T\to\infty}$ in Oseledets' theorem and the $T$ you could afford.

            Three habits follow, and they are the practical content of this chapter:

            1. **Vary $T$ and check the answer is stable.** An estimate that drifts
               monotonically — like ρ = 22.7 above — has not converged, and the
               direction of the drift tells you what it is converging to.
            2. **Check an invariant that does not depend on convergence.** For this
               system, $\sum_i\lambda_i = -(\sigma+1+\beta)$ holds at any $T$, so it
               tests the implementation while telling you nothing about convergence.
               Both facts are useful, separately.
            3. **Look at the trajectory, not only the number.** "Has it stopped
               moving?" took one line and settled a question the exponent could not.
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

    1. **Find the transition.** Set T = 400 and walk ρ from 20 upward in steps of 0.5,
       watching λ₁ and the marker on the third panel. Where does it become reliably
       positive? Compare with ρ_H = 24.74, and then re-read Section 5 before trusting
       what you found.
    2. **Break the identity.** Set T = 50. The residual on Σλᵢ stays tiny even though
       the leading exponent is badly converged. Why does poor convergence not spoil
       that check? (Because the identity holds at every $T$; it constrains the *sum*,
       which the QR accumulation gets right term by term.)
    3. **Find the periodic windows.** Move ρ to about 100 and then 92.7. λ₁ drops back
       to nearly zero — the system has fallen into a stable periodic orbit inside the
       chaotic range. Confirm it with the λ₂ value and with the phase portrait in
       chapter 6.
    4. **Narrow the histogram.** In Section 4, take τ from 0.25 to 4 MTU and watch the
       standard deviation of the local exponents fall. Roughly how does it scale with
       τ? (For an average of weakly correlated contributions you would expect
       $\tau^{-1/2}$.)

    ## What you should have seen

    At ρ = 28 the spectrum is $(+0.91,\ 0,\ -14.6)$ MTU⁻¹: one growing direction, one
    neutral one along the flow, and violent contraction onto a thin attractor. Their
    sum reproduces $-(\sigma+1+\beta)$ to five decimals at any integration length,
    while $\lambda_1$ itself is still drifting in the second decimal at T = 400.

    The local exponents span a factor of several across a single attractor, so
    "the" predictability of this system is an average over situations that differ
    substantially — and below the transition, the exponent you measure can describe a
    chaotic set the system will eventually leave.

    ## Further reading

    - Benettin, G., Galgani, L., Giorgilli, A. and Strelcyn, J.-M. (1980). Lyapunov
      characteristic exponents for smooth dynamical systems. *Meccanica*, **15**,
      9–20 — the algorithm used here.
    - Oseledets, V. I. (1968). A multiplicative ergodic theorem.
      *Trudy Moskov. Mat. Obšč.*, **19**, 179–210 *[citation needed: pages]*.
    - Sparrow, C. (1982). *The Lorenz Equations: Bifurcations, Chaos, and Strange
      Attractors.* Springer — the bifurcation structure behind Section 5.
    - Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*,
      §6.1 — Lyapunov exponents in a forecasting context.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 2 *[citation needed: pages]*.

    ---

    *System: Lorenz (1963), σ = 10, β = 8/3, ρ as set above.*
    *Spectra: Benettin with QR re-orthonormalisation at every step, RK4 tangent
    propagation, transient of 30 MTU discarded.*
    *λ₁(ρ) curve precomputed by `scripts/generate_rho_sweep.py`.*
    *Time unit: 1 MTU read as ≈ 5 days — a loose convention, not a calibration; see
    chapter 6, Section 4.*
    """
    )
    return


if __name__ == "__main__":
    app.run()
