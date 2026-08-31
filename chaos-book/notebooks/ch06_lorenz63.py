# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 6 -- The Lorenz (1963) system: the butterfly.

The strange attractor, sensitive dependence on initial conditions, ensemble
forecasting, and the bridge from a three-variable model to the real atmosphere.

Part II of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.

Phase space is drawn as STATIC 2-D projections (x-z, x-y) rather than a rotatable
3-D scene. A 3-D view of the Lorenz attractor looks impressive and reads worse: the
reader must hunt for a viewpoint before they can see the two lobes, the projection
they land on is unrepeatable, and it costs far more in the browser. The x-z
projection is the image everyone recognises as the butterfly, and it is the same
every time.

To edit:   marimo edit notebooks/ch06_lorenz63.py
To export: make nb-one NB=ch06_lorenz63
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Lorenz 63: Chaos & Predictability")


# ---------------------------------------------------------------------------
# Imports + shared Lorenz helpers + a common visual system for every figure
# ---------------------------------------------------------------------------
@app.cell
async def imports():
    import marimo as mo

    import sys

    if sys.platform == "emscripten":
        # Browser (Pyodide/WASM): install the chaoslib wheel that `make notebooks`
        # ships in the export's shared public/ folder. Every chapter exports into
        # one directory, so this resolves to /nb/public/ and the wheel is stored
        # once for the whole book rather than once per chapter.
        import micropip

        await micropip.install(
            str(
                mo.notebook_location()
                / "public"
                / "chaoslib-0.1.0-py3-none-any.whl"
            )
        )
    else:
        # Local (marimo edit): import straight from the repo checkout.
        sys.path.insert(0, str(mo.notebook_dir().parent))

    import numpy as np
    import matplotlib.pyplot as plt

    from chaoslib import integrate, lyapunov, plotting, systems

    SIGMA0, RHO0, BETA0 = 10.0, 28.0, 8.0 / 3.0

    # ---- the book's semantic palette and figure styling, from chaoslib ----
    # Defined once in chaoslib.plotting so every chapter means the same thing by
    # the same colour; re-bound to short local names here for figure code.
    # C_CONTEXT is a CSS rgba string (Plotly's format); matplotlib needs a
    # float tuple. Convert once here rather than at every call site.
    C_CONTEXT = plotting.mpl_colour(plotting.C_CONTEXT)
    C_TRUTH = plotting.C_TRUTH
    C_PERT = plotting.C_PERT
    C_SPREAD = plotting.C_SPREAD
    C_MEAN = plotting.C_MEAN
    C_FIXED = plotting.C_FIXED
    C_SAT = plotting.C_SAT
    C_START = plotting.C_START
    # Static-figure helpers: a styled row of matplotlib panels, and the closing
    # tight_layout call. The semantic colours above serve both figure kinds.
    mpl_panels = plotting.mpl_panels
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    # ---- numerics: thin adapters over tested chaoslib primitives ----
    def integrate_l63(s0, t_end, n=6000, sigma=SIGMA0, rho=RHO0, beta=BETA0,
                      transient=0.0, rtol=1e-8, atol=1e-10):
        """Integrate Lorenz 63 and return (t, y) with y shaped (3, n).

        The transposed shape is kept for the figure code below, which indexes
        components as y[0], y[1], y[2]. chaoslib returns time-leading arrays.
        """
        t_eval = np.linspace(0.0, t_end, n)
        traj = integrate.solve(
            systems.lorenz63, np.asarray(s0, float), t_eval,
            rtol=rtol, atol=atol, sigma=sigma, rho=rho, beta=beta,
        )
        if transient > 0:
            keep = t_eval >= transient
            return t_eval[keep], traj[keep].T
        return t_eval, traj.T

    def fixed_points(rho, beta):
        """The labelled fixed points O, C+ and C-, for annotating figures."""
        origin, c_plus, c_minus = systems.lorenz63_fixed_points(rho, beta)
        pts = {"O": origin}
        if c_plus is not None:
            pts["C\u207a"] = c_plus
            pts["C\u207b"] = c_minus
        return pts

    def hopf_threshold(sigma, beta):
        """Critical rho, or None where the formula's denominator is non-positive."""
        if sigma - beta - 1.0 <= 0:
            return None
        return systems.lorenz63_hopf_rho(sigma, beta)

    def ftle_estimate(seed, sigma, rho, beta, window=10.0, d0=1e-8):
        """Leading finite-time Lyapunov exponent from a twin-trajectory fit.

        Fits only the genuinely exponential stretch of the separation curve --
        above 10x the initial perturbation, below a tenth of saturation. Fitting
        the whole record instead (the tempting shortcut) biases the rate low,
        because it averages in both the initial rotation and the nonlinear
        bend-over. Returns nan when the window is too thin to fit, so the caller
        can show "unavailable" rather than a fabricated number.
        """
        grid = np.linspace(0.0, window, 900)
        separation, _ = lyapunov.twin_trajectory_growth(
            systems.lorenz63, np.asarray(seed, float), d0, grid,
            seed=42, sigma=sigma, rho=rho, beta=beta,
        )
        try:
            rate, _ = lyapunov.fit_growth_rate(grid, separation)
        except ValueError:
            return float("nan")
        return rate

    return (
        BETA0, C_CONTEXT, C_FIXED, C_MEAN, C_PERT, C_SAT, C_SPREAD, C_START,
        C_TRUTH, RHO0, SIGMA0, finish_mpl, ftle_estimate, fixed_points,
        hopf_threshold, integrate_l63, lyapunov, mo, mpl_grid, mpl_panels,
        np, plotting, plt, systems,
    )


# ---------------------------------------------------------------------------
# Attractor reference trajectory (classic params) — reused in Sections 2 & 3
# ---------------------------------------------------------------------------
@app.cell
def compute_attractor(integrate_l63, np):
    _t, attractor_ref = integrate_l63(
        [1.0, 1.0, 1.0], 80.0, n=8000, transient=10.0, rtol=1e-9, atol=1e-12,
    )
    attractor_size = float(np.mean(np.std(attractor_ref, axis=1)))
    return attractor_ref, attractor_size


# ---------------------------------------------------------------------------
# UI controls — returned so downstream cells can read .value reactively
# ---------------------------------------------------------------------------
@app.cell
def attractor_controls(mo):
    sigma_sl = mo.ui.slider(
        start=1.0, stop=20.0, step=0.5, value=10.0,
        label="σ  (Prandtl number)", show_value=True,
    )
    rho_sl = mo.ui.slider(
        start=0.5, stop=220.0, step=0.5, value=28.0,
        label="ρ  (Rayleigh number)", show_value=True,
    )
    beta_sl = mo.ui.slider(
        start=1.0, stop=4.0, step=1.0 / 3.0, value=8.0 / 3.0,
        label="β  (geometry factor)", show_value=True,
    )
    x0_in = mo.ui.number(start=-25.0, stop=25.0, step=0.1, value=1.0, label="x₀")
    y0_in = mo.ui.number(start=-30.0, stop=30.0, step=0.1, value=1.0, label="y₀")
    z0_in = mo.ui.number(start=0.0, stop=60.0, step=0.1, value=1.0, label="z₀")
    show_transient = mo.ui.checkbox(value=True, label="show the initial transient")
    return beta_sl, rho_sl, show_transient, sigma_sl, x0_in, y0_in, z0_in


@app.cell
def sdic_controls(mo):
    sep_exp = mo.ui.slider(
        start=-8, stop=-1, step=0.5, value=-4,
        label="Log₁₀ initial separation  δ₀",
        show_value=True,
    )
    # Default 15 MTU: with delta0 = 1e-4 and lambda ~ 0.85, the separation needs
    # ln(8.5/1e-4)/0.85 ~ 13 MTU to saturate. At the previous default of 10 the
    # curve stopped before the interesting part and the readout always said
    # "forecasts still agree".
    sdic_lead = mo.ui.slider(
        start=1, stop=30, step=1, value=15,
        label="Lead time (MTU)",
        show_value=True,
    )
    return sep_exp, sdic_lead


@app.cell
def ens_controls(mo):
    ic_choice = mo.ui.dropdown(
        options={
            "Slow region — spread grows late": "predictable",
            "Typical point on the attractor": "typical",
            "Fast region — spread grows early": "chaotic",
        },
        value="Typical point on the attractor",
        label="Starting location on attractor",
    )
    perturb_exp = mo.ui.slider(
        start=-6, stop=-1, step=0.5, value=-4,
        label="Log₁₀ perturbation size  δ₀",
        show_value=True,
    )
    n_members = mo.ui.slider(
        start=5, stop=50, step=5, value=20,
        label="Ensemble size  N",
        show_value=True,
    )
    # Default 20 MTU: at the previous 10 the ensemble spread never crossed the 90%
    # threshold for ANY starting point, so all three predictability regimes
    # collapsed into one and the shaded bands conveyed nothing.
    lead_time = mo.ui.slider(
        start=1, stop=30, step=1, value=20,
        label="Lead time (MTU)",
        show_value=True,
    )
    return ic_choice, lead_time, n_members, perturb_exp


# ===========================================================================
# Title and overview
# ===========================================================================
@app.cell
def display_title(mo):
    mo.md(r"""
# Chapter 6 · The Lorenz (1963) System: the Butterfly

**Part II — From regular motion to chaos.**

Scroll from top to bottom and interact with each panel before
reading the explanation beneath it — building intuition by *doing*
is more effective than reading first.

---

### Learning objectives

By the end of this notebook you will be able to:

1. **Describe** the Lorenz (1963) system and explain what each variable represents physically
2. **Demonstrate** sensitive dependence on initial conditions (SDIC) by running your own experiments
3. **Measure** the leading Lyapunov exponent from the slope of the error-growth curve
4. **Explain** why ensemble forecasting is the correct operational response to SDIC
5. **Calculate** the gain in predictable time from a given improvement in observational accuracy
6. **Distinguish** predictability of the first kind (initial-value) from the second kind (forced response)

---

### How to read this notebook

| Symbol | Meaning |
|--------|---------|
| ⚙️ **Controls** | Sliders and dropdowns you manipulate |
| 📐 **Theory** | Background equations and concepts |
| 🔬 **Experiment** | Step-by-step activity |
| 💡 **Observation** | Live readout that updates as you explore |

---

### Tutorial structure

| Section | Topic | Key concept |
|---------|-------|-------------|
| **1** | The Lorenz (1963) system | Strange attractor, deterministic chaos |
| **2** | Sensitive dependence on initial conditions | Lyapunov exponent, butterfly effect |
| **3** | Ensemble forecasting | Predictability horizon, ensemble spread |
| **4** | Connection to the real atmosphere | Error doubling time, 2nd-kind predictability |
| **📝** | Guided questions | Synthesis and quantitative reasoning |

> **Unit convention.** One *model time unit* (MTU) is read as ≈ **5 days** of
> atmospheric time throughout this chapter. Treat that as a loose convention rather
> than a calibration: it is *not* derived by matching Lorenz 63 to the atmosphere, and
> Section 4 shows exactly where the two part company.
""")
    return


# ===========================================================================
# Historical context: Lorenz's accidental discovery
# ===========================================================================
@app.cell
def cell_lorenz_story(mo):
    mo.md(r"""
---
### 🕰️ Historical context: how chaos was discovered by accident

In the winter of 1961, Edward Lorenz was running a primitive numerical weather model on the
Royal McBee LGP-30 computer at MIT.  He wanted to re-examine a particular simulation from the
middle, so instead of restarting from the beginning he typed in the intermediate values
from a printout — but he entered them rounded to three decimal places (0.506) instead of
the full six-digit precision (0.506127) stored in the computer's memory.

He expected the two runs to agree.  Instead, after a few simulated weeks, the two
solutions had **diverged completely** — not just a small discrepancy, but an entirely
different weather pattern.  At first he suspected a hardware fault.  Then the insight
hit: the tiny rounding error — about one part in a thousand — had grown exponentially
until it erased all predictive information.

> *"At this point I became rather excited.  It no longer seemed that predicting
> the weather for two weeks or a month would merely be a question of developing
> better equations."*
> — Edward Lorenz, *The Essence of Chaos* (1993)

This accidental discovery led directly to the 1963 paper that founded the mathematical
study of chaos.  The three-equation model in that paper was a deliberate simplification
of the full convection equations — a toy designed to be analytically tractable — but
it captured the essential unpredictability of the real atmosphere.

**The "butterfly effect" name** came a decade later.  At the December 1972 meeting of
the American Meteorological Society, Lorenz delivered a talk titled:

> *"Does the flap of a butterfly's wings in Brazil set off a tornado in Texas?"*

The title was chosen half-jokingly by the session organiser Philip Merilees, not by
Lorenz himself.  Lorenz used it to sharpen a point he had been making since 1963:
in a chaotic system, an *arbitrarily small* perturbation anywhere — even one too small
to measure — can, in principle, alter the large-scale evolution weeks later.
This is **not** a statement that butterflies literally cause tornadoes.
It is a statement about the *fundamental structure of deterministic predictability limits*.
""")
    return


# ===========================================================================
# Section 1 — The Lorenz (1963) system: theory
# ===========================================================================
@app.cell
def display_section1_text(mo):
    mo.md(r"""
---
## 1 · The Lorenz (1963) System

In 1963 Edward Lorenz published a three-variable model of **Rayleigh–Bénard convection** —
the buoyancy-driven overturning of a fluid heated from below.
Despite its simplicity, it became the founding example of *deterministic chaos*.

### The governing equations

$$\frac{dX}{dt} = \sigma\,(Y - X)$$
$$\frac{dY}{dt} = X\,(\rho - Z) - Y$$
$$\frac{dZ}{dt} = X\,Y - \beta\,Z$$

**What the variables represent:**

| Variable | Physical meaning |
|----------|-----------------|
| $X$ | Intensity of the convective overturning circulation |
| $Y$ | Temperature difference between ascending and descending fluid |
| $Z$ | Deviation of the vertical temperature profile from linearity |

**Classic parameter values** that produce chaotic behaviour:

| Parameter | Value | Physical role |
|-----------|-------|--------------|
| $\sigma = 10$ | Prandtl number — ratio of momentum to thermal diffusivity |
| $\rho = 28$ | Normalised Rayleigh number — strength of thermal forcing relative to dissipation |
| $\beta = 8/3$ | Geometric factor — aspect ratio of convection cells |

### Fixed points and the route to chaos

The system has **three fixed points** (where all derivatives are zero):

- **Origin** $(0, 0, 0)$: the purely conductive state (no convection).
  Always exists; unstable for $\rho > 1$.
- **Two symmetric points** $C^\pm = (\pm\sqrt{\beta(\rho-1)},\; \pm\sqrt{\beta(\rho-1)},\; \rho-1)$:
  steady convective rolls.  These become **unstable** (Hopf bifurcation) when $\rho > \rho_H \approx 24.74$.

At $\rho = 28 > 24.74$, *none* of the three fixed points is stable.
Trajectories cannot settle anywhere — they must wander forever, tracing out the **strange attractor**.

### The strange attractor

For these parameters the system is *chaotic*: trajectories are **bounded** (they
stay on the attractor forever) but **never periodic** (they never exactly repeat).
The geometric object they trace out is called a **strange attractor** — a fractal set
with non-integer dimension (~2.06).

**Explore it yourself.**  The panel below integrates the equations live from
initial condition $(x_0, y_0, z_0)$ for whatever $\sigma, \rho, \beta$ you dial in.
The 3-D view (colour = time) is paired with the $x(t)$ time series, and the three
fixed points are marked ($\text{O}$ at the origin, $C^\pm$ on the lobes).
A live readout classifies the long-term behaviour and estimates the leading
finite-time Lyapunov exponent $\lambda_1$ from a pair of nearby trajectories.

> 💡 **Things to try:** &nbsp; $\rho = 0.8$ (conduction — everything decays to O) &nbsp;·&nbsp;
> $\rho = 15$ (steady rolls — spirals onto $C^\pm$) &nbsp;·&nbsp;
> $\rho = 28$ (the classic strange attractor) &nbsp;·&nbsp;
> $\rho = 100$ (a periodic window — a single closed loop). &nbsp;
> Click and drag to rotate the 3-D plot.
""")
    return


# ===========================================================================
# Section 1 — interactive attractor explorer
# ===========================================================================
@app.cell
def display_section1_fig(
    C_CONTEXT, C_FIXED, C_START, C_TRUTH, beta_sl, finish_mpl, ftle_estimate,
    fixed_points, hopf_threshold, integrate_l63, mo, mpl_panels, np, rho_sl,
    show_transient, sigma_sl, x0_in, y0_in, z0_in,
):
    _sig, _rho, _beta = sigma_sl.value, rho_sl.value, beta_sl.value
    _s0 = [float(x0_in.value or 0.0), float(y0_in.value or 0.0), float(z0_in.value or 0.0)]
    _t_cut = 25.0

    _t, _y = integrate_l63(_s0, 75.0, n=6500, sigma=_sig, rho=_rho, beta=_beta)
    _tr = _t < _t_cut          # transient mask
    _at = ~_tr                 # settled / attractor mask

    # ---- classify the long-term behaviour ----
    _tail = _y[:, int(0.62 * _y.shape[1]):]
    _span = float(np.ptp(_tail, axis=1).max())
    _fps = fixed_points(_rho, _beta)
    if _span < 2e-2:
        _end = _tail[:, -1]
        _near = min(_fps, key=lambda k: np.linalg.norm(_end - _fps[k]))
        _kind, _ck = f"a **fixed point** (settles onto {_near})", "success"
        _lam = float("nan")
    else:
        _lam = ftle_estimate(_y[:, -1], _sig, _rho, _beta)
        if np.isfinite(_lam) and _lam > 0.03:
            _kind, _ck = "**chaotic** — a strange attractor", "danger"
        else:
            _kind, _ck = "**periodic** — a closed limit cycle", "warn"

    _rho_h = hopf_threshold(_sig, _beta)
    _rho_h_str = (
        f"ρ_H ≈ **{_rho_h:.2f}**  →  C± are "
        + ("**unstable** (ρ > ρ_H)" if _rho > _rho_h else "**stable** (ρ < ρ_H)")
        if _rho_h is not None else
        "ρ_H undefined for this σ, β (σ ≤ β + 1)"
    )
    _lam_str = (
        f"{_lam:.2f} MTU⁻¹  (τ_λ ≈ {1.0 / _lam:.1f} MTU)"
        if np.isfinite(_lam) and _lam > 0.02 else "— (no exponential divergence)"
    )

    # ---- static phase-space projections + time series ----
    # x-z first: that projection IS the butterfly, and putting it leftmost means the
    # recognisable image is the first thing the reader sees.
    _fig, _ax = mpl_panels(
        3,
        titles=(
            "Phase space  (x–z)",
            "Phase space  (x–y)",
            "Convective intensity  x(t)",
        ),
    )

    for _k, (_i, _j, _xl, _yl) in enumerate(
        [(0, 2, "x", "z"), (0, 1, "x", "y")]
    ):
        if show_transient.value and _tr.sum() > 1:
            _ax[_k].plot(
                _y[_i, _tr], _y[_j, _tr], color=C_CONTEXT, linewidth=0.6,
                linestyle=":", zorder=1,
                label="transient" if _k == 0 else None,
            )
        # Thin, semi-transparent: the attractor is thousands of near-coincident
        # passes, and a thick line hides its layered structure entirely.
        _ax[_k].plot(
            _y[_i, _at], _y[_j, _at], color=C_TRUTH, linewidth=0.35, alpha=0.65,
            zorder=2, label="settled trajectory" if _k == 0 else None,
        )
        _ax[_k].plot(
            _s0[_i], _s0[_j], marker="o", markersize=5, color=C_START,
            markeredgecolor="white", markeredgewidth=0.8, zorder=5,
            linestyle="none", label="start" if _k == 0 else None,
        )
        for _name, _p in _fps.items():
            _ax[_k].plot(
                _p[_i], _p[_j], marker="D", markersize=4.5, color=C_FIXED,
                markeredgecolor="white", markeredgewidth=0.6, zorder=6,
                linestyle="none",
                label="fixed points" if (_k == 0 and _name == "O") else None,
            )
            _ax[_k].annotate(
                _name, (_p[_i], _p[_j]), textcoords="offset points",
                xytext=(5, 4), fontsize=8, color="#92600b", zorder=7,
            )
        _ax[_k].set_xlabel(_xl)
        _ax[_k].set_ylabel(_yl)

    _ax[0].legend(loc="upper left", fontsize=7, framealpha=0.9)

    _ax[2].axvspan(0.0, _t_cut, color="#f1eefb", zorder=0)
    _ax[2].annotate("transient", (0.4, 0.96), xycoords="axes fraction",
                    fontsize=8, color="#6b6580", va="top")
    _ax[2].plot(_t, _y[0], color=C_TRUTH, linewidth=0.7)
    for _name, _p in _fps.items():
        if _name != "O":
            _ax[2].axhline(_p[0], color=C_FIXED, linewidth=0.9, linestyle=":")
            _ax[2].annotate(f"{_name}: x = {_p[0]:.1f}", (0.99, _p[0]),
                            xycoords=("axes fraction", "data"), ha="right",
                            fontsize=7, color="#92600b", va="bottom")
    _ax[2].set_xlabel("time (MTU)")
    _ax[2].set_ylabel("x")

    finish_mpl(
        _fig,
        suptitle=f"σ = {_sig:g},   ρ = {_rho:g},   β = {_beta:.3f}",
    )

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([sigma_sl, rho_sl, beta_sl], gap="2.5rem", justify="start"),
        mo.hstack([x0_in, y0_in, z0_in, show_transient], gap="1.5rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; σ = {_sig:g}, ρ = {_rho:g}, "
                f"β = {_beta:.3f}  \n"
                f"Long-term behaviour: this trajectory approaches {_kind}.  \n"
                f"Leading finite-time Lyapunov exponent λ₁ ≈ {_lam_str}  \n"
                f"Linear stability of the convective fixed points: {_rho_h_str}"
            ),
            kind=_ck,
        ),
    ])
    return


@app.cell
def display_section1_callout(mo):
    mo.callout(
        mo.md(r"""
**What makes this attractor "strange"?**  A periodic orbit would be a closed loop.
A fixed point would be a dot.  This attractor is *neither* — it is a **fractal set** with
non-integer (Hausdorff) dimension of approximately **2.06**.

Trajectories wind around the two lobes in an order that looks random but is in fact
*completely determined* by the initial condition.  The catch: two trajectories starting
from almost identical initial conditions will eventually end up on opposite lobes with
no correlation between them.  That is precisely what the next section demonstrates.
"""),
        kind="info",
    )
    return


# ===========================================================================
# SDIC — intuition and Lyapunov background
# ===========================================================================
@app.cell
def cell_sdic_intuition(mo):
    mo.md(r"""
---
### 📐 Quantifying chaos: the Lyapunov exponent

Before the interactive demonstration, let's build intuition about how we *measure* chaos.

**The key question:** If two trajectories start with separation $\delta_0$, how fast does it grow?

In a chaotic system, the answer (averaged over the attractor) is **exponentially**:

$$\delta(t) \approx \delta_0 \, e^{\lambda t}$$

where $\lambda$ is the **leading Lyapunov exponent**.  Its sign determines the system type:

| Sign of $\lambda$ | System type | Example |
|-------------------|-------------|---------|
| $\lambda < 0$ | Stable fixed point or limit cycle | Damped pendulum |
| $\lambda = 0$ | Marginally stable | Integrable Hamiltonian |
| $\lambda > 0$ | **Chaotic** — exponential divergence | Lorenz 63, real atmosphere |

**The Lorenz 63 Lyapunov spectrum** $(\lambda_1, \lambda_2, \lambda_3)$:

| Exponent | Value (MTU⁻¹) | Meaning |
|----------|--------------|---------|
| $\lambda_1$ | ≈ +0.906 | Exponential stretching — the source of chaos |
| $\lambda_2$ | ≈ 0.000 | Neutral — along the flow direction |
| $\lambda_3$ | ≈ −14.57 | Strong contraction — the attractor has zero volume |

The sum $\lambda_1 + \lambda_2 + \lambda_3 \approx -13.67$ is the **divergence** of the vector
field, confirming the system is dissipative (volume-shrinking).

The **Kaplan–Yorke dimension** estimates the fractal dimension of the attractor:
$$D_{KY} = 2 + \frac{\lambda_1}{|\lambda_3|} = 2 + \frac{0.906}{14.57} \approx 2.062$$

This non-integer value (between a surface and a volume) is the hallmark of a fractal strange attractor.

**The Lyapunov time** $\tau_\lambda = 1/\lambda_1 \approx 1.1$ MTU is the e-folding time for error growth.
Taking logarithms of the growth equation:
$$\ln\delta(t) \approx \ln\delta_0 + \lambda\,t$$
This is a **straight line on a log-scale plot** of separation vs. time, with slope $\lambda$.
That is exactly what the right-hand panel of Section 2 shows.
""")
    return


# ===========================================================================
# Section 2 — SDIC interactive
# ===========================================================================
@app.cell
def display_section2_text(mo):
    mo.md(r"""
---
## 2 · Sensitive Dependence on Initial Conditions

The defining signature of chaos is that two trajectories starting from
**arbitrarily close** initial conditions diverge exponentially fast.
This is *sensitive dependence on initial conditions* (SDIC) — the butterfly effect.

### What the panels show

| 3-D view (left) | Upper-right: separation | Lower-right: x(t) |
|-----------------|-------------------------|------------------|
| Indigo = truth **A**, rose = perturbed **B** | $\|\mathbf{A}-\mathbf{B}\|$ vs time, **log** y-axis | $x(t)$ for both runs on one axis |
| Emerald dot = shared start; squares = positions at lead time $T$ | Straight line = exponential growth; amber dotted = fitted $e^{\lambda t}$ | The two curves track, then abruptly separate |
| Grey ghost = the full attractor, for reference | Dashed red = attractor diameter (zero skill beyond it) | Dashed grey line marks when the paths split |

### How to read the log-separation plot

- **Flat or slowly rising**: error growing, but sub-exponentially (very early phase).
- **Straight line (exponential phase)**: error doubling every $\ln 2/\lambda \approx 0.77$ MTU — classic chaotic regime.
- **Levelling off at the red dashed line**: error saturated.  The two forecasts are now statistically
  independent — completely uncorrelated.  Forecast skill has dropped to zero.
""")
    return


@app.cell
def display_section2_experiment(mo):
    mo.callout(
        mo.md(r"""
**🔬 Experiment — measure the Lyapunov exponent:**

1. Set lead time = **5 MTU** and δ₀ = **10⁻⁴**.  Are the blue and red paths still together?
2. Drag lead time slowly to **20 MTU**.  At what time do they separate on the 3-D plot?
   Does the right panel enter the "exponential growth" straight-line regime?
3. Reduce δ₀ to **10⁻⁶** (two decades smaller).  How many extra MTU of predictability does that buy?
   Is the gain proportional to the change in δ₀?
4. Read off λ from the slope of the orange dotted line.  Compare to the theoretical value ≈ 0.9 MTU⁻¹.
5. Try δ₀ = **10⁻¹** (a large perturbation).  Does it diverge immediately, or is there still a coherent phase?
"""),
        kind="neutral",
    )
    return


@app.cell
def display_section2_interactive(
    C_CONTEXT, C_FIXED, C_PERT, C_SAT, C_SPREAD, C_START, C_TRUTH, attractor_ref,
    attractor_size, finish_mpl, integrate_l63, mo, mpl_grid, np, sdic_lead, sep_exp,
):
    # ---- two trajectories from (almost) the same start ----
    # A point taken from the settled attractor trajectory. The obvious-looking
    # [8.5, 8.5, 27.0] is NOT usable here: it sits 0.02 from the fixed point C+,
    # so both trajectories spiral there for tens of MTU and the separation grows
    # at ~0.09 MTU^-1 instead of ~0.9 -- the section would demonstrate the
    # opposite of its point.
    _x0 = attractor_ref[:, 0].copy()
    _delta0 = 10.0 ** sep_exp.value
    _T = sdic_lead.value
    _t, _traj_a = integrate_l63(_x0, _T, n=900, rtol=1e-11, atol=1e-13)
    _, _traj_b = integrate_l63(_x0 + np.array([_delta0, 0.0, 0.0]), _T,
                               n=900, rtol=1e-11, atol=1e-13)
    _sep = np.sqrt(np.sum((_traj_a - _traj_b) ** 2, axis=0))

    # ---- fit the leading exponent on the exponential-growth window ----
    _win = (_t > 0.5) & (_t < 0.75 * _T) & (_sep > 0) & (_sep < 0.4 * attractor_size)
    if _win.sum() > 8:
        _slope = float(np.polyfit(_t[_win], np.log(_sep[_win]), 1)[0])
        _lam = max(0.0, _slope)
        _lam_str = f"{_lam:.2f} MTU⁻¹  (≈ {_lam / 5:.2f} day⁻¹)"
    else:
        _lam = None
        _lam_str = "— (extend the lead time to see exponential growth)"

    # ---- divergence time + predictability regime ----
    _above = np.where(_sep > 0.3 * attractor_size)[0]
    _t_div = float(_t[_above[0]]) if _above.size else None
    _frac = float(_sep[-1]) / attractor_size
    if _frac > 0.8:
        _regime, _ck = "🔴 Forecasts completely uncorrelated — beyond the predictability horizon", "danger"
    elif _frac > 0.3:
        _regime, _ck = "🟠 Diverging rapidly — entering the semi-predictable regime", "warn"
    else:
        _regime, _ck = "🟢 Forecasts still agree — within the predictable window", "success"

    # ---- 2x2: two static projections on top, two diagnostics below ----
    _fig, (_axz, _axy, _axs, _axx) = mpl_grid(2, 2)

    for _a, (_i, _j, _xl, _yl, _ttl) in [
        (_axz, (0, 2, "x", "z", "Phase space  (x–z)")),
        (_axy, (0, 1, "x", "y", "Phase space  (x–y)")),
    ]:
        _a.plot(attractor_ref[_i], attractor_ref[_j], color=C_CONTEXT,
                linewidth=0.4, zorder=1)
        _a.plot(_traj_a[_i], _traj_a[_j], color=C_TRUTH, linewidth=1.1,
                zorder=3, label="truth  A")
        _a.plot(_traj_b[_i], _traj_b[_j], color=C_PERT, linewidth=1.1,
                zorder=3, label="perturbed  B")
        _a.plot(_x0[_i], _x0[_j], marker="o", markersize=5, color=C_START,
                markeredgecolor="white", markeredgewidth=0.8, linestyle="none",
                zorder=6, label="shared start")
        _a.plot(_traj_a[_i, -1], _traj_a[_j, -1], marker="s", markersize=5,
                color=C_TRUTH, linestyle="none", zorder=6, label=f"A at t = {_T}")
        _a.plot(_traj_b[_i, -1], _traj_b[_j, -1], marker="s", markersize=5,
                color=C_PERT, linestyle="none", zorder=6, label=f"B at t = {_T}")
        _a.set_xlabel(_xl)
        _a.set_ylabel(_yl)
        _a.set_title(_ttl)
    _axz.legend(loc="upper left", fontsize=6.5, framealpha=0.9, ncol=2)

    # ---- separation, log scale ----
    _axs.semilogy(_t, _sep, color=C_SPREAD, linewidth=1.8, label="|A − B|")
    _axs.fill_between(_t, 1e-300, _sep, color=C_SPREAD, alpha=0.07)
    if _lam:
        _tw = _t[_win]
        _axs.semilogy(_tw, _sep[_win][0] * np.exp(_lam * (_tw - _tw[0])),
                      color=C_FIXED, linewidth=1.2, linestyle=":",
                      label=f"e^(λt),  λ ≈ {_lam:.2f}")
    _axs.axhline(attractor_size, color=C_SAT, linewidth=1.2, linestyle="--")
    _axs.annotate("attractor diameter — fully random", (0.02, attractor_size),
                  xycoords=("axes fraction", "data"), fontsize=7,
                  color="#b91c1c", va="bottom")
    if _t_div is not None:
        _axs.axvline(_t_div, color="#64748b", linewidth=0.9, linestyle="--")
        # Horizontal, just under the top edge and offset from the line, so it does
        # not sit on top of the separation curve.
        # Mid-panel and left of the line: the top carries the saturation label and
        # the bottom-right carries the legend.
        _axs.annotate(f"split · t ≈ {_t_div:.1f}", (_t_div, 0.62),
                      xycoords=("data", "axes fraction"), fontsize=7,
                      color="#475569", va="center", ha="right",
                      xytext=(-4, 0), textcoords="offset points",
                      # The separation curve passes through this height, so the
                      # label needs its own ground to be legible.
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                                boxstyle="round,pad=0.2"))
    _axs.set_xlabel("lead time (MTU)")
    _axs.set_ylabel("|A − B|")
    _axs.set_title("Separation  (log scale)")
    _axs.set_ylim(max(_sep.min() * 0.5, 1e-14), attractor_size * 3)
    _axs.legend(loc="lower right", fontsize=7, framealpha=0.9)

    # ---- x(t) for both ----
    _axx.plot(_t, _traj_a[0], color=C_TRUTH, linewidth=1.0, label="x_A")
    _axx.plot(_t, _traj_b[0], color=C_PERT, linewidth=1.0, label="x_B")
    if _t_div is not None:
        _axx.axvline(_t_div, color="#64748b", linewidth=0.9, linestyle="--")
    _axx.set_xlabel("lead time (MTU)")
    _axx.set_ylabel("x")
    _axx.set_title("x(t) for each trajectory")
    _axx.legend(loc="upper left", fontsize=7, framealpha=0.9)

    finish_mpl(_fig, suptitle="Two trajectories from (almost) the same start")

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([sep_exp, sdic_lead], gap="4rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; δ₀ = 10^{sep_exp.value:.1f} "
                f"&nbsp;·&nbsp; lead time = {_T} MTU  \n"
                f"Fitted leading exponent: **λ ≈ {_lam_str}**  (theory: ≈ 0.9 MTU⁻¹)  \n"
                f"Final separation = **{_sep[-1]:.3g}** ({_frac:.0%} of attractor size)"
                + (f" &nbsp;·&nbsp; paths visibly split at **t ≈ {_t_div:.1f} MTU**"
                   if _t_div is not None else "")
                + f"  \n{_regime}"
            ),
            kind=_ck,
        ),
    ])
    return


@app.cell
def display_section2_weather(mo):
    mo.md(r"""
### Why this matters for weather forecasting

In the real atmosphere the leading Lyapunov exponent is $\lambda \approx 0.35\;\text{day}^{-1}$,
giving an error **doubling time** of

$$\tau_{2} = \frac{\ln 2}{\lambda} \approx \frac{0.69}{0.35\;\text{day}^{-1}} \approx 2\;\text{days}$$

This means that even a perfect 1-hour analysis error doubles in two days.
A 10-day forecast must survive five doublings — an amplification factor of $2^5 = 32$.
No conceivable improvement in observations or models can eliminate this growth,
because it is a property of the underlying flow, not of our instruments.

**The practical predictability ceiling** in the real atmosphere is approximately
**2–3 weeks** — beyond which even a perfect initial state cannot yield a useful
deterministic forecast.  This is a mathematical consequence of the Lorenz time of the atmosphere,
not a pessimistic statement about the current state of NWP.
""")
    return


# ===========================================================================
# Ensemble NWP — historical background
# ===========================================================================
@app.cell
def cell_ensemble_history(mo):
    mo.md(r"""
---
### 🕰️ The operational history of ensemble forecasting

The mathematical case for ensemble NWP was made long before it was computationally feasible:

| Year | Development |
|------|------------|
| **1963** | Lorenz shows deterministic chaos implies finite predictability |
| **1965** | Lorenz estimates the atmospheric predictability limit |
| **1969** | **Epstein** proposes stochastic-dynamic forecasting — the first ensemble concept |
| **1974** | **Leith** demonstrates Monte Carlo ensemble forecasting in a simple model |
| **1992** | ECMWF launches the **Ensemble Prediction System (EPS)** operationally |
| **1992** | NCEP launches the **Global Ensemble Forecast System (GEFS)** |
| **2002** | Ensemble Kalman filter (EnKF) applied to NWP by Hamill & Snyder |
| **2010s** | Hybrid ensemble-variational (En-Var) data assimilation adopted by major centres |
| **2020s** | Machine-learning ensemble post-processing and diffusion-model ensemble generation |

**How are operational perturbations chosen?**

Simply adding random noise (as we do below) is not optimal — it wastes ensemble members
on directions that do not grow.  Real NWP centres use more sophisticated methods:

| Method | Idea | Used by |
|--------|------|---------|
| **Bred vectors** | Evolve a perturbation for a short time, rescale, repeat — breeds fast-growing modes | NCEP (1992–) |
| **Singular vectors** | Find the perturbation that grows most over a chosen optimisation period | ECMWF (1992–) |
| **Ensemble Kalman filter** | Use the ensemble itself as the background-error covariance in data assimilation | Many regional centres |
| **Stochastic physics** | Add random noise to model tendencies to represent model uncertainty | ECMWF (2009–) |

**ECMWF EPS at a glance (2024):** 51 members (1 control + 50 perturbed),
18 km horizontal resolution, 137 vertical levels, 15-day medium-range and 46-day extended-range products.
""")
    return


# ===========================================================================
# Section 3 — Ensemble Forecasting interactive
# ===========================================================================
@app.cell
def display_section3_text(mo):
    mo.md(r"""
---
## 3 · Ensemble Forecasting

A single deterministic forecast is an *answer without an error bar*.
The operational response to SDIC is the **ensemble forecast**: integrate $N$ slightly
different trajectories from initial states that sample the analysis uncertainty.

### What the ensemble tells us

The **ensemble mean** $\bar X(t) = \frac{1}{N}\sum_{i=1}^N X_i(t)$ is a better
point forecast than any single member.

The **ensemble spread** — the RMS standard deviation across members — measures forecast uncertainty:

$$\sigma_\text{spread}(t) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left|X_i(t) - \bar X(t)\right|^2}$$

Three predictability regimes:

| Regime | Spread / attractor size | Interpretation |
|--------|------------------------|---------------|
| 🟢 **Predictable** | < 10 % | Members closely clustered; deterministic forecast reliable |
| 🟠 **Semi-predictable** | 10 – 90 % | Spread growing fast; probabilistic guidance still useful |
| 🔴 **Unpredictable** | > 90 % | Spread saturated; forecast no better than climatology |

### What the two panels show

**Left (phase space):** Faint violet lines are the $N$ members; the **indigo** line is
the *truth* (unperturbed run) and the **teal** line is the *ensemble mean*.
The emerald cloud is the ensemble at $t = 0$, the rose cloud at $t = T$.
A compact rose cloud = reliable forecast; a cloud spanning the whole attractor = no skill.
Watch the mean track the truth, then peel away once the members disagree.

**Right (spread plot):** The **violet** curve is $\sigma_\text{spread}(t)$; the dashed
grey curve is the **ensemble-mean error** $|\bar X - X_\text{truth}|$, both on a log scale.
For a well-calibrated ensemble the two curves should sit on top of each other — if the
spread runs *below* the error, the ensemble is **under-dispersive** (overconfident).
The x-position of the 🟢→🟠 transition is the **predictability horizon**.

### Why starting location matters

The attractor is not uniformly chaotic.  Near **lobe centres**, trajectories loop
coherently before switching — relatively predictable.  Near the **saddle point** at
the origin, perturbations grow much faster.  Near **lobe transitions**, which lobe
a trajectory switches to becomes sensitive to tiny perturbations.
This *flow-dependent predictability* is why modern NWP generates a fresh ensemble every 6 hours.
""")
    return


@app.cell
def display_section3_experiment(mo):
    mo.callout(
        mo.md(r"""
**🔬 Experiment — find the predictability horizon:**

1. Start with defaults (predictable region, δ₀ = 10⁻⁴, N = 20).
   Drag **lead time** slowly from 1 → 30 MTU.  Note when the green cloud fills the attractor.

2. Change **starting location** to *Near saddle point*.  Is the horizon earlier or later?
   Why might the position on the attractor matter?

3. Fix lead time = 15.  Slide **perturbation size** from 10⁻⁶ → 10⁻¹.
   Does reducing δ₀ by one decade give a proportionally longer horizon?

4. Fix lead time = 15 and δ₀ = 10⁻⁴.  Compare **N = 5** vs **N = 50**.
   Which gives a smoother, more reliable spread estimate?

5. *Bonus:* Set N = 50, δ₀ = 10⁻⁴, location = *Chaotic lobe transition*.
   Is the horizon shorter or longer than the predictable region?
"""),
        kind="neutral",
    )
    return


@app.cell
def display_section3_interactive(
    C_CONTEXT, C_MEAN, C_PERT, C_SAT, C_SPREAD, C_START, C_TRUTH, attractor_ref,
    attractor_size, finish_mpl, ic_choice, integrate_l63, lead_time, mo, mpl_grid,
    n_members, np, perturb_exp,
):
    # All three starts are points ON the attractor, chosen because their measured
    # spread-growth horizons genuinely differ (t10 of roughly 11, 9 and 6.5 MTU).
    # The previous options did not work: "predictable" was the fixed point C+ and
    # "saddle" was near the origin, and from both the ensemble spread stayed under
    # 1% of the attractor size for 20 MTU -- so the dropdown, whose whole purpose
    # is to show that WHERE you start matters, showed nothing at all.
    _ic_map = {
        "predictable": attractor_ref[:, 2400].copy(),
        "typical": attractor_ref[:, 0].copy(),
        "chaotic": attractor_ref[:, 1500].copy(),
    }
    _ic_labels = {
        "predictable": "slow region — spread grows late",
        "typical": "a typical point on the attractor",
        "chaotic": "fast region — spread grows early",
    }
    _x0 = _ic_map[ic_choice.value]
    _N = n_members.value
    _T = lead_time.value
    _pert = 10.0 ** perturb_exp.value
    _nt = 600

    _t, _truth = integrate_l63(_x0, _T, n=_nt, rtol=1e-9, atol=1e-12)
    _rng = np.random.default_rng(42)
    _perturbs = _rng.standard_normal((_N, 3)) * _pert
    _trajs = np.zeros((_N, 3, _nt))
    for _i in range(_N):
        _, _trajs[_i] = integrate_l63(_x0 + _perturbs[_i], _T, n=_nt,
                                      rtol=1e-9, atol=1e-12)

    _mean = _trajs.mean(axis=0)
    _rms_spread = np.sqrt(np.mean(np.var(_trajs, axis=0), axis=0))
    _mean_err = np.sqrt(np.mean((_mean - _truth) ** 2, axis=0))
    _t_max = float(_t[-1])

    _i10 = np.where(_rms_spread >= 0.1 * attractor_size)[0]
    _i90 = np.where(_rms_spread >= 0.9 * attractor_size)[0]
    _t10 = float(_t[_i10[0]]) if _i10.size else _t_max
    _t90 = float(_t[_i90[0]]) if _i90.size else _t_max

    _final_sat = float(_rms_spread[-1] / attractor_size)
    if _final_sat < 0.3:
        _regime, _ck = "🟢 Ensemble well-clustered — forecast trustworthy", "success"
    elif _final_sat < 0.8:
        _regime, _ck = "🟠 Spread growing rapidly — probabilistic guidance only", "warn"
    else:
        _regime, _ck = "🔴 Spread saturated — forecast is climatology", "danger"

    # ---- 2x2: projections above, spread/error and member x(t) below ----
    _fig, (_axz, _axy, _axsp, _axx) = mpl_grid(2, 2)

    for _a, (_i, _j, _xl, _yl, _ttl) in [
        (_axz, (0, 2, "x", "z", "Ensemble in phase space  (x–z)")),
        (_axy, (0, 1, "x", "y", "Ensemble in phase space  (x–y)")),
    ]:
        _a.plot(attractor_ref[_i], attractor_ref[_j], color=C_CONTEXT,
                linewidth=0.4, zorder=1)
        # Members drawn first, thin and translucent: the point is the *envelope*
        # they trace, not any individual member.
        for _m in range(_N):
            _a.plot(_trajs[_m, _i], _trajs[_m, _j], color=C_SPREAD,
                    linewidth=0.5, alpha=0.28, zorder=2,
                    label="members" if (_m == 0 and _a is _axz) else None)
        _a.plot(_truth[_i], _truth[_j], color=C_TRUTH, linewidth=1.3, zorder=4,
                label="truth" if _a is _axz else None)
        _a.plot(_mean[_i], _mean[_j], color=C_MEAN, linewidth=1.2, zorder=4,
                label="ensemble mean" if _a is _axz else None)
        _a.plot(_trajs[:, _i, 0], _trajs[:, _j, 0], marker="o", markersize=2.6,
                color=C_START, linestyle="none", zorder=5,
                label="t = 0 cloud" if _a is _axz else None)
        _a.plot(_trajs[:, _i, -1], _trajs[:, _j, -1], marker="o", markersize=2.6,
                color=C_PERT, linestyle="none", zorder=5,
                label=f"t = {_T} cloud" if _a is _axz else None)
        _a.set_xlabel(_xl)
        _a.set_ylabel(_yl)
        _a.set_title(_ttl)
    _axz.legend(loc="upper left", fontsize=6.5, framealpha=0.9, ncol=2)

    # ---- spread vs ensemble-mean error, with the three regimes shaded ----
    # Plain-text labels inside the figure: matplotlib's default font has no emoji
    # glyphs, so an emoji here renders as an empty box. The coloured shading already
    # carries the traffic-light meaning, and the emoji survive in the readout below.
    for _x0s, _x1s, _col, _lab in (
        (0.0, _t10, "#e8f7f1", "predictable"),
        (_t10, _t90, "#fdf3e3", "semi-predictable"),
        (_t90, _t_max, "#fbeaea", "unpredictable"),
    ):
        if _x0s < _x1s:
            _axsp.axvspan(_x0s, _x1s, color=_col, zorder=0)
            # Along the BOTTOM: the top of this panel already carries the
            # attractor-size and 10% threshold annotations, and three more labels
            # up there collide with both of them and with each other.
            _axsp.annotate(_lab, ((_x0s + _x1s) / 2, 0.02),
                           xycoords=("data", "axes fraction"), ha="center",
                           va="bottom", fontsize=6.5, color="#475569")
    _axsp.semilogy(_t, _rms_spread, color=C_SPREAD, linewidth=1.8,
                   label="ensemble spread", zorder=3)
    _axsp.semilogy(_t, _mean_err, color="#475569", linewidth=1.3, linestyle="--",
                   label="ensemble-mean error", zorder=3)
    _axsp.axhline(attractor_size, color=C_SAT, linewidth=1.2, linestyle="--")
    _axsp.annotate("attractor size — fully unpredictable", (0.02, attractor_size),
                   xycoords=("axes fraction", "data"), fontsize=6.5,
                   color="#b91c1c", va="bottom")
    _axsp.axhline(0.1 * attractor_size, color=C_SPREAD, linewidth=1.0, linestyle=":")
    _axsp.annotate("10 % threshold", (0.02, 0.1 * attractor_size),
                   xycoords=("axes fraction", "data"), ha="left", fontsize=6.5,
                   color="#7c3aed", va="bottom")
    _axsp.set_xlabel("lead time (MTU)")
    _axsp.set_ylabel("RMS (state units)")
    _axsp.set_title("Spread and ensemble-mean error")
    _axsp.legend(loc="center right", fontsize=7, framealpha=0.9)

    # ---- every member's x(t), the classic spaghetti plot ----
    for _m in range(_N):
        _axx.plot(_t, _trajs[_m, 0], color=C_SPREAD, linewidth=0.5, alpha=0.3)
    _axx.plot(_t, _truth[0], color=C_TRUTH, linewidth=1.3, label="truth")
    _axx.plot(_t, _mean[0], color=C_MEAN, linewidth=1.2, label="ensemble mean")
    _axx.set_xlabel("lead time (MTU)")
    _axx.set_ylabel("x")
    _axx.set_title("x(t): every member")
    _axx.legend(loc="upper left", fontsize=7, framealpha=0.9)

    finish_mpl(
        _fig,
        suptitle=f"N = {_N}  ·  δ₀ = 10^{perturb_exp.value:.1f}  ·  T = {_T} MTU  ·  "
                 f"IC: {_ic_labels[ic_choice.value]}",
    )

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([ic_choice, n_members], gap="3rem", justify="start"),
        mo.hstack([perturb_exp, lead_time], gap="3rem", justify="start"),
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; N = {_N} members &nbsp;·&nbsp; "
                f"δ₀ = 10^{perturb_exp.value:.1f} &nbsp;·&nbsp; T = {_T} MTU  \n"
                f"Predictability horizon (spread > 10 %): **t ≈ {_t10:.1f} MTU**  \n"
                f"Full saturation (spread > 90 %): **t ≈ {_t90:.1f} MTU**  \n"
                f"Final spread: **{_final_sat:.0%}** of attractor size &nbsp;·&nbsp; "
                f"spread/error ratio at T: **{_rms_spread[-1] / max(_mean_err[-1], 1e-9):.2f}**  \n"
                f"{_regime}"
            ),
            kind=_ck,
        ),
    ])
    return


@app.cell
def display_section3_calibration(mo):
    mo.md(r"""
### Ensemble spread vs. ensemble mean error

A perfectly calibrated ensemble satisfies:
$$\langle \sigma^2_\text{spread} \rangle = \langle \epsilon^2_\text{mean} \rangle$$

where $\epsilon_\text{mean} = |\bar X - X_\text{truth}|$ is the ensemble-mean error.
In practice, most NWP ensembles are **underdispersive** (spread < error) because:

1. Initial perturbations do not fully sample the true analysis error
2. Model error is not fully represented
3. Ensemble size $N$ is finite

Underdispersion means the ensemble is **overconfident**.
Calibration techniques (inflation, rank histogram adjustment) correct for this in post-processing.
""")
    return


# ===========================================================================
# Section 4 — Connection to the real atmosphere
# ===========================================================================
@app.cell
def display_section4_text(mo):
    mo.md(r"""
---
## 4 · Connection to the Real Atmosphere

The Lorenz model is a toy, but its key numbers map onto the real atmosphere
with surprising fidelity.

### Lyapunov numbers: model vs. atmosphere

| Quantity | Lorenz 63 | Real atmosphere |
|----------|-----------|----------------|
| Leading Lyapunov exponent $\lambda$ | ≈ 0.91 MTU⁻¹ | ≈ 0.35 day⁻¹ |
| Lyapunov (e-folding) time $1/\lambda$ | ≈ 1.1 MTU | ≈ 2.9 days |
| Error doubling time $\ln 2 / \lambda$ | ≈ 0.77 MTU | ≈ 2.0 days |
| Predictability horizon (spread > 10 %) | ≈ 3–5 MTU | ≈ 1–2 weeks |
| Full saturation | ≈ 6–9 MTU | ≈ 3–4 weeks |

**The two columns are independent measurements, not conversions of each other** — and
reading across the doubling-time row is where the 5-day convention breaks down. At
5 days per MTU, Lorenz 63's 0.77 MTU is ≈ 3.8 days, against the atmosphere's ≈ 2.0.
Equivalently, $\lambda \approx 0.91\ \mathrm{MTU^{-1}}$ becomes
$0.18\ \mathrm{day^{-1}}$ — about **half** the atmosphere's 0.35. Under this convention
the model is roughly half as chaotic per day as the system it stands in for.

Calibrate the other way — pick the MTU so the *doubling times* agree — and you get
1 MTU ≈ 2.6 days instead. Neither reading is wrong; they answer different questions,
and no single conversion satisfies both. What transfers from this model to the
atmosphere is the **law** — error grows exponentially, so predictability is bought
logarithmically — not the constant in front of it.

Taking the 5-day reading, ECMWF's ≈ 10 days of useful deterministic skill is roughly
2 MTU: about **2.6 error-doubling times**, or 1.8 e-folding times. (Those two are easy
to conflate. The Lyapunov time $1/\lambda$ is the e-folding time; the doubling time
$\ln 2/\lambda$ is shorter by a factor of $\ln 2$.)

### The diminishing return of better observations

Suppose you improve your analysis error from $\delta_0$ to $\delta_0 / 10$.
The extra predictable time gained is

$$\Delta t = \frac{\ln 10}{\lambda} \approx \frac{2.3}{0.35\;\text{day}^{-1}} \approx 6.5\;\text{days}$$

A factor-of-10 improvement buys only **6.5 extra days**.
A factor-of-100 improvement buys only **13 extra days**.
This logarithmic ceiling means the ≈ 2–3 week predictability limit is
**fundamental, not a consequence of inadequate technology**.

### Historical skill improvement at ECMWF

ECMWF tracks forecast skill continuously since 1980.
The 500 hPa geopotential anomaly correlation (AC) score: a score of 0.6 is
the conventional threshold for "useful" forecasting.

| Era | 500 hPa AC = 0.6 reached at... |
|-----|-------------------------------|
| 1980 | ≈ 5 days (Northern Hemisphere) |
| 1990 | ≈ 7 days |
| 2000 | ≈ 8 days |
| 2010 | ≈ 9 days |
| 2020 | ≈ 9–10 days |

The slowing rate of improvement is consistent with the **logarithmic limit** imposed by SDIC.

### Predictability of the second kind

Everything above is **predictability of the first kind**: initial-value prediction
of a specific trajectory.  There is also a **second kind**: predicting the *response
of the attractor to a sustained external forcing*.

In the Lorenz system, individual trajectories become unpredictable after ≈ 5–8 MTU,
but if you change $\rho$ (the forcing parameter), the *time-mean* of $X$ shifts
systematically — and that shift can be predicted even when individual trajectories cannot.
This is the mathematical analogue of the climate-vs-weather distinction.

| Phenomenon | Typical lead time | Mechanism |
|------------|------------------|-----------|
| El Niño / La Niña (ENSO) | 6–18 months | Slow ocean-atmosphere coupling |
| Monsoon onset | 2–4 weeks | Land–sea thermal contrast |
| Stratospheric sudden warmings | 2–3 weeks | Wave-mean-flow interaction |
| Long-term climate change | Decades–centuries | Radiative forcing from GHGs |

| Predictability type | Question asked | Chaotic limit applies? |
|---------------------|----------------|----------------------|
| **1st kind** | Where will this air mass be in 10 days? | Yes — hard ceiling |
| **2nd kind** | How will the *average* temperature change if CO₂ doubles? | No — signal persists |

Climate projections are a predictability-of-the-second-kind problem.
Their uncertainty comes from *model structural error* and *scenario uncertainty*,
not from the butterfly effect.
""")
    return


@app.cell
def display_section4_callout(mo):
    mo.callout(
        mo.md(r"""
**Key take-aways from this tutorial**

1. **Chaos is irreducible:** SDIC means that no finite improvement in initial
   conditions can extend deterministic forecast skill indefinitely.
   The atmosphere has a hard predictability ceiling near 2–3 weeks.

2. **Ensembles are the correct response:** A probabilistic forecast communicates
   the *distribution* of possible futures honestly.  A deterministic forecast
   beyond the predictability horizon is overconfident by construction.

3. **The two kinds of predictability are different problems:**
   Weather forecasting (1st kind) is limited by chaos.
   Climate projection (2nd kind) is not — but faces other sources of uncertainty.

4. **Improving observations has diminishing returns:**
   Each decade of improvement in $\delta_0$ buys only $\ln(10)/\lambda$ extra days.
   For the atmosphere that is ≈ 6.5 days per decade.

5. **Flow-dependent predictability matters:**
   Not all weather patterns are equally predictable.
   Ensemble spread is the operational estimate of this situation-dependent uncertainty.
"""),
        kind="info",
    )
    return


# ===========================================================================
# Guided Questions
# ===========================================================================
@app.cell
def display_questions(mo):
    mo.md(r"""
---
## 📝 Guided Questions

Work through these with a neighbour (~15 min).  We will discuss as a group.

---

**Q1 — Predictability horizon** *(Section 3)*

Set δ₀ = 10⁻⁴, N = 20, starting location = *Predictable region*.
Drag lead time slowly from 1 → 30 MTU.

- At what lead time does the green start-cloud scatter across the entire attractor?
- Which colour zone does the spread plot enter at that point?
- Switch to *Chaotic lobe transition*.  Is the horizon earlier or later?
  Why might some regions of the attractor be more predictable than others?

---

**Q2 — Sensitivity to perturbation size** *(Section 3)*

Fix lead time = 15 MTU.  Slide δ₀ from 10⁻⁶ → 10⁻¹.

- Does reducing δ₀ by **one decade** give you a proportionally longer horizon?
- Using $\lambda \approx 0.9\;\text{MTU}^{-1}$, calculate the expected extra
  predictable time: $\Delta t = \ln(10)/\lambda = ?$  Does your experiment agree?
- What does this imply for the practical benefit of improving atmospheric
  observations by an order of magnitude?

---

**Q3 — Ensemble size** *(Section 3)*

Fix lead time = 15 MTU and δ₀ = 10⁻⁴.  Compare N = 5 vs N = 50.

- How noisy is the spread estimate with N = 5?
  Could you reliably identify the predictability horizon from it?
- At what N does the spread curve look smooth enough to trust?
- ECMWF uses 51 members.  Based on your experiments, does that seem justified?

---

**Q4 — Quantitative connection to the real atmosphere** *(Sections 2 & 4)*

In L63 the error **doubling** time is $\ln 2/\lambda_1 \approx 0.77$ MTU and the
**e-folding** time is $1/\lambda_1 \approx 1.1$ MTU — do not use them
interchangeably. ECMWF achieves useful skill to ≈ 10 days.

- Taking 1 MTU ≈ 5 days, how many *doubling* times is a 10-day forecast? How many
  *e-folding* times? (They differ by a factor of $\ln 2$; you should get ≈ 2.6 and
  ≈ 1.8.)
- If ECMWF could reduce analysis error by a factor of 100, how many extra days
  of deterministic predictability would that buy?  Use $\Delta t = \ln(100) / 0.35\;\text{day}^{-1}$.
- Is this gain worth the cost of a factor-100 improvement in observations?

---

**Q5 — Predictability of the second kind** *(Section 4, bonus)*

Set starting location = *Near saddle point*, δ₀ = 10⁻⁶, N = 20.
Watch the ensemble saturate — individual trajectories become completely uncorrelated.

- Suppose you care not about any specific trajectory but about the
  **long-run time-average** of $X$.  Would that remain predictable after the Lorenz time?  Why?
- If someone changed $\rho$ from 28 to 30, could you predict the *new* time-mean of $X$
  even though individual trajectories are still chaotic?
- How does this connect to the distinction between weather forecasting
  (**predictability of the 1st kind**) and climate projection (**predictability of the 2nd kind**)?

---

*System: Lorenz (1963), σ = 10, ρ = 28, β = 8/3.*
*Integration: RK45 (scipy), rtol = 10⁻⁹.*
*Ensemble perturbations: iid Gaussian with seed = 42.*
*Time unit: 1 MTU read as ≈ 5 days — a loose convention, not a calibration; see Section 4.*
""")
    return


# ===========================================================================
# Further reading
# ===========================================================================
@app.cell
def cell_further_reading(mo):
    mo.md(r"""
---
## 📚 Further Reading

### Original papers

- **Lorenz, E. N. (1963)**. *Deterministic nonperiodic flow.*
  Journal of the Atmospheric Sciences, 20(2), 130–141.
  The founding paper.  Remarkably readable for a mathematical landmark.

- **Lorenz, E. N. (1965)**. *A study of the predictability of a 28-variable atmospheric model.*
  Tellus, 17(3), 321–333.  First estimate of the atmospheric predictability limit.

- **Lorenz, E. N. (1975)**. *Climatic predictability.*
  GARP Publication Series No. 16, 132–136.
  Introduces the first/second-kind predictability distinction.

- **Epstein, E. S. (1969)**. *Stochastic dynamic prediction.*
  Tellus, 21(6), 739–759.  The first formal ensemble forecasting proposal.

- **Palmer, T. N. (2000)**. *Predicting uncertainty in forecasts of weather and climate.*
  Reports on Progress in Physics, 63(2), 71.
  Excellent review connecting chaos theory to operational NWP.

### Books

- **Lorenz, E. N. (1993)**. *The Essence of Chaos.* University of Washington Press.
  Lorenz's own account, written for a general audience.  Highly recommended.

- **Gleick, J. (1987)**. *Chaos: Making a New Science.* Viking Penguin.
  Popular science account of the chaos revolution; contains the Lorenz story.

- **Palmer, T. N. & Hagedorn, R. (Eds.) (2006)**. *Predictability of Weather and Climate.*
  Cambridge University Press.  Comprehensive graduate-level treatment.

- **Kalnay, E. (2003)**. *Atmospheric Modelling, Data Assimilation and Predictability.*
  Cambridge University Press.  Standard NWP textbook; Chapter 6 covers ensembles.

---
*Notebook by Aneesh C. Subramanian.*
*Built with [marimo](https://marimo.io), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [Matplotlib](https://matplotlib.org).*
""")
    return


if __name__ == "__main__":
    app.run()
