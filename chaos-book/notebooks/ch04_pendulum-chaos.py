# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "plotly",
# ]
# ///
"""Chapter 4 -- Regular motion and why it is predictable.

Single vs. double pendulum: the minimal demonstration that nonlinearity alone
does not produce chaos -- phase-space dimension does.

Part II of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.

To edit:   marimo edit notebooks/ch04_pendulum-chaos.py
To export: make nb-one NB=ch04_pendulum-chaos
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Single vs. Double Pendulum")


# ---------------------------------------------------------------------------
# Imports
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
    import plotly.graph_objects as go

    from chaoslib import integrate, plotting, systems

    return go, integrate, mo, np, plotting, systems


# ---------------------------------------------------------------------------
# Physical constants (fixed, not exposed as sliders — the point of this
# notebook is the qualitative difference between the two systems, not a
# parameter study of mass/length ratios)
# ---------------------------------------------------------------------------
@app.cell
def constants():
    G = 9.81  # m/s^2
    L_SINGLE = 1.0  # m — single-pendulum rod length
    M1, M2 = 1.0, 1.0  # kg — double-pendulum point masses
    L1, L2 = 1.0, 1.0  # m — double-pendulum rod lengths
    return G, L1, L2, L_SINGLE, M1, M2


# ---------------------------------------------------------------------------
# Governing equations -- thin adapters over the tested chaoslib primitives
# ---------------------------------------------------------------------------
@app.cell
def ode_functions(G, L1, L2, L_SINGLE, M1, M2, systems):
    # The equations of motion live in `chaoslib.systems`, tested there against
    # energy conservation and the exact elliptic-integral period. These are thin
    # adapters that bind this chapter's fixed constants, so the printed equations
    # and the stepped equations cannot drift apart.
    def single_pendulum_rhs(t, s):
        return systems.pendulum(t, s, g=G, length=L_SINGLE)

    def double_pendulum_rhs(t, s):
        return systems.double_pendulum(t, s, g=G, l1=L1, l2=L2, m1=M1, m2=M2)

    return double_pendulum_rhs, single_pendulum_rhs


# ---------------------------------------------------------------------------
# UI controls
# ---------------------------------------------------------------------------
@app.cell
def single_pendulum_controls(mo):
    theta0_deg = mo.ui.slider(
        start=5, stop=175, step=5, value=30,
        label="Initial angle θ₀ (degrees)",
        show_value=True,
    )
    return (theta0_deg,)


@app.cell
def double_pendulum_controls(mo):
    dp_theta1_deg = mo.ui.slider(
        start=-170, stop=170, step=5, value=120,
        label="θ₁ initial angle (degrees)",
        show_value=True,
    )
    dp_theta2_deg = mo.ui.slider(
        start=-170, stop=170, step=5, value=-10,
        label="θ₂ initial angle (degrees)",
        show_value=True,
    )
    dp_pert_exp = mo.ui.slider(
        start=-8, stop=-2, step=0.5, value=-6,
        label="Log₁₀ perturbation size  δ₀ (rad)",
        show_value=True,
    )
    dp_lead_time = mo.ui.slider(
        start=1, stop=30, step=1, value=12,
        label="Lead time (s)",
        show_value=True,
    )
    return dp_lead_time, dp_pert_exp, dp_theta1_deg, dp_theta2_deg


# ===========================================================================
# Title and overview
# ===========================================================================
@app.cell
def display_title(mo):
    mo.md(r"""
# Chapter 4 · One Pendulum, Two Pendulums: From Perfectly Regular to Chaotic

**Part II — From regular motion to chaos.**

**The forecasting question.** Weather models are nonlinear, and nonlinearity is
usually blamed for the two-week forecast limit. But nonlinearity by itself is not
enough: the pendulum below is thoroughly nonlinear and perfectly predictable
forever. What actually has to be true of a system before its forecasts can fail?

This notebook compares the simplest possible *regular* dynamical system — the
single pendulum — with the simplest possible *chaotic* one — the double
pendulum. Interact with each phase portrait before reading on: the qualitative
difference between the two is visible immediately, and the equations explain
*why* it has to be that way.

---

### Learning objectives

By the end of this notebook you will be able to:

1. **Write down** the exact equations of motion for the single and double pendulum
2. **Derive** the small-angle (linear, SHM) approximation from the exact single-pendulum
   equation, and state when it breaks down
3. **Explain**, using the Poincaré–Bendixson theorem, why the single pendulum can never
   be chaotic — at any amplitude
4. **Demonstrate** sensitive dependence on initial conditions in the double pendulum via
   a twin-trajectory experiment, and estimate its Lyapunov exponent
5. **Connect** the dimension-counting argument here to the Lorenz 63 system

---

### How to read this notebook

| Symbol | Meaning |
|--------|---------|
| ⚙️ **Controls** | Sliders you manipulate |
| 📐 **Theory** | Background equations and concepts |
| 🔬 **Experiment** | Step-by-step activity |
| 💡 **Observation** | Live readout that updates as you explore |

---

### Notebook structure

| Section | Topic | Key concept |
|---------|-------|-------------|
| **1** | The single pendulum | Closed orbits, energy conservation, never chaotic |
| **2** | The double pendulum | Sensitive dependence, Lyapunov exponent, chaos |
| **3** | Comparison | The Poincaré–Bendixson dimension argument |
| **📝** | Guided questions | Synthesis |
""")
    return


# ===========================================================================
# Historical context
# ===========================================================================
@app.cell
def cell_pendulum_story(mo):
    mo.md(r"""
---
### 🕰️ Historical context: from clocks to chaos

Christiaan Huygens analysed the pendulum's motion in the 1650s–70s to build the
first accurate pendulum clocks, and derived the small-angle period formula
$T = 2\pi\sqrt{L/g}$ that is still taught today. For nearly two centuries the
pendulum was physics' best example of *perfectly predictable* motion — exactly
the kind of system 19th-century determinism was built on.

The double pendulum tells the opposite story. Its equations of motion follow
from classical mechanics known since Lagrange, but its chaotic behaviour was
only fully *appreciated* once digital computers made it possible to integrate
the equations for long enough, and to compare two barely-different initial
conditions — the same twin-experiment method used in Section 2 below, and the
same one Lorenz used to discover chaos in his weather model (see the
<a href="/part2/ch06_lorenz63/" target="_top">Lorenz 63 chapter</a>). Poincaré had
already found this sensitive dependence, in the 3-body problem, seventy years
before Lorenz — but without a computer to visualise it, the result stayed a
mathematical curiosity for decades.

Today the double pendulum is one of the most common classroom demonstrations
of deterministic chaos — cheap to build, easy to film, and its divergence is
visible to the naked eye in under a minute.
""")
    return


# ===========================================================================
# Section 1 — The single pendulum: theory
# ===========================================================================
@app.cell
def display_section1_text(mo):
    mo.md(r"""
---
## 1 · The Single Pendulum — Regular, Not Chaotic

### Equations of motion

A point mass $m$ on a massless rigid rod of length $L$, released from rest at angle
$\theta_0$ from the vertical, obeys Newton's second law for rotation:

$$\frac{d\theta}{dt} = \omega \qquad \frac{d\omega}{dt} = -\frac{g}{L}\sin\theta$$

This is the **full, exact, nonlinear** equation — no approximation has been made yet.

### The small-angle (linear) approximation

For $|\theta| \ll 1$ rad, $\sin\theta \approx \theta$, and the equation becomes **linear**:

$$\frac{d\omega}{dt} \approx -\frac{g}{L}\theta
\qquad\Longrightarrow\qquad
\ddot\theta + \omega_0^2\,\theta = 0,\qquad \omega_0 = \sqrt{g/L}$$

This is simple harmonic motion (SHM): $\theta(t) = \theta_0\cos(\omega_0 t)$, with a
period $T_0 = 2\pi/\omega_0$ that is **independent of amplitude** — the classic (and
only approximately true) statement that "the pendulum period doesn't depend on how
far you pull it back."

### Why the *exact* nonlinear pendulum is still never chaotic

The single pendulum conserves energy per unit mass,

$$E = \tfrac12\omega^2 - \frac{g}{L}\cos\theta \;=\; \text{constant},$$

so every trajectory is confined to one energy contour in the 2-D phase plane
$(\theta,\omega)$ — a **closed curve** (for bounded swinging motion) or a smoothly
periodic open curve (if the pendulum has enough energy to go over the top). Two
trajectories starting on nearby contours **stay on nearby contours forever** —
nothing here can grow exponentially, however large the amplitude.

This is not a special property of the pendulum — it is forced by the
**Poincaré–Bendixson theorem**: a continuous-time autonomous system with only a
**2-dimensional** phase space cannot have chaotic trajectories. Bounded orbits in
2-D can only approach a fixed point, approach a periodic orbit, or (as here) trace
out a one-parameter family of periodic curves. There simply is not enough room in a
2-D phase space for trajectories to stretch and fold without crossing themselves —
and trajectories of an autonomous ODE are never allowed to cross.
""")
    return


# ===========================================================================
# Section 1 — interactive phase portrait
# ===========================================================================
@app.cell
def display_section1_interactive(
    G, L_SINGLE, go, integrate, mo, np, single_pendulum_rhs, systems,
    theta0_deg,
):
    _K = G / L_SINGLE  # = omega_0^2
    _omega0 = np.sqrt(_K)
    _theta0 = np.deg2rad(theta0_deg.value)
    _theta_grid = np.linspace(-2 * np.pi, 2 * np.pi, 800)

    def _libration_branch(theta0):
        _mask = np.cos(_theta_grid) >= np.cos(theta0)
        _w = np.full_like(_theta_grid, np.nan)
        _w[_mask] = np.sqrt(2 * _K * (np.cos(_theta_grid[_mask]) - np.cos(theta0)))
        return _w

    _fig = go.Figure()

    # reference libration contours at fixed amplitudes
    for _th0_deg_ref in (20, 60, 100, 140, 178):
        _w = _libration_branch(np.deg2rad(_th0_deg_ref))
        for _sign in (1, -1):
            _fig.add_trace(go.Scatter(
                x=np.rad2deg(_theta_grid), y=_sign * np.rad2deg(_w),
                mode="lines", line=dict(color="rgba(140,140,140,0.35)", width=1),
                showlegend=False, hoverinfo="skip",
            ))

    # separatrix (theta0 -> 180 deg)
    _w_sep = 2 * _omega0 * np.abs(np.cos(_theta_grid / 2))
    for _sign, _leg in ((1, True), (-1, False)):
        _fig.add_trace(go.Scatter(
            x=np.rad2deg(_theta_grid), y=_sign * np.rad2deg(_w_sep),
            mode="lines", line=dict(color="firebrick", width=1.5, dash="dash"),
            name="Separatrix (θ₀ → 180°)", showlegend=_leg, hoverinfo="skip",
        ))

    # rotation curves (energy above the separatrix — pendulum goes over the top)
    for _f in (0.3, 0.9):
        _E = _K * (1 + _f)
        _w_rot = np.sqrt(np.maximum(2 * (_E + _K * np.cos(_theta_grid)), 0))
        for _sign in (1, -1):
            _fig.add_trace(go.Scatter(
                x=np.rad2deg(_theta_grid), y=_sign * np.rad2deg(_w_rot),
                mode="lines", line=dict(color="rgba(100,150,255,0.35)", width=1, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

    # small-angle (SHM) ellipse, for the *selected* amplitude
    _theta_lin = np.linspace(-_theta0, _theta0, 200)
    _w_lin = _theta0 * _omega0 * np.sqrt(np.maximum(1 - (_theta_lin / _theta0) ** 2, 0))
    for _sign, _leg in ((1, True), (-1, False)):
        _fig.add_trace(go.Scatter(
            x=np.rad2deg(_theta_lin), y=_sign * np.rad2deg(_w_lin),
            mode="lines", line=dict(color="orange", width=1.5, dash="dot"),
            name="Small-angle (SHM) approximation", showlegend=_leg,
        ))

    # exact selected orbit, analytic curve
    _w_exact = _libration_branch(_theta0)
    for _sign, _leg in ((1, True), (-1, False)):
        _fig.add_trace(go.Scatter(
            x=np.rad2deg(_theta_grid), y=_sign * np.rad2deg(_w_exact),
            mode="lines", line=dict(color="royalblue", width=3),
            name=f"Exact orbit, θ₀ = {theta0_deg.value}°", showlegend=_leg,
        ))

    # numerically-integrated trajectory overlaid — confirms it traces the same curve
    _T_exact = systems.pendulum_period_exact(_theta0, g=G, length=L_SINGLE)
    _t_eval = np.linspace(0, 1.5 * _T_exact, 300)
    _traj = integrate.solve(
        single_pendulum_rhs, [_theta0, 0.0], _t_eval, rtol=1e-10, atol=1e-12
    )
    _fig.add_trace(go.Scatter(
        x=np.rad2deg(_traj[:, 0]), y=np.rad2deg(_traj[:, 1]),
        mode="markers", marker=dict(size=3, color="black", opacity=0.5),
        name="Numerically integrated trajectory",
    ))

    _fig.update_layout(
        height=520,
        title=dict(
            text="Single-pendulum phase portrait  (θ, ω) — every orbit is a closed curve",
            x=0.5, font_size=13,
        ),
        xaxis=dict(title="θ (degrees)", range=[-360, 360], gridcolor="#ebebeb"),
        yaxis=dict(title="ω (degrees/s)", gridcolor="#ebebeb"),
        margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.01, font_size=10, bgcolor="rgba(255,255,255,0.88)"),
    )

    _T_lin = 2 * np.pi / _omega0

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        theta0_deg,
        _fig,
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; θ₀ = {theta0_deg.value}°  \n"
                f"Small-angle period estimate: **{_T_lin:.3f} s**  \n"
                f"Exact period (elliptic integral): **{_T_exact:.3f} s**  \n"
                f"Exact period is **{100 * (_T_exact / _T_lin - 1):.1f}%** longer than the SHM estimate  \n"
                f"🟢 Bounded, closed orbit at every amplitude — this system cannot become chaotic."
            ),
            kind="success",
        ),
    ])
    return


# ===========================================================================
# Section 1 — callout
# ===========================================================================
@app.cell
def display_section1_callout(mo):
    mo.callout(
        mo.md(r"""
**Try it:** drag θ₀ all the way to 175° — right up against the unstable equilibrium at
θ = 180°. Even here, arbitrarily close to the separatrix, the orbit is still a single
closed curve, and the period simply grows (logarithmically) longer — it never becomes
irregular or sensitive to initial conditions. A 1-DOF conservative system is *always*
regular, no matter how nonlinear its equation of motion is.
"""),
        kind="info",
    )
    return


# ===========================================================================
# Section 2 — The double pendulum: theory
# ===========================================================================
@app.cell
def display_section2_text(mo):
    mo.md(r"""
---
## 2 · The Double Pendulum — Chaotic

### Setup

Attach a second pendulum (mass $m_2$, rod length $L_2$) to the bob of the first
(mass $m_1$, rod length $L_1$). The state is now four numbers,
$(\theta_1,\theta_2,\omega_1,\omega_2)$ — the phase space is **4-dimensional**.

### Equations of motion

Applying the Euler–Lagrange equations to the double-pendulum Lagrangian gives the
exact (no small-angle approximation) coupled nonlinear system:

$$\dot\theta_1=\omega_1,\qquad \dot\theta_2=\omega_2$$

$$\dot\omega_1=\frac{-g(2m_1+m_2)\sin\theta_1-m_2 g\sin(\theta_1-2\theta_2)
-2\sin(\theta_1-\theta_2)\,m_2\!\left(\omega_2^2 L_2+\omega_1^2 L_1\cos(\theta_1-\theta_2)\right)}
{L_1\left(2m_1+m_2-m_2\cos(2\theta_1-2\theta_2)\right)}$$

$$\dot\omega_2=\frac{2\sin(\theta_1-\theta_2)\left(\omega_1^2 L_1(m_1+m_2)+g(m_1+m_2)\cos\theta_1
+\omega_2^2 L_2 m_2\cos(\theta_1-\theta_2)\right)}
{L_2\left(2m_1+m_2-m_2\cos(2\theta_1-2\theta_2)\right)}$$

These equations look intimidating, but the physics is simple: each rod's angular
acceleration depends on gravity, on the *other* rod's angle and angular velocity
(through the $\sin(\theta_1-\theta_2)$ coupling terms), and on centripetal feedback
($\omega^2$ terms). It is exactly this coupling that breaks the argument used for
the single pendulum above.

### Why this system *can* be chaotic

The double pendulum still conserves total energy, but energy conservation only
removes **one** dimension from a **4-dimensional** phase space — leaving a
3-dimensional energy surface for trajectories to wander on. That is enough room for
trajectories to stretch and fold without crossing — exactly the mechanism behind the
Lorenz attractor in the
<a href="/part2/ch06_lorenz63/" target="_top">Lorenz 63 chapter</a>.
The Poincaré–Bendixson restriction that protected the single pendulum **does not
apply** once the phase space has 3 or more dimensions.

Whether a *given* double-pendulum trajectory is actually chaotic depends on its
energy: at low energy (small swings) the motion is close to two decoupled linear
oscillators and looks regular; at high energy — especially anywhere near
$\theta_1,\theta_2=\pm180°$ — the motion is almost always chaotic.
""")
    return


# ===========================================================================
# Section 2 — experiment
# ===========================================================================
@app.cell
def display_section2_experiment(mo):
    mo.callout(
        mo.md(r"""
**🔬 Experiment — find the transition from regular to chaotic:**

1. Set θ₁ = θ₂ = 20° (small swings) and δ₀ = 10⁻⁶.  Watch the phase portrait and the
   separation plot.  Do the two trajectories stay together?
2. Now set θ₁ = 120°, θ₂ = -10°, keeping δ₀ = 10⁻⁶ (the same tiny perturbation).
   What happens to the separation curve now?
3. With θ₁ = 120° fixed, slide δ₀ from 10⁻⁸ up to 10⁻³.  Does starting *closer*
   together delay the divergence, or prevent it entirely?
4. Try θ₁ = θ₂ = 170° — close to the most energetic configuration available.
   How quickly do the trajectories separate compared to case 2?
"""),
        kind="neutral",
    )
    return


# ===========================================================================
# Section 2 — interactive twin-trajectory experiment
# ===========================================================================
@app.cell
def display_section2_interactive(
    G, L1, L2, dp_lead_time, dp_pert_exp, dp_theta1_deg, dp_theta2_deg,
    double_pendulum_rhs, go, integrate, mo, np,
):
    _tau = np.sqrt((L1 + L2) / G)  # characteristic timescale, used to non-dimensionalize omega

    _th1_0 = np.deg2rad(dp_theta1_deg.value)
    _th2_0 = np.deg2rad(dp_theta2_deg.value)
    _delta0 = 10.0 ** dp_pert_exp.value
    _T = dp_lead_time.value
    _t_eval = np.linspace(0, _T, 2000)

    _s0_a = np.array([_th1_0, _th2_0, 0.0, 0.0])
    _s0_b = _s0_a + np.array([_delta0, 0.0, 0.0, 0.0])

    _traj_a = integrate.solve(
        double_pendulum_rhs, _s0_a, _t_eval, rtol=1e-10, atol=1e-12
    )
    _traj_b = integrate.solve(
        double_pendulum_rhs, _s0_b, _t_eval, rtol=1e-10, atol=1e-12
    )

    _th1_a, _th2_a, _w1_a, _w2_a = _traj_a.T
    _th1_b, _th2_b, _w1_b, _w2_b = _traj_b.T

    # full 4-D state separation, non-dimensionalized (angles in rad, rates scaled by tau)
    _sep = np.sqrt(
        (_th1_a - _th1_b) ** 2 + (_th2_a - _th2_b) ** 2
        + (_tau * (_w1_a - _w1_b)) ** 2 + (_tau * (_w2_a - _w2_b)) ** 2
    )

    # Lyapunov estimate from early exponential growth
    _early = (_t_eval < _T / 3) & (_sep > 0)
    if _early.sum() > 5 and _sep[_early][0] > 0:
        _log_sep = np.log(_sep[_early] + 1e-20)
        _slope, _ = np.polyfit(_t_eval[_early], _log_sep, 1)
        _lambda_est = float(_slope)
        _lambda_str = f"{_lambda_est:.3f} s⁻¹" if _lambda_est > 0 else "≈ 0 (no measurable growth yet)"
    else:
        _lambda_est = None
        _lambda_str = "—"

    # ---- phase-space figure: (theta1, omega1) projection of the full 4-D state ----
    _fig_phase = go.Figure()
    _fig_phase.add_trace(go.Scatter(
        x=np.rad2deg(_th1_a), y=np.rad2deg(_w1_a),
        mode="lines", line=dict(color="royalblue", width=2), name="Baseline (A)",
    ))
    _fig_phase.add_trace(go.Scatter(
        x=np.rad2deg(_th1_b), y=np.rad2deg(_w1_b),
        mode="lines", line=dict(color="crimson", width=2), name="Perturbed (B)",
    ))
    _fig_phase.add_trace(go.Scatter(
        x=[np.rad2deg(_th1_a[0])], y=[np.rad2deg(_w1_a[0])],
        mode="markers", marker=dict(size=9, color="limegreen"),
        name=f"Shared start (δ₀ = 10^{dp_pert_exp.value:.1f})",
    ))
    _fig_phase.update_layout(
        height=480,
        title=dict(text="Double-pendulum phase portrait — (θ₁, ω₁) projection of the 4-D state",
                   x=0.5, font_size=12),
        xaxis=dict(title="θ₁ (degrees)", gridcolor="#ebebeb"),
        yaxis=dict(title="ω₁ (degrees/s)", gridcolor="#ebebeb"),
        margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.01, font_size=10, bgcolor="rgba(255,255,255,0.88)"),
    )

    # ---- separation figure ----
    _fig_sep = go.Figure()
    _fig_sep.add_trace(go.Scatter(
        x=_t_eval, y=_sep, mode="lines",
        line=dict(color="darkorchid", width=2.5),
        fill="tozeroy", fillcolor="rgba(148,0,211,0.07)",
        name="|state A − state B|",
    ))
    if _lambda_est and _lambda_est > 0:
        _t_ref = _t_eval[_early]
        _fig_sep.add_trace(go.Scatter(
            x=_t_ref, y=_sep[_early][0] * np.exp(_lambda_est * (_t_ref - _t_ref[0])),
            mode="lines", line=dict(color="orange", width=1.5, dash="dot"),
            name=f"e^(λt),  λ ≈ {_lambda_est:.3f} s⁻¹",
        ))
    _fig_sep.update_layout(
        height=480,
        title=dict(text="Separation between the two trajectories (log scale)", x=0.5, font_size=12),
        xaxis=dict(title="Time (s)", gridcolor="#ebebeb"),
        yaxis=dict(title="Non-dimensional state separation", type="log", gridcolor="#ebebeb"),
        margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="white",
        legend=dict(x=0.01, y=0.99, font_size=10, bgcolor="rgba(255,255,255,0.88)"),
    )

    if _lambda_est is None:
        _regime = "⚪ Not enough early-time samples to estimate λ — try a longer lead time."
        _ck = "neutral"
    elif _lambda_est > 0.05:
        _regime = "🔴 Exponential divergence detected — this trajectory pair is chaotic."
        _ck = "danger"
    elif _lambda_est > 0.0:
        _regime = "🟠 Slow growth — near the edge between regular and chaotic."
        _ck = "warn"
    else:
        _regime = "🟢 No measurable divergence yet — on this time horizon this looks like a regular (non-chaotic) trajectory."
        _ck = "success"

    mo.vstack([
        mo.md("### ⚙️ Controls"),
        mo.hstack([dp_theta1_deg, dp_theta2_deg], gap="3rem"),
        mo.hstack([dp_pert_exp, dp_lead_time], gap="3rem"),
        mo.hstack([_fig_phase, _fig_sep], widths=[1, 1]),
        mo.callout(
            mo.md(
                f"**💡 Live readout** &nbsp;|&nbsp; "
                f"θ₁₀ = {dp_theta1_deg.value}°, θ₂₀ = {dp_theta2_deg.value}° &nbsp;·&nbsp; "
                f"δ₀ = 10^{dp_pert_exp.value:.1f} rad &nbsp;·&nbsp; T = {_T} s  \n"
                f"Final separation: **{_sep[-1]:.3g}** (non-dimensional)  \n"
                f"Estimated Lyapunov exponent λ: **{_lambda_str}**  \n"
                f"{_regime}"
            ),
            kind=_ck,
        ),
    ])
    return


# ===========================================================================
# Section 3 — comparison
# ===========================================================================
@app.cell
def display_section3_comparison(mo):
    mo.md(r"""
---
## 3 · Why One Pendulum Is Never Chaotic and Two (Almost) Always Can Be

| | Single pendulum | Double pendulum |
|---|---|---|
| Degrees of freedom | 1 | 2 |
| Phase-space dimension | 2  $(\theta,\omega)$ | 4  $(\theta_1,\theta_2,\omega_1,\omega_2)$ |
| Energy surface dimension | 1 (a curve) | 3 (a volume) |
| Governing equations | Linear *only* for small angle | Nonlinear at *every* amplitude |
| Integrable? | Yes, always (closed-form via elliptic integrals) | No, in general |
| Lyapunov exponent λ | Exactly 0 (periodic/quasi-periodic) | > 0 for most high-energy initial conditions |
| Long-term predictability | Perfect, at any amplitude | Bounded by the same SDIC argument as <a href="/part2/ch06_lorenz63/" target="_top">Lorenz 63</a> |

**The general principle** (Poincaré–Bendixson theorem): a continuous-time
autonomous system needs **at least 3 phase-space dimensions** to be chaotic. The
single pendulum is stuck at 2 — no amount of nonlinearity can rescue it. The
double pendulum's extra hinge buys the third (and fourth) dimension, and that is
the *entire* difference between perfect predictability and chaos. This is the same
dimensional argument that makes the Lorenz 63 system (3 dimensions, $X,Y,Z$) the
*minimal* continuous chaotic system — see the
<a href="/part2/ch06_lorenz63/" target="_top">Lorenz 63 chapter</a>
for a full treatment of what happens once a system crosses that threshold.
""")
    return


# ===========================================================================
# Guided Questions
# ===========================================================================
@app.cell
def display_questions(mo):
    mo.md(r"""
---
## 📝 Guided Questions

---

**Q1 — Dimension counting**

- How many numbers does it take to fully specify the *instantaneous* state of a single
  pendulum?  Of a double pendulum?
- Using the Poincaré–Bendixson theorem, explain in one sentence why the single pendulum
  can never be chaotic, however large its amplitude.

---

**Q2 — Amplitude and period** *(Section 1)*

Slide θ₀ from 10° to 170°.

- At what amplitude does the exact period first differ from the small-angle estimate by
  more than 10%?
- Does the exact orbit ever stop being a single closed curve in the phase portrait?

---

**Q3 — Finding the chaotic threshold** *(Section 2)*

Fix δ₀ = 10⁻⁶.  Starting from θ₁ = θ₂ = 10°, slowly increase both angles together.

- Roughly where does the separation curve change from "flat" to "exponential growth"?
- Is there a sharp threshold, or a gradual transition?

---

**Q4 — Comparing to Lorenz 63**

The <a href="/part2/ch06_lorenz63/" target="_top">Lorenz 63 chapter</a> estimates a Lyapunov
exponent λ ≈ 0.9 MTU⁻¹ (dimensionless model time). Here you estimated λ in s⁻¹ for a
physical double pendulum.

- Why can't these two λ values be compared directly without knowing a real physical
  pendulum's rod length and the atmosphere's characteristic timescale?
- What do the two systems have in common structurally, despite being physically
  unrelated?

---

*Single pendulum: $g/L = 9.81\ \text{s}^{-2}$.  Double pendulum: $m_1=m_2=L_1=L_2=1$ (SI units).*
*Integration: RK45 (scipy), rtol = 10⁻¹⁰.*
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

### Papers

- **Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992)**. *Chaos in a double
  pendulum.* American Journal of Physics, 60(6), 491–499. The classic, accessible
  treatment of exactly this system.
- **Poincaré, H. (1890)**. *Sur le problème des trois corps et les équations de la
  dynamique.* The original discovery of sensitive dependence, in the three-body
  problem — predating Lorenz by seventy years.

### Books

- **Strogatz, S. H. (2015)**. *Nonlinear Dynamics and Chaos.* Westview Press.
  Chapter 6 covers the Poincaré–Bendixson theorem in full; Chapter 9 covers the
  Lorenz system.
- **Tabor, M. (1989)**. *Chaos and Integrability in Nonlinear Dynamics.* Wiley.
  A rigorous treatment of why some systems (like the single pendulum) are
  integrable and others are not.
- **Baker, G. L. & Blackburn, J. A. (2005)**. *The Pendulum: A Case Study in
  Physics.* Oxford University Press. A whole book on how much physics the humble
  pendulum contains — through to the chaotic double pendulum.

---
*Notebook by Aneesh C. Subramanian — Chaos and Predictability.*
*Built with [marimo](https://marimo.io), [NumPy](https://numpy.org), [SciPy](https://scipy.org), [Plotly](https://plotly.com).*
""")
    return


if __name__ == "__main__":
    app.run()
