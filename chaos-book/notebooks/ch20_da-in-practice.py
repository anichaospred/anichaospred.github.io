# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "plotly",
# ]
# ///
"""Chapter 20 -- Data assimilation in practice.

3D-Var, 4D-Var and the ensemble Kalman filter on Lorenz 63, and the
logarithmic return on better observations.

Part V of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.

To edit:   marimo edit notebooks/ch20_da-in-practice.py
To export: make nb-one NB=ch20_da-in-practice
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 20: Data Assimilation in Practice")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
@app.cell
async def imports():
    import marimo as mo

    import sys

    if sys.platform == "emscripten":
        # Browser (Pyodide/WASM): install the chaoslib wheel that `make notebooks`
        # ships in the export's shared public/ folder.
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
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from chaoslib import assimilate, integrate, lyapunov, plotting, systems

    return (
        assimilate,
        go,
        integrate,
        lyapunov,
        make_subplots,
        mo,
        np,
        plotting,
        systems,
    )


# ---------------------------------------------------------------------------
# Experiment configuration and the twin experiment
# ---------------------------------------------------------------------------
@app.cell
def config():
    # Lorenz 63 at the classical parameters -- chaotic, and the system every
    # earlier chapter in Part III used.
    SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0

    DT_MODEL = 0.01   # model time step, MTU
    T_SPINUP = 5.0    # spin-up onto the attractor
    T_ASSIM = 5.0     # assimilation experiment length
    DT_OBS = 0.2      # observation interval, MTU
    OBS_SIGMA = 2.0   # observation error std, per component
    BG_SIGMA = 3.0    # background error std, per component
    return (
        BETA,
        BG_SIGMA,
        DT_MODEL,
        DT_OBS,
        OBS_SIGMA,
        RHO,
        SIGMA,
        T_ASSIM,
        T_SPINUP,
    )


@app.cell
def helpers(BETA, DT_MODEL, RHO, SIGMA, integrate, np, systems):
    def propagate(x0, t_end, dt=DT_MODEL):
        """Integrate Lorenz 63 from x0 for t_end, returning the trajectory.

        Adaptive RK45 at tight tolerance via chaoslib. The returned array is
        time-leading, and lands exactly on t_end so that assimilation windows
        and tangent-linear propagators agree on the interval.
        """
        if t_end <= 0.0:
            return np.atleast_2d(np.asarray(x0, dtype=float))
        n = max(2, int(round(t_end / dt)) + 1)
        return integrate.solve(
            systems.lorenz63,
            np.asarray(x0, dtype=float),
            np.linspace(0.0, t_end, n),
            rtol=1e-9,
            atol=1e-12,
            sigma=SIGMA,
            rho=RHO,
            beta=BETA,
        )

    def rmse(a, b):
        """Root-mean-square error per component between two states."""
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    return propagate, rmse


@app.cell
def twin_experiment(
    BG_SIGMA, DT_MODEL, DT_OBS, OBS_SIGMA, T_ASSIM, T_SPINUP, np, propagate
):
    # A fixed seed: every number quoted in the text below is reproducible, and a
    # reader who changes it should expect the RMSE values to move by ~10%.
    _rng = np.random.default_rng(42)

    # ---- spin-up: land on the attractor before the experiment starts ----
    x_start = propagate(np.array([1.0, 1.0, 20.0]), T_SPINUP)[-1]

    # ---- nature run: this is "the truth", known only to us ----
    n_steps = int(round(T_ASSIM / DT_MODEL)) + 1
    t_assim = np.linspace(0.0, T_ASSIM, n_steps)
    truth = propagate(x_start, T_ASSIM)

    # ---- observations: the truth, sampled sparsely and corrupted ----
    obs_idx = np.arange(0, n_steps, int(round(DT_OBS / DT_MODEL)))
    obs_times = t_assim[obs_idx]
    H = np.eye(3)  # all three components observed; H = I keeps the algebra visible
    R = np.eye(3) * OBS_SIGMA**2
    observations = truth[obs_idx] + _rng.normal(
        0.0, OBS_SIGMA, (obs_idx.size, 3)
    )

    # ---- background: a deliberately wrong first guess ----
    x_bg = x_start + _rng.normal(0.0, BG_SIGMA, 3)
    B = np.eye(3) * BG_SIGMA**2

    bg_error_0 = float(np.linalg.norm(x_bg - x_start))
    return (
        B,
        H,
        R,
        bg_error_0,
        n_steps,
        obs_idx,
        obs_times,
        observations,
        t_assim,
        truth,
        x_bg,
        x_start,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 20 · Data Assimilation in Practice

    **Part V — The machinery of prediction.**

    **The forecasting question.** Every forecast starts from a state nobody knows.
    The atmosphere is observed at scattered points, by instruments that disagree,
    at times that do not line up — and from that, a forecast centre has to produce
    a complete, physically consistent initial condition for a model with a hundred
    million degrees of freedom. Data assimilation is how. This chapter runs the
    three algorithms that do it, on a system small enough to watch, and then asks
    the question that decides observing-system policy: **if you could make every
    observation ten times more accurate, how much forecast would you buy?**

    The answer is not "ten times more". It is a fixed number of days — and then
    the next factor of ten buys the same fixed number again.
    """
    )
    return


@app.cell(hide_code=True)
def orientation(mo):
    mo.md(
        r"""
    > *"All models are wrong, but some are useful — and with data assimilation, we
    > can make them more useful."*

    Data assimilation (DA) is the discipline of **combining a dynamical model with
    noisy observations** to obtain the best possible estimate of the true state of
    a system. It sits at the heart of numerical weather prediction, ocean
    reanalysis, and increasingly of hybrid AI/physics forecasting systems.

    Three foundational algorithms, on the Lorenz 63 system as a low-dimensional but
    genuinely chaotic testbed:

    | Algorithm | Temporal scope | Background covariance |
    |-----------|---------------|----------------------|
    | **3D-Var** | Single analysis time | Static $\mathbf{B}$ |
    | **4D-Var** | Time window $[t_0, t_N]$ | Static $\mathbf{B}$, implicit flow-dependence |
    | **EnKF**  | Sequential / cycling | Flow-dependent, from the ensemble |

    ### What you need before this chapter

    - Lorenz 63 and sensitive dependence — chapter 6.
    - Error growth and saturation — chapter 9.
    - The tangent linear and adjoint models — chapter 15. 4D-Var *is* the adjoint,
      put to work.
    - Ensembles and spread–skill — chapter 17.

    Everything numerical here comes from `chaoslib.assimilate`, which is tested
    against the linear-Gaussian Kalman filter, against the adjoint identity, and —
    for the 4D-Var gradient — against central differences to one part in $10^9$.
    """
    )
    return


# ===========================================================================
# 1. The twin experiment
# ===========================================================================
@app.cell(hide_code=True)
def sec1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · The twin-experiment framework

    We work in a **perfect-model, twin-experiment** setup — the standard way to test
    an assimilation scheme, because it is the only setting in which the truth is
    actually known:

    1. **Nature run** (truth): integrate Lorenz 63 from a known initial condition.
    2. **Observations**: sample the truth every $\Delta t_{\rm obs}$ and add Gaussian
       noise $\mathcal{N}(0, \mathbf{R})$.
    3. **Background** $\mathbf{x}^b$: a deliberately perturbed initial condition,
       standing in for our prior knowledge.
    4. **Analysis** $\mathbf{x}^a$: what the algorithm produces, and what we score.

    "Perfect model" means the same equations generate the truth and the forecast, so
    every error we measure is an *initial-condition* error. Chapter 21 drops that
    assumption, and the picture changes.

    The symbols follow the notation page, which follows Kalnay (2003): $\mathbf{x}^b$
    background, $\mathbf{x}^a$ analysis, $\mathbf{y}$ observations, $\mathbf{H}$ the
    observation operator, $\mathbf{B}$ and $\mathbf{R}$ the background and observation
    error covariances, and innovations $d = \mathbf{y} - \mathbf{H}\mathbf{x}^b$.
    """
    )
    return


@app.cell(hide_code=True)
def sec1_fig(
    bg_error_0, go, mo, np, obs_idx, observations, plotting, propagate, truth, x_bg
):
    _fig = go.Figure()
    # Faint attractor for context, so the 5-MTU experiment is visibly a small
    # excursion on a much larger object.
    _ctx = propagate(np.array([1.0, 1.0, 20.0]), 60.0)[1000:]
    _fig.add_scatter3d(
        x=_ctx[:, 0], y=_ctx[:, 1], z=_ctx[:, 2], mode="lines",
        line=dict(width=1, color=plotting.C_CONTEXT), name="attractor",
        hoverinfo="skip",
    )
    _fig.add_scatter3d(
        x=truth[:, 0], y=truth[:, 1], z=truth[:, 2], mode="lines",
        line=dict(width=5, color=plotting.C_TRUTH), name="truth (nature run)",
    )
    _fig.add_scatter3d(
        x=observations[:, 0], y=observations[:, 1], z=observations[:, 2],
        mode="markers",
        marker=dict(size=4, color=plotting.C_OBS, symbol="diamond"),
        name="observations",
    )
    # A free forecast from the wrong background: what DA has to beat.
    _free = propagate(x_bg, 5.0)
    _fig.add_scatter3d(
        x=_free[:, 0], y=_free[:, 1], z=_free[:, 2], mode="lines",
        line=dict(width=4, color=plotting.C_BG, dash="dash"),
        name="free forecast from background",
    )
    _fig.add_scatter3d(
        x=[truth[0, 0]], y=[truth[0, 1]], z=[truth[0, 2]], mode="markers",
        marker=dict(size=7, color=plotting.C_START), name="start",
    )
    plotting.style3d(_fig, height=520, title="The twin experiment on the L63 attractor")

    mo.vstack([
        _fig,
        mo.md(
            f"""Initial background error $\\|\\mathbf{{x}}^b - \\mathbf{{x}}^t\\|$ =
            **{bg_error_0:.3f}**, against a climatological spread of order 10 — so the
            background starts *close* to the truth and still diverges from it
            completely within a few MTU. That divergence, not the initial error, is
            what makes assimilation necessary: without repeated correction the grey
            forecast is worthless long before the experiment ends."""
        ),
    ])
    return


# ===========================================================================
# 2. 3D-Var
# ===========================================================================
@app.cell(hide_code=True)
def sec2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · 3D-Var — three-dimensional variational assimilation

    ### Theory

    3D-Var finds the analysis $\mathbf{x}^a$ by minimising the **cost function**

    $$
    \mathcal{J}(\mathbf{x}) =
    \underbrace{\frac{1}{2}(\mathbf{x} - \mathbf{x}^b)^T \mathbf{B}^{-1}(\mathbf{x} - \mathbf{x}^b)}_{\mathcal{J}_b \text{ — background term}}
    +
    \underbrace{\frac{1}{2}(\mathbf{y} - \mathbf{H}\mathbf{x})^T \mathbf{R}^{-1}(\mathbf{y} - \mathbf{H}\mathbf{x})}_{\mathcal{J}_o \text{ — observation term}}
    $$

    The two terms are a tug of war: stay near what you believed, stay near what you
    measured, each weighted by how much you trust it. The gradient the minimiser
    needs is

    $$
    \nabla_{\mathbf{x}} \mathcal{J} = \mathbf{B}^{-1}(\mathbf{x} - \mathbf{x}^b) - \mathbf{H}^T \mathbf{R}^{-1}(\mathbf{y} - \mathbf{H}\mathbf{x}),
    $$

    and setting it to zero gives, for linear $\mathbf{H}$, the closed form

    $$
    \mathbf{x}^a = \mathbf{x}^b + \mathbf{K}(\mathbf{y} - \mathbf{H}\mathbf{x}^b), \qquad
    \mathbf{K} = \mathbf{B} \mathbf{H}^T (\mathbf{H} \mathbf{B} \mathbf{H}^T + \mathbf{R})^{-1}.
    $$

    `chaoslib.assimilate.three_dvar_update` evaluates that closed form, and the
    library's tests check it against the Kalman filter analysis — they must agree
    exactly, because for the same $\mathbf{B}$ they are the same estimator.
    Operational systems minimise $\mathcal{J}$ iteratively instead, not because the
    algebra is wrong but because $\mathbf{H}$ is nonlinear and $\mathbf{B}$ is far too
    large to invert.

    ### Key limitations

    * **Atemporal**: uses observations at a single time — no memory of the trajectory.
    * **Static $\mathbf{B}$**: the same background covariance on every day of the
      year, whatever the flow is doing. This is the limitation that everything else
      in this chapter exists to fix.
    * Simple and fast — and still used operationally in many regional systems.
    """
    )
    return


@app.cell
def run_3dvar(
    B, H, R, assimilate, np, obs_idx, observations, propagate, rmse, t_assim, truth, x_bg
):
    def cycle_3dvar(x_background, obs, obs_indices, times):
        """Cycling 3D-Var: analyse, forecast to the next observation, repeat.

        The cycling is the part that matters operationally. A single analysis is
        just a weighted average; it is the repetition -- analysis, forecast,
        analysis -- that keeps a model locked onto reality for years at a time.
        """
        x_curr = np.asarray(x_background, dtype=float).copy()
        analyses, backgrounds = [], []
        for k, idx in enumerate(obs_indices):
            if k > 0:
                x_curr = propagate(
                    x_curr, float(times[idx] - times[obs_indices[k - 1]])
                )[-1]
            backgrounds.append((float(times[idx]), x_curr.copy()))
            x_curr = assimilate.three_dvar_update(x_curr, B, obs[k], H, R)
            analyses.append((float(times[idx]), x_curr.copy()))
        return analyses, backgrounds

    analyses_3dvar, backgrounds_3dvar = cycle_3dvar(
        x_bg, observations, obs_idx, t_assim
    )

    rmse_3dvar_an = float(
        np.sqrt(np.mean([
            (truth[obs_idx[k]] - xa) ** 2
            for k, (_, xa) in enumerate(analyses_3dvar)
        ]))
    )
    rmse_3dvar_bg = float(
        np.sqrt(np.mean([
            (truth[obs_idx[k]] - xb) ** 2
            for k, (_, xb) in enumerate(backgrounds_3dvar)
        ]))
    )
    return analyses_3dvar, backgrounds_3dvar, cycle_3dvar, rmse_3dvar_an, rmse_3dvar_bg


@app.cell
def component_panels(go, make_subplots, np, plotting):
    def component_figure(
        t_full, truth_full, obs_t, obs_v, series, title, height=560
    ):
        """Three stacked panels, one per state component: the book's DA figure.

        `series` is a list of (label, times, values, colour, mode) so each of the
        three algorithms below draws itself the same way -- the reader compares
        panels rather than relearning a layout.
        """
        _fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
            subplot_titles=("X", "Y", "Z"),
        )
        for _row, _c in enumerate([0, 1, 2], start=1):
            _show = _row == 1
            _fig.add_scatter(
                x=t_full, y=truth_full[:, _c], mode="lines",
                line=dict(width=2.2, color=plotting.C_TRUTH),
                name="truth", legendgroup="truth", showlegend=_show,
                row=_row, col=1,
            )
            _fig.add_scatter(
                x=obs_t, y=obs_v[:, _c], mode="markers",
                marker=dict(size=6, color=plotting.C_OBS, symbol="diamond-open",
                            line=dict(width=1.4)),
                name="observations", legendgroup="obs", showlegend=_show,
                row=_row, col=1,
            )
            for _label, _ts, _vs, _colour, _mode in series:
                _fig.add_scatter(
                    x=_ts, y=np.asarray(_vs)[:, _c], mode=_mode,
                    line=dict(width=2, color=_colour, dash="dot"
                              if _mode == "lines" and _label.startswith("background")
                              else "solid"),
                    marker=dict(size=7, color=_colour),
                    name=_label, legendgroup=_label, showlegend=_show,
                    row=_row, col=1,
                )
        plotting.style2d(_fig, height=height, title=title)
        _fig.update_xaxes(title_text="time (MTU)", row=3, col=1)
        return _fig

    return (component_figure,)


@app.cell(hide_code=True)
def fig_3dvar(
    analyses_3dvar, backgrounds_3dvar, component_figure, mo, np, obs_times,
    observations, plotting, rmse_3dvar_an, rmse_3dvar_bg, t_assim, truth
):
    _an_t = np.array([a[0] for a in analyses_3dvar])
    _an_v = np.array([a[1] for a in analyses_3dvar])
    _bg_t = np.array([b[0] for b in backgrounds_3dvar])
    _bg_v = np.array([b[1] for b in backgrounds_3dvar])

    _fig = component_figure(
        t_assim, truth, obs_times, observations,
        [
            ("background (before analysis)", _bg_t, _bg_v, plotting.C_BG, "markers"),
            ("3D-Var analysis", _an_t, _an_v, plotting.C_ANALYSIS, "markers"),
        ],
        "3D-Var: cycling analyses against the truth",
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""**Background RMSE {rmse_3dvar_bg:.3f} → analysis RMSE
            {rmse_3dvar_an:.3f}.** Each analysis pulls the state back towards the
            truth, and the forecast between observations pushes it away again. The
            analysis is never as good as the observations are accurate
            ($\\sigma_o = 2$) *and* never as bad as the background it started from —
            it sits between, which is exactly what a weighted average of the two
            should do."""
        ),
    ])
    return


# ===========================================================================
# 3. 4D-Var
# ===========================================================================
@app.cell(hide_code=True)
def sec3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · 4D-Var — four-dimensional variational assimilation

    ### Theory

    4D-Var extends 3D-Var to **a time window** $[t_0, t_N]$. We seek the initial
    condition $\mathbf{x}_0$ that, when propagated by the model $\mathcal{M}$, best
    fits *all* the observations in the window at once:

    $$
    \mathcal{J}(\mathbf{x}_0) =
    \frac{1}{2}(\mathbf{x}_0 - \mathbf{x}_0^b)^T \mathbf{B}^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b)
    +
    \sum_{k=0}^{N}
    \frac{1}{2}(\mathbf{y}_k - \mathbf{H}\mathbf{x}_k)^T \mathbf{R}^{-1}(\mathbf{y}_k - \mathbf{H}\mathbf{x}_k),
    \qquad \mathbf{x}_k = \mathcal{M}_{0 \to t_k}(\mathbf{x}_0).
    $$

    The analysis is no longer a state at one instant but a **trajectory** — the model
    is used as a constraint, so the answer is automatically dynamically consistent.

    ### The gradient is where the adjoint earns its keep

    $$
    \nabla_{\mathbf{x}_0}\mathcal{J} = \mathbf{B}^{-1}(\mathbf{x}_0 - \mathbf{x}_0^b)
    - \sum_k \mathbf{M}_{0 \to t_k}^{\!\top}\,\mathbf{H}^T \mathbf{R}^{-1}(\mathbf{y}_k - \mathbf{H}\mathbf{x}_k)
    $$

    Each innovation is carried back to the start of the window by the **adjoint**
    $\mathbf{M}^{\!\top}$ built in chapter 15. The cost is one adjoint application per
    observation time — *not* one model run per degree of freedom, which is what a
    finite-difference gradient would need. For Lorenz 63 that is the difference
    between 3 extra model runs and 1; for a global model it is the difference
    between feasible and impossible, and it is the whole reason 4D-Var exists.

    This chapter uses the **exact adjoint gradient**, via
    `chaoslib.assimilate.four_dvar_analysis`. The library's test suite checks it
    against central differences of the same cost function and requires agreement to
    better than one part in $10^7$; it currently agrees to about one part in $10^9$.
    That test is not decoration — it caught two real interval bugs in the propagator
    while this chapter was being written, either of which left the gradient about 7%
    wrong in a way no amount of tuning the minimiser would have revealed.

    ### Advantage over 3D-Var

    * Observations spread over a **window** all constrain one trajectory.
    * Flow-dependence arrives implicitly: the model's own dynamics propagate
      information from an observation late in the window back to the start.
    * The minimiser is quasi-Newton (L-BFGS-B). Fixed-step steepest descent is not a
      viable substitute here — with accurate observations $\mathbf{R}^{-1}$ is large,
      and any step big enough to converge sends the Lorenz trajectory to overflow.
    """
    )
    return


@app.cell
def run_4dvar(
    B, BETA, DT_MODEL, H, R, RHO, SIGMA, assimilate, np, obs_idx, observations,
    propagate, systems, t_assim, truth, x_bg
):
    WINDOW_SIZE = 5  # observations per 4D-Var window -> 0.8 MTU, ~1 Lyapunov time

    def cycle_4dvar(x_background, obs, obs_indices, times, window=WINDOW_SIZE):
        """Cycling 4D-Var: fit one window, advance to the next, repeat."""
        x_curr = np.asarray(x_background, dtype=float).copy()
        analyses = []
        n_obs = obs_indices.size
        for k in range(0, n_obs, window):
            end = min(k + window, n_obs)
            t0 = float(times[obs_indices[k]])
            # Observation times measured from the START of this window.
            window_obs = [
                (float(times[obs_indices[j]] - t0), obs[j]) for j in range(k, end)
            ]
            x0 = assimilate.four_dvar_analysis(
                systems.lorenz63,
                systems.lorenz63_jacobian,
                x_curr,
                B,
                window_obs,
                H,
                R,
                dt=DT_MODEL,
                max_iterations=60,
                sigma=SIGMA,
                rho=RHO,
                beta=BETA,
            )
            analyses.append((t0, x0.copy()))
            if end < n_obs:
                x_curr = propagate(
                    x0, float(times[obs_indices[end]] - t0)
                )[-1]
        return analyses

    analyses_4dvar = cycle_4dvar(x_bg, observations, obs_idx, t_assim)

    rmse_4dvar = float(
        np.sqrt(np.mean([
            (truth[obs_idx[k * WINDOW_SIZE]] - xa) ** 2
            for k, (_, xa) in enumerate(analyses_4dvar)
            if k * WINDOW_SIZE < obs_idx.size
        ]))
    )
    return WINDOW_SIZE, analyses_4dvar, cycle_4dvar, rmse_4dvar


@app.cell(hide_code=True)
def fig_4dvar(
    WINDOW_SIZE, analyses_4dvar, component_figure, mo, np, obs_times, observations,
    plotting, rmse_3dvar_an, rmse_4dvar, t_assim, truth
):
    _an_t = np.array([a[0] for a in analyses_4dvar])
    _an_v = np.array([a[1] for a in analyses_4dvar])
    _fig = component_figure(
        t_assim, truth, obs_times, observations,
        [("4D-Var analysis (window start)", _an_t, _an_v, plotting.C_ANALYSIS, "markers")],
        f"4D-Var: one analysis per {WINDOW_SIZE}-observation window",
    )
    _better = 100.0 * (rmse_3dvar_an - rmse_4dvar) / rmse_3dvar_an
    mo.vstack([
        _fig,
        mo.md(
            f"""**4D-Var analysis RMSE {rmse_4dvar:.3f}**, against
            {rmse_3dvar_an:.3f} for 3D-Var — about **{_better:.0f}% better** from the
            same observations and the same $\\mathbf{{B}}$.

            The improvement comes entirely from *when* the observations are used.
            3D-Var sees five observations one at a time and forgets each before the
            next; 4D-Var fits one trajectory through all five, so an observation at
            the end of the window constrains the state at the beginning. There are
            far fewer analysis points on this figure than on the last one, and the
            result is still better."""
        ),
    ])
    return


# ===========================================================================
# 4. Ensemble Kalman filter
# ===========================================================================
@app.cell(hide_code=True)
def sec4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · The ensemble Kalman filter

    ### Theory

    The EnKF (Evensen 1994) replaces the static $\mathbf{B}$ with a
    **flow-dependent** background error covariance estimated from an ensemble of $N$
    model trajectories:

    $$
    \mathbf{B}^f \approx \mathbf{P}^f = \frac{1}{N-1}\sum_{i=1}^{N}
    (\mathbf{x}^f_i - \bar{\mathbf{x}}^f)(\mathbf{x}^f_i - \bar{\mathbf{x}}^f)^T
    $$

    **Forecast step** — propagate each member through the full nonlinear model:
    $\mathbf{x}^f_i(t+1) = \mathcal{M}(\mathbf{x}^a_i(t))$.

    **Analysis step** (perturbed-observation form):

    $$
    \mathbf{x}^a_i = \mathbf{x}^f_i + \mathbf{K}(\mathbf{y}_i - \mathbf{H}\mathbf{x}^f_i),
    \quad \mathbf{y}_i = \mathbf{y} + \boldsymbol{\epsilon}_i,
    \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \mathbf{R}),
    \qquad
    \mathbf{K} = \mathbf{P}^f \mathbf{H}^T (\mathbf{H} \mathbf{P}^f \mathbf{H}^T + \mathbf{R})^{-1}
    $$

    Each member assimilates its *own* perturbed observation. Without that
    perturbation the analysis ensemble comes out systematically too narrow — it would
    report more confidence than it has earned.

    ### Key strengths

    * Fully nonlinear forecast step, and **no adjoint required** — which is why the
      EnKF spread to ocean, land and coupled systems long before variational methods
      did.
    * $\mathbf{P}^f$ adapts to the local geometry of the attractor, collapsing onto
      the unstable manifold where errors are actually growing.
    * It delivers an ensemble, so the forecast is probabilistic by construction
      rather than as an afterthought.

    ### Practical notes

    * **Ensemble size $N$ is critical.** Too small and $\mathbf{P}^f$ is rank
      deficient with spurious long-range correlations, and the filter diverges — it
      becomes so confident in a wrong background that it stops listening to the
      observations at all.
    * The standard remedies are **inflation** (multiply the spread by $1+\delta$) and
      **localisation** (taper distant covariances, e.g. `chaoslib.assimilate.gaspari_cohn`).
      Localisation does nothing in a 3-variable system where every variable is close
      to every other; in a global model it is indispensable.

    Move the sliders below. A larger ensemble costs a proportionally longer run, so
    $N = 50$ takes a noticeably longer time in the browser than $N = 20$.
    """
    )
    return


@app.cell(hide_code=True)
def enkf_controls(mo):
    n_members = mo.ui.slider(
        start=10, stop=50, step=10, value=20, label="ensemble size $N$"
    )
    # The range extends BELOW 1: this configuration comes out over-dispersed, so
    # inflation alone cannot reach reliability. Deflation is a diagnostic here,
    # not an operational technique -- see the discussion under section 6.
    inflation = mo.ui.slider(
        start=0.85, stop=1.30, step=0.05, value=1.05,
        label="inflation (below 1 = deflation)",
    )
    mo.hstack([n_members, inflation], justify="start", gap=2)
    return inflation, n_members


@app.cell
def run_enkf(
    BG_SIGMA, H, R, assimilate, inflation, n_members, np, obs_idx, observations,
    propagate, t_assim, truth, x_bg
):
    def cycle_enkf(n, alpha, seed=7):
        """Cycling perturbed-observation EnKF. Returns means, spreads, ensembles.

        The ensemble is centred on the BACKGROUND, not on the truth. That matters
        for fairness: 3D-Var and 4D-Var above both start from x_bg, so centring
        the ensemble on the true state would hand the EnKF a head start the other
        two do not get. (It turns out to make little difference here -- the filter
        forgets its initial condition within a few analyses -- but a comparison
        that needs that excuse is not a comparison.)
        """
        _rng = np.random.default_rng(seed)
        ens = np.asarray(x_bg, dtype=float) + _rng.normal(
            0.0, BG_SIGMA, (int(n), 3)
        )
        means, spreads, final = [], [], None
        prev = 0
        for k, idx in enumerate(obs_idx):
            seg = float(t_assim[idx] - t_assim[prev])
            if seg > 0:
                # One propagation per member. The forecast step is embarrassingly
                # parallel in principle -- here it is a Python loop, which is what
                # makes N the dominant cost in the browser.
                ens = np.array([propagate(m, seg)[-1] for m in ens])
            ens = assimilate.enkf_update(
                ens, observations[k], H, R, inflation=float(alpha), seed=1000 + k
            )
            means.append(ens.mean(axis=0))
            # Spread as RMS about the mean, per component -- directly comparable
            # with the per-component RMSE below.
            spreads.append(float(np.sqrt(np.mean(np.var(ens, axis=0, ddof=1)))))
            final = ens.copy()
            prev = idx
        return np.array(means), np.array(spreads), final

    enkf_means, enkf_spreads, enkf_final = cycle_enkf(
        n_members.value, inflation.value
    )
    enkf_errors = np.array([
        float(np.sqrt(np.mean((truth[obs_idx[k]] - m) ** 2)))
        for k, m in enumerate(enkf_means)
    ])
    rmse_enkf = float(np.sqrt(np.mean(enkf_errors**2)))
    return cycle_enkf, enkf_errors, enkf_final, enkf_means, enkf_spreads, rmse_enkf


@app.cell(hide_code=True)
def fig_enkf(
    component_figure, enkf_means, inflation, mo, n_members, np, obs_times,
    observations, plotting, rmse_3dvar_an, rmse_4dvar, rmse_enkf, t_assim, truth
):
    _fig = component_figure(
        t_assim, truth, obs_times, observations,
        [("EnKF ensemble mean", obs_times, enkf_means, plotting.C_MEAN, "markers")],
        f"EnKF: ensemble-mean analyses (N = {n_members.value}, "
        f"inflation = {inflation.value:.2f})",
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""**EnKF analysis RMSE {rmse_enkf:.3f}** — against
            {rmse_4dvar:.3f} (4D-Var) and {rmse_3dvar_an:.3f} (3D-Var).

            Do not read that ordering as a general ranking of the three methods. It
            is one realisation, on a 3-variable perfect-model problem where all
            components are observed at every analysis time — about the friendliest
            possible setting for an ensemble method, and one where localisation,
            the EnKF's main practical difficulty, cannot bite. Try $N = 10$ and
            inflation $1.0$ and watch the filter start to lose the truth."""
        ),
    ])
    return


# ===========================================================================
# 5. Side by side, and reliability
# ===========================================================================
@app.cell(hide_code=True)
def sec5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · Side by side

    All three on one axis. The quantity plotted is the per-component RMS error of
    each method's analysis against the truth, at every analysis time.
    """
    )
    return


@app.cell(hide_code=True)
def fig_comparison(
    WINDOW_SIZE, analyses_3dvar, analyses_4dvar, backgrounds_3dvar, enkf_errors, go,
    make_subplots, mo, np, obs_times, plotting, rmse_3dvar_an, rmse_4dvar, rmse_enkf,
    truth, obs_idx
):
    _err = lambda seq, idxs: np.array([
        float(np.sqrt(np.mean((truth[idxs[k]] - x) ** 2)))
        for k, (_, x) in enumerate(seq)
    ])
    _e_bg = _err(backgrounds_3dvar, obs_idx)
    _e_3d = _err(analyses_3dvar, obs_idx)
    _t_4d = np.array([a[0] for a in analyses_4dvar])
    _e_4d = np.array([
        float(np.sqrt(np.mean((truth[obs_idx[k * WINDOW_SIZE]] - xa) ** 2)))
        for k, (_, xa) in enumerate(analyses_4dvar)
        if k * WINDOW_SIZE < obs_idx.size
    ])

    _fig = make_subplots(
        rows=1, cols=2, column_widths=[0.66, 0.34],
        subplot_titles=("Analysis error through the experiment", "Mean over the experiment"),
    )
    _fig.add_scatter(x=obs_times, y=_e_bg, mode="lines+markers",
                     line=dict(color=plotting.C_BG, dash="dot", width=2),
                     name="background (no analysis)", row=1, col=1)
    _fig.add_scatter(x=obs_times, y=_e_3d, mode="lines+markers",
                     line=dict(color=plotting.C_PERT, width=2), name="3D-Var",
                     row=1, col=1)
    _fig.add_scatter(x=_t_4d, y=_e_4d, mode="lines+markers",
                     line=dict(color=plotting.C_ANALYSIS, width=2), name="4D-Var",
                     marker=dict(size=9), row=1, col=1)
    _fig.add_scatter(x=obs_times, y=enkf_errors, mode="lines+markers",
                     line=dict(color=plotting.C_MEAN, width=2), name="EnKF",
                     row=1, col=1)
    _fig.add_bar(
        x=["3D-Var", "4D-Var", "EnKF"],
        y=[rmse_3dvar_an, rmse_4dvar, rmse_enkf],
        marker_color=[plotting.C_PERT, plotting.C_ANALYSIS, plotting.C_MEAN],
        showlegend=False, row=1, col=2,
    )
    plotting.style2d(_fig, height=440, title="Three assimilation schemes, same observations")
    _fig.update_yaxes(title_text="analysis RMSE", type="log", row=1, col=1)
    _fig.update_xaxes(title_text="time (MTU)", row=1, col=1)
    _fig.update_yaxes(title_text="mean RMSE", row=1, col=2)

    mo.vstack([
        _fig,
        mo.md(
            """The left panel is logarithmic, because the background error and the
            analysis errors differ by more than an order of magnitude. Note that the
            background curve is *not* monotonic: it is the error of a forecast
            launched from the previous analysis, so it inherits whatever that
            analysis got wrong and then amplifies it at the local growth rate — which,
            as chapter 7 showed, varies by a factor of several around the attractor."""
        ),
    ])
    return


@app.cell(hide_code=True)
def sec6_text(mo):
    mo.md(
        r"""
    ---
    ## 6 · Is the ensemble honest? Spread against error

    An ensemble makes a claim about its own uncertainty, and that claim can be
    checked. For a **reliable** ensemble the RMS spread equals the RMS error of the
    ensemble mean, at every lead time — the truth should look like just another
    member.

    Two ways to fail:

    * **Under-dispersed** (spread below error): the ensemble is overconfident. This
      is the characteristic failure of operational systems, and the reason inflation
      exists.
    * **Over-dispersed** (spread above error): the ensemble hedges. Less dangerous,
      but it wastes members and blunts every probabilistic forecast built from it.
    """
    )
    return


@app.cell(hide_code=True)
def fig_reliability(
    enkf_errors, enkf_spreads, go, inflation, mo, n_members, np, obs_times,
    make_subplots, plotting
):
    _fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=("Spread and error against time", "Spread vs error"),
    )
    _fig.add_scatter(x=obs_times, y=enkf_spreads, mode="lines+markers",
                     line=dict(color=plotting.C_SPREAD, width=2),
                     name="ensemble spread", row=1, col=1)
    _fig.add_scatter(x=obs_times, y=enkf_errors, mode="lines+markers",
                     line=dict(color=plotting.C_MEAN, width=2),
                     name="error of the mean", row=1, col=1)

    _lim = float(max(enkf_spreads.max(), enkf_errors.max())) * 1.1
    _fig.add_scatter(x=[0, _lim], y=[0, _lim], mode="lines",
                     line=dict(color=plotting.C_SAT, dash="dash", width=1.5),
                     name="perfect reliability", row=1, col=2)
    _fig.add_scatter(x=enkf_errors, y=enkf_spreads, mode="markers",
                     marker=dict(size=8, color=plotting.C_SPREAD, opacity=0.75),
                     showlegend=False, row=1, col=2)
    plotting.style2d(_fig, height=420,
                     title=f"EnKF reliability (N = {n_members.value}, "
                           f"inflation = {inflation.value:.2f})")
    _fig.update_xaxes(title_text="time (MTU)", row=1, col=1)
    _fig.update_xaxes(title_text="error of the ensemble mean", row=1, col=2)
    _fig.update_yaxes(title_text="ensemble spread", row=1, col=2)

    _ratio = float(np.sqrt(np.mean(enkf_spreads**2) / np.mean(enkf_errors**2)))
    _verdict = (
        "close to reliable" if 0.8 < _ratio < 1.25
        else ("over-dispersed — it is hedging" if _ratio >= 1.25
              else "under-dispersed — it is overconfident")
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""Spread/error ratio = **{_ratio:.2f}** — {_verdict}.

            In the right panel a reliable ensemble scatters along the dashed line;
            points above it mean more spread than error.

            **This configuration is persistently over-dispersed, and inflation cannot
            fix it** — inflation only ever widens the ensemble, so it moves the cloud
            the wrong way. Reaching the diagonal here needs the slider *below* 1, and
            deflation is a diagnostic rather than something any operational centre
            does. Two reasons this case behaves so unusually: all three components are
            observed at every analysis time, which is an extraordinarily rich network,
            and the model is perfect, so there is no model error inflating the true
            uncertainty. Operational systems have the opposite problem — sparse
            observations and an imperfect model make them **under**-dispersed, which is
            precisely why inflation was invented.

            Note also how little the ratio moves as you change $N$ and the inflation,
            compared with how much the RMSE moves. Calibration and accuracy are
            genuinely different properties, and tuning for one does not tune the
            other. The ratio is computed from only {enkf_errors.size} analyses, so treat small
            changes as noise."""
        ),
    ])
    return


# ===========================================================================
# 7. Observation frequency
# ===========================================================================
@app.cell(hide_code=True)
def sec7_text(mo):
    mo.md(
        r"""
    ---
    ## 7 · Observing less often

    The first question an observing-system planner asks is not "how accurate?" but
    "how often?". Below, the same 3D-Var and EnKF cycling is repeated with the
    observation interval stretched from 0.1 to 0.8 MTU.
    """
    )
    return


@app.cell
def obs_frequency(
    B, BG_SIGMA, DT_MODEL, H, OBS_SIGMA, R, assimilate, cycle_3dvar, np,
    propagate, t_assim, truth, x_bg, x_start
):
    def sweep_observation_interval(intervals, n_members=20, alpha=1.05, seed=11):
        """Analysis RMSE against observation interval, for 3D-Var and the EnKF."""
        out = []
        for dt_obs in intervals:
            _rng = np.random.default_rng(seed)
            step = max(1, int(round(dt_obs / DT_MODEL)))
            idx = np.arange(0, t_assim.size, step)
            obs = truth[idx] + _rng.normal(0.0, OBS_SIGMA, (idx.size, 3))

            an3, _ = cycle_3dvar(x_bg, obs, idx, t_assim)
            r3 = float(np.sqrt(np.mean([
                (truth[idx[k]] - xa) ** 2 for k, (_, xa) in enumerate(an3)
            ])))

            ens = np.asarray(x_start, float) + _rng.normal(
                0.0, BG_SIGMA, (n_members, 3)
            )
            errs, prev = [], 0
            for k, i in enumerate(idx):
                seg = float(t_assim[i] - t_assim[prev])
                if seg > 0:
                    ens = np.array([propagate(m, seg)[-1] for m in ens])
                ens = assimilate.enkf_update(
                    ens, obs[k], H, R, inflation=alpha, seed=2000 + k
                )
                errs.append(np.mean((ens.mean(axis=0) - truth[i]) ** 2))
                prev = i
            out.append((dt_obs, r3, float(np.sqrt(np.mean(errs)))))
        return out

    interval_sweep = sweep_observation_interval([0.1, 0.2, 0.4, 0.6, 0.8])
    return interval_sweep, sweep_observation_interval


@app.cell(hide_code=True)
def fig_obs_frequency(go, interval_sweep, mo, np, plotting):
    _d = np.array([r[0] for r in interval_sweep])
    _r3 = np.array([r[1] for r in interval_sweep])
    _re = np.array([r[2] for r in interval_sweep])
    _fig = go.Figure()
    _fig.add_scatter(x=_d, y=_r3, mode="lines+markers",
                     line=dict(color=plotting.C_PERT, width=2), name="3D-Var")
    _fig.add_scatter(x=_d, y=_re, mode="lines+markers",
                     line=dict(color=plotting.C_MEAN, width=2), name="EnKF (N=20)")
    plotting.style2d(_fig, height=420, title="Analysis error vs observation interval")
    _fig.update_xaxes(title_text="observation interval Δt_obs (MTU)")
    _fig.update_yaxes(title_text="analysis RMSE")
    mo.vstack([
        _fig,
        mo.md(
            f"""Between $\\Delta t_{{\\rm obs}} = {_d[0]}$ and ${_d[-1]}$ MTU the
            3D-Var analysis error grows from {_r3[0]:.2f} to {_r3[-1]:.2f} and the
            EnKF's from {_re[0]:.2f} to {_re[-1]:.2f}.

            The mechanism is the one chapter 6 established: between observations the
            error grows at roughly $e^{{\\lambda t}}$, so doubling the gap does not
            double the error — it squares the amplification factor. Observing gaps
            hurt exponentially, which is why forecast centres fight so hard for
            temporal coverage rather than only for instrument precision."""
        ),
    ])
    return


# ===========================================================================
# 8. The logarithmic return on observations -- the chapter's centrepiece
# ===========================================================================
@app.cell(hide_code=True)
def sec8_text(mo):
    mo.md(
        r"""
    ---
    ## 8 · The logarithmic return on better observations

    Now the question that decides budgets. Suppose an observing-system upgrade
    reduces the analysis error by a factor of ten. **How much forecast does that
    buy?**

    The naive expectation is "ten times better forecasts". The answer is quite
    different, and it follows from one line of algebra. If errors grow exponentially,
    an initial error $\delta_0$ reaches a fixed useful-forecast threshold $\delta_c$
    at

    $$
    t_c = \frac{1}{\lambda}\ln\!\frac{\delta_c}{\delta_0}.
    $$

    Reducing $\delta_0$ by a factor of 10 therefore adds

    $$
    \Delta t = \frac{\ln 10}{\lambda}
    $$

    to the horizon — **the same fixed increment every time**, no matter how accurate
    the observations already were. Predictability is bought in units of $\ln$, not in
    proportion.

    The experiment below measures that directly. It seeds analysis errors of
    prescribed size $\delta_0$ at many points on the attractor, forecasts from each,
    averages the error curves, and reads off the lead time at which the mean error
    first crosses a fraction of the climatological saturation level. Averaging over
    launches matters: chapter 7 showed local growth rates vary by a factor of several,
    so a single forecast tells you about one trajectory rather than about the system.
    """
    )
    return


@app.cell(hide_code=True)
def logret_controls(mo):
    threshold = mo.ui.slider(
        start=0.2, stop=0.8, step=0.1, value=0.5,
        label="useful-forecast threshold (fraction of saturation)",
    )
    n_launch = mo.ui.slider(
        start=8, stop=48, step=8, value=32, label="forecast launches to average"
    )
    mo.hstack([threshold, n_launch], justify="start", gap=2)
    return n_launch, threshold


@app.cell
def logret(BETA, DT_MODEL, RHO, SIGMA, integrate, n_launch, np, systems, threshold):
    T_FORECAST = 22.0  # long enough for delta0 = 1e-6 to reach saturation

    # One long truth run, reused for every launch. Vectorised: all launches are
    # stepped simultaneously as an ensemble, so the whole sweep costs about as
    # much as a single 22-MTU integration times the number of amplitudes.
    _grid = np.linspace(0.0, 120.0, int(round(120.0 / DT_MODEL)) + 1)
    _long = integrate.rk4(
        systems.lorenz63, np.array([1.0, 1.0, 20.0]), _grid,
        sigma=SIGMA, rho=RHO, beta=BETA,
    )
    truth_long = _long[2000:]  # drop the spin-up

    # Climatological saturation: RMS difference between randomly paired states.
    _r = np.random.default_rng(0)
    _perm = _r.permutation(truth_long.shape[0])
    sat_rms = float(np.sqrt(np.mean((truth_long - truth_long[_perm]) ** 2)))

    n_fc = int(round(T_FORECAST / DT_MODEL)) + 1
    t_fc = np.linspace(0.0, T_FORECAST, n_fc)
    _stride = max(1, (truth_long.shape[0] - n_fc) // int(n_launch.value))
    launch_idx = np.arange(0, truth_long.shape[0] - n_fc, _stride)[: int(n_launch.value)]

    def horizon_for(delta0, seed=1):
        """Mean forecast-error curve and the lead time it crosses the threshold."""
        _rng = np.random.default_rng(seed)
        x0 = truth_long[launch_idx]
        d = _rng.normal(size=x0.shape)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        fc = integrate.rk4(
            systems.lorenz63, x0 + delta0 * d, t_fc,
            sigma=SIGMA, rho=RHO, beta=BETA,
        )
        tru = np.stack([truth_long[i : i + n_fc] for i in launch_idx], axis=1)
        curve = np.sqrt(np.mean((fc - tru) ** 2, axis=2)).mean(axis=1)
        crossed = np.flatnonzero(curve >= float(threshold.value) * sat_rms)
        horizon = float(t_fc[crossed[0]]) if crossed.size else float("nan")
        return horizon, curve

    deltas = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    horizons, curves = [], []
    for _d0 in deltas:
        _h, _c = horizon_for(_d0)
        horizons.append(_h)
        curves.append(_c)
    horizons = np.array(horizons)

    # Fit horizon = -(1/lambda) ln(delta0) + c. The slope IS 1/lambda if the law holds.
    _ok = np.isfinite(horizons)
    slope, _icpt = np.polyfit(-np.log(deltas[_ok]), horizons[_ok], 1)
    gain_per_decade = float(slope * np.log(10.0))
    return (
        T_FORECAST,
        curves,
        deltas,
        gain_per_decade,
        horizons,
        sat_rms,
        slope,
        t_fc,
    )


@app.cell(hide_code=True)
def fig_logret(
    curves, deltas, gain_per_decade, go, horizons, lyapunov, make_subplots, mo, np,
    plotting, sat_rms, slope, systems, t_fc, threshold
):
    _fig = make_subplots(
        rows=1, cols=2, column_widths=[0.58, 0.42],
        subplot_titles=(
            "Mean forecast error, by initial error size",
            "Horizon against −ln δ₀",
        ),
    )
    _shades = ["#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed", "#6d28d9", "#4c1d95"]
    for _i, (_d0, _c) in enumerate(zip(deltas, curves)):
        _fig.add_scatter(
            x=t_fc, y=_c, mode="lines",
            line=dict(width=2, color=_shades[_i % len(_shades)]),
            name=f"δ₀ = {_d0:.0e}", row=1, col=1,
        )
    _fig.add_hline(
        y=float(threshold.value) * sat_rms, line=dict(color=plotting.C_SAT, dash="dash"),
        row=1, col=1,
    )
    _fig.add_hline(y=sat_rms, line=dict(color=plotting.C_CONTEXT, dash="dot"), row=1, col=1)

    _x = -np.log(deltas)
    _ok = np.isfinite(horizons)
    _fig.add_scatter(
        x=_x[_ok], y=horizons[_ok], mode="markers",
        marker=dict(size=11, color=plotting.C_ANALYSIS), name="measured horizon",
        row=1, col=2,
    )
    _fit = slope * _x + (horizons[_ok].mean() - slope * _x[_ok].mean())
    _fig.add_scatter(
        x=_x, y=_fit, mode="lines",
        line=dict(color=plotting.C_SAT, dash="dash", width=2),
        name=f"fit: {slope:.2f} MTU per e-fold", row=1, col=2,
    )
    plotting.style2d(_fig, height=470,
                     title="Predictability is bought in units of ln")
    _fig.update_xaxes(title_text="forecast lead time (MTU)", row=1, col=1)
    _fig.update_yaxes(title_text="RMS error", type="log", row=1, col=1)
    _fig.update_xaxes(title_text="−ln δ₀", row=1, col=2)
    _fig.update_yaxes(title_text="horizon (MTU)", row=1, col=2)

    _lam = lyapunov.lyapunov_spectrum(
        systems.lorenz63, systems.lorenz63_jacobian, np.array([1.0, 1.0, 20.0]),
        dt=0.01, t_final=200.0, t_transient=20.0,
    )[0]
    _theory_slope = 1.0 / _lam
    _theory_decade = np.log(10.0) / _lam
    _rows = "\n".join(
        f"| {d:.0e} | {h:.2f} |" for d, h in zip(deltas, horizons)
    )
    mo.vstack([
        _fig,
        mo.md(
            f"""
| initial error δ₀ | horizon (MTU) |
|---|---|
{_rows}

**Measured slope {slope:.3f} MTU per e-fold of initial error, against the theoretical
$1/\\lambda_1 = {_theory_slope:.3f}$.** Each decade of analysis-error reduction buys
**{gain_per_decade:.2f} MTU** of forecast; theory says
$\\ln 10/\\lambda_1 = {_theory_decade:.2f}$.

Look at the left panel. The curves are *parallel* on a logarithmic error axis, each
displaced by a fixed horizontal distance for each factor of ten in $\\delta_0$. That
parallel displacement **is** the logarithmic law: reducing the initial error does not
change how fast error grows, it only changes where the growth starts. Six orders of
magnitude in observation quality buys about a factor of four in forecast range.

At the conventional reading of 1 MTU $\\approx$ 5 days this is roughly
{gain_per_decade * 5:.0f} days per decade. Do not carry that number to the real
atmosphere: the atmosphere's leading exponent is about $0.35\\ \\mathrm{{day}}^{{-1}}$,
which gives $\\ln 10/\\lambda \\approx 6.6$ days. Lorenz 63 under the 5-day convention
is *less* chaotic per day than the atmosphere it stands in for, so it flatters the
value of better observations. The **law** transfers; the constant does not. That
distinction is the whole reason chapter 3 insists on a hierarchy of models rather than
one favourite.
"""
        ),
    ])
    return


# ===========================================================================
# 9. Takeaways
# ===========================================================================
@app.cell(hide_code=True)
def takeaways(mo):
    mo.md(
        r"""
    ---
    ## 9 · What to take away

    | Feature | 3D-Var | 4D-Var | EnKF |
    |---------|-------|-------|------|
    | **Background covariance** | Static $\mathbf{B}$ | Static $\mathbf{B}$ | Flow-dependent $\mathbf{P}^f$ |
    | **Temporal scope** | Single time | Time window | Sequential |
    | **Model adjoint** | Not required | **Required** | Not required |
    | **Ensemble** | No | No | Yes |
    | **Parallelism** | Low | Low | High (per member) |
    | **Typical use** | Regional NWP, fast cycling | Global NWP (ECMWF) | Ensemble NWP, ocean, land |

    ### The three results worth remembering

    1. **Cycling is what makes assimilation work.** A single analysis is a weighted
       average. It is the repetition — analyse, forecast, analyse — that holds a model
       to reality indefinitely against exponential error growth.
    2. **Using observations at the right *time* is worth as much as having more of
       them.** 4D-Var beat 3D-Var here using the same observations and the same
       $\mathbf{B}$, with far fewer analysis points, purely by fitting one trajectory
       through a window.
    3. **Better observations pay logarithmically.** Each factor of ten buys the same
       fixed increment of forecast range, $\ln 10/\lambda$. This is why the practical
       forecast horizon has advanced by roughly a day per decade for fifty years
       rather than exploding, and why it will keep advancing slowly rather than
       suddenly — see chapter 22.

    ### Connections to methods you may already know

    * **3D-Var** is Tikhonov-regularised least squares — ridge regression with
      $\mathbf{B}^{-1}$ as the regularisation matrix.
    * **4D-Var** is **backpropagation through time** for a physics model: the adjoint
      is exactly the reverse-mode automatic-differentiation graph. If you have trained
      a recurrent network, you have run 4D-Var's inner loop.
    * **The EnKF** is a Monte-Carlo Kalman filter, closely related to particle methods
      in Bayesian deep learning.
    * Hybrid systems now combine variational assimilation with learned forecast
      models, which makes the adjoint available for free through the AD framework —
      chapter 29 takes that up.

    ### Try this

    1. Set the EnKF to $N = 10$ with inflation $1.0$ and watch for **filter
       divergence** — the moment the ensemble becomes so narrow that the analysis
       stops responding to observations. Then raise the inflation until it recovers.
    2. Find the inflation that centres the reliability scatter on the diagonal. Is the
       value that gives the best RMSE the same as the value that gives the best
       reliability? (It usually is not, and that tension is real.)
    3. In section 8, change the useful-forecast threshold from 0.5 to 0.2 and to 0.8.
       The horizons all move — does the *slope* move? It should not, and the reason it
       does not is the point of the whole section.

    ### What you should have seen

    3D-Var, 4D-Var and the EnKF all keep the analysis close to the truth while a free
    forecast from the same background diverges completely. 4D-Var improves on 3D-Var
    from timing alone. And the forecast horizon grows like the **logarithm** of
    observation accuracy, with a slope set by $1/\lambda_1$ — measured here to within
    a few percent of the value chapter 7 computed from the dynamics alone, by a
    completely independent route.

    ### Further reading

    * Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability.*
      Cambridge University Press — chapter 5 for the algorithms, §6.1 for
      predictability.
    * Evensen, G. (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer.
    * Bocquet, M. et al. (2023). *A guide to ensemble Kalman methods with
      implementation in Python.* arXiv:2305.00087.
    * Bauer, P., Thorpe, A. and Brunet, G. (2015). The quiet revolution of numerical
      weather prediction. *Nature*, **525**, 47–55 — the historical record of the
      "day per decade" improvement.
    * Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, chapter 8 *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
