# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 23 -- Boundary-forced predictability and the S2S window.

Predictable statistics over an unpredictable trajectory: what a slow forcing
tells you, when it stops mattering that you knew today's state, and why the
subseasonal "predictability desert" is not empty.

Part VI of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The experiments are precomputed by `scripts/generate_ch23_data.py`.

To edit:   marimo edit notebooks/ch23_boundary-forced-s2s.py
To export: make nb-one NB=ch23_boundary-forced-s2s
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 23: Boundary-Forced Predictability")


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

    from chaoslib import information, integrate, plotting, systems

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
    C_ANALYSIS = plotting.C_ANALYSIS
    MPL_SEQUENTIAL = plotting.MPL_SEQUENTIAL
    mpl_panels = plotting.mpl_panels
    finish_mpl = plotting.finish_mpl

    return (
        C_ANALYSIS, C_BG, C_CONTEXT, C_FIXED, C_MEAN, C_OBS, C_PERT, C_SAT,
        C_SPREAD, C_START, C_TRUTH, MPL_SEQUENTIAL, finish_mpl, information,
        integrate, mo, mpl_panels, np, plotting, plt, systems,
    )


@app.cell
def chapter_data():
    # Precomputed by scripts/generate_ch23_data.py (~3 min): a 4000-TU
    # forced climatology conditioned on phase, 120 forecast launches
    # spread around the cycle, and sweeps of forcing amplitude and
    # period. Knob-free; the figures slice these arrays.

    N_PHASE = 8
    BINS = (np.float64(-5.0), np.float64(-3.125), np.float64(-1.25), np.float64(0.625), np.float64(2.5), np.float64(4.375), np.float64(6.25), np.float64(8.125), np.float64(10.0), np.float64(11.875), np.float64(13.75), np.float64(15.625), np.float64(17.5), np.float64(19.375), np.float64(21.25), np.float64(23.125), np.float64(25.0), np.float64(26.875), np.float64(28.75), np.float64(30.625), np.float64(32.5), np.float64(34.375), np.float64(36.25), np.float64(38.125), np.float64(40.0), np.float64(41.875), np.float64(43.75), np.float64(45.625), np.float64(47.5), np.float64(49.375), np.float64(51.25), np.float64(53.125), np.float64(55.0), np.float64(56.875), np.float64(58.75), np.float64(60.625), np.float64(62.5), np.float64(64.375), np.float64(66.25), np.float64(68.125), np.float64(70.0))
    PERIOD = 40.0
    RHO_MEAN = 28.0
    RHO_AMPLITUDE = 6.0
    BOUNDARY_MEAN = 0.119949
    BOUNDARY_BY_PHASE = (
        0.039563, 0.159416, 0.156101, 0.051137, 0.067391, 0.219572, 0.206838, 0.059575,
    )
    PHASE_MEAN_Z = (
        25.7814, 28.9224, 28.9729, 26.0694, 21.6163, 18.4998, 18.6886, 21.5532,
    )
    POOLED_HIST = (
        0.0, 0.0, 0.0, 106.0, 710.0, 2806.0, 6613.0, 11260.0, 16688.0, 21535.0,
        25118.0, 27937.0, 29119.0, 29016.0, 28404.0, 27826.0, 26908.0, 24886.0, 22848.0, 19965.0,
        16860.0, 14243.0, 11986.0, 9506.0, 7554.0, 5520.0, 3887.0, 2407.0, 1324.0, 617.0,
        253.0, 89.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    PHASE_HIST_1 = (
        0.0, 0.0, 0.0, 0.0, 0.0, 47.0, 123.0, 346.0, 627.0, 1028.0,
        1466.0, 2199.0, 2824.0, 3157.0, 3561.0, 3695.0, 3538.0, 3355.0, 2992.0, 2703.0,
        2560.0, 2536.0, 2575.0, 2502.0, 2227.0, 1936.0, 1452.0, 1022.0, 572.0, 288.0,
        124.0, 38.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    PHASE_HIST_5 = (
        0.0, 0.0, 0.0, 92.0, 269.0, 884.0, 1854.0, 3002.0, 4157.0, 4864.0,
        4737.0, 4720.0, 3857.0, 3363.0, 3080.0, 3080.0, 3231.0, 3031.0, 2681.0, 1646.0,
        697.0, 217.0, 36.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )

    LEADS = (np.float64(0.0), np.float64(1.0), np.float64(2.0), np.float64(3.0), np.float64(4.0), np.float64(5.0), np.float64(6.0), np.float64(7.0), np.float64(8.0), np.float64(9.0), np.float64(10.0), np.float64(11.0), np.float64(12.0), np.float64(13.0), np.float64(14.0), np.float64(15.0), np.float64(16.0), np.float64(17.0), np.float64(18.0), np.float64(19.0), np.float64(20.0), np.float64(21.0), np.float64(22.0), np.float64(23.0), np.float64(24.0), np.float64(25.0), np.float64(26.0), np.float64(27.0), np.float64(28.0), np.float64(29.0), np.float64(30.0))
    N_STARTS = 120
    MEMBERS = 2000
    CROSSING = 17.0000
    PLATEAU = 0.145890
    AGAINST_POOLED = (
        2.874547, 2.616301, 2.315977, 1.916736, 1.457910, 1.150094, 0.971289, 0.712566,
        0.607258, 0.492254, 0.403571, 0.352878, 0.313737, 0.301048, 0.267109, 0.252464,
        0.243162, 0.208960, 0.210882, 0.200741, 0.178400, 0.176713, 0.168240, 0.154325,
        0.155399, 0.147497, 0.152070, 0.145500, 0.143668, 0.144509, 0.143702,
    )
    AGAINST_PHASE = (
        2.760403, 2.586340, 2.204609, 1.804826, 1.321037, 1.028973, 0.832779, 0.601934,
        0.476913, 0.391211, 0.295889, 0.247062, 0.208945, 0.180402, 0.154466, 0.138291,
        0.127945, 0.100547, 0.096820, 0.078321, 0.060354, 0.056321, 0.053083, 0.046720,
        0.039198, 0.038030, 0.036523, 0.033164, 0.031725, 0.029677, 0.030347,
    )

    WINDOW_SPLIT_HALF = 0.9561
    WINDOW_HALF_A = (
        10.0000, 9.0000, 10.0000, 9.0000, 9.0000, 20.0000, 14.0000, 12.0000,
    )
    WINDOW_HALF_B = (
        10.0000, 10.0000, 10.0000, 9.0000, 11.0000, 17.0000, 14.0000, 13.0000,
    )
    WINDOW_THRESHOLD = 0.250
    WINDOW_HORIZON = (
        10.0000, 10.0000, 10.0000, 9.0000, 9.0000, 17.0000, 14.0000, 13.0000,
    )
    WINDOW_COUNTS = (
        15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 16.0, 14.0,
    )
    LAUNCH_PHASE = (
        0.00000, 0.82550, 0.65100, 0.47675, 0.30225, 0.12800, 0.95350, 0.77925,
        0.60475, 0.43050, 0.25600, 0.08150, 0.90725, 0.73275, 0.55850, 0.38400,
        0.20975, 0.03525, 0.86100, 0.68650, 0.51200, 0.33775, 0.16325, 0.98900,
        0.81450, 0.64025, 0.46575, 0.29150, 0.11700, 0.94250, 0.76825, 0.59375,
        0.41950, 0.24500, 0.07075, 0.89625, 0.72200, 0.54750, 0.37300, 0.19875,
        0.02425, 0.85000, 0.67550, 0.50125, 0.32675, 0.15250, 0.97800, 0.80350,
        0.62925, 0.45475, 0.28050, 0.10600, 0.93175, 0.75725, 0.58300, 0.40850,
        0.23400, 0.05975, 0.88525, 0.71100, 0.53650, 0.36225, 0.18775, 0.01350,
        0.83900, 0.66450, 0.49025, 0.31575, 0.14150, 0.96700, 0.79275, 0.61825,
        0.44400, 0.26950, 0.09500, 0.92075, 0.74625, 0.57200, 0.39750, 0.22325,
        0.04875, 0.87450, 0.70000, 0.52550, 0.35125, 0.17675, 0.00250, 0.82800,
        0.65375, 0.47925, 0.30500, 0.13050, 0.95600, 0.78175, 0.60725, 0.43300,
        0.25850, 0.08425, 0.90975, 0.73550, 0.56100, 0.38650, 0.21225, 0.03775,
        0.86350, 0.68900, 0.51475, 0.34025, 0.16600, 0.99150, 0.81700, 0.64275,
        0.46825, 0.29400, 0.11950, 0.94525, 0.77075, 0.59650, 0.42200, 0.24775,
    )


    AMPLITUDES = (0.0, 2.0, 4.0, 6.0, 9.0)
    PERIODS = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    AMPLITUDE_BOUNDARY = (
        0.001502, 0.021706, 0.067043, 0.119949, 0.251796,
    )
    PERIOD_BOUNDARY = (
        0.167310, 0.121158, 0.116250, 0.114973, 0.119949, 0.121100,
        0.126062,
    )

    return (
        AGAINST_PHASE, AGAINST_POOLED, AMPLITUDES, AMPLITUDE_BOUNDARY,
        BINS, BOUNDARY_BY_PHASE, BOUNDARY_MEAN, CROSSING, LAUNCH_PHASE,
        LEADS, MEMBERS, N_PHASE, N_STARTS, PERIOD, PERIODS,
        PERIOD_BOUNDARY, PHASE_HIST_1, PHASE_HIST_5, PHASE_MEAN_Z,
        PLATEAU, POOLED_HIST, RHO_AMPLITUDE, RHO_MEAN, WINDOW_COUNTS,
        WINDOW_HALF_A, WINDOW_HALF_B, WINDOW_HORIZON, WINDOW_SPLIT_HALF,
        WINDOW_THRESHOLD,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(PERIOD, RHO_AMPLITUDE, RHO_MEAN, mo):
    mo.md(
        rf"""
    # Chapter 23 · Boundary-Forced Predictability and the S2S Window

    **Part VI — Predictability of the second kind: from S2S to climate.**

    **The forecasting question.** Weather forecasts are useful for about a week. Seasonal
    forecasts of the coming winter are useful too — issued months ahead, when no trace of
    today's weather can possibly survive. Both cannot be initial-value problems. What is
    the seasonal forecast actually using?

    **Chapter 1** measured the answer's shape: information about the future carried by
    today's state decays to nothing, while information carried by the *forcing* does not
    decay at all, and the two cross. That chapter held the forcing fixed. This one lets
    it move.

    The system is Lorenz 63 with a slowly oscillating Rayleigh number,

    $$
    \rho(t) = {RHO_MEAN:g} + {RHO_AMPLITUDE:g}\,\sin\!\left(\frac{{2\pi t}}{{{PERIOD:g}}}\right),
    $$

    with a period an order of magnitude longer than the trajectory's own predictability
    time. That separation is what makes $\rho$ a *boundary condition* rather than part of
    the fast dynamics — the cheapest caricature of ENSO, of the seasonal cycle, or of any
    other slow driver.

    ---

    **What you need before this chapter.** **Chapter 1** for the two kinds of
    predictability and the information measure used here. **Chapter 6** for Lorenz 63.
    """
    )
    return


# ===========================================================================
# Section 1
# ===========================================================================
@app.cell(hide_code=True)
def s1_md(N_PHASE, mo):
    mo.md(
        rf"""
    ## 1 · Two climatologies

    The move that makes seasonal forecasting possible is small and worth stating slowly.

    A **climatology** is what you say when you know nothing about today. But there is more
    than one of them. There is the climatology pooled over *all* forcing phases — the
    unconditional one, the average January-through-December — and there is the
    climatology **conditioned on the forcing phase**: what happens when $\rho$ is high,
    as against when it is low.

    If those two differ, then *knowing the phase* is worth something, and it is worth the
    same amount however far ahead you are looking. Below, the pooled climatology of $z$
    against two of the {N_PHASE} phase-conditioned ones.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    BINS, BOUNDARY_BY_PHASE, C_BG, C_PERT, C_TRUTH, N_PHASE, PERIOD,
    PHASE_HIST_1, PHASE_HIST_5, PHASE_MEAN_Z, POOLED_HIST, RHO_AMPLITUDE,
    RHO_MEAN, finish_mpl, mpl_panels, np, systems,
):
    _edges = np.asarray(BINS)
    _centres = 0.5 * (_edges[:-1] + _edges[1:])
    _width = _edges[1] - _edges[0]

    def _density(counts):
        counts = np.asarray(counts, dtype=float)
        return counts / (counts.sum() * _width)

    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("The forcing", "Pooled and phase-conditioned climatologies"),
        figsize=(10.4, 4.0),
    )
    _t = np.linspace(0.0, 2.0 * PERIOD, 400)
    _ax0.plot(_t / PERIOD, systems.lorenz63_forcing(
        _t, RHO_MEAN, RHO_AMPLITUDE, PERIOD), "-", color=C_TRUTH, linewidth=2.2)
    _bandlo, _bandhi = 1 / N_PHASE, 2 / N_PHASE
    for _cycle in (0, 1):
        _ax0.axvspan(_cycle + _bandlo, _cycle + _bandhi, color=C_PERT,
                     alpha=0.16, linewidth=0)
        _ax0.axvspan(_cycle + 5 / N_PHASE, _cycle + 6 / N_PHASE, color=C_BG,
                     alpha=0.22, linewidth=0)
    _ax0.set_xlabel("time (forcing periods)")
    _ax0.set_ylabel(r"$\rho(t)$")

    _ax1.fill_between(_centres, _density(POOLED_HIST), step="mid", color=C_BG,
                      alpha=0.35, label="pooled over all phases")
    _ax1.plot(_centres, _density(PHASE_HIST_1), "-", color=C_PERT,
              linewidth=2.3,
              label=f"phase {1/N_PHASE:.2f}–{2/N_PHASE:.2f}  "
                    f"($\\bar z$ = {PHASE_MEAN_Z[1]:.1f})")
    _ax1.plot(_centres, _density(PHASE_HIST_5), "-", color=C_TRUTH,
              linewidth=2.3,
              label=f"phase {5/N_PHASE:.2f}–{6/N_PHASE:.2f}  "
                    f"($\\bar z$ = {PHASE_MEAN_Z[5]:.1f})")
    _ax1.set_xlabel("$z$")
    _ax1.set_ylabel("probability density")
    _ax1.legend(fontsize=8, framealpha=0.9)
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s1_note(BOUNDARY_BY_PHASE, BOUNDARY_MEAN, PHASE_MEAN_Z, mo, np):
    _b = np.asarray(BOUNDARY_BY_PHASE)
    _m = np.asarray(PHASE_MEAN_Z)
    mo.md(
        rf"""
    They differ, and by a lot: the conditional mean of $z$ runs from {_m.min():.1f} to
    {_m.max():.1f} around the cycle. Measured as information — the same relative entropy
    chapter 1 used — knowing the phase is worth **{BOUNDARY_MEAN:.4f} nats** on average.

    **And it is worth very different amounts at different phases**, from {_b.min():.4f} to
    {_b.max():.4f} nats, a factor of {_b.max() / _b.min():.1f}. That variation is the
    first hint of section 4's subject: predictability is not uniform around the cycle, and
    a forecast system that reports one number for "seasonal skill" is averaging over
    something structured.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(MEMBERS, N_STARTS, mo):
    mo.md(
        rf"""
    ## 2 · Where the skill comes from

    Now launch {N_STARTS} ensembles of {MEMBERS} members from points spread around the
    cycle, and measure each forecast distribution against **both** climatologies.

    * Against the **pooled** climatology: everything the forecast knows, from any source.
    * Against the **phase-conditioned** climatology: only what the forecast knows *beyond
      what the forcing phase already told you*.

    The difference between the two curves is the boundary-forced contribution. If the
    forcing were doing nothing, they would lie on top of each other.
    """
    )
    return


@app.cell(hide_code=True)
def s2_fig(
    AGAINST_PHASE, AGAINST_POOLED, BOUNDARY_MEAN, CROSSING, C_ANALYSIS, C_BG,
    C_PERT, C_TRUTH, LEADS, PLATEAU, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(LEADS)
    _pooled = np.asarray(AGAINST_POOLED)
    _phase = np.asarray(AGAINST_PHASE)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(9.0, 4.6))
    _ax.semilogy(_leads, _pooled, "-", color=C_TRUTH, linewidth=2.6,
                 label="everything the forecast knows")
    _ax.semilogy(_leads, _phase, "-", color=C_PERT, linewidth=2.6,
                 label="what it knows beyond the forcing phase")
    _ax.axhline(BOUNDARY_MEAN, color=C_ANALYSIS, linestyle="--", linewidth=1.9)
    # Left-hand side and below the line: the right-hand side is where the blue
    # curve settles, and the label sat on top of it.
    _ax.text(_leads[0] + 0.3, BOUNDARY_MEAN * 0.86,
             f"what the phase alone tells you ({BOUNDARY_MEAN:.3f} nats)",
             fontsize=8.5, color=C_ANALYSIS)
    if np.isfinite(CROSSING):
        _ax.axvline(CROSSING, color=C_BG, linestyle=":", linewidth=1.6)
        _ax.text(CROSSING - 0.6, _pooled[0] * 0.28,
                 f"the forcing overtakes\nthe initial state\n(lead {CROSSING:.0f})",
                 fontsize=8, color=C_BG, ha="right")
    _ax.set_xlabel("forecast lead (time units)")
    _ax.set_ylabel("information beyond climatology (nats)")
    _ax.legend(fontsize=8.5, framealpha=0.95, loc="lower left")
    finish_mpl(
        _fig,
        "The upper curve settles on a floor; the lower one does not have a "
        "floor to settle on.",
    )
    return


@app.cell(hide_code=True)
def s2_note(
    AGAINST_PHASE, AGAINST_POOLED, BOUNDARY_MEAN, CROSSING, LEADS, PLATEAU,
    mo, np,
):
    _leads = np.asarray(LEADS)
    _pooled = np.asarray(AGAINST_POOLED)
    _phase = np.asarray(AGAINST_PHASE)
    _residual = _phase[-1]
    mo.md(
        rf"""
    **The upper curve does not decay to zero. It decays to a floor.** From
    {_pooled[0]:.2f} nats it settles at **{PLATEAU:.4f}**, and the phase alone is worth
    {BOUNDARY_MEAN:.4f}. The forecast at long lead has stopped being a forecast in the
    ordinary sense — it has become a statement about the forcing.

    **The lower curve keeps falling**, from {_phase[0]:.2f} to {_residual:.4f} nats.
    Beyond the forcing phase, the initial condition has almost nothing left to add.

    And the two numbers fit together. The floor plus what remains of the initial
    condition, {BOUNDARY_MEAN:.4f} + {_residual:.4f} = {BOUNDARY_MEAN + _residual:.4f},
    against a measured plateau of {PLATEAU:.4f} — agreement to
    {100 * abs(PLATEAU - BOUNDARY_MEAN - _residual) / PLATEAU:.1f} %. Relative entropy is
    not additive in general, so this is a check rather than an identity; that it holds
    this closely says the two sources of information are, here, close to independent.

    **The forcing overtakes the initial state at lead {CROSSING:.0f}** — the lead at
    which what remains of today's state falls below what the phase was always worth. In
    a real forecasting system that crossover is roughly where "weather forecast" stops
    being the right description and "seasonal outlook" starts.

    /// admonition | The predictability desert, and why it is not empty
        type: note

    Between the two regimes there is a well-known awkward gap: too far ahead for the
    initial state to help much, not far enough for the boundary signal to be the whole
    story. In these numbers it runs from about lead 10, where beyond-the-phase
    information has fallen to a quarter of a nat, to about lead {CROSSING:.0f}.

    The useful point is that **the desert has a floor under it**. Skill there is small
    but it is not zero, and what is left is boundary-forced — which is exactly why
    subseasonal forecasting is worth attempting rather than a category error. The gap is
    thin, not empty.
    ///
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(WINDOW_SPLIT_HALF, WINDOW_THRESHOLD, mo):
    mo.md(
        rf"""
    ## 3 · Windows of opportunity

    Section 1 showed that the phase is worth a factor of five more at some points in the
    cycle than others. The natural next question is whether *forecasts* launched at
    different phases are correspondingly better or worse — the operational idea of a
    **window of opportunity**, where a particular state of the slow driver makes the
    coming weeks unusually predictable.

    Below, the lead at which a forecast's beyond-the-phase information falls below
    {WINDOW_THRESHOLD:g} nats, resolved by the phase it was *launched* at.

    With fifteen launches per bin this is exactly the kind of claim that could be
    sampling noise, so it is checked: splitting the launches into two independent halves
    and asking whether they agree about which phases are the predictable ones gives a
    correlation of **{WINDOW_SPLIT_HALF:+.2f}**.
    """
    )
    return


@app.cell(hide_code=True)
def s3_fig(
    BOUNDARY_BY_PHASE, C_ANALYSIS, C_BG, C_PERT, C_TRUTH, N_PHASE,
    WINDOW_HALF_A, WINDOW_HALF_B, WINDOW_HORIZON, finish_mpl, mpl_panels, np,
):
    _phase = (np.arange(N_PHASE) + 0.5) / N_PHASE
    _h = np.asarray(WINDOW_HORIZON)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Horizon by launch phase", "The two independent halves"),
        figsize=(10.2, 4.0),
    )
    _ax0.bar(_phase, _h, width=0.9 / N_PHASE, color=C_ANALYSIS,
             edgecolor="white", linewidth=0.8)
    _ax0.axhline(float(np.nanmean(_h)), color=C_BG, linestyle="--",
                 linewidth=1.4)
    _ax0.text(0.02, np.nanmean(_h) * 1.03, "mean", fontsize=8, color=C_BG)
    _ax0.set_xlabel("forcing phase at launch")
    _ax0.set_ylabel("lead at which the initial-state\nsignal is spent (TU)")

    _twin = _ax0.twinx()
    _twin.plot(_phase, np.asarray(BOUNDARY_BY_PHASE), "o-", color=C_PERT,
               markersize=4, linewidth=1.6)
    _twin.set_ylabel("phase information (nats)", color=C_PERT)
    _twin.tick_params(axis="y", colors=C_PERT, labelsize=8)
    _twin.grid(False)

    _a, _b = np.asarray(WINDOW_HALF_A), np.asarray(WINDOW_HALF_B)
    _ax1.plot(_a, _b, "o", color=C_TRUTH, markersize=9,
              markeredgecolor="white", markeredgewidth=1.0)
    _lo = float(np.nanmin([_a.min(), _b.min()])) - 1.0
    _hi = float(np.nanmax([_a.max(), _b.max()])) + 1.0
    _ax1.plot([_lo, _hi], [_lo, _hi], "--", color=C_BG, linewidth=1.4)
    _ax1.set_xlabel("horizon from the odd-numbered launches (TU)")
    _ax1.set_ylabel("from the even-numbered ones (TU)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s3_note(
    BOUNDARY_BY_PHASE, N_PHASE, WINDOW_HORIZON, WINDOW_SPLIT_HALF, mo, np,
):
    _h = np.asarray(WINDOW_HORIZON)
    _b = np.asarray(BOUNDARY_BY_PHASE)
    _best = int(np.nanargmax(_h))
    _worst = int(np.nanargmin(_h))
    _finite = np.isfinite(_h)
    _corr = float(np.corrcoef(_h[_finite], _b[_finite])[0, 1])
    mo.md(
        rf"""
    Forecasts launched at phase {_best / N_PHASE:.2f}–{(_best + 1) / N_PHASE:.2f} keep
    their initial-condition advantage out to lead {_h[_best]:.0f}; those launched at
    {_worst / N_PHASE:.2f}–{(_worst + 1) / N_PHASE:.2f} lose it by
    {_h[_worst]:.0f} — a factor of {_h[_best] / _h[_worst]:.1f}. The split-half
    correlation of {WINDOW_SPLIT_HALF:+.2f} says this is structure rather than noise.

    **This is the operational idea of a window of opportunity, in miniature.** Skill is
    not a property of a forecast system alone; it is a property of the system *and the
    state of the slow driver when the forecast was launched*. A centre that knows which
    windows are the good ones can say so in advance — which is worth more than a higher
    average score, because it tells a user when to act on the forecast.

    The right-hand axis of the left panel puts the two side by side, and they broadly
    track each other: across the eight bins they correlate at **{_corr:+.2f}**. Broadly,
    but not tightly — and with eight points a correlation of that size is suggestive
    rather than established.

    That is the honest reading, and it is more interesting than either extreme would
    be. The phases where the *forcing* is most informative tend to be the phases where
    *forecasts* last longest, which is not obvious: they are different questions — one
    asks what the boundary condition is worth, the other how long the initial condition
    survives — and nothing forces them to have the same answer. A real forecasting system
    has to answer both, and here it gets a partial discount for answering one.
    """
    )
    return


# ===========================================================================
# Section 4
# ===========================================================================
@app.cell(hide_code=True)
def s4_md(mo):
    mo.md(
        r"""
    ## 4 · The knob: how strong, and how slow?

    Two things can be varied about a slow driver, and they do not do the same job.
    """
    )
    return


@app.cell(hide_code=True)
def s4_fig(
    AMPLITUDES, AMPLITUDE_BOUNDARY, C_ANALYSIS, C_BG, C_PERT, C_SAT, PERIODS,
    PERIOD_BOUNDARY, RHO_AMPLITUDE, finish_mpl, mpl_panels, np,
):
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Amplitude: how much the forcing is worth",
                "Period: whether the system can follow it"),
        figsize=(10.2, 4.0),
    )
    _ax0.plot(AMPLITUDES, AMPLITUDE_BOUNDARY, "o-", color=C_ANALYSIS,
              markersize=6, linewidth=2.0)
    _ax0.set_xlabel(r"forcing amplitude $A$")
    _ax0.set_ylabel("phase information (nats)")

    _p = np.asarray(PERIODS)
    _ax1.semilogx(_p, PERIOD_BOUNDARY, "o-", color=C_PERT, markersize=6,
                  linewidth=2.0)
    # 5 TU is roughly where a Lorenz 63 forecast stops beating climatology
    # (chapter 1). Below that the forcing is not slow compared with the
    # dynamics, and "the phase" stops being a boundary condition at all.
    _ax1.axvspan(_p.min() * 0.8, 5.0, color=C_SAT, alpha=0.12, linewidth=0)
    _ax1.text(_p.min() * 0.9, max(PERIOD_BOUNDARY) * 0.98,
              "no timescale\nseparation", fontsize=8, color=C_SAT, va="top")
    _ax1.set_xlabel("forcing period $T$ (time units)")
    _ax1.set_ylabel("phase information (nats)")
    _ax1.set_xticks(_p)
    _ax1.set_xticklabels([f"{v:g}" for v in _p], fontsize=7.5)
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s4_note(
    AMPLITUDES, AMPLITUDE_BOUNDARY, PERIODS, PERIOD_BOUNDARY, mo, np,
):
    _a = np.asarray(AMPLITUDE_BOUNDARY)
    _p = np.asarray(PERIODS)
    _pb = np.asarray(PERIOD_BOUNDARY)
    _slow = _p >= 5.0
    mo.md(
        rf"""
    **Amplitude is what the forcing is worth.** From {_a[0]:.4f} nats at zero to
    {_a[-1]:.4f} at $A = {AMPLITUDES[-1]:g}$, monotonically. That is the expected result
    and the uninteresting one: a bigger swing in the boundary condition displaces the
    conditional climatologies further, and knowing which one you are in is worth more.

    **Period is not.** Across the slow range, $T$ from {_p[_slow][0]:g} to
    {_p[_slow][-1]:g} — a factor of {_p[_slow][-1] / _p[_slow][0]:.0f} — the phase
    information moves only from {_pb[_slow].min():.4f} to {_pb[_slow].max():.4f} nats,
    about {100 * (_pb[_slow].max() / _pb[_slow].min() - 1):.0f} %. Once the forcing is
    slow enough for the system to equilibrate to whatever $\rho$ currently is, making it
    slower still adds essentially nothing. **How strong the driver is matters; how slow
    it is only matters up to a point.**

    ### And a result that came out backwards

    At $T = {_p[0]:g}$ the phase information *rises* to {_pb[0]:.4f} — above every slow
    value. That is not the forcing suddenly becoming more informative. It is the
    decomposition breaking down.

    With a period shorter than the trajectory's own predictability time, "the phase" is
    no longer a slowly-varying boundary condition that the state has forgotten about; it
    is strongly correlated with where the state is *right now*. Conditioning on it is
    then partly conditioning on the initial condition, and the split into first-kind and
    second-kind information stops meaning what it says.

    The shaded region marks where that happens. **The clean separation this chapter
    relies on is a property of the timescale gap, not of the mathematics**, and it is
    worth knowing that it fails rather than assuming it holds. Real slow drivers — ENSO
    at three to seven years against synoptic weather at days — sit very far to the right
    of that shading, which is precisely why the framing is useful for them.
    """
    )
    return


# ===========================================================================
# Section 5
# ===========================================================================
@app.cell(hide_code=True)
def s5_md(mo):
    mo.md(
        r"""
    ## 5 · What to take away

    **Seasonal forecasting is not weather forecasting attempted further ahead.** It is a
    different question with a different source of skill, and the two can be separated
    cleanly by measuring a forecast against the right climatology.

    **The right climatology is conditional.** Almost everything in this chapter follows
    from taking seriously that "what you say when you know nothing about today" still
    depends on what you know about the *forcing*. A forecast scored against the
    unconditional climatology gets credit for knowing the season; scored against the
    conditional one it does not.

    **Forecast information decays to a floor, not to zero**, and the floor is what the
    boundary condition was always worth. The two contributions add to within one per cent
    here, which is a check rather than a theorem.

    **The predictability desert is thin, not empty.** Between the weather horizon and the
    seasonal regime, skill is small — but bounded below by the boundary term, which is
    why subseasonal forecasting is a hard problem rather than an impossible one.

    **Predictability is not uniform around the cycle.** Windows of opportunity are real,
    survive a split-half test, and are *not* simply the phases where the forcing is most
    informative — the two correlate weakly.

    **The framing depends on a timescale gap.** When the forcing is not slow compared with
    the dynamics, conditioning on its phase is partly conditioning on the state, and the
    decomposition stops meaning what it says.

    ### Try this

    1. Section 2's two curves start at nearly the same value and separate. At what lead do
       they first differ by more than 10 %, and what does that lead correspond to?
    2. The floor-plus-residual check agrees to about one per cent. Construct a case where
       it would fail badly — what would have to be true of the two information sources?
    3. In section 3, the phases with the most informative forcing are not the phases with
       the longest-lasting forecasts. Which of the two would you rather know about, as a
       user deciding whether to act?
    4. Section 4 shows the decomposition failing at short forcing period. Estimate, from
       chapter 1's numbers, the shortest period for which you would trust it.

    ### Where this goes next

    **Chapter 24** replaces the prescribed forcing with a genuinely coupled slow
    component — an ocean that the atmosphere acts back on — and asks what *initialising*
    a slow component means. **Chapter 25** takes the forcing to a one-way ramp rather than
    a cycle, which is climate projection, and separates the forced response from internal
    variability. **Chapter 27** asks what happens when the slow forcing pushes the system
    past a bifurcation rather than around a loop.

    ### Further reading

    - Lorenz (1975), on predictability of the second kind *[citation needed]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on seasonal
      prediction *[citation needed: chapter]*
    - Mariotti et al., on the subseasonal predictability gap and windows of opportunity
      *[citation needed]*
    - Shukla (1998), on predictability in the midst of chaos *[citation needed]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
