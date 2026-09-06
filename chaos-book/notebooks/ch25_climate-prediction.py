# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 25 -- Climate prediction and projection.

Why the attractor's statistics are predictable when its trajectory is not;
forced response against internal variability; and two times of emergence that
answer different questions.

Part VI of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The experiments are precomputed by `scripts/generate_ch25_data.py`.

To edit:   marimo edit notebooks/ch25_climate-prediction.py
To export: make nb-one NB=ch25_climate-prediction
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 25: Climate Prediction and Projection")


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

    from chaoslib import integrate, plotting, systems

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
        C_SPREAD, C_START, C_TRUTH, MPL_SEQUENTIAL, finish_mpl, integrate, mo,
        mpl_panels, np, plotting, plt, systems,
    )


@app.cell
def chapter_data():
    # Precomputed by scripts/generate_ch25_data.py (~1 min): a ramped
    # and a control ensemble of 400 from shared start states, a tight
    # forecast ensemble, and sweeps of ensemble size and ramp rate.
    # All statistics are 20-TU window means: climate is a time average.

    FORECAST_LEADS = (np.float64(0.0), np.float64(2.0), np.float64(4.0), np.float64(6.0), np.float64(8.0), np.float64(10.0), np.float64(12.0), np.float64(14.0), np.float64(16.0), np.float64(18.0), np.float64(20.0), np.float64(22.0), np.float64(24.0), np.float64(26.0), np.float64(28.0), np.float64(30.0), np.float64(32.0), np.float64(34.0), np.float64(36.0), np.float64(38.0), np.float64(40.0), np.float64(42.0), np.float64(44.0), np.float64(46.0), np.float64(48.0), np.float64(50.0), np.float64(52.0), np.float64(54.0), np.float64(56.0), np.float64(58.0), np.float64(60.0))
    FORECAST_MEMBERS = 200
    WINDOW = 20.0
    CLIM_SD = 8.614005
    MEMBER_RMSE = (
        8.752429, 9.554646, 10.044191, 10.371465, 10.686302, 10.874164, 11.454356, 11.995368,
        12.325930, 12.449236, 12.534826, 12.626177, 12.595108, 12.602538, 12.608274, 12.682556,
        12.756942, 12.664634, 12.626886, 12.629918, 12.687895, 12.541275, 12.552918, 12.609017,
        12.643801, 12.607589, 12.652382, 12.757838, 12.802362, 12.758663, 12.776320,
    )
    FORCED_TRUTH = (
        24.023182, 24.114435, 23.921653, 24.020521, 24.041470, 23.868405, 24.173613, 24.301160,
        24.637737, 24.393656, 24.427980, 24.457323, 24.628869, 24.652244, 24.839849, 25.001687,
        24.813131, 24.962207, 25.141805, 25.391390, 25.125554, 25.330527, 25.585583, 25.742263,
        25.715352, 25.848652, 26.009638, 25.889329, 25.816676, 25.815000, 26.314535,
    )
    FORCED_ENSEMBLE = (
        23.780476, 23.786697, 23.838961, 23.920991, 23.969040, 23.995353, 24.153112, 24.240079,
        24.387961, 24.464874, 24.527330, 24.638209, 24.734457, 24.835042, 24.941104, 25.070914,
        25.126920, 25.244477, 25.347949, 25.442668, 25.543873, 25.662913, 25.782971, 25.862730,
        25.969540, 26.079407, 26.154465, 26.183191, 26.243738, 26.312227, 26.369272,
    )
    MEAN_ERROR = (
        0.242706, 0.327738, 0.082692, 0.099530, 0.072430, 0.126948, 0.020500, 0.061082,
        0.249776, 0.071219, 0.099350, 0.180886, 0.105588, 0.182798, 0.101255, 0.069227,
        0.313789, 0.282270, 0.206144, 0.051278, 0.418320, 0.332386, 0.197388, 0.120467,
        0.254189, 0.230755, 0.144827, 0.293862, 0.427062, 0.497227, 0.054737,
    )


    RATE = 0.05
    MEMBERS = 400
    RHO_START = 28.0
    THRESHOLDS = (1.0, 2.0)
    HORIZON = 320.0
    TIMES = (
        10.0000, 12.0000, 14.0000, 16.0000, 18.0000, 20.0000, 22.0000, 24.0000, 26.0000, 28.0000,
        30.0000, 32.0000, 34.0000, 36.0000, 38.0000, 40.0000, 42.0000, 44.0000, 46.0000, 48.0000,
        50.0000, 52.0000, 54.0000, 56.0000, 58.0000, 60.0000, 62.0000, 64.0000, 66.0000, 68.0000,
        70.0000, 72.0000, 74.0000, 76.0000, 78.0000, 80.0000, 82.0000, 84.0000, 86.0000, 88.0000,
        90.0000, 92.0000, 94.0000, 96.0000, 98.0000, 100.0000, 102.0000, 104.0000, 106.0000, 108.0000,
        110.0000, 112.0000, 114.0000, 116.0000, 118.0000, 120.0000, 122.0000, 124.0000, 126.0000, 128.0000,
        130.0000, 132.0000, 134.0000, 136.0000, 138.0000, 140.0000, 142.0000, 144.0000, 146.0000, 148.0000,
        150.0000, 152.0000, 154.0000, 156.0000, 158.0000, 160.0000, 162.0000, 164.0000, 166.0000, 168.0000,
        170.0000, 172.0000, 174.0000, 176.0000, 178.0000, 180.0000, 182.0000, 184.0000, 186.0000, 188.0000,
        190.0000, 192.0000, 194.0000, 196.0000, 198.0000, 200.0000, 202.0000, 204.0000, 206.0000, 208.0000,
        210.0000, 212.0000, 214.0000, 216.0000, 218.0000, 220.0000, 222.0000, 224.0000, 226.0000, 228.0000,
        230.0000, 232.0000, 234.0000, 236.0000, 238.0000, 240.0000, 242.0000, 244.0000, 246.0000, 248.0000,
        250.0000, 252.0000, 254.0000, 256.0000, 258.0000, 260.0000, 262.0000, 264.0000, 266.0000, 268.0000,
        270.0000, 272.0000, 274.0000, 276.0000, 278.0000, 280.0000, 282.0000, 284.0000, 286.0000, 288.0000,
        290.0000, 292.0000, 294.0000, 296.0000, 298.0000, 300.0000, 302.0000, 304.0000, 306.0000, 308.0000,
        310.0000,
    )
    SIGNAL = (
        0.482833, 0.595442, 0.681422, 0.784112, 0.876160, 0.966225, 1.085887, 1.171754, 1.257865, 1.392153,
        1.471533, 1.549106, 1.677466, 1.773132, 1.874909, 1.988751, 2.075417, 2.200686, 2.290958, 2.398212,
        2.520018, 2.632429, 2.704116, 2.806582, 2.916301, 3.021416, 3.108760, 3.195303, 3.304442, 3.381789,
        3.481254, 3.601454, 3.725149, 3.814355, 3.909277, 3.991487, 4.128080, 4.234778, 4.326090, 4.434137,
        4.536239, 4.618462, 4.713666, 4.844471, 4.935736, 5.042662, 5.128835, 5.230633, 5.331895, 5.427404,
        5.522014, 5.624598, 5.733491, 5.820553, 5.921731, 6.043196, 6.127672, 6.219308, 6.312101, 6.437006,
        6.523935, 6.625382, 6.720398, 6.821714, 6.935950, 7.034147, 7.144032, 7.248929, 7.360144, 7.430913,
        7.534067, 7.649089, 7.743914, 7.830476, 7.927412, 8.021470, 8.121474, 8.232544, 8.328510, 8.441348,
        8.572089, 8.628238, 8.728062, 8.848020, 8.955328, 9.061067, 9.167611, 9.238661, 9.352490, 9.462647,
        9.537385, 9.668803, 9.783700, 9.871066, 9.957045, 10.048749, 10.132815, 10.256810, 10.349447, 10.434160,
        10.530505, 10.615164, 10.703499, 10.802527, 10.904834, 11.003007, 11.114436, 11.207674, 11.302304, 11.418171,
        11.515770, 11.598775, 11.703082, 11.797249, 11.900635, 11.995713, 12.096284, 12.210589, 12.301607, 12.388515,
        12.473355, 12.603772, 12.682691, 12.793066, 12.879666, 12.977066, 13.078000, 13.162484, 13.256261, 13.354669,
        13.471731, 13.548428, 13.665938, 13.745742, 13.843249, 13.947633, 14.037783, 14.135682, 14.243695, 14.345313,
        14.437054, 14.544970, 14.645619, 14.735700, 14.831936, 14.929870, 15.042800, 15.126770, 15.218243, 15.302084,
        15.415089,
    )
    INTERNAL = (
        8.594602, 8.586344, 8.600785, 8.591542, 8.596646, 8.592550, 8.597634, 8.591307, 8.593395, 8.605593,
        8.597870, 8.586732, 8.594794, 8.611021, 8.589191, 8.592397, 8.587301, 8.598692, 8.586478, 8.583873,
        8.600772, 8.609253, 8.594096, 8.587760, 8.610990, 8.612085, 8.612250, 8.605647, 8.606825, 8.607945,
        8.595318, 8.609333, 8.622094, 8.606162, 8.595355, 8.583584, 8.608925, 8.607478, 8.607143, 8.591004,
        8.607997, 8.592992, 8.593267, 8.608192, 8.614039, 8.624432, 8.601566, 8.604985, 8.608438, 8.617004,
        8.606419, 8.611859, 8.610073, 8.596402, 8.600416, 8.599842, 8.599560, 8.594643, 8.597707, 8.610058,
        8.593637, 8.603695, 8.600039, 8.615844, 8.614301, 8.613025, 8.621819, 8.622361, 8.618283, 8.609165,
        8.612649, 8.616734, 8.610450, 8.595016, 8.589805, 8.588972, 8.593274, 8.598836, 8.603482, 8.608930,
        8.625614, 8.602499, 8.601745, 8.622219, 8.627568, 8.629299, 8.622236, 8.620303, 8.622499, 8.615008,
        8.617701, 8.627983, 8.638112, 8.624913, 8.618969, 8.623451, 8.621076, 8.615174, 8.614938, 8.635781,
        8.608765, 8.600648, 8.598461, 8.590431, 8.598171, 8.599712, 8.602072, 8.607258, 8.603191, 8.597939,
        8.611826, 8.605434, 8.606560, 8.607799, 8.605705, 8.597975, 8.604443, 8.618476, 8.607310, 8.594969,
        8.592280, 8.607795, 8.598282, 8.607204, 8.608097, 8.604543, 8.601717, 8.584352, 8.593419, 8.597101,
        8.613817, 8.599138, 8.613347, 8.607419, 8.597567, 8.609922, 8.606395, 8.606408, 8.616155, 8.617055,
        8.610981, 8.617238, 8.613526, 8.612869, 8.608906, 8.609313, 8.612618, 8.622040, 8.599554, 8.600062,
        8.593596,
    )
    TOE_SINGLE = (
        171.6500, float("nan"),
    )
    TOE_ENSEMBLE = (
        10.0000, 17.7500,
    )

    SIZES = (2, 5, 10, 20, 50, 100, 200, 400)
    RATES = (0.0125, 0.025, 0.05, 0.1)
    RATE_PRODUCT = (
        1.19788, 1.22000, 1.20700, 1.23900,
    )
    SIZE_PRODUCT = (
        212.1320, 310.5898, 324.9873, 327.2262, 343.7953, 342.3000, 341.3912, 355.0000,
    )
    SIZE_TOE = (
        150.0000, 138.9000, 102.7700, 73.1700, 48.6200, 34.2300, 24.1400, 17.7500,
    )
    RATE_TOE_SINGLE = (
        float("nan"), float("nan"), float("nan"), 173.5100,
    )
    RATE_TOE_ENSEMBLE = (
        95.8300, 48.8000, 24.1400, 12.3900,
    )

    INIT_MAX_DIFF = 0.092871
    INIT_MEAN_DIFF = 0.016217
    INIT_INTERNAL = 8.605765
    RESPONSE_A = (
        0.482833, 0.595442, 0.681422, 0.784112, 0.876160, 0.966225, 1.085887, 1.171754, 1.257865, 1.392153,
        1.471533, 1.549106, 1.677466, 1.773132, 1.874909, 1.988751, 2.075417, 2.200686, 2.290958, 2.398212,
        2.520018, 2.632429, 2.704116, 2.806582, 2.916301, 3.021416, 3.108760, 3.195303, 3.304442, 3.381789,
        3.481254, 3.601454, 3.725149, 3.814355, 3.909277, 3.991487, 4.128080, 4.234778, 4.326090, 4.434137,
        4.536239, 4.618462, 4.713666, 4.844471, 4.935736, 5.042662, 5.128835, 5.230633, 5.331895, 5.427404,
        5.522014, 5.624598, 5.733491, 5.820553, 5.921731, 6.043196, 6.127672, 6.219308, 6.312101, 6.437006,
        6.523935, 6.625382, 6.720398, 6.821714, 6.935950, 7.034147, 7.144032, 7.248929, 7.360144, 7.430913,
        7.534067, 7.649089, 7.743914, 7.830476, 7.927412, 8.021470, 8.121474, 8.232544, 8.328510, 8.441348,
        8.572089, 8.628238, 8.728062, 8.848020, 8.955328, 9.061067, 9.167611, 9.238661, 9.352490, 9.462647,
        9.537385, 9.668803, 9.783700, 9.871066, 9.957045, 10.048749, 10.132815, 10.256810, 10.349447, 10.434160,
        10.530505, 10.615164, 10.703499, 10.802527, 10.904834, 11.003007, 11.114436, 11.207674, 11.302304, 11.418171,
        11.515770, 11.598775, 11.703082, 11.797249, 11.900635, 11.995713, 12.096284, 12.210589, 12.301607, 12.388515,
        12.473355, 12.603772, 12.682691, 12.793066, 12.879666, 12.977066, 13.078000, 13.162484, 13.256261, 13.354669,
        13.471731, 13.548428, 13.665938, 13.745742, 13.843249, 13.947633, 14.037783, 14.135682, 14.243695, 14.345313,
        14.437054, 14.544970, 14.645619, 14.735700, 14.831936, 14.929870, 15.042800, 15.126770, 15.218243, 15.302084,
        15.415089,
    )
    RESPONSE_B = (
        0.475938, 0.584794, 0.671875, 0.764158, 0.884489, 0.965602, 1.056284, 1.157373, 1.293296, 1.384679,
        1.479018, 1.572561, 1.688594, 1.787867, 1.861831, 2.000299, 2.097877, 2.176695, 2.265321, 2.353522,
        2.469956, 2.571932, 2.676073, 2.796204, 2.896548, 2.990892, 3.094029, 3.211652, 3.315469, 3.428839,
        3.520122, 3.637835, 3.722570, 3.831031, 3.931572, 4.032163, 4.123297, 4.230382, 4.329277, 4.449384,
        4.548788, 4.642061, 4.761516, 4.814441, 4.944734, 5.023110, 5.143471, 5.229803, 5.329602, 5.424198,
        5.530630, 5.621188, 5.735169, 5.841888, 5.909197, 6.022520, 6.112534, 6.227071, 6.335416, 6.438132,
        6.541985, 6.630918, 6.716415, 6.820426, 6.932056, 7.063148, 7.147466, 7.232061, 7.338537, 7.430539,
        7.531948, 7.636119, 7.730278, 7.860060, 7.948150, 8.022662, 8.143243, 8.240132, 8.328602, 8.445742,
        8.539421, 8.644456, 8.746051, 8.827711, 8.932103, 9.050761, 9.144853, 9.250226, 9.361502, 9.445584,
        9.542639, 9.662414, 9.744231, 9.841580, 9.961250, 10.036625, 10.137566, 10.252924, 10.331474, 10.434655,
        10.536632, 10.612775, 10.733616, 10.845587, 10.931186, 11.028235, 11.124755, 11.200008, 11.308652, 11.429760,
        11.521867, 11.612469, 11.706735, 11.803568, 11.894808, 12.017107, 12.089351, 12.212122, 12.326724, 12.371731,
        12.488085, 12.599570, 12.690848, 12.778763, 12.888039, 12.968980, 13.094086, 13.173434, 13.250968, 13.389380,
        13.468741, 13.560869, 13.661453, 13.772855, 13.861121, 13.959442, 14.060893, 14.154032, 14.261164, 14.346787,
        14.458526, 14.564701, 14.653182, 14.729990, 14.840212, 14.936189, 15.041405, 15.134417, 15.234865, 15.320657,
        15.419573,
    )

    return (
        CLIM_SD, FORCED_ENSEMBLE, FORCED_TRUTH, FORECAST_LEADS,
        FORECAST_MEMBERS, HORIZON, INIT_INTERNAL, INIT_MAX_DIFF,
        INIT_MEAN_DIFF, INTERNAL, MEAN_ERROR, MEMBERS, MEMBER_RMSE, RATE,
        RATES, RATE_PRODUCT, RATE_TOE_ENSEMBLE, RATE_TOE_SINGLE,
        RESPONSE_A, RESPONSE_B, RHO_START, SIGNAL, SIZES, SIZE_PRODUCT,
        SIZE_TOE, THRESHOLDS, TIMES, TOE_ENSEMBLE, TOE_SINGLE, WINDOW,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(RATE, RHO_START, mo):
    mo.md(
        rf"""
    # Chapter 25 · Climate Prediction and Projection

    **Part VI — Predictability of the second kind.**

    **The forecasting question.** The question this whole book has been circling. If
    weather is unpredictable beyond a fortnight, on what basis is anyone confident about
    2100?

    The answer is that they are different questions. Weather forecasting asks for the
    **trajectory on the attractor**. Projection asks how the **attractor itself**
    responds to a change in forcing. One of those becomes impossible after two weeks; the
    other does not, and this chapter measures both on the same system to show why.

    Lorenz 63 with a one-way ramp:

    $$
    \rho(t) = {RHO_START:g} + {RATE:g}\,t .
    $$

    Chapter 23's forcing went round a loop; chapter 24's was generated internally by the
    system. This one only goes up, which is the shape of an emissions scenario.

    /// admonition | What this model is not
        type: warning

    Lorenz 63 has no thermodynamics, and $\rho$ is not a greenhouse gas. Nothing here is
    a statement about how much the atmosphere warms, or when. What transfers is the
    **structure** of a forced-response problem — a distribution moving under a trend,
    against internal variability that does not shrink — and the *reasoning* that goes
    with it. Every number below is a property of Lorenz 63.
    ///

    ---

    **What you need before this chapter.** **Chapter 1** for the two kinds of
    predictability. **Chapter 24** for initialised prediction, which this chapter
    contrasts with directly.
    """
    )
    return


# ===========================================================================
# Section 1
# ===========================================================================
@app.cell(hide_code=True)
def s1_md(FORECAST_MEMBERS, WINDOW, mo):
    mo.md(
        rf"""
    ## 1 · The trajectory goes; the statistics stay

    Start with the two questions side by side. An ensemble of {FORECAST_MEMBERS} members
    from a tight blob, under the ramp, and two things measured about it:

    * how far an **individual member** is from the truth — the weather-forecasting
      question;
    * how well the **ensemble's mean** matches the truth's mean — the climate question.

    Both averaged over a {WINDOW:g}-time-unit window, because *climate is a time average*
    and an instantaneous mean of a chaotic variable is not one.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    CLIM_SD, C_ANALYSIS, C_BG, C_PERT, C_TRUTH, FORCED_ENSEMBLE, FORCED_TRUTH,
    FORECAST_LEADS, MEMBER_RMSE, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(FORECAST_LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("An individual forecast", "The ensemble's mean"),
        figsize=(10.2, 4.0),
    )
    _ax0.plot(_leads, np.asarray(MEMBER_RMSE) / CLIM_SD, "-", color=C_PERT,
              linewidth=2.5)
    _ax0.axhline(np.sqrt(2.0), color=C_BG, linestyle="--", linewidth=1.5)
    # Left end, just above the line: the curve is at ~1.0 there, so the label
    # clears both it and the dashed line it names.
    _ax0.text(_leads[0] + 1.0, np.sqrt(2.0) * 1.04, r"$\sqrt{2}$ — saturation",
              ha="left", va="bottom", fontsize=8, color=C_BG)
    _ax0.set_xlabel("lead (time units)")
    _ax0.set_ylabel("RMSE / climatological spread")
    _ax0.set_ylim(0, 1.8)

    _ax1.plot(_leads, np.asarray(FORCED_TRUTH), "-", color=C_TRUTH,
              linewidth=2.8, label="truth, windowed mean")
    _ax1.plot(_leads, np.asarray(FORCED_ENSEMBLE), "--", color=C_ANALYSIS,
              linewidth=2.2, label="ensemble, windowed mean")
    _ax1.set_xlabel("lead (time units)")
    _ax1.set_ylabel("mean $z$")
    _ax1.legend(fontsize=8.5, framealpha=0.9)
    finish_mpl(
        _fig,
        "The same forecast, asked the weather question and the climate question",
    )
    return


@app.cell(hide_code=True)
def s1_note(
    CLIM_SD, FORCED_ENSEMBLE, FORCED_TRUTH, FORECAST_LEADS, MEAN_ERROR,
    MEMBER_RMSE, WINDOW, mo, np,
):
    _rmse = np.asarray(MEMBER_RMSE) / CLIM_SD
    _err = np.asarray(MEAN_ERROR)
    _half = len(_err) // 2
    mo.md(
        rf"""
    **The individual forecast is gone.** Its error saturates at {_rmse[-1]:.2f}
    climatological spreads — near the $\sqrt2 \approx 1.41$ that two independent draws
    from the same distribution give, and slightly above it because the ramp is widening
    the attractor within each averaging window. A member is no longer a forecast; it is a
    sample of the climatology.

    **The ensemble's mean is not gone at all.** It tracks the truth's windowed mean to
    {np.mean(_err[_half:]):.3f} — about
    {100 * np.mean(_err[_half:]) / CLIM_SD:.0f} % of a climatological spread — while
    both rise together from {FORCED_TRUTH[0]:.1f} to {FORCED_TRUTH[-1]:.1f}.

    That is the whole argument of climate projection in one figure, and it is not a
    paradox. The two panels ask different questions of the same numbers. One asks *where
    the system is*, which chaos destroys. The other asks *what the distribution is*,
    which the forcing controls.

    One detail worth reading correctly: every point is a {WINDOW:g}-unit window average
    centred on that lead, so the leftmost point already contains the ensemble's rapid
    initial divergence. The actual spread at $t=0$ is 0.05, far below the bottom of the
    left-hand axis.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(MEMBERS, RATE, mo):
    mo.md(
        rf"""
    ## 2 · Forced response and internal variability

    Now the standard decomposition. Two ensembles of {MEMBERS} members from the **same**
    set of climatological start states: one with the ramp, one without.

    * the difference of their means is the **forced response** — what the forcing did;
    * the control's spread is **internal variability** — what the system does anyway.

    Everything in a projection turns on how those two compare.
    """
    )
    return


@app.cell(hide_code=True)
def s2_fig(
    C_ANALYSIS, C_BG, C_PERT, C_TRUTH, INTERNAL, RATE, RHO_START, SIGNAL,
    TIMES, finish_mpl, mpl_panels, np,
):
    _t = np.asarray(TIMES)
    _s = np.asarray(SIGNAL)
    _n = np.asarray(INTERNAL)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Response and variability", "Signal-to-noise"),
        figsize=(10.2, 4.0),
    )
    _ax0.fill_between(_t, -_n, _n, color=C_BG, alpha=0.3,
                      label="internal variability ($\\pm 1$ sd)")
    _ax0.plot(_t, _s, "-", color=C_PERT, linewidth=2.8,
              label="forced response")
    _ax0.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.2)
    _ax0.set_xlabel("time (time units)")
    _ax0.set_ylabel("$z$ anomaly")
    _ax0.legend(fontsize=8.5, framealpha=0.9, loc="upper left")

    _ax1.plot(_t, np.abs(_s) / _n, "-", color=C_ANALYSIS, linewidth=2.6)
    for _level, _label in ((1.0, "S/N = 1"), (2.0, "S/N = 2")):
        _ax1.axhline(_level, color=C_BG, linestyle=":", linewidth=1.5)
        _ax1.text(_t[0], _level * 1.04, _label, fontsize=8, color=C_BG)
    _ax1.set_xlabel("time (time units)")
    _ax1.set_ylabel("signal / internal variability")
    finish_mpl(
        _fig,
        f"$\\rho$ rises from {RHO_START:g} to "
        f"{RHO_START + RATE * _t[-1]:.0f} over this run",
    )
    return


@app.cell(hide_code=True)
def s2_note(INTERNAL, SIGNAL, TIMES, mo, np):
    _s = np.asarray(SIGNAL)
    _n = np.asarray(INTERNAL)
    mo.md(
        rf"""
    **The response grows without bound. The variability does not change at all.** Over
    the whole run internal variability stays between {_n.min():.3f} and {_n.max():.3f} —
    a spread of {100 * (_n.max() / _n.min() - 1):.1f} % — while the forced response goes
    from {_s[0]:.2f} to {_s[-1]:.2f}.

    That is why the signal-to-noise ratio rises: not because the system becomes quieter,
    but because the signal outgrows a noise that stays put. If internal variability *did*
    shrink under forcing, detection would be a far easier problem and this chapter would
    be much shorter.

    It is also why a projection can be confident about the mean and say nothing whatever
    about any particular year. The shaded band never narrows.
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(MEMBERS, mo):
    mo.md(
        rf"""
    ## 3 · Two times of emergence

    "Time of emergence" is when the forced signal becomes detectable against the noise.
    But *whose* noise? Two different questions hide under one phrase, and they differ by
    a factor of $\sqrt{{K}}$:

    * **A single realisation** — the real world, of which we have exactly one — must
      beat the full internal variability $\sigma$. This is what an observational record
      can detect.
    * **An ensemble of $K$** must beat only the standard error of its own mean,
      $\sigma/\sqrt{{K}}$. This is what a modelling centre can detect.

    Both are legitimate. Quoting one while meaning the other is not.
    """
    )
    return


@app.cell(hide_code=True)
def s3_fig(
    C_ANALYSIS, C_BG, C_PERT, MEMBERS, SIZES, SIZE_PRODUCT, SIZE_TOE,
    RATES, RATE_TOE_ENSEMBLE, RATE_TOE_SINGLE, finish_mpl, mpl_panels, np,
):
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Emergence against ensemble size",
                "Emergence against ramp rate"),
        figsize=(10.2, 4.0),
    )
    _k = np.asarray(SIZES, dtype=float)
    _toe = np.asarray(SIZE_TOE)
    _ax0.loglog(_k, _toe, "o-", color=C_ANALYSIS, markersize=6, linewidth=2.0,
                label="measured")
    _reference = float(np.median(np.asarray(SIZE_PRODUCT)))
    _ax0.loglog(_k, _reference / np.sqrt(_k), "--", color=C_BG, linewidth=1.6,
                label=r"$\propto 1/\sqrt{K}$")
    _ax0.set_xlabel("ensemble members $K$")
    _ax0.set_ylabel("time of emergence (S/N > 2)")
    _ax0.set_xticks(_k)
    _ax0.set_xticklabels([f"{int(v)}" for v in _k], fontsize=7.5)
    _ax0.legend(fontsize=8, framealpha=0.9)

    _r = np.asarray(RATES)
    _ens = np.asarray(RATE_TOE_ENSEMBLE)
    _single = np.asarray(RATE_TOE_SINGLE)
    _ax1.loglog(_r, _ens, "o-", color=C_ANALYSIS, markersize=6, linewidth=2.0,
                label=f"ensemble of 200")
    _finite = np.isfinite(_single)
    _ax1.loglog(_r[_finite], _single[_finite], "s", color=C_PERT, markersize=8,
                label="single realisation")
    _product = float(np.median(_r * _ens))
    _ax1.loglog(_r, _product / _r, "--", color=C_BG, linewidth=1.6,
                label=r"$\propto 1/\mathrm{rate}$")
    _ax1.set_xlabel("ramp rate")
    _ax1.set_ylabel("time of emergence (S/N > 2)")
    _ax1.set_xticks(_r)
    _ax1.set_xticklabels([f"{v:g}" for v in _r], fontsize=7.5)
    _ax1.legend(fontsize=8, framealpha=0.9)
    finish_mpl(_fig, "Dashed lines are the predicted laws, not fits")
    return


@app.cell(hide_code=True)
def s3_note(
    HORIZON, MEMBERS, RATE_PRODUCT, SIZE_PRODUCT, TOE_ENSEMBLE,
    TOE_SINGLE, mo, np,
):
    _single = np.asarray(TOE_SINGLE)
    _ens = np.asarray(TOE_ENSEMBLE)
    _rp = np.asarray(RATE_PRODUCT)
    _sp = np.asarray(SIZE_PRODUCT)
    mo.md(
        rf"""
    At the standard threshold of two, a **single realisation never detects the trend**
    within the {HORIZON:.0f} time units of this run, while an
    ensemble of {MEMBERS} detects it at $t = {_ens[1]:.0f}$. At a threshold of one, the
    single realisation needs $t = {_single[0]:.0f}$ and the ensemble detects it
    immediately.

    **The two scaling laws are laws, not fits.** Signal grows as $\text{{rate}}\times t$
    and ensemble noise falls as $\sigma/\sqrt K$, so emergence should go as
    $1/\text{{rate}}$ and as $1/\sqrt K$. Measured:

    * $\mathrm{{ToE}} \times \text{{rate}}$ = {", ".join(f"{v:.2f}" for v in _rp)} —
      constant to {100 * (_rp.max() / _rp.min() - 1):.0f} %;
    * $\mathrm{{ToE}} \times \sqrt K$ = {", ".join(f"{v:.0f}" for v in _sp)} — constant
      from $K = 10$ upward, and breaking at the smallest ensembles where the noise
      estimate is itself made from a handful of members.

    The dashed lines in the figure are those laws with a single constant, not regressions
    through the points.

    /// admonition | Which number should be quoted?
        type: note

    The ensemble figure is the smaller and more flattering one, and it is the *wrong*
    answer to the question most people are asking. "When will this be detectable?" almost
    always means detectable in **the observed record** — of which there is one
    realisation, not {MEMBERS}. The ensemble number answers a different question: when a
    modelling centre, averaging away internal variability it can generate at will, can
    establish that a response exists.

    Both are useful. A statement that does not say which is being used is not.
    ///
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
    ## 4 · Initialisation is irrelevant here

    Chapter 24 found that initialising a slow component was worth about eight time units
    of predictability. The natural question is what initialisation buys a *projection*.

    Below: two ensembles started from entirely different sets of climatological states —
    drawn from separate stretches of a long control run, sharing nothing — each with its
    own control, each giving a forced response.
    """
    )
    return


@app.cell(hide_code=True)
def s4_fig(
    C_ANALYSIS, C_BG, C_PERT, INIT_INTERNAL, RESPONSE_A, RESPONSE_B, TIMES,
    finish_mpl, mpl_panels, np,
):
    _t = np.asarray(TIMES)
    _a, _b = np.asarray(RESPONSE_A), np.asarray(RESPONSE_B)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Two ensembles, different start states",
                "Their difference"),
        figsize=(10.2, 4.0),
    )
    _ax0.fill_between(_t, -INIT_INTERNAL, INIT_INTERNAL, color=C_BG, alpha=0.25,
                      label="internal variability")
    _ax0.plot(_t, _a, "-", color=C_ANALYSIS, linewidth=2.8, label="start set A")
    _ax0.plot(_t, _b, "--", color=C_PERT, linewidth=2.0, label="start set B")
    _ax0.set_ylabel("forced response")
    _ax0.legend(fontsize=8.5, framealpha=0.9, loc="upper left")

    _ax1.plot(_t, _a - _b, "-", color=C_PERT, linewidth=2.0)
    _ax1.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.2)
    _ax1.set_ylim(-INIT_INTERNAL, INIT_INTERNAL)
    _ax1.text(_t[0], INIT_INTERNAL * 0.82,
              "axis spans $\\pm 1$ sd of internal variability", fontsize=8,
              color=C_BG)
    _ax1.set_ylabel("difference")

    for _ax in (_ax0, _ax1):
        _ax.set_xlabel("time (time units)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s4_note(INIT_INTERNAL, INIT_MAX_DIFF, INIT_MEAN_DIFF, mo):
    mo.md(
        rf"""
    The two responses differ by at most **{INIT_MAX_DIFF:.3f}** — about
    {100 * INIT_MAX_DIFF / INIT_INTERNAL:.0f} % of one internal-variability standard
    deviation — and by {INIT_MEAN_DIFF:.3f} on average. On the right-hand panel, whose
    axis spans a full $\pm 1$ sd, the difference is a flat line.

    **So initialisation buys a projection nothing.** That is not a defect; it is the
    definition. A projection is a statement about the *distribution* the forcing
    produces, and the distribution has forgotten the initial condition — which is exactly
    what chapter 1 measured as first-kind information decaying to zero.

    Put beside chapter 24, this is the cleanest statement of the difference between the
    two enterprises:

    | | initialise? | what it predicts |
    |---|---|---|
    | **decadal prediction** | yes — worth ~8 TU | the trajectory of a slow variable |
    | **climate projection** | no — worth nothing | the distribution under a forcing |

    It also explains a practical asymmetry. A decadal prediction system needs an
    observing system, an assimilation system and a re-forecast archive, and chapter 24
    showed how each of those can go wrong. A projection needs none of them — and pays for
    it by being unable to tell you about any particular decade.
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

    **Two questions, one system.** The trajectory becomes unpredictable and the
    statistics do not. An individual member saturates at $\sqrt2$ climatological spreads
    while the ensemble mean tracks the truth to a few per cent of one. Both are true at
    the same time, of the same forecast.

    **Internal variability does not shrink under forcing.** The signal-to-noise ratio
    rises because the signal grows, not because the system quietens. A projection is
    therefore confident about the mean and silent about any particular year, permanently.

    **There are two times of emergence and they differ by $\sqrt K$.** One is what a
    single observed record can detect; the other is what a large ensemble can establish.
    The second is smaller and more flattering, and is usually the wrong answer to the
    question being asked.

    **Emergence scales as $1/\text{rate}$ and $1/\sqrt{K}$** — laws, checked here to a
    few per cent, not fitted.

    **Initialisation is worth nothing to a projection**, where chapter 24 found it worth
    eight time units to a prediction. That single contrast separates the two enterprises
    more sharply than any amount of definition.

    ### Try this

    1. Section 1's two panels use the same forecast. Which of them would change if the
       ensemble had 20 members instead of 200, and which would not?
    2. Internal variability stays flat here. Name a physical mechanism that would make it
       *grow* under forcing, and say what that would do to the emergence time.
    3. The single-realisation emergence at S/N > 2 never arrives within the run. Using the
       $1/\text{rate}$ law, estimate the ramp rate at which it would arrive by $t = 200$.
    4. Section 4 shows initialisation is worthless for the forced response. At what lead
       does it stop being worth anything — and how would you measure that crossover with
       chapter 24's machinery?

    ### Where this goes next

    **Chapter 26** asks what "initialising" means for components slower still — the deep
    ocean, the carbon cycle, ice sheets — where the memory outlives the observational
    record. **Chapter 27** asks what happens when a ramp like this one carries the system
    past a bifurcation rather than smoothly along, which is where the reassuring
    distinction between trajectory and statistics breaks down.

    ### Further reading

    - Hawkins & Sutton (2009), on the sources of uncertainty in climate projections
      *[citation needed]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*
      *[citation needed: chapter]*
    - Deser et al., on internal variability and large ensembles *[citation needed]*
    - Lorenz (1975), on predictability of the second kind *[citation needed]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
