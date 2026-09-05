# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 1 -- What is predictability?

Four things the word is used for, separated and then measured: practical
against intrinsic limits, and predictability of the first kind against the
second. The unifying idea is that a forecast is a probability distribution, and
skill is its distance from climatology.

Part I of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The experiments are precomputed by `scripts/generate_ch01_data.py`.

To edit:   marimo edit notebooks/ch01_what-is-predictability.py
To export: make nb-one NB=ch01_what-is-predictability
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 1: What is Predictability?")


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
    # Precomputed by scripts/generate_ch01_data.py (~20 s). Lorenz 63:
    # 200 scored cases three ways, and the two kinds of predictability
    # measured in the same units. Knob-free; the selector slices.

    SCORE_LEADS = (np.float64(0.0), np.float64(0.25), np.float64(0.5), np.float64(0.75), np.float64(1.0), np.float64(1.25), np.float64(1.5), np.float64(1.75), np.float64(2.0), np.float64(2.25), np.float64(2.5), np.float64(2.75), np.float64(3.0), np.float64(3.25), np.float64(3.5), np.float64(3.75), np.float64(4.0), np.float64(4.25), np.float64(4.5), np.float64(4.75), np.float64(5.0), np.float64(5.25), np.float64(5.5), np.float64(5.75), np.float64(6.0), np.float64(6.25), np.float64(6.5), np.float64(6.75), np.float64(7.0), np.float64(7.25), np.float64(7.5), np.float64(7.75), np.float64(8.0), np.float64(8.25), np.float64(8.5), np.float64(8.75), np.float64(9.0), np.float64(9.25), np.float64(9.5), np.float64(9.75), np.float64(10.0), np.float64(10.25), np.float64(10.5), np.float64(10.75), np.float64(11.0), np.float64(11.25), np.float64(11.5), np.float64(11.75), np.float64(12.0))
    SCORE_CASES = 200
    SCORE_MEMBERS = 50
    CLIMATE_SIGMA = 8.560063
    RMSE_CROSSING = 5.7500
    CRPS_CROSSING = 5.5000
    RMSE_SINGLE = (
        0.048936, 0.102396, 0.734018, 0.616738, 0.500177, 1.755492, 1.270588, 2.875721, 3.020888, 3.589162,
        2.868352, 1.921354, 4.477081, 4.775546, 4.942528, 4.387380, 5.295988, 6.084893, 7.345146, 7.644806,
        6.529417, 7.074284, 7.837636, 9.267907, 9.021417, 9.185768, 8.895825, 9.188056, 8.800878, 10.036000,
        9.916157, 10.023977, 11.082343, 11.050777, 10.183041, 11.276790, 11.224045, 9.863640, 10.853692, 11.149566,
        11.052520, 11.708462, 11.157856, 11.036691, 11.520064, 11.598744, 11.604356, 11.034539, 10.515477,
    )
    RMSE_ENS = (
        0.006950, 0.017202, 0.088720, 0.307754, 0.366091, 1.317137, 0.916211, 1.185950, 1.231439, 2.583631,
        1.359576, 1.664609, 2.505947, 2.175841, 2.758238, 3.630535, 3.903062, 4.140629, 5.431565, 4.816519,
        4.310204, 4.890613, 5.365330, 6.149252, 6.393519, 7.041570, 6.888985, 6.390370, 6.157757, 7.354008,
        6.467112, 6.915265, 7.895796, 7.295223, 7.197333, 8.301969, 7.860603, 7.091586, 7.648438, 7.935983,
        7.953001, 8.015727, 8.113104, 7.824371, 7.812873, 7.866378, 8.208881, 7.800521, 7.762535,
    )
    RMSE_CLIM = (
        8.632600, 8.966301, 8.563331, 8.173748, 8.589480, 8.610066, 8.655372, 9.068326, 8.652790, 8.686490,
        8.164777, 8.349946, 8.273946, 8.600612, 8.283397, 8.794811, 8.892496, 8.714631, 9.033300, 8.120088,
        8.051093, 8.489108, 8.633685, 8.806355, 9.029528, 8.668059, 8.834532, 8.483797, 8.382237, 8.527408,
        8.327084, 8.246017, 9.130808, 8.705203, 8.576080, 9.040507, 8.234715, 8.210440, 8.564764, 8.389560,
        8.906044, 8.814111, 8.630059, 8.647549, 8.522959, 8.503150, 8.782260, 8.289592, 8.369530,
    )
    CRPS_SINGLE = (
        0.039107, 0.059941, 0.217517, 0.233103, 0.229715, 0.416222, 0.491059, 0.828338, 1.034765, 1.198179,
        1.200665, 1.122071, 1.780678, 2.052153, 2.288346, 2.316959, 2.628951, 3.331123, 4.038376, 4.473943,
        4.173509, 4.568849, 5.156389, 6.071983, 6.122081, 6.248373, 6.025217, 6.415052, 6.459082, 7.217649,
        7.324580, 7.364715, 8.356088, 8.192791, 7.752801, 9.014278, 8.650585, 7.595717, 8.762727, 8.785516,
        8.560989, 9.074363, 8.582055, 8.996077, 9.311672, 9.213247, 9.436062, 8.817320, 8.292086,
    )
    CRPS_ENS = (
        0.012251, 0.020486, 0.054737, 0.087023, 0.102401, 0.181653, 0.219734, 0.259051, 0.312461, 0.466908,
        0.404235, 0.498277, 0.692883, 0.669599, 0.896267, 1.105919, 1.251422, 1.468612, 1.965367, 1.949145,
        1.892727, 2.102622, 2.473728, 2.746295, 3.019499, 3.409855, 3.382395, 3.252475, 3.156948, 3.755123,
        3.407227, 3.596053, 4.154165, 3.874137, 3.876453, 4.453248, 4.312520, 3.912447, 4.122414, 4.389804,
        4.410380, 4.412113, 4.508636, 4.400368, 4.427114, 4.421178, 4.627405, 4.430147, 4.381571,
    )
    CRPS_CLIM = (
        5.068318, 5.251502, 4.941660, 4.789556, 5.027513, 5.009678, 5.086323, 5.266868, 4.907064, 5.124728,
        4.760596, 4.901086, 4.879130, 5.083554, 4.877506, 5.221453, 5.154937, 5.065367, 5.348922, 4.805979,
        4.746289, 4.927789, 5.029526, 5.033676, 5.318535, 5.155008, 5.144523, 4.982666, 4.803650, 5.023621,
        4.849710, 4.833634, 5.512993, 5.112528, 4.982908, 5.392030, 4.869150, 4.825624, 4.994005, 4.838499,
        5.211454, 5.089376, 5.013920, 5.079033, 5.016218, 4.944079, 5.122293, 4.821284, 4.800687,
    )



    LEADS = (np.float64(0.0), np.float64(0.5), np.float64(1.0), np.float64(1.5), np.float64(2.0), np.float64(2.5), np.float64(3.0), np.float64(3.5), np.float64(4.0), np.float64(4.5), np.float64(5.0), np.float64(5.5), np.float64(6.0), np.float64(6.5), np.float64(7.0), np.float64(7.5), np.float64(8.0), np.float64(8.5), np.float64(9.0), np.float64(9.5), np.float64(10.0), np.float64(10.5), np.float64(11.0), np.float64(11.5), np.float64(12.0), np.float64(12.5), np.float64(13.0), np.float64(13.5), np.float64(14.0), np.float64(14.5), np.float64(15.0), np.float64(15.5), np.float64(16.0), np.float64(16.5), np.float64(17.0), np.float64(17.5), np.float64(18.0), np.float64(18.5), np.float64(19.0), np.float64(19.5), np.float64(20.0))
    BINS = (np.float64(-5.0), np.float64(-3.125), np.float64(-1.25), np.float64(0.625), np.float64(2.5), np.float64(4.375), np.float64(6.25), np.float64(8.125), np.float64(10.0), np.float64(11.875), np.float64(13.75), np.float64(15.625), np.float64(17.5), np.float64(19.375), np.float64(21.25), np.float64(23.125), np.float64(25.0), np.float64(26.875), np.float64(28.75), np.float64(30.625), np.float64(32.5), np.float64(34.375), np.float64(36.25), np.float64(38.125), np.float64(40.0), np.float64(41.875), np.float64(43.75), np.float64(45.625), np.float64(47.5), np.float64(49.375), np.float64(51.25), np.float64(53.125), np.float64(55.0), np.float64(56.875), np.float64(58.75), np.float64(60.625), np.float64(62.5), np.float64(64.375), np.float64(66.25), np.float64(68.125), np.float64(70.0))
    RHO_BASE = 28.0
    RHO_SHIFTS = (29.0, 30.0, 32.0, 36.0)
    N_STARTS = 32
    MEMBERS = 3000
    VARIABLE_NAME = 'z'
    NOISE_FLOOR = 0.009736
    FIRST_MONOTONE = True
    FIRST_KIND = (
        2.755746, 2.779948, 2.681941, 2.406492, 2.182422, 2.069596, 1.895403, 1.570857, 1.457112, 1.264306,
        1.142926, 0.858398, 0.793117, 0.656288, 0.620039, 0.484270, 0.397881, 0.350696, 0.296504, 0.258965,
        0.221778, 0.201865, 0.178026, 0.155712, 0.137582, 0.124046, 0.113609, 0.102206, 0.091589, 0.088075,
        0.075531, 0.071681, 0.068980, 0.059482, 0.056884, 0.055103, 0.047900, 0.044959, 0.045330, 0.038088,
        0.037678,
    )
    FIRST_KIND_SPREAD = (
        0.400300, 0.442692, 0.622773, 0.617985, 0.542444, 0.763003, 0.664450, 0.747387, 0.556232, 0.543895,
        0.622105, 0.428616, 0.494129, 0.419443, 0.559209, 0.419583, 0.244479, 0.256047, 0.197704, 0.153987,
        0.138666, 0.126970, 0.109640, 0.099535, 0.088632, 0.070158, 0.071373, 0.064145, 0.048686, 0.053725,
        0.048536, 0.038043, 0.038428, 0.034874, 0.028642, 0.030906, 0.028377, 0.021940, 0.023084, 0.018497,
        0.017985,
    )
    FIRST_KIND_ERROR = (
        0.070764, 0.078258, 0.110092, 0.109245, 0.095891, 0.134881, 0.117459, 0.132121, 0.098329, 0.096148,
        0.109974, 0.075769, 0.087350, 0.074148, 0.098855, 0.074173, 0.043218, 0.045263, 0.034949, 0.027221,
        0.024513, 0.022445, 0.019382, 0.017595, 0.015668, 0.012402, 0.012617, 0.011339, 0.008606, 0.009497,
        0.008580, 0.006725, 0.006793, 0.006165, 0.005063, 0.005464, 0.005016, 0.003879, 0.004081, 0.003270,
        0.003179,
    )
    SECOND_KIND = (
        0.014521, 0.050520, 0.213444, 0.838503,
    )
    CROSSINGS = (
        float("nan"), 18.0000, 10.5000, 6.0000,
    )
    CLIM_HIST = (
        0.0, 0.0, 0.0, 0.0, 122.0, 530.0, 1020.0, 1629.0, 3120.0, 4436.0,
        5742.0, 6786.0, 7186.0, 7391.0, 6684.0, 5661.0, 5025.0, 4722.0, 4606.0, 4880.0,
        4922.0, 4201.0, 3324.0, 2218.0, 1191.0, 414.0, 175.0, 16.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    SHOW_LEADS = (0.0, 5.0, 12.0)
    SHOW_START = 0
    SHOW_START_Z = 25.3153
    FORECAST_HIST_000 = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    FORECAST_HIST_050 = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 821.0, 2177.0,
        2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    FORECAST_HIST_120 = (
        0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 37.0, 116.0, 431.0,
        736.0, 559.0, 367.0, 277.0, 141.0, 91.0, 66.0, 50.0, 37.0, 32.0,
        33.0, 3.0, 6.0, 3.0, 2.0, 1.0, 7.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    WARM_HIST = (
        0.0, 0.0, 0.0, 0.0, 8.0, 50.0, 161.0, 299.0, 587.0, 1029.0,
        1642.0, 2228.0, 3332.0, 4163.0, 4805.0, 5398.0, 5838.0, 5990.0, 5930.0, 5678.0,
        5456.0, 4935.0, 4880.0, 4557.0, 4250.0, 3894.0, 3265.0, 2697.0, 2023.0, 1442.0,
        750.0, 443.0, 188.0, 63.0, 18.0, 2.0, 0.0, 0.0, 0.0, 0.0,
    )
    WARM_RHO = 36.0

    return (
        BINS, CLIMATE_SIGMA, CLIM_HIST, CROSSINGS, CRPS_CLIM,
        CRPS_CROSSING, CRPS_ENS, CRPS_SINGLE, FIRST_KIND,
        FIRST_KIND_ERROR, FIRST_KIND_SPREAD, FIRST_MONOTONE,
        FORECAST_HIST_000, FORECAST_HIST_050, FORECAST_HIST_120, LEADS,
        MEMBERS, NOISE_FLOOR, N_STARTS, RHO_BASE, RHO_SHIFTS, RMSE_CLIM,
        RMSE_CROSSING, RMSE_ENS, RMSE_SINGLE, SCORE_CASES, SCORE_LEADS,
        SCORE_MEMBERS, SECOND_KIND, SHOW_LEADS, SHOW_START, SHOW_START_Z,
        VARIABLE_NAME, WARM_HIST, WARM_RHO,
    )

# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 1 · What is Predictability?

    **Part I — What predictability means.**

    **The forecasting question.** Two forecasts can both be wrong and one of them still
    be excellent. So before any mathematics: what exactly is being claimed when someone
    says a forecast is good, or that the atmosphere is predictable to about a fortnight?

    The word *predictability* is doing at least four jobs.

    | | |
    |---|---|
    | **Practical** | the limit set by today's observations, models and computers — it moves |
    | **Intrinsic** | the limit set by the dynamics themselves — it does not |
    | **First kind** | what today's *state* tells you about the future |
    | **Second kind** | what the *forcing* tells you, whatever today's state is |

    This chapter separates them and then measures the last two **in the same units, on
    the same system**, which is what makes them comparable rather than merely
    contrasted. Everything runs on the Lorenz 63 system — three equations,
    **chapter 6** — because chapter 1 has to be readable before chapter
    6 has introduced anything.

    The central move of the whole book is made here: **a forecast is a probability
    distribution, not a trajectory**, and skill is the distance between that distribution
    and climatology.
    """
    )
    return


# ===========================================================================
# Section 1
# ===========================================================================
@app.cell(hide_code=True)
def s1_md(SCORE_CASES, SCORE_MEMBERS, mo):
    mo.md(
        rf"""
    ## 1 · One forecast, three ways of being right

    Start with the question as usually asked: *how far ahead is this forecast any good?*

    Below, the same {SCORE_CASES} Lorenz 63 cases forecast three ways, all starting from
    **the same initial uncertainty** — so the comparison is about what is done with that
    uncertainty, not about who was given a better starting point.

    * a **single run** from one perturbed initial state, the classical deterministic
      forecast;
    * an **ensemble** of {SCORE_MEMBERS} runs from {SCORE_MEMBERS} perturbed states;
    * **climatology** — ignore today entirely and quote the long-run distribution.

    And scored two ways: root-mean-square error, which asks how far the forecast is from
    the truth, and CRPS, which asks how good the whole *distribution* is and reduces to
    the absolute error for a single run, so the comparison is fair.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    CLIMATE_SIGMA, CRPS_CLIM, CRPS_ENS, CRPS_SINGLE, C_ANALYSIS, C_BG, C_PERT,
    RMSE_CLIM, RMSE_ENS, RMSE_SINGLE, SCORE_LEADS, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(SCORE_LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Root-mean-square error", "CRPS (scores the whole distribution)"),
        figsize=(10.2, 4.1),
    )
    for _ax, _sets in (
        (_ax0, ((RMSE_SINGLE, C_PERT, "single run"),
                (RMSE_ENS, C_ANALYSIS, "ensemble mean"),
                (RMSE_CLIM, C_BG, "climatology"))),
        (_ax1, ((CRPS_SINGLE, C_PERT, "single run"),
                (CRPS_ENS, C_ANALYSIS, "ensemble"),
                (CRPS_CLIM, C_BG, "climatology"))),
    ):
        for _values, _colour, _label in _sets:
            _ax.plot(_leads, np.asarray(_values), "-", color=_colour,
                     linewidth=2.1, label=_label)
        _ax.set_xlabel("forecast lead (time units)")
        _ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    _ax0.set_ylabel(f"RMSE in $z$   (climatological $\\sigma$ = {CLIMATE_SIGMA:.1f})")
    _ax1.set_ylabel("CRPS in $z$")
    finish_mpl(_fig, "The same information, presented three ways")
    return


@app.cell(hide_code=True)
def s1_note(
    CRPS_CLIM, CRPS_CROSSING, CRPS_ENS, CRPS_SINGLE, RMSE_CLIM, RMSE_CROSSING,
    RMSE_ENS, RMSE_SINGLE, SCORE_LEADS, mo, np,
):
    _leads = np.asarray(SCORE_LEADS)
    _last = -1
    _ens_beats_clim_rmse = np.asarray(RMSE_ENS)[_last] < np.asarray(RMSE_CLIM)[_last]
    _ens_beats_clim_crps = np.asarray(CRPS_ENS)[_last] < np.asarray(CRPS_CLIM)[_last]
    mo.md(
        rf"""
    **The single run stops being worth anything at lead {RMSE_CROSSING:.1f} by RMSE and
    {CRPS_CROSSING:.1f} by CRPS.** Past that point a forecaster who ignored today's
    weather entirely and quoted the climatological average would have done better. Not
    "less accurate than it used to be" — *actively worse than knowing nothing*.

    **The ensemble never crosses.** At lead {_leads[_last]:g} its RMSE is
    {np.asarray(RMSE_ENS)[_last]:.2f} against climatology's
    {np.asarray(RMSE_CLIM)[_last]:.2f}, and its CRPS
    {np.asarray(CRPS_ENS)[_last]:.2f} against {np.asarray(CRPS_CLIM)[_last]:.2f}. Same
    model, same initial uncertainty, same physics — and it remains useful long after the
    single run has become misleading.

    That is not a trick. The ensemble mean is closer to climatology than any individual
    run is, and *that is the correct thing for a best estimate of a partly unpredictable
    quantity to be*. Chapter 22 makes this precise and shows the cost:
    a damped forecast has too little variance and too few extremes, which is why
    operational centres still issue an undamped deterministic run alongside the ensemble.

    So the opening question was badly posed. **"How far ahead is this forecast good?" has
    no answer until you say what is being forecast, how it is scored, and what it is being
    compared against.** Chapter 22 gives that its own chapter and finds horizons a factor
    of 2.5 apart across perfectly defensible definitions.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(RHO_BASE, mo):
    mo.md(
        rf"""
    ## 2 · Two kinds of predictability

    Lorenz drew a distinction that organises this whole book *[citation needed]*.

    **Predictability of the first kind** is what today's state tells you about tomorrow.
    It is an initial-value problem: given where the atmosphere is now, where will it be?
    Chaos limits it, because two states too close to distinguish today become
    unrecognisably different later. Parts II to V of this book are about the first kind.

    **Predictability of the second kind** is what the *forcing* tells you, whatever
    today's state is. It is a boundary-value problem: if the sun brightens, or carbon
    dioxide doubles, the atmosphere's **statistics** change in ways that have nothing to
    do with today's weather. Part VI is about the second kind.

    The distinction is easy to state and easy to blur, so here it is as an experiment.
    Lorenz 63 has a parameter $\rho$ — the Rayleigh number, the strength of the heating.
    Raising it is the cleanest analogue of a change in forcing this system has: it does
    not move the state, it changes the attractor the state lives on.

    Two questions, then, and **both are measured in nats by the same estimator on the
    same variable**, which is the only reason they can be compared at all:

    * how much does an ensemble started from today's state tell you about $z$ at lead
      $t$, beyond what climatology already says?
    * how much does knowing $\rho$ has changed tell you, beyond the same climatology?
    """
    )
    return


@app.cell(hide_code=True)
def s2_control(SHOW_LEADS, mo):
    lead_pick = mo.ui.radio(
        options={f"{v:g} TU": f"{int(v * 10):03d}" for v in SHOW_LEADS},
        value=f"{SHOW_LEADS[0]:g} TU",
        label="forecast lead",
        inline=True,
    )
    lead_pick
    return (lead_pick,)


@app.cell(hide_code=True)
def s2_fig(
    BINS, CLIM_HIST, C_ANALYSIS, C_BG, C_PERT, FORECAST_HIST_000,
    FORECAST_HIST_050, FORECAST_HIST_120, SHOW_LEADS, WARM_HIST, WARM_RHO,
    RHO_BASE, finish_mpl, lead_pick, mpl_panels, np,
):
    _hists = {"000": FORECAST_HIST_000, "050": FORECAST_HIST_050,
              "120": FORECAST_HIST_120}
    _edges = np.asarray(BINS)
    _centres = 0.5 * (_edges[:-1] + _edges[1:])
    _width = _edges[1] - _edges[0]

    def _norm(counts):
        counts = np.asarray(counts, dtype=float)
        return counts / (counts.sum() * _width)

    _key = lead_pick.value
    _lead = [v for v in SHOW_LEADS if f"{int(v * 10):03d}" == _key][0]

    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.8, 4.2))
    _ax.fill_between(_centres, _norm(CLIM_HIST), step="mid", color=C_BG,
                     alpha=0.35, label=f"climatology, $\\rho={RHO_BASE:g}$")
    _ax.plot(_centres, _norm(WARM_HIST), "-", color=C_PERT, linewidth=2.2,
             label=f"climatology, $\\rho={WARM_RHO:g}$  (second kind)")
    _ax.plot(_centres, _norm(_hists[_key]), "-", color=C_ANALYSIS,
             linewidth=2.4,
             label=f"forecast at lead {_lead:g} TU  (first kind)")
    _ax.set_xlabel("$z$")
    _ax.set_ylabel("probability density")
    _ax.legend(fontsize=8.5, framealpha=0.9)
    finish_mpl(
        _fig,
        "One forecast case. A forecast is a distribution, and skill is its "
        "distance from climatology.",
    )
    return


@app.cell(hide_code=True)
def s2_note(
    BINS, CLIM_HIST, FORECAST_HIST_000, FORECAST_HIST_050, FORECAST_HIST_120,
    RHO_BASE, SHOW_LEADS, WARM_RHO, mo, np,
):
    _edges = np.asarray(BINS)
    _centres = 0.5 * (_edges[:-1] + _edges[1:])

    def _span(counts):
        occupied = _centres[np.asarray(counts) > 0]
        return float(occupied.min()), float(occupied.max())

    _s0 = _span(FORECAST_HIST_000)
    _s1 = _span(FORECAST_HIST_050)
    _s2 = _span(FORECAST_HIST_120)
    _clim = _span(CLIM_HIST)
    mo.md(
        rf"""
    Step the lead and watch the blue curve. This is **one** forecast case, chosen for
    illustration; the quantitative curve in section 3 averages over
    thirty-two of them.

    At lead {SHOW_LEADS[0]:g} the ensemble occupies $z \in
    [{_s0[0]:.0f}, {_s0[1]:.0f}]$ — a spike. It knows almost exactly where the system is.
    At lead {SHOW_LEADS[1]:g} it has widened to $[{_s1[0]:.0f}, {_s1[1]:.0f}]$, still
    narrow and clearly **displaced** from the grey climatology: much less certain, but
    still saying something specific and useful. By lead {SHOW_LEADS[2]:g} it spans
    $[{_s2[0]:.0f}, {_s2[1]:.0f}]$ against climatology's
    $[{_clim[0]:.0f}, {_clim[1]:.0f}]$ — it has become the climatology, and has nothing
    left to say that climatology did not already say.

    Notice that the middle stage is the interesting one, and the one a single-run forecast
    cannot express. "Somewhere between {_s1[0]:.0f} and {_s1[1]:.0f}, probably near
    {0.5 * (_s1[0] + _s1[1]):.0f}" is a genuinely useful statement. A single number is
    either that statement stripped of its uncertainty, or — if the run happens to sit at
    the edge of the spread — actively misleading.

    The red curve does not move with lead at all. It is the climatology of a *differently
    forced* system, $\rho = {WARM_RHO:g}$ against {RHO_BASE:g}, and its displacement from
    the grey one is permanent. **Predictability of the second kind does not decay**,
    because it was never about today's state in the first place.
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(MEMBERS, N_STARTS, NOISE_FLOOR, mo):
    mo.md(
        rf"""
    ## 3 · One number for both

    Turn each of those distances into a number: the **relative entropy** of the forecast
    distribution against climatology, in nats. It is zero when the forecast says nothing
    climatology did not, and grows as the forecast becomes sharper and more displaced.
    Chapter 10 develops it properly; here it is just a ruler that both kinds fit on.

    Two honesties before the figure. The curve is averaged over {N_STARTS} starting
    points, because a single start is **not** monotone — whether a forecast blob sits
    where the climatology is thick or thin is an accident of where it began, not a fact
    about predictability. And the estimator has a **noise floor**:
    {MEMBERS} members in 40 bins give
    {NOISE_FLOOR:.4f} nats even when the forecast *is* the climatology. That floor is
    measured, not assumed, and no decaying curve can go below it.
    """
    )
    return


@app.cell(hide_code=True)
def s3_fig(
    CROSSINGS, C_ANALYSIS, C_BG, C_PERT, C_SAT, C_TRUTH, FIRST_KIND,
    FIRST_KIND_ERROR, LEADS, NOISE_FLOOR, RHO_BASE, RHO_SHIFTS, SECOND_KIND,
    finish_mpl, mpl_panels, np, plt,
):
    _leads = np.asarray(LEADS)
    _first = np.asarray(FIRST_KIND)
    _error = np.asarray(FIRST_KIND_ERROR)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(9.2, 4.8))
    _ax.semilogy(_leads, _first, "-", color=C_TRUTH, linewidth=2.6,
                 label="first kind: what today's state tells you")
    _ax.fill_between(_leads, np.maximum(_first - _error, 1e-4), _first + _error,
                     color=C_TRUTH, alpha=0.18, linewidth=0)

    _cmap = plt.get_cmap("plasma")
    for _i, (_rho, _value, _cross) in enumerate(
        zip(RHO_SHIFTS, SECOND_KIND, CROSSINGS)
    ):
        _colour = _cmap(0.12 + 0.62 * _i / max(1, len(RHO_SHIFTS) - 1))
        _ax.axhline(_value, color=_colour, linestyle="--", linewidth=1.7)
        _ax.text(_leads[-1], _value * 1.08,
                 f"$\\rho\\,{{\\to}}\\,{_rho:g}$", ha="right", fontsize=8,
                 color=_colour)
        if np.isfinite(_cross):
            _ax.plot([_cross], [_value], "o", color=_colour, markersize=9,
                     markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    _ax.axhline(NOISE_FLOOR, color=C_BG, linestyle=":", linewidth=1.6)
    _ax.text(_leads[0], NOISE_FLOOR * 1.15, "estimator noise floor",
             fontsize=8, color=C_BG)
    _ax.set_xlabel("forecast lead (time units)")
    _ax.set_ylabel("information beyond climatology (nats)")
    _ax.set_ylim(NOISE_FLOOR * 0.5, _first.max() * 2.0)
    _ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
    finish_mpl(
        _fig,
        "Dashed lines are constants: the forcing signal never decays. "
        "Circles mark where the two cross.",
    )
    return


@app.cell(hide_code=True)
def s3_note(
    CROSSINGS, FIRST_KIND, LEADS, NOISE_FLOOR, RHO_BASE, RHO_SHIFTS,
    SECOND_KIND, mo, np,
):
    _first = np.asarray(FIRST_KIND)
    _leads = np.asarray(LEADS)
    _second = np.asarray(SECOND_KIND)
    _cross = np.asarray(CROSSINGS)
    _rows = "\n".join(
        f"    | {RHO_BASE:g} → {r:g} | {s:.4f} | "
        + ("never, within " + f"{_leads[-1]:g}" if not np.isfinite(c) else f"{c:.1f}")
        + " |"
        for r, s, c in zip(RHO_SHIFTS, _second, _cross)
    )
    mo.md(
        rf"""
    The blue curve falls from {_first[0]:.2f} nats to {_first[-1]:.3f} — a factor of
    {_first[0] / _first[-1]:.0f} — and it is still falling. The dashed lines do not fall
    at all.

    | forcing change | second-kind information | today's state is worth more out to lead |
    |---|---|---|
{_rows}

    **Read the third column as the whole of Part VI in one number.** A large enough change
    in the forcing tells you more about the distant future than today's weather does, and
    the bigger the change the sooner that happens. For $\rho \to {RHO_SHIFTS[-1]:g}$ the
    crossover is at lead {_cross[-1]:.1f}; for $\rho \to {RHO_SHIFTS[1]:g}$ it is
    {_cross[1]:.1f}; for the smallest change tested it does not happen within the
    {_leads[-1]:g} time units measured at all — the forcing signal is real but still
    smaller than what today's state knows.

    This is why seasonal forecasting, and climate projection beyond it, are not simply
    weather forecasting attempted further ahead. **They are a different question**, and
    they get their answer from a different place.

    /// admonition | Two cautions about this figure
        type: warning

    **Relative entropy is not a forecast.** It says how much the forecast distribution
    differs from climatology, not whether it is *right* — a confidently wrong forecast
    scores highly. Chapters 17 and 22 supply scores that penalise being wrong.

    **The crossover leads are properties of Lorenz 63, not of the atmosphere.** The
    conventional reading of one Lorenz time unit as a few atmospheric days is a
    convention, not a derivation, and a three-variable system has no scale-dependent
    error cascade of the kind chapters 12 and 14 show matters most. What transfers is the
    *structure* — one signal decaying, one not, and a crossing between them.
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
    ## 4 · Practical and intrinsic

    One more pair, cutting across the first. Both of the limits above are **intrinsic**:
    they assume a perfect model and a known forcing, and ask what the dynamics allow. A
    real forecast is nowhere near that, and is held back instead by the **practical**
    limit — imperfect observations, an imperfect model, finite computers.

    The distinction matters because the two behave completely differently under
    improvement, and the book measures both:

    * The **practical** limit moves. Chapter 22 measures it advancing at
      $\ln 10 / \lambda_1$ — about 1.4 time units for every factor of ten taken off the
      initial error — which is roughly the day-per-decade that operational forecasting has
      in fact delivered for forty years.
    * The **intrinsic** limit is a different quantity. Chapter 12 argues it is set by how
      error cascades between scales, chapter 14 shows the exponent that decides whether it
      is finite at all, and chapter 22 finds a system where seven decades of better
      initial conditions buy **eight times less** than in a single-scale one.

    Confusing the two is the commonest error in discussions of forecast limits. "Weather
    is unpredictable beyond two weeks" is a claim about the *intrinsic* limit; "our
    forecasts are useful to about a week" is a claim about the *practical* one; and the
    gap between them is the room left for improvement.
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

    **A forecast is a probability distribution.** Everything else in this book follows
    from taking that seriously. The single-run forecast is the special case that throws
    the distribution away, and section 1 shows what that costs: it becomes worse than
    useless at a lead where the ensemble is still informative.

    **"How far ahead is it good?" is not a well-posed question** until you say what is
    forecast, how it is scored, and against what. Chapter 22 finds a factor of 2.5 between
    defensible answers for one set of forecasts.

    **Two kinds of predictability, measurable in the same units.** What today's state
    tells you decays to nothing. What the forcing tells you does not decay at all. They
    cross, and where they cross depends on how large the forcing change is.

    **Practical and intrinsic limits are different quantities.** One moves with better
    observations and models; the other does not. Most public confusion about forecast
    limits is a conflation of these two.

    ### Try this

    1. In section 2, step the lead through 0, 5 and 12. At which lead would you say the
       forecast has "stopped being useful" — and does your answer change if you imagine
       being asked whether $z$ will exceed 40?
    2. Section 1's ensemble beats climatology at every lead shown. Does that mean an
       ensemble forecast is never useless? What would have to be true of the ensemble for
       it to lose?
    3. The crossover for $\rho \to 29$ never happens within 20 time units. Is that because
       the forcing signal is too small, or because the measurement is too short? Which
       number in section 3 settles it?
    4. Relative entropy rewards a sharp forecast. Construct a forecast that scores highly
       and is completely wrong, and say which chapter's scores would catch it.

    ### Where this goes next

    **Chapter 2** asks how anyone ever computed a forecast at all, and why the first
    attempt failed. **Chapter 3** lays out the hierarchy of models this book uses and
    defends the claim that three equations can teach something true about $10^9$.
    Then Part II starts the mathematics properly.

    ### Further reading

    - Lorenz (1975), on predictability of the first and second kind *[citation needed]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, introduction
      *[citation needed: chapter]*
    - Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, ch. 1
      *[citation needed]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
