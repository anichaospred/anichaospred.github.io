# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 17 -- Probabilistic forecast design.

Five ways to build an ensemble and what each one costs; why maximising growth
is not the same as being calibrated; reliability against resolution; and the
value of a probability to a decision.

Part V of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The cycling experiments are precomputed by `scripts/generate_ch17_data.py`.

To edit:   marimo edit notebooks/ch17_probabilistic-forecast-design.py
To export: make nb-one NB=ch17_probabilistic-forecast-design
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 17: Probabilistic Forecast Design")


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

    from chaoslib import ensemble, integrate, plotting, systems

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
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    # One colour per construction, used consistently in every figure.
    METHOD_COLOUR = {
        "EDA": plotting.C_ANALYSIS,
        "GAUSS": plotting.C_BG,
        "BRED": plotting.C_PERT,
        "BREDORTH": plotting.C_MEAN,
        "SV": plotting.C_FIXED,
    }
    METHOD_LABEL = {
        "EDA": "EDA (analysis ensemble)",
        "GAUSS": "isotropic random",
        "BRED": "bred vectors",
        "BREDORTH": "bred, orthogonalised",
        "SV": "singular vectors",
    }

    return (
        C_ANALYSIS, C_BG, C_CONTEXT, C_FIXED, C_MEAN, C_OBS, C_PERT, C_SAT,
        C_SPREAD, C_START, C_TRUTH, METHOD_COLOUR, METHOD_LABEL,
        MPL_SEQUENTIAL, ensemble, finish_mpl, integrate, mo, mpl_grid,
        mpl_panels, np, plotting, plt, systems,
    )


@app.cell
def chapter_data():
    # Precomputed by scripts/generate_ch17_data.py (~5 min). The cycling
    # LETKF, the breeding cycle maintained alongside it, 500 forecast cases
    # x 5 constructions x 7 leads, and the scoring. Knob-free: the slider
    # slices these arrays.

    L63_CYCLES = (2, 4, 6, 10, 16, 25, 40)
    L63_CYCLE_TIME = 0.25
    L63_LAMBDA1 = 0.906
    L63_RANDOM_ANGLE = 52.2232
    L63_N_VECTORS = 3
    L63_EFOLDS = (
        0.45300, 0.90600, 1.35900, 2.26500, 3.62400, 5.66250, 9.06000,
    )
    L63_ANGLE = (
        24.1576, 12.2774, 7.4057, 3.8243, 1.6365, 0.1248, 0.0864,
    )
    L63_ANGLE_ORTH = (
        90.0000, 90.0000, 90.0000, 90.0000, 90.0000, 90.0000, 90.0000,
    )
    L96_CYCLES = (2, 6, 12, 25, 50, 100, 200)
    L96_CYCLE_TIME = 0.05
    L96_LAMBDA1 = 1.67
    L96_RANDOM_ANGLE = 84.1541
    L96_N_VECTORS = 4
    L96_EFOLDS = (
        0.16700, 0.50100, 1.00200, 2.08750, 4.17500, 8.35000, 16.70000,
    )
    L96_ANGLE = (
        81.1378, 73.7212, 72.2319, 65.8764, 49.3258, 26.4238, 22.4836,
    )
    L96_ANGLE_ORTH = (
        90.0000, 90.0000, 90.0000, 90.0000, 90.0000, 90.0000, 90.0000,
    )

    DEGEN_BRED_ALIGNED = 0.390000
    DEGEN_BRED_ANTI = 0.460000
    DEGEN_BRED_MIDDLE = 0.150000
    DEGEN_BRED_HIST = (
        46.0, 3.0, 2.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 38.0,
    )
    DEGEN_BREDORTH_ALIGNED = 0.050000
    DEGEN_BREDORTH_ANTI = 0.000000
    DEGEN_BREDORTH_MIDDLE = 0.950000
    DEGEN_BREDORTH_HIST = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 52.0, 43.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0,
    )
    DEGEN_BINS = (np.float64(-1.0), np.float64(-0.9), np.float64(-0.8), np.float64(-0.7), np.float64(-0.6), np.float64(-0.5), np.float64(-0.4), np.float64(-0.3), np.float64(-0.2), np.float64(-0.1), np.float64(0.0), np.float64(0.1), np.float64(0.2), np.float64(0.3), np.float64(0.4), np.float64(0.5), np.float64(0.6), np.float64(0.7), np.float64(0.8), np.float64(0.9), np.float64(1.0))

    LEADS = (0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
    METHODS = ('EDA', 'GAUSS', 'BRED', 'BREDORTH', 'SV')
    MEMBERS = 20
    N_CASES = 500
    ANALYSIS_ERROR = 0.229779
    RANK_LEAD = 1.0
    SPREAD_EDA = (
        0.245380, 0.297142, 0.403402, 0.652558, 1.339218, 2.052371, 2.634352,
    )
    ERROR_EDA = (
        0.229779, 0.278128, 0.378956, 0.631942, 1.337274, 2.043140, 2.633650,
    )
    CRPS_EDA = (
        0.123995, 0.146185, 0.189272, 0.289293, 0.600467, 0.996601, 1.375663,
    )
    RANKS_EDA = (
        856.0, 971.0, 992.0, 932.0, 976.0, 912.0, 973.0, 958.0, 983.0, 932.0, 986.0, 946.0, 964.0, 964.0, 958.0, 992.0, 929.0, 1012.0, 995.0, 909.0, 860.0,
    )
    SPREAD_GAUSS = (
        0.229751, 0.244545, 0.352303, 0.616699, 1.370935, 2.119441, 2.708150,
    )
    ERROR_GAUSS = (
        0.235521, 0.283793, 0.389280, 0.655468, 1.390592, 2.104853, 2.683447,
    )
    CRPS_GAUSS = (
        0.132323, 0.154534, 0.200012, 0.307400, 0.635060, 1.038773, 1.411664,
    )
    RANKS_GAUSS = (
        863.0, 785.0, 891.0, 911.0, 928.0, 985.0, 969.0, 974.0, 1020.0, 1095.0, 1086.0, 1047.0, 1062.0, 1011.0, 987.0, 999.0, 966.0, 869.0, 853.0, 795.0, 904.0,
    )
    SPREAD_BRED = (
        0.232090, 0.274060, 0.361046, 0.562248, 1.070451, 1.567431, 2.019667,
    )
    ERROR_BRED = (
        0.233055, 0.281884, 0.384222, 0.645344, 1.441507, 2.286944, 3.007081,
    )
    CRPS_BRED = (
        0.156512, 0.184047, 0.236341, 0.360008, 0.756868, 1.285205, 1.798223,
    )
    RANKS_BRED = (
        5009.0, 537.0, 243.0, 131.0, 72.0, 61.0, 75.0, 48.0, 2615.0, 892.0, 590.0, 893.0, 2617.0, 50.0, 53.0, 67.0, 93.0, 156.0, 228.0, 514.0, 5056.0,
    )
    SPREAD_BREDORTH = (
        0.229779, 0.260813, 0.366042, 0.614135, 1.342316, 2.074592, 2.674253,
    )
    ERROR_BREDORTH = (
        0.235038, 0.284076, 0.387701, 0.648721, 1.368348, 2.065328, 2.642248,
    )
    CRPS_BREDORTH = (
        0.128467, 0.151545, 0.195816, 0.300100, 0.616788, 1.010632, 1.382854,
    )
    RANKS_BREDORTH = (
        827.0, 907.0, 935.0, 975.0, 973.0, 1026.0, 998.0, 967.0, 937.0, 949.0, 1002.0, 943.0, 979.0, 970.0, 1024.0, 952.0, 994.0, 982.0, 975.0, 922.0, 763.0,
    )
    SPREAD_SV = (
        0.235749, 0.329251, 0.606661, 1.089818, 1.753494, 2.351850, 2.822266,
    )
    ERROR_SV = (
        0.229779, 0.278212, 0.380650, 0.652574, 1.426723, 2.142185, 2.722269,
    )
    CRPS_SV = (
        0.137680, 0.158180, 0.204536, 0.319715, 0.660662, 1.068053, 1.437890,
    )
    RANKS_SV = (
        473.0, 830.0, 1143.0, 1205.0, 1261.0, 1151.0, 1071.0, 877.0, 858.0, 760.0, 762.0, 765.0, 864.0, 912.0, 1033.0, 1108.0, 1262.0, 1199.0, 1126.0, 882.0, 458.0,
    )

    SIZE_EDA_ASYMPTOTE = 0.575907
    SIZE_EDA_SLOPE = 0.563491
    SIZE_EDA = (
        0.709407, 0.682867, 0.632148, 0.614209, 0.600467, float("nan"), float("nan"),
    )
    SIZE_GAUSS_ASYMPTOTE = 0.602099
    SIZE_GAUSS_SLOPE = 0.649174
    SIZE_GAUSS = (
        0.767355, 0.705273, 0.667519, 0.648305, 0.635060, 0.624037, 0.619251,
    )
    SIZES = (4, 6, 10, 14, 20, 30, 40)
    SIZE_LEAD = 1.0

    BRIER_EDA = (
        0.055840, 0.008807, 0.162967, 0.210000,
    )
    RELDIAG_F_EDA = (
        0.01088, 0.17069, 0.29762, 0.42692, 0.53667, 0.66250, 0.81351, 0.97344,
    )
    RELDIAG_O_EDA = (
        0.01404, 0.06897, 0.33333, 0.38462, 0.46667, 0.75000, 0.89189, 0.92708,
    )
    RELDIAG_N_EDA = (
        285.0, 29.0, 21.0, 13.0, 15.0, 4.0, 37.0, 96.0,
    )
    BRIER_GAUSS = (
        0.062500, 0.003370, 0.150870, 0.210000,
    )
    RELDIAG_F_GAUSS = (
        0.01159, 0.17353, 0.30250, 0.42750, 0.54565, 0.67500, 0.81136, 0.97303,
    )
    RELDIAG_O_GAUSS = (
        0.01384, 0.11765, 0.35000, 0.30000, 0.60870, 0.65000, 0.86364, 0.95506,
    )
    RELDIAG_N_GAUSS = (
        289.0, 17.0, 20.0, 20.0, 23.0, 20.0, 22.0, 89.0,
    )
    BRIER_BRED = (
        0.068330, 0.006598, 0.148268, 0.210000,
    )
    RELDIAG_F_BRED = (
        0.00286, 0.15833, 0.30000, 0.40769, 0.57200, 0.65000, 0.82500, 0.99631,
    )
    RELDIAG_O_BRED = (
        0.02857, 0.33333, 0.33333, 0.30769, 0.64000, 1.00000, 1.00000, 0.90984,
    )
    RELDIAG_N_BRED = (
        315.0, 6.0, 3.0, 26.0, 25.0, 1.0, 2.0, 122.0,
    )
    BRIER_BREDORTH = (
        0.057360, 0.005030, 0.157670, 0.210000,
    )
    RELDIAG_F_BREDORTH = (
        0.01246, 0.17353, 0.30500, 0.42917, 0.55588, 0.67778, 0.80167, 0.97473,
    )
    RELDIAG_O_BREDORTH = (
        0.01365, 0.17647, 0.15000, 0.50000, 0.41176, 0.72222, 0.90000, 0.93548,
    )
    RELDIAG_N_BREDORTH = (
        293.0, 17.0, 20.0, 12.0, 17.0, 18.0, 30.0, 93.0,
    )
    BRIER_SV = (
        0.058345, 0.008466, 0.160121, 0.210000,
    )
    RELDIAG_F_SV = (
        0.02658, 0.17714, 0.28235, 0.42143, 0.54773, 0.68611, 0.81591, 0.94589,
    )
    RELDIAG_O_SV = (
        0.01761, 0.08571, 0.29412, 0.42857, 0.63636, 0.50000, 0.90909, 0.97260,
    )
    RELDIAG_N_SV = (
        284.0, 35.0, 17.0, 7.0, 22.0, 18.0, 44.0, 73.0,
    )
    RECAL_METHOD = 'EDA'
    RELDIAG_BINS = 8
    EVENT_THRESHOLD = 4.880938
    EVENT_BASE_RATE = 0.300000
    EVENT_LEAD = 1.0
    EVENT_SITE = 0
    RECAL_BEFORE = (
        0.055840, 0.008807, 0.162967, 0.210000,
    )
    RECAL_AFTER = (
        0.047033, 0.000000, 0.162967, 0.210000,
    )

    VALUE_METHOD = 'EDA'
    VALUE_PROB_POSITIVE = 19
    VALUE_DET_POSITIVE = 16
    COST_LOSS = (np.float64(0.05), np.float64(0.1), np.float64(0.15), np.float64(0.2), np.float64(0.25), np.float64(0.3), np.float64(0.35), np.float64(0.4), np.float64(0.45), np.float64(0.5), np.float64(0.55), np.float64(0.6), np.float64(0.65), np.float64(0.7), np.float64(0.75), np.float64(0.8), np.float64(0.85), np.float64(0.9), np.float64(0.95))
    VALUE_PROB = (
        0.660000, 0.742857, 0.797143, 0.828571, 0.848571, 0.862857, 0.848205, 0.831111, 0.810909, 0.786667,
        0.761481, 0.730000, 0.694286, 0.664444, 0.640000, 0.613333, 0.602222, 0.580000, 0.546667,
    )
    VALUE_DET = (
        -0.300000, 0.357143, 0.576190, 0.685714, 0.751429, 0.795238, 0.782051, 0.766667, 0.748485, 0.726667,
        0.700000, 0.666667, 0.623810, 0.566667, 0.486667, 0.366667, 0.166667, -0.233333, -1.433333,
    )

    return (
        ANALYSIS_ERROR, BRIER_BRED, BRIER_BREDORTH, BRIER_EDA,
        BRIER_GAUSS, BRIER_SV, COST_LOSS, CRPS_BRED, CRPS_BREDORTH,
        CRPS_EDA, CRPS_GAUSS, CRPS_SV, DEGEN_BINS, DEGEN_BREDORTH_ALIGNED,
        DEGEN_BREDORTH_ANTI, DEGEN_BREDORTH_HIST, DEGEN_BREDORTH_MIDDLE,
        DEGEN_BRED_ALIGNED, DEGEN_BRED_ANTI, DEGEN_BRED_HIST,
        DEGEN_BRED_MIDDLE, ERROR_BRED, ERROR_BREDORTH, ERROR_EDA,
        ERROR_GAUSS, ERROR_SV, EVENT_BASE_RATE, EVENT_LEAD, EVENT_SITE,
        EVENT_THRESHOLD, L63_ANGLE, L63_ANGLE_ORTH, L63_CYCLES,
        L63_CYCLE_TIME, L63_EFOLDS, L63_LAMBDA1, L63_N_VECTORS,
        L63_RANDOM_ANGLE, L96_ANGLE, L96_ANGLE_ORTH, L96_CYCLES,
        L96_CYCLE_TIME, L96_EFOLDS, L96_LAMBDA1, L96_N_VECTORS,
        L96_RANDOM_ANGLE, LEADS, MEMBERS, METHODS, N_CASES, RANKS_BRED,
        RANKS_BREDORTH, RANKS_EDA, RANKS_GAUSS, RANKS_SV, RANK_LEAD,
        RECAL_AFTER, RECAL_BEFORE, RECAL_METHOD, RELDIAG_BINS,
        RELDIAG_F_BRED, RELDIAG_F_BREDORTH, RELDIAG_F_EDA,
        RELDIAG_F_GAUSS, RELDIAG_F_SV, RELDIAG_N_BRED, RELDIAG_N_BREDORTH,
        RELDIAG_N_EDA, RELDIAG_N_GAUSS, RELDIAG_N_SV, RELDIAG_O_BRED,
        RELDIAG_O_BREDORTH, RELDIAG_O_EDA, RELDIAG_O_GAUSS, RELDIAG_O_SV,
        SIZES, SIZE_EDA, SIZE_EDA_ASYMPTOTE, SIZE_EDA_SLOPE, SIZE_GAUSS,
        SIZE_GAUSS_ASYMPTOTE, SIZE_GAUSS_SLOPE, SIZE_LEAD, SPREAD_BRED,
        SPREAD_BREDORTH, SPREAD_EDA, SPREAD_GAUSS, SPREAD_SV, VALUE_DET,
        VALUE_DET_POSITIVE, VALUE_METHOD, VALUE_PROB, VALUE_PROB_POSITIVE,
    )

# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 17 · Probabilistic Forecast Design

    **Part V — The machinery of prediction.**

    **The forecasting question.** You have run an ensemble. Now what is it *for*?

    Chapter 19 treated an ensemble purely as a covariance estimator — a device for
    building $\mathbf{P}^f$. That is one use, and it is not the one the public sees.
    The forecast that goes out is a *probability*: 30 % chance of frost tonight. And a
    probability makes a claim that can be checked, which a single number cannot.

    Three questions follow, and they are separate.

    **How should the ensemble be built?** There are several ways to choose the
    perturbations, and the obvious criterion — pick the directions that grow fastest —
    turns out to be the wrong one.

    **What makes a probability forecast good?** Not accuracy alone. A forecast can be
    sharp and wrong, or honest and useless, and the standard scores mix these together
    in ways that have to be pulled apart deliberately.

    **What is it worth?** This is the question that justifies the whole expense of
    running an ensemble, and it has an answer that no accuracy score can give.

    ---

    **What you need before this chapter.** **Chapter 7** for Lyapunov vectors, which
    bred vectors approximate. **Chapter 16** for singular vectors. **Chapter 19** for
    the LETKF that supplies the analyses here — the analysis-error distribution the
    ensembles are supposed to sample is the one they are scored against.
    """
    )
    return


# ===========================================================================
# Section 1
# ===========================================================================
@app.cell(hide_code=True)
def s1_md(mo):
    mo.md(
        r"""
    ## 1 · Five ways to perturb, and one that eats itself

    The perturbations have to come from somewhere. Five candidates:

    | Construction | Idea | Needs |
    |---|---|---|
    | **Isotropic random** | perturb in every direction equally | nothing |
    | **Bred vectors** | perturb, integrate, difference, rescale, repeat | the nonlinear model |
    | **Bred, orthogonalised** | as above, re-orthogonalised each cycle | the nonlinear model |
    | **Singular vectors** | the fastest-growing directions over a window | the tangent linear *and* adjoint |
    | **EDA** | the analysis ensemble from the assimilation itself | a working ensemble DA system |

    Breeding (Toth & Kalnay 1993) is the cheapest way to find growing directions and
    needs **no adjoint at all**, which is why it was NCEP's operational scheme while
    ECMWF ran singular vectors. Perturb the analysis, integrate the perturbed and
    control runs one cycle, take the difference, rescale it back, and repeat: after a
    few e-foldings the perturbation has forgotten its initial condition and points
    along the locally fastest-growing direction.

    That last sentence contains the problem. *The* fastest-growing direction — the same
    one, every time. Independently seeded bred vectors converge onto each other, and an
    ensemble built from several of them samples one direction repeatedly.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    C_PERT, C_TRUTH, L63_ANGLE, L63_ANGLE_ORTH, L63_EFOLDS, L63_N_VECTORS,
    L63_RANDOM_ANGLE, L96_ANGLE, L96_ANGLE_ORTH, L96_EFOLDS, L96_N_VECTORS,
    L96_RANDOM_ANGLE, C_MEAN, finish_mpl, mpl_panels, np,
):
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.6, 4.2))
    for _efolds, _angle, _orth, _random, _colour, _label, _n in (
        (L63_EFOLDS, L63_ANGLE, L63_ANGLE_ORTH, L63_RANDOM_ANGLE, C_TRUTH,
         "Lorenz 63 ($n=3$, one positive exponent)", L63_N_VECTORS),
        (L96_EFOLDS, L96_ANGLE, L96_ANGLE_ORTH, L96_RANDOM_ANGLE, C_PERT,
         "Lorenz 96 ($n=40$, thirteen positive)", L96_N_VECTORS),
    ):
        _ax.semilogx(np.asarray(_efolds), np.asarray(_angle), "o-",
                     color=_colour, markersize=5, linewidth=1.9, label=_label)
        _ax.axhline(_random, color=_colour, linestyle=":", linewidth=1.2,
                    alpha=0.7)
    _ax.axhline(90.0, color=C_MEAN, linestyle="--", linewidth=1.5)
    _ax.text(np.asarray(L96_EFOLDS)[-1], 87.0,
             "orthogonalised each cycle: exactly 90°", ha="right", fontsize=8,
             color=C_MEAN)
    _ax.text(np.asarray(L96_EFOLDS)[0], L96_RANDOM_ANGLE - 6.0,
             "unbred random directions", fontsize=7.5, color="#6b6580")
    _ax.set_xlabel("e-foldings of breeding  ($\\lambda_1 \\times$ total time)")
    _ax.set_ylabel("mean pairwise angle between bred vectors (degrees)")
    _ax.set_ylim(0, 95)
    _ax.legend(fontsize=8, framealpha=0.9, loc="lower left")
    finish_mpl(_fig, "Bred vectors collapse onto each other")
    return


@app.cell(hide_code=True)
def s1_note(
    L63_ANGLE, L63_EFOLDS, L63_RANDOM_ANGLE, L96_ANGLE, L96_EFOLDS,
    L96_RANDOM_ANGLE, mo, np,
):
    _a63 = np.asarray(L63_ANGLE)
    _a96 = np.asarray(L96_ANGLE)
    _e63 = np.asarray(L63_EFOLDS)
    _e96 = np.asarray(L96_EFOLDS)
    mo.md(
        rf"""
    Both systems collapse, and **the rate depends on how well separated the leading
    Lyapunov exponent is**. Lorenz 63 has a single positive exponent, so the leading
    direction dominates quickly: from {L63_RANDOM_ANGLE:.0f}° unbred down to
    {_a63[3]:.1f}° after {_e63[3]:.1f} e-foldings and {_a63[-1]:.1f}° by
    {_e63[-1]:.0f}. Lorenz 96 has thirteen positive exponents of similar size, so the
    set converges to a *subspace* rather than a line and levels off near
    {_a96[-1]:.0f}° — still a severe loss of diversity from {L96_RANDOM_ANGLE:.0f}°,
    but not total.

    Re-orthogonalising after **each** cycle holds the vectors at exactly 90°, and this
    is not a cosmetic fix: it converts breeding into the Gram–Schmidt construction of
    [chapter 7]'s Lyapunov spectrum, so the vectors converge to the leading Lyapunov
    *vectors* instead of all to the leading one.

    Two details that are easy to get wrong, both of which this chapter got wrong first.
    Orthogonalising **once at the end** is useless — by then the set is nearly rank
    one, and orthogonalising a collapsed set manufactures its extra directions out of
    rounding error. And breeding must run **along** the trajectory being perturbed: an
    earlier version bred forward from the analysis for several time units, producing
    vectors that belonged to a state that far downstream. Here the perturbations are
    carried as persistent state across the assimilation cycle, which is what the
    operational scheme did.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(ANALYSIS_ERROR, MEMBERS, N_CASES, mo):
    mo.md(
        rf"""
    ## 2 · Spread is a claim, and it can be checked

    A well-built ensemble makes a specific, falsifiable claim: **the truth is
    statistically indistinguishable from a member.** One immediate consequence is the
    calibration identity — RMS spread should equal the RMS error of the ensemble mean,
    at every lead. Too little spread and the ensemble is overconfident; too much and it
    hedges.

    All five constructions below start from the *same* analysis (a cycling LETKF on
    Lorenz 96, chapter 19's configuration, analysis error {ANALYSIS_ERROR:.4f} per
    component), use the *same* {MEMBERS} members, and are rescaled to the *same* total
    perturbation amplitude. So the comparison is about **direction and nothing else**,
    over {N_CASES} forecast cases.
    """
    )
    return


@app.cell(hide_code=True)
def s2_control(LEADS, mo):
    lead_pick = mo.ui.slider(
        steps=[float(v) for v in LEADS],
        value=float(LEADS[len(LEADS) // 2]),
        label="forecast lead for the rank histograms (TU)",
        show_value=True,
        full_width=True,
    )
    lead_pick
    return (lead_pick,)


@app.cell(hide_code=True)
def s2_fig(
    CRPS_BRED, CRPS_BREDORTH, CRPS_EDA, CRPS_GAUSS, CRPS_SV, ERROR_BRED,
    ERROR_BREDORTH, ERROR_EDA, ERROR_GAUSS, ERROR_SV, LEADS, METHODS,
    METHOD_COLOUR, METHOD_LABEL, SPREAD_BRED, SPREAD_BREDORTH, SPREAD_EDA,
    SPREAD_GAUSS, SPREAD_SV, finish_mpl, mpl_panels, np,
):
    _spread = {"EDA": SPREAD_EDA, "GAUSS": SPREAD_GAUSS, "BRED": SPREAD_BRED,
               "BREDORTH": SPREAD_BREDORTH, "SV": SPREAD_SV}
    _error = {"EDA": ERROR_EDA, "GAUSS": ERROR_GAUSS, "BRED": ERROR_BRED,
              "BREDORTH": ERROR_BREDORTH, "SV": ERROR_SV}
    _crps = {"EDA": CRPS_EDA, "GAUSS": CRPS_GAUSS, "BRED": CRPS_BRED,
             "BREDORTH": CRPS_BREDORTH, "SV": CRPS_SV}
    _leads = np.asarray(LEADS)

    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Calibration: spread / error", "CRPS, relative to the best"),
        figsize=(10.2, 4.1),
    )
    for _name in METHODS:
        _ratio = np.asarray(_spread[_name]) / np.asarray(_error[_name])
        _ax0.plot(_leads, _ratio, "o-", color=METHOD_COLOUR[_name],
                  markersize=5, linewidth=1.8, label=METHOD_LABEL[_name])
    _ax0.axhline(1.0, color="#8b8698", linestyle="--", linewidth=1.4)
    _ax0.text(_leads[0], 1.02, "calibrated", fontsize=7.5, color="#6b6580")
    # Explicit limits: the bred-vector curve reaches 0.71, which is the whole
    # point of the panel, and matplotlib's default view clipped it against the
    # bottom edge where it was invisible.
    _stacked = np.array([
        np.asarray(_spread[n]) / np.asarray(_error[n]) for n in METHODS
    ])
    _ax0.set_ylim(_stacked.min() - 0.13, _stacked.max() + 0.07)
    _ax0.set_ylabel("RMS spread / RMS error of the mean")
    _ax0.set_xlabel("forecast lead (TU)")
    _ax0.legend(fontsize=7, framealpha=0.9, loc="lower right")

    # Ratio to the best at each lead: the absolute CRPS grows by an order of
    # magnitude across these leads, which would hide every difference between
    # methods on a shared axis.
    _stack = np.array([np.asarray(_crps[n]) for n in METHODS])
    _best = _stack.min(axis=0)
    for _index, _name in enumerate(METHODS):
        _ax1.plot(_leads, _stack[_index] / _best, "o-",
                  color=METHOD_COLOUR[_name], markersize=5, linewidth=1.8)
    _ax1.axhline(1.0, color="#8b8698", linestyle="--", linewidth=1.4)
    _ax1.set_ylabel("CRPS / best CRPS at that lead")
    _ax1.set_xlabel("forecast lead (TU)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s2_ranks(
    MEMBERS, METHODS, METHOD_COLOUR, METHOD_LABEL, RANKS_BRED, RANKS_BREDORTH,
    RANKS_EDA, RANKS_GAUSS, RANKS_SV, RANK_LEAD, finish_mpl, mpl_panels, np,
):
    _ranks = {"EDA": RANKS_EDA, "GAUSS": RANKS_GAUSS, "BRED": RANKS_BRED,
              "BREDORTH": RANKS_BREDORTH, "SV": RANKS_SV}
    _fig, _axes = mpl_panels(
        ncols=5,
        titles=tuple(METHOD_LABEL[m] for m in METHODS),
        figsize=(14.0, 3.2),
    )
    _bins = np.arange(MEMBERS + 1)
    for _ax, _name in zip(_axes, METHODS):
        _counts = np.asarray(_ranks[_name], dtype=float)
        _flat = _counts.sum() / _counts.size
        _ax.bar(_bins, _counts / _flat, width=0.9,
                color=METHOD_COLOUR[_name], edgecolor="white", linewidth=0.4)
        _ax.axhline(1.0, color="#4a4460", linestyle="--", linewidth=1.2)
        _ax.set_ylim(0, 2.6)
        _ax.set_xlabel("rank of the truth")
    _axes[0].set_ylabel("count / flat")
    finish_mpl(
        _fig,
        f"Rank histograms at lead {RANK_LEAD} TU. Flat is calibrated; "
        f"U-shaped is under-spread, domed is over-spread.",
    )
    return


@app.cell(hide_code=True)
def s2_note(
    CRPS_BRED, CRPS_BREDORTH, ERROR_BRED, ERROR_BREDORTH, ERROR_GAUSS,
    ERROR_SV, LEADS, SPREAD_BRED, SPREAD_BREDORTH, SPREAD_GAUSS, SPREAD_SV,
    mo, np,
):
    _leads = np.asarray(LEADS)
    _ratio = lambda sp, er: np.asarray(sp) / np.asarray(er)   # noqa: E731
    _bred = _ratio(SPREAD_BRED, ERROR_BRED)
    _orth = _ratio(SPREAD_BREDORTH, ERROR_BREDORTH)
    _sv = _ratio(SPREAD_SV, ERROR_SV)
    _gauss = _ratio(SPREAD_GAUSS, ERROR_GAUSS)
    _peak = int(np.argmax(_sv))
    _dip = int(np.argmin(_gauss))
    _gain = 100 * (
        1 - np.asarray(CRPS_BREDORTH)[-1] / np.asarray(CRPS_BRED)[-1]
    )
    mo.md(
        rf"""
    Four findings, and the first two are the chapter's point.

    **Plain bred vectors are the worst construction, and they get worse with lead.**
    Their spread/error ratio falls from {_bred[0]:.2f} to **{_bred[-1]:.2f}** at lead
    {_leads[-1]:g} TU — the ensemble becomes steadily more overconfident — and their
    CRPS is worst at every lead. This is section 1's collapse, cashed out: if every
    member grows along the same direction, the ensemble spread grows at one exponent
    while the true error spreads into a thirteen-dimensional unstable subspace.
    **Orthogonalising fixes it**, improving CRPS at lead {_leads[-1]:g} by
    {_gain:.0f} % and restoring the ratio to {_orth[-1]:.3f}.

    **Singular vectors are over-dispersed, and score worse than random perturbations
    at short lead.** Their ratio peaks at **{_sv[_peak]:.2f}** at lead
    {_leads[_peak]:g} TU. This is not a failure of the method; it is the method working
    exactly as designed. Singular vectors maximise growth over their window, so an
    ensemble built from them grows faster than the analysis error actually does.
    **Maximising growth and sampling the analysis-error distribution are different
    objectives**, and only the second is what a probability forecast needs.

    **The EDA wins at every lead**, because it is the only construction that samples
    the analysis-error distribution rather than guessing at its shape. It is the
    operational answer for exactly this reason.

    **Isotropic perturbations are under-dispersed early and recover late** — the ratio
    dips to {_gauss[_dip]:.2f} at lead {_leads[_dip]:g} and returns to
    {_gauss[-1]:.2f} by lead {_leads[-1]:g}. A random direction has little projection
    on the growing subspace, so it grows slowly at first; once the growing directions
    dominate, it catches up.
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(SIZE_LEAD, mo):
    mo.md(
        rf"""
    ## 3 · How many members?

    The most expensive decision in an operational forecasting system. CRPS at lead
    {SIZE_LEAD:g} TU against ensemble size, for the best and the naive construction.
    """
    )
    return


@app.cell(hide_code=True)
def s3_fig(
    C_ANALYSIS, C_BG, SIZES, SIZE_EDA, SIZE_EDA_ASYMPTOTE, SIZE_EDA_SLOPE,
    SIZE_GAUSS, SIZE_GAUSS_ASYMPTOTE, SIZE_GAUSS_SLOPE, SIZE_LEAD, finish_mpl,
    mpl_panels, np,
):
    _k = np.asarray(SIZES, dtype=float)
    _fine = np.linspace(_k.min(), _k.max(), 200)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("CRPS against ensemble size", r"the same, against $1/k$"),
        figsize=(9.8, 4.0),
    )
    for _values, _asym, _slope, _colour, _label in (
        (SIZE_EDA, SIZE_EDA_ASYMPTOTE, SIZE_EDA_SLOPE, C_ANALYSIS, "EDA"),
        (SIZE_GAUSS, SIZE_GAUSS_ASYMPTOTE, SIZE_GAUSS_SLOPE, C_BG,
         "isotropic random"),
    ):
        _v = np.asarray(_values)
        _good = np.isfinite(_v)
        _ax0.plot(_k[_good], _v[_good], "o", color=_colour, markersize=6,
                  label=_label)
        _ax0.plot(_fine, _asym + _slope / _fine, "-", color=_colour,
                  linewidth=1.5, alpha=0.8)
        _ax0.axhline(_asym, color=_colour, linestyle=":", linewidth=1.2)
        _ax1.plot(1.0 / _k[_good], _v[_good], "o", color=_colour, markersize=6)
        _ax1.plot(1.0 / _fine, _asym + _slope / _fine, "-", color=_colour,
                  linewidth=1.5, alpha=0.8)
    _ax0.set_xlabel("ensemble size $k$")
    _ax0.set_ylabel(f"CRPS at lead {SIZE_LEAD:g} TU")
    _ax0.legend(fontsize=8, framealpha=0.9)
    _ax1.set_xlabel("$1/k$")
    _ax1.set_ylabel("CRPS")
    finish_mpl(
        _fig,
        "Dotted lines are the fitted asymptotes: what an infinite ensemble "
        "would score.",
    )
    return


@app.cell(hide_code=True)
def s3_note(
    MEMBERS, SIZES, SIZE_EDA, SIZE_EDA_ASYMPTOTE, SIZE_EDA_SLOPE, SIZE_GAUSS,
    SIZE_GAUSS_ASYMPTOTE, SIZE_GAUSS_SLOPE, mo, np,
):
    _k = np.asarray(SIZES, dtype=float)
    _gap = 100 * (SIZE_GAUSS_ASYMPTOTE / SIZE_EDA_ASYMPTOTE - 1.0)
    _pen20 = 100 * SIZE_EDA_SLOPE / MEMBERS / SIZE_EDA_ASYMPTOTE
    mo.md(
        rf"""
    CRPS fits $a + b/k$ closely — the right-hand panel is a straight line — and the
    two fitted parts say different things.

    **The $1/k$ term is mostly the estimator, not the ensemble.**
    $b = {SIZE_EDA_SLOPE:.3f}$ for the EDA against ${SIZE_GAUSS_SLOPE:.3f}$ for
    isotropic perturbations — the same order, differing by
    {100 * abs(SIZE_GAUSS_SLOPE / SIZE_EDA_SLOPE - 1):.0f} %, and both costing about
    **{_pen20:.0f} %** at $k = {MEMBERS}$. Most of it is the finite-sample bias of the
    CRPS estimator itself, which knows nothing about how the members were chosen. The
    two are not identical, and the EDA figure is the less well determined of the pair:
    its fit rests on five points to the Gaussian's seven, because the assimilation only
    ran {MEMBERS} members.

    **The asymptote is where construction shows.** ${SIZE_EDA_ASYMPTOTE:.4f}$ against
    ${SIZE_GAUSS_ASYMPTOTE:.4f}$, a gap of {_gap:.0f} %, and **no ensemble size closes
    it**: an infinite ensemble of the wrong directions is still worse than a small one
    of the right ones. Read the two panels together and the operational implication is
    blunt — going from {int(_k[0])} to {int(_k[-1])} members buys
    {100 * (1 - np.asarray(SIZE_GAUSS)[-1] / np.asarray(SIZE_GAUSS)[0]):.0f} %, while
    fixing the construction buys {_gap:.0f} % for free.

    The EDA curve stops at $k = {MEMBERS}$ because the assimilation ran
    {MEMBERS} members and there are no more to draw. Resampling with replacement would
    duplicate members and understate the spread, which is the very quantity being
    scored.
    """
    )
    return


# ===========================================================================
# Section 4
# ===========================================================================
@app.cell(hide_code=True)
def s4_md(EVENT_BASE_RATE, EVENT_LEAD, EVENT_SITE, EVENT_THRESHOLD, mo):
    mo.md(
        rf"""
    ## 4 · Reliability and resolution are different things

    Now score a **decision-shaped** forecast: the probability that site
    {EVENT_SITE} exceeds {EVENT_THRESHOLD:.2f} at lead {EVENT_LEAD:g} TU, an event with
    a base rate of {EVENT_BASE_RATE:.2f}. The Brier score is the mean squared error of
    the probability, and Murphy's decomposition splits it into three parts that behave
    completely differently:

    $$
    \underbrace{{\mathrm{{BS}}}}_{{\text{{what you are scored on}}}} =
      \underbrace{{\mathrm{{REL}}}}_{{\text{{calibration error}}}}
      - \underbrace{{\mathrm{{RES}}}}_{{\text{{information}}}}
      + \underbrace{{\mathrm{{UNC}}}}_{{\text{{the event's own difficulty}}}}
    $$

    Lower reliability is better; *higher* resolution is better; uncertainty depends only
    on the base rate and no forecast system can touch it. With the bins taken as the
    distinct forecast values — and an ensemble of $k$ members can only issue
    $0, 1/k, \ldots, 1$ — the identity is **exact**, which `chaoslib` asserts as a test.

    The reason to bother splitting it: **reliability is fixable after the fact and
    resolution is not.**
    """
    )
    return


@app.cell(hide_code=True)
def s4_fig(
    BRIER_BRED, BRIER_BREDORTH, BRIER_EDA, BRIER_GAUSS, BRIER_SV, METHODS,
    METHOD_COLOUR, METHOD_LABEL, RELDIAG_F_EDA, RELDIAG_N_EDA, RELDIAG_O_EDA,
    finish_mpl, mpl_panels, np,
):
    _brier = {"EDA": BRIER_EDA, "GAUSS": BRIER_GAUSS, "BRED": BRIER_BRED,
              "BREDORTH": BRIER_BREDORTH, "SV": BRIER_SV}
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Reliability against resolution", "Reliability diagram, EDA"),
        figsize=(10.0, 4.2),
    )
    for _name in METHODS:
        _bs, _rel, _res, _unc = np.asarray(_brier[_name])
        _ax0.plot(_rel, _res, "o", color=METHOD_COLOUR[_name], markersize=11,
                  markeredgecolor="white", markeredgewidth=1.0,
                  label=f"{METHOD_LABEL[_name]}  (BS {_bs:.4f})")
    # No axis inversion. An inverted axis plus a "-> worse" label said the
    # opposite of what the axis did: with the inversion, *better* reliability
    # was on the right. Plain axis, plain label, arrow for the good corner.
    _ax0.set_xlabel("reliability  (lower is better $\\leftarrow$)")
    _ax0.set_ylabel("resolution  (higher is better $\\uparrow$)")
    # Upper left is the only free quadrant: the five points run from
    # bottom-left to top-right, and a legend at lower right sat on top of the
    # bred-vector marker.
    _ax0.legend(fontsize=7, framealpha=0.95, loc="upper left")

    _f = np.asarray(RELDIAG_F_EDA)
    _o = np.asarray(RELDIAG_O_EDA)
    _n = np.asarray(RELDIAG_N_EDA, dtype=float)
    _ax1.plot([0, 1], [0, 1], "--", color="#8b8698", linewidth=1.4)
    # Only join bins holding a meaningful number of cases. Connecting every bin
    # drew a zigzag through one- and two-case bins whose observed frequency can
    # only be 0 or 1, which looked like a wildly unreliable forecast and was
    # entirely sampling noise.
    _solid = _n >= 0.02 * _n.sum()
    _ax1.plot(_f[_solid], _o[_solid], "-", color=METHOD_COLOUR["EDA"],
              linewidth=1.6, alpha=0.75, zorder=2)
    _ax1.scatter(_f, _o, s=18 + 260 * _n / _n.max(),
                 color=METHOD_COLOUR["EDA"], alpha=0.85, edgecolor="white",
                 linewidth=0.8, zorder=3)
    _ax1.set_xlabel("forecast probability")
    _ax1.set_ylabel("observed frequency")
    _ax1.set_xlim(-0.03, 1.03)
    _ax1.set_ylim(-0.03, 1.03)
    _ax1.text(0.04, 0.95, "marker area $\\propto$ number of cases;\n"
              "line joins the well-populated bins", fontsize=7.5,
              color="#6b6580", va="top")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s4_note(RECAL_AFTER, RECAL_BEFORE, RECAL_METHOD, METHOD_LABEL, mo, np):
    _b = np.asarray(RECAL_BEFORE)
    _a = np.asarray(RECAL_AFTER)
    mo.md(
        rf"""
    The left panel separates two ways of being good, and the constructions do not agree
    on which they are. **{METHOD_LABEL[RECAL_METHOD]} has the most resolution and the
    worst reliability** — it carries the most information about this event and states
    it least honestly.

    That combination is the good one to have, because the honesty is repairable.
    Relabel each forecast probability with the frequency the event actually attained
    when that probability was issued — no new information, no new model runs, just a
    lookup table:

    | | Brier score | reliability | resolution |
    |---|---|---|---|
    | before | {_b[0]:.4f} | {_b[1]:.5f} | {_b[2]:.5f} |
    | recalibrated | {_a[0]:.4f} | {_a[1]:.5f} | {_a[2]:.5f} |

    Reliability is gone; **resolution has not moved at all**, to five decimal places.
    The recalibration is a monotone relabelling, so it cannot change which cases are
    ranked above which, and resolution depends on the forecasts only through that
    ranking. The Brier score falls by
    {100 * (1 - _a[0] / _b[0]):.0f} % for the price of a lookup table.

    /// admonition | This recalibration is in-sample, and that is why REL is exactly zero
        type: warning

    The mapping is fitted on the same cases it is then scored on. That is why
    reliability comes out at **exactly** {_a[1]:.5f} rather than merely small. A real
    system fits the mapping on a training period and applies it out of sample, where
    reliability improves substantially but does not vanish.

    The claim being demonstrated is the *structural* one — that resolution is untouched
    by recalibration — which does not depend on the sample. The size of the Brier gain
    does, and should be read as an upper bound.
    ///
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
    ## 5 · What is a probability worth?

    Every score so far measures whether the forecast is *right*. None measures whether
    it is *useful*, and those are different questions with different answers.

    Put a decision behind it. A user pays $C$ to protect against an event that would
    otherwise cost them $L$. Only the ratio $\alpha = C/L$ matters, and the optimal
    strategy is simple: **protect whenever the forecast probability exceeds
    $\alpha$.** Comparing the expected expense against what a user with only the
    climatology would spend, and against what a user with a perfect forecast would
    spend, gives the relative value

    $$
    V(\alpha) = \frac{E_{\text{clim}} - E_{\text{forecast}}}
                     {E_{\text{clim}} - E_{\text{perfect}}},
    $$

    which is 1 for a perfect forecast, 0 for one no better than climatology, and
    negative for one that is worse than useless.

    The essential point is that **$V$ is a curve, not a number.** Different users sit at
    different $\alpha$. A council gritting roads has a small $C/L$; a factory shutting a
    production line has a large one. A single deterministic forecast forces one
    threshold on all of them.
    """
    )
    return


@app.cell(hide_code=True)
def s5_fig(
    COST_LOSS, C_ANALYSIS, C_PERT, VALUE_DET, VALUE_METHOD, VALUE_PROB,
    METHOD_LABEL, finish_mpl, mpl_panels, np,
):
    _alpha = np.asarray(COST_LOSS)
    _prob = np.asarray(VALUE_PROB)
    _det = np.asarray(VALUE_DET)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.8, 4.4))
    _ax.plot(_alpha, _prob, "o-", color=C_ANALYSIS, markersize=5, linewidth=2.0,
             label=f"probabilistic ({METHOD_LABEL[VALUE_METHOD]}, recalibrated)")
    _ax.plot(_alpha, _det, "s-", color=C_PERT, markersize=5, linewidth=2.0,
             label="deterministic (the same forecast, one threshold)")
    _ax.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.3)
    _ax.fill_between(_alpha, _det, 0.0, where=_det < 0.0, color=C_PERT,
                     alpha=0.16, linewidth=0)
    _ax.text(_alpha[-1], -0.25, "worse than\nno forecast at all", ha="right",
             fontsize=8, color=C_PERT)
    _ax.set_xlabel(r"cost-loss ratio  $\alpha = C/L$")
    _ax.set_ylabel(r"relative economic value  $V(\alpha)$")
    _ax.set_ylim(min(-0.6, _det.min() * 1.05), 1.02)
    _ax.legend(fontsize=8.5, framealpha=0.9, loc="lower left")
    finish_mpl(_fig, "The same information, served two ways")
    return


@app.cell(hide_code=True)
def s5_note(
    COST_LOSS, VALUE_DET, VALUE_DET_POSITIVE, VALUE_PROB, VALUE_PROB_POSITIVE,
    mo, np,
):
    _alpha = np.asarray(COST_LOSS)
    _prob = np.asarray(VALUE_PROB)
    _det = np.asarray(VALUE_DET)
    mo.md(
        rf"""
    Both forecasts contain **exactly the same information** — the deterministic one is
    the probabilistic one pushed through a 50 % threshold. What differs is who can use
    it.

    The probabilistic forecast has useful value at
    **{VALUE_PROB_POSITIVE} of {_alpha.size}** cost-loss ratios; the deterministic one
    at {VALUE_DET_POSITIVE}. Its peak is higher ({_prob.max():.3f} against
    {_det.max():.3f}) — a calibrated probability beats a point forecast even for the
    user it suits best. And at the extremes the deterministic forecast is
    **actively harmful**: $V = {_det.min():.2f}$ at $\alpha = {_alpha[int(np.argmin(_det))]:.2f}$,
    where a user would have done better ignoring it entirely and following the
    climatology. The probabilistic forecast never falls below {_prob.min():.2f}.

    This is the argument for ensemble forecasting that no accuracy score can make. The
    ensemble is not primarily a device for being more often right; it is a device for
    **letting every user apply their own threshold**. A deterministic forecast has
    already made that decision on their behalf, and it cannot have made it correctly for
    more than one of them.
    """
    )
    return


# ===========================================================================
# Section 6
# ===========================================================================
@app.cell(hide_code=True)
def s6_md(mo):
    mo.md(
        r"""
    ## 6 · What to take away

    **Growing fastest is the wrong objective.** Singular vectors are, by construction,
    the fastest-growing perturbations — and an ensemble built from them is
    over-dispersed by half at medium lead and scores worse than isotropic noise at
    short lead. A probability forecast needs to *sample the analysis-error
    distribution*, which is a different thing.

    **Bred vectors collapse, and the collapse is expensive.** Independently seeded bred
    vectors converge onto each other, at a rate set by how well separated the leading
    Lyapunov exponent is. The resulting ensemble grows along one direction while the
    truth spreads into thirteen, giving spread/error of 0.71 at lead 2. Orthogonalising
    each cycle recovers most of the loss.

    **The EDA wins because it is the only construction that answers the right
    question.** It samples the distribution the analysis error actually has.

    **Ensemble size buys less than construction.** CRPS is $a + b/k$: the $1/k$ term is
    the estimator's finite-sample bias and is the same for every construction, while the
    asymptote $a$ is where the construction shows — and no ensemble size closes a gap in
    $a$.

    **Reliability is repairable; resolution is not.** Split the Brier score before
    judging a forecast system. A badly calibrated, high-resolution forecast is a good
    forecast with a fixable problem; a well-calibrated, low-resolution one is honestly
    useless.

    **Value is a curve, not a number.** The case for ensembles is not that they are more
    often right, but that they let every user apply their own threshold — and a
    deterministic forecast is worse than useless for users at the extremes.

    ### Try this

    1. In section 2, slide the lead and watch the singular-vector rank histogram change
       shape. At which lead is it flattest, and what does that tell you about the
       window the singular vectors were computed over?
    2. Bred vectors are under-dispersed and singular vectors over-dispersed. Would an
       ensemble built from *both* be calibrated? What would its rank histogram look
       like, and would CRPS reward it?
    3. Section 3 fits CRPS $= a + b/k$. Predict $b$ for a 100-member ensemble of the
       *wrong* directions, and say which of $a$ and $b$ you could improve by buying more
       computer time.
    4. The recalibration in section 4 is in-sample. Design the out-of-sample version,
       and say what you expect to happen to reliability and to resolution.
    5. Section 5's deterministic forecast thresholds at 50 %. Find the threshold that
       maximises its value at $\alpha = 0.1$, and explain why no single choice can serve
       both ends of the range.

    ### Where this goes next

    **Chapter 22** asks how any of this is verified when the truth is itself only
    observed — every score here compared a forecast against a known truth, which no
    operational centre ever has. **Chapter 21** is where the model error this chapter
    ignored comes back: all five constructions here assumed a perfect model, and
    perturbing the initial state cannot represent an error in the equations.

    ### Further reading

    - Toth & Kalnay (1993, 1997), breeding and the NCEP ensemble *[citation needed]*
    - Molteni et al. (1996), the ECMWF singular-vector ensemble *[citation needed]*
    - Buizza & Palmer (1995), on singular vectors and ensemble design
      *[citation needed]*
    - Murphy (1973), the Brier score decomposition *[citation needed]*
    - Hersbach (2000), on CRPS and its decomposition *[citation needed]*
    - Richardson (2000), on the relative economic value of ensemble forecasts
      *[citation needed]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, ch. 10
      *[citation needed: chapter number]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
