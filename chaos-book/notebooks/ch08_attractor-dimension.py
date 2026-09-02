# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 8 -- Attractors, fractal dimension, and entropy.

Two independent routes to the same number, and the several ways each of them
will quietly give you a different one.

Part III of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
Figures are static matplotlib, matching chapters 5-7 and 11.

To edit:   marimo edit notebooks/ch08_attractor-dimension.py
To export: make nb-one NB=ch08_attractor-dimension
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 8: Attractor Dimension")


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

    from chaoslib import dimension, plotting

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
    mpl_panels = plotting.mpl_panels
    finish_mpl = plotting.finish_mpl

    LAMBDA1_L63 = 0.9056  # chapter 7, pinned in chaoslib's tests
    DAYS_PER_MTU = 5.0

    return (
        C_BG,
        C_CONTEXT,
        C_FIXED,
        C_MEAN,
        C_OBS,
        C_PERT,
        C_SAT,
        C_SPREAD,
        C_START,
        C_TRUTH,
        DAYS_PER_MTU,
        LAMBDA1_L63,
        dimension,
        finish_mpl,
        mo,
        mpl_panels,
        np,
    )


# ---------------------------------------------------------------------------
# Precomputed curves
# ---------------------------------------------------------------------------
@app.cell
def curve_data():
    # From scripts/generate_ch08_data.py.
    #
    # This chapter splits along an unusually clean line. Every expensive
    # computation here is knob-FREE -- forming C(r) is O(N^2) in the sample and
    # does not depend on anything the reader chooses -- while the one thing the
    # reader does choose, the scaling window, costs a polyfit over a stored
    # curve. So the curves are precomputed and every slider re-fits them, which
    # is instant and is also exactly the pedagogical point: one fixed curve,
    # with the window moving across it, rather than a curve that changes shape
    # underneath the reader.
    #
    # The radius ranges are deliberately much wider than any sensible fit
    # window, so that the window can be dragged into the saturated and the
    # noise-dominated regions and be seen to return a confident wrong answer.
    L63_DIAMETER = 72.819781
    L63_D2 = 2.0579
    L63_DKY = 2.0618
    L63_HKS = 0.9010
    HENON_DIAMETER = 2.670240
    HENON_D2 = 1.1919
    L63_BOX_N = 19251
    EMBED_DIMENSIONS = (2, 3, 4, 5, 6)
    EMBED_LAGS = (10, 20, 30)
    L63_RADII = (
        0.0218459, 0.0251433, 0.0289384, 0.0333063, 0.0383335,
        0.0441194, 0.0507787, 0.0584432, 0.0672644, 0.0774172,
        0.0891024, 0.102551, 0.11803, 0.135845, 0.15635, 0.179949,
        0.20711, 0.23837, 0.27435, 0.315759, 0.363419, 0.418273,
        0.481406, 0.554069, 0.637698, 0.733951, 0.844732, 0.972234,
        1.11898, 1.28788, 1.48227, 1.706, 1.9635, 2.25986, 2.60096,
        2.99355, 3.44538, 3.96542, 4.56396, 5.25283, 6.04568, 6.9582,
        8.00846, 9.21724, 10.6085, 12.2097, 14.0526, 16.1737, 18.6149,
        21.4246, 24.6583, 28.3802, 32.6639, 37.5941, 43.2684, 49.7993,
        57.3159, 65.967, 75.9239, 87.3837,
    )
    L63_C_THEILER_0 = (
        2.697899e-07, 2.697899e-07, 5.395799e-07, 1.348950e-06,
        1.753635e-06, 2.023425e-06, 2.967689e-06, 3.642164e-06,
        4.856219e-06, 7.823908e-06, 1.079160e-05, 1.551292e-05,
        2.171809e-05, 3.048626e-05, 4.114297e-05, 5.503715e-05,
        7.486671e-05, 1.042738e-04, 1.435282e-04, 1.939790e-04,
        2.666874e-04, 3.573368e-04, 4.802261e-04, 6.483052e-04,
        8.867995e-04, 1.189369e-03, 1.602552e-03, 2.153059e-03,
        2.885403e-03, 3.870811e-03, 5.168231e-03, 6.900282e-03,
        9.186752e-03, 1.221852e-02, 1.626402e-02, 2.159534e-02,
        2.868042e-02, 3.809380e-02, 5.040013e-02, 6.624071e-02,
        8.633237e-02, 1.116161e-01, 1.430444e-01, 1.818526e-01,
        2.285918e-01, 2.848595e-01, 3.524447e-01, 4.332260e-01,
        5.283665e-01, 6.359167e-01, 7.461760e-01, 8.422320e-01,
        9.130781e-01, 9.612333e-01, 9.896871e-01, 9.991869e-01,
        9.999985e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00,
    )
    L63_C_THEILER_10 = (
        2.711967e-07, 2.711967e-07, 5.423935e-07, 1.355984e-06,
        1.762779e-06, 2.033976e-06, 2.983164e-06, 3.661156e-06,
        4.881541e-06, 7.864705e-06, 1.084787e-05, 1.559381e-05,
        2.183134e-05, 3.064523e-05, 4.135750e-05, 5.532413e-05,
        7.525709e-05, 1.048175e-04, 1.442767e-04, 1.949905e-04,
        2.680780e-04, 3.592001e-04, 4.827302e-04, 6.516858e-04,
        8.914237e-04, 1.195571e-03, 1.610909e-03, 2.164286e-03,
        2.900178e-03, 3.889368e-03, 5.191790e-03, 6.929483e-03,
        9.219604e-03, 1.225769e-02, 1.630340e-02, 2.162957e-02,
        2.869967e-02, 3.809460e-02, 5.037100e-02, 6.620034e-02,
        8.628002e-02, 1.115598e-01, 1.429989e-01, 1.817847e-01,
        2.284729e-01, 2.847160e-01, 3.522884e-01, 4.330632e-01,
        5.281750e-01, 6.357332e-01, 7.460419e-01, 8.421560e-01,
        9.130356e-01, 9.612299e-01, 9.896995e-01, 9.991845e-01,
        9.999985e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00,
    )
    L63_C_THEILER_50 = (
        2.769354e-07, 2.769354e-07, 5.538709e-07, 1.384677e-06,
        1.800080e-06, 2.077016e-06, 3.046290e-06, 3.738628e-06,
        4.846370e-06, 7.477257e-06, 1.052355e-05, 1.509298e-05,
        2.118556e-05, 2.977056e-05, 4.057104e-05, 5.441781e-05,
        7.463410e-05, 1.044047e-04, 1.437295e-04, 1.941317e-04,
        2.673812e-04, 3.583544e-04, 4.824215e-04, 6.531522e-04,
        8.936706e-04, 1.198023e-03, 1.613564e-03, 2.166327e-03,
        2.901453e-03, 3.891220e-03, 5.190601e-03, 6.924632e-03,
        9.214473e-03, 1.225301e-02, 1.630277e-02, 2.163461e-02,
        2.870837e-02, 3.809745e-02, 5.035669e-02, 6.615904e-02,
        8.620280e-02, 1.114586e-01, 1.428865e-01, 1.816313e-01,
        2.282912e-01, 2.845171e-01, 3.520533e-01, 4.328288e-01,
        5.279591e-01, 6.355825e-01, 7.459709e-01, 8.421408e-01,
        9.130536e-01, 9.612581e-01, 9.897138e-01, 9.991850e-01,
        9.999985e-01, 1.000000e+00, 1.000000e+00, 1.000000e+00,
    )
    L63_SPECTRUM = (
        0.894763, 0.006198, -14.567526,
    )
    HENON_RADII = (
        0.000801072, 0.000948818, 0.00112381, 0.00133108, 0.00157658,
        0.00186736, 0.00221177, 0.0026197, 0.00310287, 0.00367514,
        0.00435297, 0.00515581, 0.00610673, 0.00723303, 0.00856705,
        0.0101471, 0.0120186, 0.0142353, 0.0168608, 0.0199705,
        0.0236538, 0.0280163, 0.0331836, 0.0393038, 0.0465528,
        0.0551388, 0.0653083, 0.0773535, 0.0916203, 0.108518,
        0.128533, 0.152239, 0.180317, 0.213574, 0.252965, 0.29962,
        0.354881, 0.420334, 0.497858, 0.589681, 0.698439, 0.827256,
        0.979832, 1.16055, 1.37459, 1.62812, 1.9284, 2.28407, 2.70533,
        3.20429,
    )
    HENON_C = (
        1.316559e-04, 1.632119e-04, 2.016223e-04, 2.470165e-04,
        3.074126e-04, 3.804830e-04, 4.690726e-04, 5.695605e-04,
        6.924221e-04, 8.385627e-04, 1.015225e-03, 1.223831e-03,
        1.495290e-03, 1.861159e-03, 2.318980e-03, 2.880134e-03,
        3.577083e-03, 4.394307e-03, 5.383924e-03, 6.584605e-03,
        8.010317e-03, 9.798665e-03, 1.203746e-02, 1.483599e-02,
        1.821986e-02, 2.221816e-02, 2.707792e-02, 3.294178e-02,
        4.038525e-02, 4.949977e-02, 6.047946e-02, 7.378964e-02,
        9.046480e-02, 1.116164e-01, 1.394125e-01, 1.740722e-01,
        2.165790e-01, 2.658109e-01, 3.227227e-01, 3.886986e-01,
        4.651436e-01, 5.468191e-01, 6.273028e-01, 7.021504e-01,
        7.794283e-01, 8.630332e-01, 9.351720e-01, 9.828051e-01,
        1.000000e+00, 1.000000e+00,
    )
    CANTOR_SCALES = (
        1e-06, 1.65609e-06, 2.74265e-06, 4.54208e-06, 7.52211e-06,
        1.24573e-05, 2.06305e-05, 3.4166e-05, 5.65821e-05,
        9.37052e-05, 0.000155185, 0.000257, 0.000425616, 0.00070486,
        0.00116731, 0.00193318, 0.00320153, 0.00530203, 0.00878066,
        0.0145416, 0.0240822, 0.0398825, 0.0660491, 0.109383,
        0.181149, 0.3,
    )
    CANTOR_EXP_Q0 = (
        -9.37152, -9.06889, -8.73986, -8.42945, -8.11552, -7.7907,
        -7.4793, -7.16162, -6.85541, -6.47697, -6.2106, -5.89164,
        -5.59842, -5.23111, -4.95583, -4.61512, -4.27667, -3.97029,
        -3.61092, -3.2581, -2.99573, -2.70805, -2.30259, -2.07944,
        -1.60944, -1.38629,
    )
    CANTOR_EXP_Q1 = (
        -9.26671, -8.95362, -8.64401, -8.32729, -8.01673, -7.69818,
        -7.38307, -7.07706, -6.75158, -6.421, -6.05448, -5.80431,
        -5.47399, -5.13389, -4.84645, -4.51656, -4.19579, -3.88742,
        -3.55572, -3.15872, -2.91238, -2.6231, -2.2196, -1.81463,
        -1.58044, -1.28025,
    )
    CANTOR_EXP_Q2 = (
        -9.19696, -8.88045, -8.57852, -8.26563, -7.94899, -7.64773,
        -7.31372, -7.03125, -6.67871, -6.38746, -5.96574, -5.75436,
        -5.40183, -5.07724, -4.7858, -4.45157, -4.14476, -3.82469,
        -3.52043, -3.0885, -2.87312, -2.55496, -2.18148, -1.69689,
        -1.55574, -1.20476,
    )
    CANTOR_OCCUPANCY = (
        17.02, 23.04, 32.02, 43.67, 59.77, 82.71, 112.9, 155.2, 210.7,
        307.7, 401.6, 552.5, 740.7, 1070, 1408, 1980, 2778, 3774,
        5405, 7692, 1e+04, 1.333e+04, 2e+04, 2.5e+04, 4e+04, 5e+04,
    )
    KOCH_SCALES = (
        1e-05, 1.49695e-05, 2.24087e-05, 3.35448e-05, 5.0215e-05,
        7.51696e-05, 0.000112525, 0.000168445, 0.000252155,
        0.000377464, 0.000565047, 0.000845849, 0.0012662, 0.00189544,
        0.00283739, 0.00424744, 0.00635821, 0.00951795, 0.0142479,
        0.0213285, 0.0319278, 0.0477944, 0.0715461, 0.107101,
        0.160326, 0.24,
    )
    KOCH_EXP_Q0 = (
        -11.0904, -11.0904, -11.0904, -11.0904, -11.0904, -11.0904,
        -11.0904, -10.9363, -10.5458, -10.079, -9.57859, -9.07989,
        -8.57998, -8.06212, -7.55171, -6.96885, -6.45677, -6.03309,
        -5.47646, -4.96284, -4.40672, -3.82864, -3.46574, -2.77259,
        -2.30259, -1.94591,
    )
    KOCH_EXP_Q1 = (
        -11.0904, -11.0904, -11.0904, -11.0904, -11.0904, -11.0904,
        -11.0904, -10.8888, -10.4611, -9.94858, -9.42714, -8.91309,
        -8.39632, -7.88705, -7.36309, -6.79579, -6.28296, -5.84486,
        -5.28959, -4.73866, -4.12514, -3.63229, -3.2192, -2.63504,
        -2.15304, -1.65885,
    )
    KOCH_EXP_Q2 = (
        -11.0904, -11.0904, -11.0904, -11.0904, -11.0904, -11.0904,
        -11.0904, -10.8285, -10.3871, -9.85438, -9.33567, -8.82423,
        -8.30589, -7.79991, -7.27589, -6.7159, -6.20911, -5.7502,
        -5.21088, -4.65593, -4.05767, -3.58341, -3.10876, -2.56267,
        -2.08306, -1.554,
    )
    KOCH_OCCUPANCY = (
        1, 1, 1, 1, 1, 1, 1, 1.167, 1.724, 2.749, 4.535, 7.467, 12.31,
        20.66, 34.42, 61.65, 102.9, 157.2, 274.2, 458.3, 799.2, 1425,
        2048, 4096, 6554, 9362,
    )
    SIERPINSKI_SCALES = (
        0.0001, 0.000140591, 0.000197659, 0.000277892, 0.000390693,
        0.00054928, 0.000772241, 0.0010857, 0.00152641, 0.002146,
        0.00301709, 0.00424177, 0.00596356, 0.00838425, 0.0117875,
        0.0165723, 0.0232992, 0.0327567, 0.0460531, 0.0647466,
        0.0910282, 0.127978, 0.179926, 0.25296, 0.35564, 0.5,
    )
    SIERPINSKI_EXP_Q0 = (
        -12.1679, -12.1403, -12.0957, -12.0175, -11.8942, -11.6976,
        -11.4077, -11.023, -10.5652, -10.0535, -9.55648, -9.00541,
        -8.47324, -7.95121, -7.40123, -6.83626, -6.32972, -5.77144,
        -5.1985, -4.65396, -4.09434, -3.55535, -2.99573, -2.48491,
        -1.94591, -1.60944,
    )
    SIERPINSKI_EXP_Q1 = (
        -12.1536, -12.1163, -12.0569, -11.9549, -11.8016, -11.5687,
        -11.2502, -10.854, -10.4018, -9.89741, -9.39946, -8.85396,
        -8.32115, -7.79671, -7.24486, -6.68044, -6.16666, -5.60529,
        -5.05428, -4.49883, -3.96007, -3.40537, -2.85263, -2.31729,
        -1.74039, -1.21185,
    )
    SIERPINSKI_EXP_Q2 = (
        -12.132, -12.0809, -12.0024, -11.873, -11.6929, -11.436,
        -11.1102, -10.7223, -10.2876, -9.79831, -9.31125, -8.77636,
        -8.2502, -7.7238, -7.17709, -6.61476, -6.09846, -5.53955,
        -4.9912, -4.42969, -3.895, -3.34102, -2.77858, -2.23424,
        -1.64125, -1.10249,
    )
    SIERPINSKI_OCCUPANCY = (
        1.039, 1.068, 1.117, 1.208, 1.366, 1.663, 2.222, 3.264, 5.159,
        8.607, 14.15, 24.55, 41.8, 70.45, 122.1, 214.8, 356.5, 623.1,
        1105, 1905, 3333, 5714, 1e+04, 1.667e+04, 2.857e+04, 4e+04,
    )
    L63_BOX_SCALES = (
        0.002, 0.00257866, 0.00332474, 0.00428668, 0.00552694,
        0.00712604, 0.00918781, 0.0118461, 0.0152735, 0.0196926,
        0.0253902, 0.0327364, 0.0422079, 0.0544199, 0.0701652,
        0.090466, 0.11664, 0.150388, 0.193899, 0.25,
    )
    L63_BOX_EXP_Q0 = (
        -9.81629, -9.78437, -9.72824, -9.64368, -9.51119, -9.31407,
        -9.03503, -8.71637, -8.33014, -7.90802, -7.49387, -7.06133,
        -6.6053, -6.1506, -5.67332, -5.22036, -4.76217, -4.29046,
        -3.91202, -3.3322,
    )
    L63_BOX_OCCUPANCY = (
        1.05, 1.084, 1.147, 1.248, 1.425, 1.735, 2.294, 3.155, 4.642,
        7.08, 10.71, 16.51, 26.05, 41.05, 66.15, 104.1, 164.5, 263.7,
        385, 687.5,
    )
    EMBED_D2 = (
        1.7202, 1.6738, 1.7055, 1.8708, 1.9454, 1.9866, 1.9504,
        2.0038, 1.9812, 1.9878, 2.0092, 1.9728, 2.0045, 1.9804,
        1.9906,
    )
    return (
        CANTOR_EXP_Q0,
        CANTOR_EXP_Q1,
        CANTOR_EXP_Q2,
        CANTOR_OCCUPANCY,
        CANTOR_SCALES,
        EMBED_D2,
        EMBED_DIMENSIONS,
        EMBED_LAGS,
        HENON_C,
        HENON_D2,
        HENON_DIAMETER,
        HENON_RADII,
        KOCH_EXP_Q0,
        KOCH_EXP_Q1,
        KOCH_EXP_Q2,
        KOCH_OCCUPANCY,
        KOCH_SCALES,
        L63_BOX_EXP_Q0,
        L63_BOX_N,
        L63_BOX_OCCUPANCY,
        L63_BOX_SCALES,
        L63_C_THEILER_0,
        L63_C_THEILER_10,
        L63_C_THEILER_50,
        L63_D2,
        L63_DIAMETER,
        L63_DKY,
        L63_HKS,
        L63_RADII,
        L63_SPECTRUM,
        SIERPINSKI_EXP_Q0,
        SIERPINSKI_EXP_Q1,
        SIERPINSKI_EXP_Q2,
        SIERPINSKI_OCCUPANCY,
        SIERPINSKI_SCALES,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 8 · Attractors, Fractal Dimension, and Entropy

    **Part III — Quantifying chaos and predictability.**

    **The forecasting question.** A forecast model has $10^7$ variables. Does
    that mean an analysis has to constrain $10^7$ numbers?

    No — because the model's trajectory does not visit most of its state space.
    It settles onto an attractor, and the attractor has a dimension of its own,
    generally far smaller than the state dimension and generally not an integer.
    That number is what an observing system actually has to pin down, and this
    chapter is about measuring it.

    Two quantities, both of which chapter 7 left owing:

    $$D_{KY} = j + \frac{\sum_{i=1}^{j}\lambda_i}{|\lambda_{j+1}|},
      \qquad
      C(r) \sim r^{D_2} .$$

    The first, the **Kaplan–Yorke dimension**, comes from the Lyapunov spectrum
    — from the *dynamics*, computed by integrating the tangent equations. The
    second, the **correlation dimension**, counts pairs of points closer than
    $r$ on a sampled trajectory — from the *geometry*, using nothing but a cloud
    of points. They are not two ways of writing the same calculation; they share
    no intermediate quantity. Agreement between them is therefore evidence, and
    for Lorenz 63 they agree to 0.2 %.

    Along with the third quantity that falls out of the same spectrum:

    $$h_{KS} = \sum_{\lambda_i > 0}\lambda_i$$

    the **Kolmogorov–Sinai entropy**, which measures how fast the system
    destroys information about its own initial condition — in bits per day, if
    you like, and that is the rate at which observations must arrive to keep a
    forecast alive.

    **Most of this chapter is about how easily these numbers are got wrong.**
    Not as a caveat at the end, but as the substance: a dimension estimate is a
    slope fitted over a range you chose, and choosing badly returns a clean fit
    with a small residual and the wrong answer. Nothing in the return value
    tells you.

    ---

    ## What is here

    | Section | The question |
    |---|---|
    | 1 | Can box counting recover a dimension we know exactly? |
    | 2 | Why not use it on Lorenz 63? |
    | 3 | The correlation sum, and the two ways to misread it |
    | 4 | Geometry against dynamics: $D_2$ against $D_{KY}$ |
    | 5 | Entropy: what the attractor costs you per day |
    | 6 | Dimension from one observed variable |
    """
    )
    return


# ===========================================================================
# 1. Box counting, checked against exact answers
# ===========================================================================
@app.cell(hide_code=True)
def s1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · Start with sets whose dimension is known

    Cover the set with boxes of side $\varepsilon$, count the occupied ones, and
    look at how that count grows as the boxes shrink:

    $$N(\varepsilon) \sim \varepsilon^{-D_0}
      \quad\Longrightarrow\quad
      D_0 = -\lim_{\varepsilon\to0}\frac{\ln N(\varepsilon)}{\ln\varepsilon}.$$

    For a smooth curve $N \propto \varepsilon^{-1}$ and $D_0 = 1$; for a filled
    region, $D_0 = 2$. For the three sets below it is neither, and — because
    each is built by exact self-similar recursion — we know the answer in closed
    form before measuring anything:

    | set | construction | $D_0$ |
    |---|---|---|
    | Cantor | keep 2 pieces of scale $1/3$ | $\ln2/\ln3 = 0.6309$ |
    | Koch curve | keep 4 pieces of scale $1/3$ | $\ln4/\ln3 = 1.2619$ |
    | Sierpiński | keep 3 pieces of scale $1/2$ | $\ln3/\ln2 = 1.5850$ |

    In general $N$ pieces at scale $r$ gives $D_0 = \ln N/\ln(1/r)$. These are
    the calibration for everything else in the chapter: an estimator that cannot
    return 1.5850 here has no business being pointed at an attractor.

    **Move the fit window.** The curve is fixed; the shaded band is what gets
    fitted. There is a right answer available and you can miss it.
    """
    )
    return


@app.cell(hide_code=True)
def s1_controls(mo):
    fractal_choice = mo.ui.dropdown(
        options={
            "Sierpiński triangle  (D₀ = ln3/ln2 = 1.5850)": "sierpinski",
            "Koch curve  (D₀ = ln4/ln3 = 1.2619)": "koch",
            "Cantor set  (D₀ = ln2/ln3 = 0.6309)": "cantor",
        },
        value="Sierpiński triangle  (D₀ = ln3/ln2 = 1.5850)",
        label="reference set",
    )
    window_lo = mo.ui.slider(
        start=-5.0, stop=-1.4, step=0.1, value=-2.5,
        label="fit window: log₁₀ of the finer edge", show_value=True,
    )
    window_width = mo.ui.slider(
        start=0.4, stop=3.0, step=0.1, value=1.0,
        label="fit window: width in decades", show_value=True,
    )
    return fractal_choice, window_lo, window_width


@app.cell(hide_code=True)
def s1_figure(
    CANTOR_EXP_Q0,
    CANTOR_OCCUPANCY,
    CANTOR_SCALES,
    C_CONTEXT,
    C_PERT,
    C_SAT,
    C_TRUTH,
    KOCH_EXP_Q0,
    KOCH_OCCUPANCY,
    KOCH_SCALES,
    SIERPINSKI_EXP_Q0,
    SIERPINSKI_OCCUPANCY,
    SIERPINSKI_SCALES,
    dimension,
    finish_mpl,
    fractal_choice,
    mo,
    mpl_panels,
    np,
    window_lo,
    window_width,
):
    _sets = {
        "cantor": (CANTOR_SCALES, CANTOR_EXP_Q0, CANTOR_OCCUPANCY, "Cantor set"),
        "koch": (KOCH_SCALES, KOCH_EXP_Q0, KOCH_OCCUPANCY, "Koch curve"),
        "sierpinski": (
            SIERPINSKI_SCALES, SIERPINSKI_EXP_Q0, SIERPINSKI_OCCUPANCY,
            "Sierpiński triangle",
        ),
    }
    _key = str(fractal_choice.value)
    _scales, _exponents, _occupancy, _label = _sets[_key]
    _scales = np.asarray(_scales)
    # For q = 0 the stored exponent is -ln N(eps), so the box count is its
    # negative exponential and the slope of the exponent IS D_0.
    _exponents = np.asarray(_exponents)
    _occupancy = np.asarray(_occupancy)
    _counts = np.exp(-_exponents)
    _exact = dimension.REFERENCE_DIMENSIONS[_key]
    _vetted = dimension.REFERENCE_WINDOWS[_key]

    _lo = 10.0 ** float(window_lo.value)
    _hi = _lo * 10.0 ** float(window_width.value)
    _inside = (_scales >= _lo) & (_scales <= _hi)
    _n_inside = int(_inside.sum())

    _fig, _ax = mpl_panels(
        3,
        titles=(f"{_label}: box count", "Local slope", "Points per box"),
        height=3.6,
    )
    _ax[0].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[0].loglog(_scales, _counts, marker="o", markersize=3, color=C_TRUTH,
                  linewidth=1.3)
    _ax[0].set_xlabel(r"box size $\varepsilon$")
    _ax[0].set_ylabel(r"occupied boxes $N(\varepsilon)$")

    _slopes = np.gradient(_exponents, np.log(_scales))
    _ax[1].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[1].axhline(_exact, color=C_SAT, linewidth=1.3, linestyle="--",
                   label=f"exact {_exact:.4f}")
    _ax[1].semilogx(_scales, _slopes, marker="o", markersize=3, color=C_TRUTH,
                    linewidth=1.3)
    _ax[1].set_xlabel(r"box size $\varepsilon$")
    _ax[1].set_ylabel(r"$d\ln N / d\ln(1/\varepsilon)$")
    _ax[1].set_ylim(0.0, max(2.2, _exact + 0.7))
    _ax[1].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    _ax[2].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[2].axhline(dimension.MIN_BOX_OCCUPANCY, color=C_PERT, linewidth=1.3,
                   linestyle="--",
                   label=f"{dimension.MIN_BOX_OCCUPANCY:g} points/box")
    _ax[2].loglog(_scales, _occupancy, marker="o", markersize=3, color=C_TRUTH,
                  linewidth=1.3)
    _ax[2].set_xlabel(r"box size $\varepsilon$")
    _ax[2].set_ylabel("mean points per occupied box")
    _ax[2].legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    if _n_inside >= 3:
        _fitted = float(np.polyfit(np.log(_scales[_inside]), _exponents[_inside], 1)[0])
        _min_occupancy = float(_occupancy[_inside].min())
        _error = abs(_fitted - _exact)
        _starved = _min_occupancy < dimension.MIN_BOX_OCCUPANCY
        _verdict = (
            f"""| | |
|---|---|
| fit window | {_lo:.2e} to {_hi:.2e} ({_n_inside} scales, {float(window_width.value):.1f} decades) |
| measured $D_0$ | **{_fitted:.4f}** |
| exact $D_0$ | {_exact:.4f} |
| error | {_error:.4f} |
| lowest occupancy in the window | {_min_occupancy:.1f} points per box |

The vetted window for this set is {_vetted[0]:.0e} to {_vetted[1]:.0e}, which
gives an error below 0.01."""
        )
        if _starved:
            _verdict += f"""

**This window is starved.** At {_min_occupancy:.1f} points per occupied box the
count is limited by the sample, not by the set: as occupancy approaches 1, every
point gets its own box, $N(\\varepsilon)$ stops responding to
$\\varepsilon$ at all, and the slope falls toward zero. It is measuring how many
points you drew."""
        elif _error > 0.05:
            _verdict += f"""

**This window is in the wrong place** — off by {_error:.3f} — but the occupancy
is fine, so the sample is not the problem. Look at the middle panel: the local
slope is not on its plateau here. At coarse $\\varepsilon$ the boxes are bigger
than the structure and the set looks like whatever it is embedded in."""
    else:
        _fitted = float("nan")
        _verdict = (
            f"Only {_n_inside} scales fall inside "
            f"{_lo:.2e}–{_hi:.2e}; widen the window."
        )

    mo.vstack([
        mo.hstack([fractal_choice, window_lo, window_width], justify="start"),
        _fig,
        mo.md(_verdict),
        mo.md(
            r"""**The Cantor set is worth a look for a different reason.** Its
            local slope does not sit on a plateau at all — it oscillates, with
            period $\ln 3$ in $\ln\varepsilon$, because the set's structure
            recurs at powers of three while the box ladder does not. This is
            *lacunarity*, and it means the honest fit averages over several
            periods of the oscillation rather than finding a flat stretch. Widen
            the window to 2.5 decades and the estimate lands within 0.002; take
            a narrow window anywhere and it will be off by up to 0.06 depending
            on where you happen to put it."""
        ),
    ])
    return


# ===========================================================================
# 2. Why box counting is the wrong tool in three dimensions
# ===========================================================================
@app.cell(hide_code=True)
def s2_figure(
    C_CONTEXT,
    C_PERT,
    C_SAT,
    C_TRUTH,
    L63_BOX_EXP_Q0,
    L63_BOX_N,
    L63_BOX_OCCUPANCY,
    L63_BOX_SCALES,
    L63_D2,
    dimension,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _scales = np.asarray(L63_BOX_SCALES)
    _exponents = np.asarray(L63_BOX_EXP_Q0)
    _occupancy = np.asarray(L63_BOX_OCCUPANCY)
    _slopes = np.gradient(_exponents, np.log(_scales))
    _usable = _occupancy >= dimension.MIN_BOX_OCCUPANCY

    _fig, _ax = mpl_panels(
        2,
        titles=("Lorenz 63: box counting", "…and the sample runs out first"),
        height=3.5,
    )
    _ax[0].axhline(L63_D2, color=C_SAT, linewidth=1.3, linestyle="--",
                   label=f"$D_2$ from pair counting: {L63_D2:.3f}")
    _ax[0].semilogx(_scales, _slopes, marker="o", markersize=3.4, color=C_TRUTH,
                    linewidth=1.3, label="local slope")
    if (~_usable).any():
        _ax[0].axvspan(_scales.min(), _scales[_usable].min(), color=C_CONTEXT,
                       zorder=0)
        _ax[0].annotate("starved", (np.sqrt(_scales.min() * _scales[_usable].min()), 0.35),
                        ha="center", fontsize=6.5, color="#6b7280")
    _ax[0].set_xlabel(r"box size $\varepsilon$ (fraction of the extent)")
    _ax[0].set_ylabel(r"$d\ln N/d\ln(1/\varepsilon)$")
    _ax[0].set_ylim(0.0, 2.6)
    _ax[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    _ax[1].axhline(dimension.MIN_BOX_OCCUPANCY, color=C_PERT, linewidth=1.3,
                   linestyle="--", label=f"{dimension.MIN_BOX_OCCUPANCY:g} points/box")
    _ax[1].loglog(_scales, _occupancy, marker="o", markersize=3.4, color=C_TRUTH,
                  linewidth=1.3)
    _ax[1].set_xlabel(r"box size $\varepsilon$")
    _ax[1].set_ylabel("mean points per occupied box")
    _ax[1].legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"{L63_BOX_N} samples of the Lorenz attractor")

    # Report the typical slope over the usable range, not the maximum: the
    # single coarsest scale divides the attractor into a handful of boxes and
    # throws up a spuriously high value that would read as near-success.
    _usable_slopes = _slopes[_usable]
    _typical = float(np.median(_usable_slopes)) if _usable.any() else float("nan")
    _spurious = float(_usable_slopes.max()) if _usable.any() else float("nan")
    _decades = (
        np.log10(_scales[_usable].max() / _scales[_usable].min())
        if _usable.any() else 0.0
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 2 · Why not just use that on Lorenz 63?

    Because in three dimensions the sample runs out before the scaling does.
    Box counting spreads $N$ points over $\varepsilon^{-d}$ boxes, and the count
    of boxes needed grows with the *embedding* dimension whether or not the
    attractor fills it. In one and two dimensions Section 1 had hundreds of
    points per box across two decades of scale. Here it does not."""
        ),
        _fig,
        mo.md(
            f"""
With **{L63_BOX_N} samples** the usable range — occupancy above
{dimension.MIN_BOX_OCCUPANCY:g} points per box — is only
**{_decades:.1f} decades** wide, and across it the local slope sits around
**{_typical:.2f}**, well short of the {L63_D2:.3f} that pair counting gets from
the same trajectory. There is no plateau to find: the curve is still climbing
where the sample gives out. (It does touch {_spurious:.2f} at the single
coarsest scale, but there the boxes are a fifth of the attractor and there are
about five of them, so that number is an artefact of having almost no boxes
rather than a measurement.)

Getting two clean decades in three dimensions would need of order $10^6$ points,
and a fourth dimension a hundred times more again.

**The fix is to stop counting boxes and start counting pairs.** The correlation
sum uses all $N(N-1)/2$ *pairs* of points rather than distributing $N$ points
over a grid, so it extracts far more from the same sample and does not care how
many dimensions the attractor is embedded in — only how many it fills. That is
Section 3, and it is why every dimension quoted for a real attractor is a
correlation dimension.
"""
        ),
    ])
    return


# ===========================================================================
# 3. The correlation sum, and the two ways to misread it
# ===========================================================================
@app.cell(hide_code=True)
def s3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · Counting pairs

    $$C(r) = \frac{2}{N(N-1-2w)}\sum_{j>i+w}\Theta\bigl(r - \|x_i - x_j\|\bigr)$$

    — the fraction of pairs closer together than $r$. On a $D_2$-dimensional
    set, doubling $r$ multiplies the number of neighbours by $2^{D_2}$, so
    $C(r) \sim r^{D_2}$ and $D_2$ is a slope on log axes.

    That relation holds over a **window** and nowhere else. Above it every pair
    is already counted and $C \to 1$. Below it there are so few pairs that
    $C$ moves in visible steps of $1/N_{\rm pairs}$. And $w$ — the **Theiler
    window** — excludes pairs closer than $w$ samples in *time*, which matters
    because consecutive samples are near each other for reasons that have
    nothing to do with the attractor's geometry.

    Both controls below are traps with a preferred direction, and neither
    direction is the one usually warned about. Find them.
    """
    )
    return


@app.cell(hide_code=True)
def s3_controls(mo):
    theiler_choice = mo.ui.dropdown(
        options={"w = 0 (no exclusion)": "0", "w = 10": "10", "w = 50": "50"},
        value="w = 50",
        label="Theiler window",
    )
    fit_lo = mo.ui.slider(
        start=-3.5, stop=-0.3, step=0.1, value=-2.1,
        label="fit window: log₁₀(r / diameter), finer edge", show_value=True,
    )
    fit_width = mo.ui.slider(
        start=0.3, stop=3.6, step=0.1, value=0.8,
        label="fit window: width in decades", show_value=True,
    )
    return fit_lo, fit_width, theiler_choice


@app.cell(hide_code=True)
def s3_figure(
    C_CONTEXT,
    C_PERT,
    C_SAT,
    C_TRUTH,
    L63_C_THEILER_0,
    L63_C_THEILER_10,
    L63_C_THEILER_50,
    L63_DIAMETER,
    L63_DKY,
    L63_RADII,
    dimension,
    finish_mpl,
    fit_lo,
    fit_width,
    mo,
    mpl_panels,
    np,
    theiler_choice,
):
    _curves = {
        "0": np.asarray(L63_C_THEILER_0),
        "10": np.asarray(L63_C_THEILER_10),
        "50": np.asarray(L63_C_THEILER_50),
    }
    _w = str(theiler_choice.value)
    _radii = np.asarray(L63_RADII)
    _c = _curves[_w]
    _fraction = _radii / L63_DIAMETER

    _lo = 10.0 ** float(fit_lo.value)
    _hi = min(1.2, _lo * 10.0 ** float(fit_width.value))
    _inside = (_fraction >= _lo) & (_fraction <= _hi) & (_c > 0.0)
    _n_inside = int(_inside.sum())

    _fig, _ax = mpl_panels(
        2,
        titles=(r"$C(r)$ for Lorenz 63", "Local slope"),
        height=3.5,
    )
    _ax[0].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[0].loglog(_fraction, np.maximum(_c, 1e-9), marker="o", markersize=2.6,
                  color=C_TRUTH, linewidth=1.2)
    if _n_inside >= 3:
        _fit = np.polyfit(np.log(_radii[_inside]), np.log(_c[_inside]), 1)
        _ax[0].loglog(_fraction[_inside],
                      np.exp(np.polyval(_fit, np.log(_radii[_inside]))),
                      color=C_PERT, linewidth=2.0, linestyle="--", label="the fit")
        _ax[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    _ax[0].set_xlabel("r / attractor diameter")
    _ax[0].set_ylabel("C(r)")

    _ok = _c > 0.0
    _slopes = np.gradient(np.log(_c[_ok]), np.log(_radii[_ok]))
    _ax[1].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[1].axhline(L63_DKY, color=C_SAT, linewidth=1.3, linestyle="--",
                   label=f"$D_{{KY}}$ = {L63_DKY:.3f}")
    _ax[1].semilogx(_fraction[_ok], _slopes, marker="o", markersize=2.6,
                    color=C_TRUTH, linewidth=1.2)
    _ax[1].set_xlabel("r / attractor diameter")
    _ax[1].set_ylabel(r"$d\ln C/d\ln r$")
    _ax[1].set_ylim(0.0, 3.4)
    _ax[1].legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"4000 samples, Theiler window w = {_w}")

    if _n_inside >= 3:
        _d2, _used = dimension.fit_dimension(
            _radii, _c, (_lo * L63_DIAMETER, _hi * L63_DIAMETER)
        )
        _cmax = float(_c[_inside].max())
        if _cmax > 0.4:
            _diagnosis = (
                f"""**Saturated.** $C$ reaches {_cmax:.2f} inside this window: almost
every pair is already counted, so the curve is flattening toward 1 and the slope
toward 0. This is the low-biased failure, and it is the easiest to spot because
the answer is absurd."""
            )
        elif _c[_inside].min() < 3e-5:
            _diagnosis = (
                f"""**Noise floor.** The smallest $C$ in this window is
{_c[_inside].min():.1e} — a handful of pairs out of eight million. The curve
moves in visible steps there and the fitted slope is **biased high**, which is
the failure the textbooks mention least and which is easiest to mistake for a
real answer."""
            )
        elif abs(_hi / _lo) > 100.0:
            _diagnosis = (
                """**Too greedy.** This window spans more than two decades, which
means it necessarily includes both the saturated top and the noisy bottom. The
answer comes out plausible — within a few percent — and wrong. This is the
dangerous case."""
            )
        else:
            _diagnosis = (
                f"""This window sits on the plateau. Measured
$D_2$ = **{_d2:.4f}** against $D_{{KY}}$ = {L63_DKY:.4f} from the dynamics —
Section 4 takes that comparison seriously."""
            )
        _readout = f"""| | |
|---|---|
| fit window | {_lo:.1e} to {_hi:.1e} of the diameter ({_used} radii) |
| measured $D_2$ | **{_d2:.4f}** |
| $D_{{KY}}$ from the spectrum | {L63_DKY:.4f} |
| $C$ across the window | {_c[_inside].min():.2e} to {_cmax:.2e} |

{_diagnosis}"""
    else:
        _readout = f"Only {_n_inside} radii inside the window; widen it."

    mo.vstack([
        mo.hstack([theiler_choice, fit_lo, fit_width], justify="start"),
        _fig,
        mo.md(_readout),
        mo.callout(
            mo.md(
                r"""### The Theiler window, and why its sign surprised me

Switch $w$ from 50 to 0 with the window on the plateau. The estimate moves, but
not far — because these 4000 samples were taken from a trajectory of 770,000
steps, so consecutive *samples* are already about 0.19 MTU apart and largely
decorrelated.

Sample densely instead and the effect is large and has a definite direction. On
a trajectory sampled every 0.01 MTU with no subsampling, the same estimator
returns **2.139** at $w=0$, then 2.118, 2.084, **2.039** at $w = 10, 50, 200$ —
a monotone decrease. So temporal correlation biases $D_2$ **high** here, and the
mechanism is visible: the trajectory moves 1.3 % of the attractor's diameter per
step, which lands *inside* the fit window, so the excess of adjacent pairs
appears as a bump in $C(r)$ right there, and the bump's rising flank steepens
the local slope from 1.92 to 2.22.

The warning usually given is the opposite — that temporal correlation makes the
estimator report the trajectory's smooth one-dimensional curve, biasing *low*.
That happens too, when the sampling is dense enough to put the step scale
*below* the fit window. Which bias you get depends on where your sampling
interval falls relative to your fit window, which is not something you can
correct for afterwards. It is something you have to look at."""
            ),
            kind="warn",
        ),
    ])
    return


# ===========================================================================
# 4. Geometry against dynamics
# ===========================================================================
@app.cell(hide_code=True)
def s4_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_TRUTH,
    HENON_C,
    HENON_D2,
    HENON_DIAMETER,
    HENON_RADII,
    L63_C_THEILER_50,
    L63_D2,
    L63_DIAMETER,
    L63_DKY,
    L63_RADII,
    L63_SPECTRUM,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _spectrum = np.asarray(L63_SPECTRUM)
    _fig, _ax = mpl_panels(
        3,
        titles=("Two routes, one attractor", "Hénon map",
                "Where $D_{KY}$ comes from"),
        height=3.5,
    )

    _frac = np.asarray(L63_RADII) / L63_DIAMETER
    _c = np.asarray(L63_C_THEILER_50)
    _ok = _c > 0.0
    _slopes = np.gradient(np.log(_c[_ok]), np.log(np.asarray(L63_RADII)[_ok]))
    _ax[0].axhline(L63_DKY, color=C_SAT, linewidth=1.4, linestyle="--",
                   label=f"$D_{{KY}}$ = {L63_DKY:.4f}  (dynamics)")
    _ax[0].axhline(L63_D2, color=C_MEAN, linewidth=1.4, linestyle=":",
                   label=f"$D_2$ = {L63_D2:.4f}  (geometry)")
    _ax[0].semilogx(_frac[_ok], _slopes, marker="o", markersize=2.6,
                    color=C_TRUTH, linewidth=1.2, label="local slope")
    _ax[0].set_xlabel("r / diameter")
    _ax[0].set_ylabel(r"$d\ln C/d\ln r$")
    _ax[0].set_ylim(0.0, 3.4)
    _ax[0].legend(loc="lower left", fontsize=6.0, framealpha=0.9)

    _hfrac = np.asarray(HENON_RADII) / HENON_DIAMETER
    _hc = np.asarray(HENON_C)
    _hok = _hc > 0.0
    _hslopes = np.gradient(np.log(_hc[_hok]), np.log(np.asarray(HENON_RADII)[_hok]))
    _ax[1].axhline(HENON_D2, color=C_MEAN, linewidth=1.4, linestyle=":",
                   label=f"$D_2$ = {HENON_D2:.3f}")
    # 1.22 is the literature D_2 for the Henon map. Its box-counting D_0 is
    # about 1.26, and comparing a D_2 curve against that would be comparing
    # against the wrong quantity -- D_0 >= D_2 always.
    _ax[1].axhline(1.22, color=C_SAT, linewidth=1.3, linestyle="--",
                   label="literature $D_2 \\approx 1.22$")
    _ax[1].semilogx(_hfrac[_hok], _hslopes, marker="o", markersize=2.6,
                    color=C_TRUTH, linewidth=1.2)
    _ax[1].set_xlabel("r / diameter")
    _ax[1].set_ylabel(r"$d\ln C/d\ln r$")
    _ax[1].set_ylim(0.0, 2.4)
    _ax[1].legend(loc="lower left", fontsize=6.0, framealpha=0.9)

    _partial = np.cumsum(_spectrum)
    _index = np.arange(1, _spectrum.size + 1)
    _ax[2].axhline(0.0, color=C_SAT, linewidth=1.2, linestyle="--")
    _ax[2].plot(_index, _partial, marker="o", markersize=6, color=C_TRUTH,
                linewidth=1.6, label=r"$\sum_{i\leq j}\lambda_i$")
    for _i, _value in zip(_index, _partial):
        _ax[2].annotate(f"{_value:+.3f}", (_i, _value), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.5, color="#4b5563")
    _ax[2].set_xticks(_index)
    _ax[2].set_xlabel("j")
    _ax[2].set_ylabel("partial sum of exponents")
    _ax[2].legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    _j = int(np.searchsorted(-_partial, 0.0, side="left"))
    _difference = abs(L63_D2 - L63_DKY)

    mo.vstack([
        mo.md(
            r"""---
    ## 4 · Geometry against dynamics

    Now the comparison the chapter exists for. Nothing computed in Section 3
    touches the Lyapunov spectrum: $C(r)$ counts distances between points on a
    trajectory. Nothing in $D_{KY}$ touches the geometry: it is built from
    growth rates obtained by integrating the tangent equations and
    re-orthonormalising. Two calculations sharing no intermediate quantity."""
        ),
        _fig,
        mo.md(
            f"""
| | Lorenz 63 |
|---|---|
| $D_2$, from counting pairs | **{L63_D2:.4f}** |
| $D_{{KY}}$, from the Lyapunov spectrum | **{L63_DKY:.4f}** |
| difference | {_difference:.4f} ({100 * _difference / L63_DKY:.1f} %) |
| literature | ≈ 2.05 and ≈ 2.06 *[citation needed]* |

**They agree to {100 * _difference / L63_DKY:.1f} %.** That is the chapter's
one real result, and it earns two conclusions. Both estimators work — which
matters because Section 1 showed how easily either can be made to return
nonsense. And the **Kaplan–Yorke conjecture** holds here: the dimension implied
by the growth rates really is the dimension of the set. It is a conjecture, not
a theorem, and this is what testing it looks like.

The third panel shows where $D_{{KY}}$ comes from. The partial sums of
{', '.join(f'{v:+.4f}' for v in _spectrum)} run
{', '.join(f'{v:+.3f}' for v in _partial)}: still positive after {_j} exponents,
negative after {_j + 1}. So the attractor contains {_j} directions' worth of
volume and part of a {_j + 1}th — a fraction
$|\\sum_{{i\\le {_j}}}\\lambda_i| / |\\lambda_{{{_j + 1}}}|$ =
{_partial[_j - 1]:.4f}/{abs(_spectrum[_j]):.4f} = {_partial[_j - 1] / abs(_spectrum[_j]):.4f}
of it. Hence {_j} + {_partial[_j - 1] / abs(_spectrum[_j]):.4f} = {L63_DKY:.4f}.
The non-integer part is not a measurement error; it is the statement that the
attractor is more than a surface and less than a volume.

The Hénon map is the independent second case: $D_2$ = {HENON_D2:.3f} from a
two-dimensional map, against a literature $D_2$ near 1.22
*[citation needed: Grassberger and Procaccia (1983)]*. Note that the number
usually quoted for Hénon, 1.26, is its **box-counting** $D_0$; since
$D_0 \\ge D_1 \\ge D_2$ always, with equality only for a uniform measure, the
two are not interchangeable and a $D_2$ estimate should not be checked against
a $D_0$ value. The gap between them, about 0.04, is a statement that the
attractor's natural measure is not uniform along it.

The plateau is also flatter than Lorenz 63's, because the map has no trajectory
to correlate: successive iterates are already far apart, so the Theiler problem
barely exists.
"""
        ),
    ])
    return


# ===========================================================================
# 5. Entropy
# ===========================================================================
@app.cell(hide_code=True)
def s5_figure(
    C_PERT,
    C_SPREAD,
    C_TRUTH,
    DAYS_PER_MTU,
    L63_HKS,
    L63_SPECTRUM,
    LAMBDA1_L63,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _spectrum = np.asarray(L63_SPECTRUM)
    _positive = _spectrum[_spectrum > 0.0]
    _bits_per_mtu = L63_HKS / np.log(2.0)
    _bits_per_day = _bits_per_mtu / DAYS_PER_MTU

    # How long a given initial precision survives: an error of size d0 grows to
    # the attractor scale in (1/lambda) ln(D/d0).
    _digits = np.arange(1, 13)
    _horizon = np.log(10.0**_digits) / LAMBDA1_L63

    _fig, _ax = mpl_panels(
        2,
        titles=("Information destroyed per unit time",
                "What a decimal digit buys"),
        height=3.5,
    )
    _ax[0].bar(np.arange(_positive.size), _positive / np.log(2.0),
               color=C_SPREAD, width=0.5,
               label=f"total {_bits_per_mtu:.3f} bits/MTU")
    _ax[0].set_xticks(np.arange(_positive.size))
    _ax[0].set_xticklabels([f"$\\lambda_{i + 1}$" for i in range(_positive.size)])
    _ax[0].set_ylabel("bits per MTU")
    _ax[0].legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    _ax[1].plot(_digits, _horizon * DAYS_PER_MTU, marker="o", markersize=4.5,
                color=C_TRUTH, linewidth=1.5)
    _ax[1].set_xlabel("decimal digits of initial precision")
    _ax[1].set_ylabel("days until the error saturates")
    _ax[1].set_xticks(_digits[::2])
    finish_mpl(_fig)

    mo.vstack([
        mo.md(
            r"""---
    ## 5 · What the attractor costs you per day

    The same spectrum gives a third quantity, and by **Pesin's identity** it is
    just the positive part of it:

    $$h_{KS} = \sum_{\lambda_i > 0}\lambda_i .$$

    The Kolmogorov–Sinai entropy is a *rate of information loss*. Specify the
    initial state to some precision; the trajectory pulls apart, and after a
    while your specification no longer distinguishes states that are now far
    apart on the attractor. $h_{KS}$ is how fast that happens, in nats — or
    divide by $\ln 2$ for bits."""
        ),
        _fig,
        mo.md(
            f"""
For Lorenz 63, $h_{{KS}}$ = **{L63_HKS:.4f}** nats per MTU =
**{_bits_per_mtu:.3f} bits per MTU** = {_bits_per_day:.3f} bits per day at the
book's 5-days-per-MTU convention. The system has only one positive exponent, so
Pesin's sum has one term and $h_{{KS}} = \\lambda_1$ exactly — a coincidence of
low dimension. Lorenz 96 at $N = 40$ has thirteen, summing to 10.2 nats per time
unit, and chapter 11 shows that total growing *linearly* with the size of the
domain while $\\lambda_1$ stays put.

**The right-hand panel is the operational reading.** Each extra decimal digit of
initial precision buys $\\ln 10/\\lambda_1$ = {np.log(10)/LAMBDA1_L63:.3f} MTU =
**{np.log(10) / LAMBDA1_L63 * DAYS_PER_MTU:.1f} days** of forecast, and buys the
same amount again for the next digit, and the next. Twelve digits — which is
about all double precision has — buys {_horizon[-1] * DAYS_PER_MTU:.0f} days.
That is the whole predictability problem in one line: the return on better
observations is **logarithmic**, so it is not that better observing systems do
not help, it is that they help by a fixed increment per order of magnitude
rather than proportionally.

Chapter 20 measures exactly this constant on a cycling assimilation system and
gets 6.5 days per decade of analysis-error reduction. It is the same
$\\ln 10/\\lambda_1$, arrived at from the operational end.

And $h_{{KS}}$ says what it costs to *hold* a forecast rather than improve it.
The system destroys {_bits_per_day:.2f} bits per day about its own state, so an
observing system must supply information at that rate simply to stop the
analysis degrading. That is what data assimilation is for, and Part V is about
how it is done.
"""
        ),
    ])
    return


# ===========================================================================
# 6. Dimension from one observed variable
# ===========================================================================
@app.cell(hide_code=True)
def s6_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    EMBED_D2,
    EMBED_DIMENSIONS,
    EMBED_LAGS,
    L63_D2,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _ms = np.asarray(EMBED_DIMENSIONS)
    _lags = np.asarray(EMBED_LAGS)
    _table = np.asarray(EMBED_D2).reshape(_ms.size, _lags.size)

    _fig, _ax = mpl_panels(
        2,
        titles=(r"$D_2$ from $x(t)$ alone", "Saturation is the diagnostic"),
        height=3.5,
    )
    # Explicit palette colours: leaving plot() to matplotlib's default cycle
    # puts non-book colours in the figure.
    for _j, (_lag, _colour) in enumerate(zip(_lags, (C_TRUTH, C_MEAN, C_SPREAD))):
        _ax[0].plot(_ms, _table[:, _j], marker="o", markersize=4.5, linewidth=1.4,
                    color=_colour,
                    label=f"lag {_lag} steps = {_lag * 0.01:.2f} MTU")
    _ax[0].axhline(L63_D2, color=C_SAT, linewidth=1.4, linestyle="--",
                   label=f"full state: {L63_D2:.3f}")
    _ax[0].set_xlabel("embedding dimension m")
    _ax[0].set_ylabel(r"$D_2$")
    _ax[0].set_xticks(_ms)
    _ax[0].legend(loc="lower right", fontsize=6.0, framealpha=0.9)

    _mean = _table.mean(axis=1)
    _ax[1].axhline(0.0, color="#94a3b8", linewidth=0.9)
    _ax[1].plot(_ms[1:], np.diff(_mean), marker="o", markersize=4.5,
                color=C_PERT, linewidth=1.5)
    _ax[1].set_xlabel("embedding dimension m")
    _ax[1].set_ylabel(r"change in $D_2$ from $m-1$")
    _ax[1].set_xticks(_ms[1:])
    finish_mpl(_fig, suptitle="Delay embedding of the x component of Lorenz 63")

    _rows = "\n".join(
        f"| {m} | " + " | ".join(f"{_table[i, j]:.4f}" for j in range(_lags.size)) + " |"
        for i, m in enumerate(_ms)
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 6 · You do not get the state vector

    Everything so far assumed a cloud of points *in the state space*. An
    observing system does not deliver that. It delivers a few scalar time
    series, and often one.

    **Takens' theorem** says that is enough. For almost any smooth observable
    $s(t)$, the delay vectors

    $$\bigl(s(t),\ s(t+\ell),\ \ldots,\ s(t+(m-1)\ell)\bigr)$$

    reconstruct a set diffeomorphic to the attractor once $m > 2D$, so its
    dimension, its Lyapunov exponents and its topology all survive
    *[citation needed: Takens (1981)]*. Below is that claim tested: $D_2$
    measured from the $x$ component of Lorenz 63 alone, with $y$ and $z$ never
    used."""
        ),
        _fig,
        mo.md(
            f"""
| m | lag {_lags[0]} | lag {_lags[1]} | lag {_lags[2]} |
|---|---|---|---|
{_rows}

At $m = 2$ the estimate is capped near {_table[0].mean():.2f}: a
two-dimensional embedding cannot hold a {L63_D2:.2f}-dimensional set without the
reconstruction intersecting itself, so what is measured is the embedding, not the
attractor. By $m = 4$ it has reached {_table[2].mean():.3f} and it stops moving —
successive increments are {', '.join(f'{v:+.3f}' for v in np.diff(_mean))} — and
the converged value is within {abs(_mean[-1] - L63_D2):.3f} of the
{L63_D2:.3f} measured from the full three-dimensional state.

**That saturation is the whole method.** The criterion $m > 2D$ needs the $D$ you
are trying to measure, so it cannot be applied directly. What is done instead is
to raise $m$ until the answer stops changing, and the second panel is that test:
the increments fall from {np.diff(_mean)[0]:+.3f} to
{np.diff(_mean)[-1]:+.3f}. The lag matters much less — 0.10 to 0.30 MTU all give
the same converged answer, differing by
{_table[-1].max() - _table[-1].min():.3f} at $m = {_ms[-1]}$.

This is why any of this is more than a numerical exercise. Attractor dimensions
have been estimated for real geophysical records on exactly this basis, and the
estimates have been argued over for decades, largely because a real record is
short and noisy and the scaling window is correspondingly narrow — which is
Section 3's problem with less data and no known answer to check against
*[citation needed: on dimension estimates from climate records]*.
"""
        ),
    ])
    return


# ===========================================================================
# 7. Closing
# ===========================================================================
@app.cell(hide_code=True)
def closing(mo):
    mo.md(
        r"""
    ---
    ## Try this

    1. **Get 1.585 wrong three ways.** In Section 1, keep the Sierpiński
       triangle and find (a) a window that starves, (b) a window that is too
       coarse, (c) a window whose answer is wrong by less than 0.05 and
       therefore looks fine. Which of the three panels warns you in each case?
    2. **Fit the whole curve.** In Section 3, set the finer edge to $10^{-3.5}$
       and the width to 3.6 decades — the entire measured range. You get
       something close to 2, and it is wrong for reasons that no diagnostic in
       the figure flags. Explain why this is more dangerous than getting 0.19.
    3. **Find the noise floor's direction.** Put a narrow window at the fine
       end. Is the answer biased high or low? Now explain it from the fact that
       $C(r)$ moves in steps of $1/N_{\rm pairs}$.
    4. **Predict the Theiler effect before switching it.** Section 3's callout
       says the bias is high, and that the textbook warning says low. Before
       changing $w$, work out from the sampling interval which one applies to
       *this* curve, then check.
    5. **Cost out an observing system.** Section 5 gives
       $\ln10/\lambda_1 \approx 3.8$ days per decimal digit. If the analysis
       error were reduced by a factor of 100, how much forecast time is bought?
       Now do the same for Lorenz 96, where $\lambda_1 = 1.67$ per time unit and
       a time unit is five days. Why is the answer smaller?

    ## What you should have seen

    Box counting recovers three exactly known dimensions — $\ln2/\ln3$,
    $\ln4/\ln3$, $\ln3/\ln2$ — to better than 0.01, but only inside a window
    that has to be found rather than assumed, and only while there are more than
    about ten points per occupied box. On the Lorenz attractor, in three
    dimensions, 19,000 samples do not supply that for even one clean decade,
    which is why pair counting replaces box counting for anything real.

    The correlation dimension of Lorenz 63 is **2.058**, and the Kaplan–Yorke
    dimension from its Lyapunov spectrum is **2.062** — a 0.2 % difference
    between a calculation that used only distances between points and one that
    used only growth rates from the tangent equations. That agreement is the
    chapter's result. The same curve, fitted over the wrong window, returns
    **0.19** (saturated), **2.51** (noise floor) or a plausible and wrong
    **1.92** (everything at once), with nothing in the output to distinguish
    them.

    Pesin's identity makes $h_{KS} = 0.901$ nats per MTU = 1.30 bits per MTU:
    the rate at which the system destroys information about its own initial
    state, and therefore the rate at which observations must arrive to hold a
    forecast steady. Each decimal digit of extra initial precision buys
    $\ln10/\lambda_1 = 3.8$ days, and the next digit buys 3.8 days again — the
    logarithmic return that chapter 20 measures from the operational end.

    And all of it can be done from a single observed variable. $D_2$ from $x(t)$
    alone rises with the embedding dimension and saturates at 1.99 by $m = 4$,
    within 0.07 of the full-state answer, with $y$ and $z$ never used.

    ## Further reading

    - Grassberger, P. and Procaccia, I. (1983). Characterization of strange
      attractors. *Physical Review Letters*, **50**, 346–349.
    - Kaplan, J. L. and Yorke, J. A. (1979). Chaotic behavior of
      multidimensional difference equations *[citation needed: pages]*.
    - Takens, F. (1981). Detecting strange attractors in turbulence
      *[citation needed: pages]*.
    - Theiler, J. (1986). Spurious dimension from correlation algorithms applied
      to limited time-series data. *Physical Review A*, **34**, 2427–2432 — the
      trap in Section 3, and the window named after it.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 2 *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
