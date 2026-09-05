# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 22 -- Forecast verification and the practical horizon.

How far ahead a forecast is useful, and how much that answer depends on the
score, the threshold, the post-processing and what you verify against.

Part V of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The cycling experiments are precomputed by `scripts/generate_ch22_data.py`.

To edit:   marimo edit notebooks/ch22_verification.py
To export: make nb-one NB=ch22_verification
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 22: Forecast Verification")


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

    from chaoslib import integrate, plotting, systems, verification

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

    return (
        C_ANALYSIS, C_BG, C_CONTEXT, C_FIXED, C_MEAN, C_OBS, C_PERT, C_SAT,
        C_SPREAD, C_START, C_TRUTH, MPL_SEQUENTIAL, finish_mpl, integrate, mo,
        mpl_grid, mpl_panels, np, plotting, plt, systems, verification,
    )


@app.cell
def chapter_data():
    # Precomputed by scripts/generate_ch22_data.py (~2 min): a cycling
    # LETKF archive keeping truth, analysis and observations separately,
    # 367 forecasts scored five ways at 51 leads, and a seven-decade
    # initial-error sweep in two systems. Knob-free; the slider slices.

    LEADS = (np.float64(0.0), np.float64(0.1), np.float64(0.2), np.float64(0.3), np.float64(0.4), np.float64(0.5), np.float64(0.6), np.float64(0.7), np.float64(0.8), np.float64(0.9), np.float64(1.0), np.float64(1.1), np.float64(1.2), np.float64(1.3), np.float64(1.4), np.float64(1.5), np.float64(1.6), np.float64(1.7), np.float64(1.8), np.float64(1.9), np.float64(2.0), np.float64(2.1), np.float64(2.2), np.float64(2.3), np.float64(2.4), np.float64(2.5), np.float64(2.6), np.float64(2.7), np.float64(2.8), np.float64(2.9), np.float64(3.0), np.float64(3.1), np.float64(3.2), np.float64(3.3), np.float64(3.4), np.float64(3.5), np.float64(3.6), np.float64(3.7), np.float64(3.8), np.float64(3.9), np.float64(4.0), np.float64(4.1), np.float64(4.2), np.float64(4.3), np.float64(4.4), np.float64(4.5), np.float64(4.6), np.float64(4.7), np.float64(4.8), np.float64(4.9), np.float64(5.0))
    CLIMATE_VARIANCE = 13.264559
    CLIMATE_SIGMA = 3.642054
    N_FORECAST_CASES = 367
    N_ENSEMBLE_CASES = 120
    EVENT_THRESHOLD = 4.606035
    SCORE_ACC = (
        0.997881, 0.996894, 0.995389, 0.993208, 0.990132, 0.985799, 0.979558, 0.970830, 0.958802, 0.943089,
        0.923495, 0.899921, 0.871215, 0.838373, 0.802542, 0.764950, 0.728391, 0.691934, 0.655280, 0.615601,
        0.573930, 0.533667, 0.499328, 0.466626, 0.434415, 0.398746, 0.361512, 0.332094, 0.303913, 0.276682,
        0.258949, 0.250016, 0.238030, 0.215845, 0.193382, 0.175508, 0.157897, 0.142803, 0.131649, 0.118781,
        0.108988, 0.101234, 0.093597, 0.087840, 0.086231, 0.085717, 0.085747, 0.084432, 0.076396, 0.069958,
        0.062413,
    )
    SCORE_RMSE = (
        0.235541, 0.285205, 0.347619, 0.421938, 0.508562, 0.610070, 0.731863, 0.874319, 1.039422, 1.221642,
        1.415709, 1.618234, 1.834857, 2.054829, 2.270643, 2.476159, 2.660291, 2.832824, 2.998132, 3.167924,
        3.335888, 3.492146, 3.620991, 3.740903, 3.856290, 3.976165, 4.093695, 4.185808, 4.274204, 4.354164,
        4.404428, 4.434849, 4.474310, 4.539953, 4.602339, 4.649382, 4.697366, 4.743961, 4.781691, 4.817667,
        4.844637, 4.868890, 4.895785, 4.915466, 4.916823, 4.914548, 4.913065, 4.912631, 4.931014, 4.945780,
        4.965892,
    )
    SCORE_MSESS = (
        0.995766, 0.993796, 0.990788, 0.986429, 0.980277, 0.971612, 0.959141, 0.941674, 0.917552, 0.886068,
        0.846898, 0.799886, 0.742627, 0.677034, 0.605559, 0.530879, 0.458493, 0.386147, 0.312625, 0.232743,
        0.149518, 0.068136, -0.001874, -0.069035, -0.135760, -0.207538, -0.280093, -0.338715, -0.396244, -0.448749,
        -0.482353, -0.503691, -0.531164, -0.576624, -0.620158, -0.652818, -0.686547, -0.719840, -0.746947, -0.773459,
        -0.794149, -0.812771, -0.833499, -0.848810, -0.849837, -0.848080, -0.846772, -0.845974, -0.859267, -0.870127,
        -0.884805,
    )
    SCORE_CRPS = (
        0.124628, 0.148686, 0.177623, 0.210830, 0.252408, 0.297764, 0.349617, 0.409687, 0.470163, 0.536871,
        0.607551, 0.680922, 0.759244, 0.845615, 0.936501, 1.023764, 1.100529, 1.174436, 1.256684, 1.322788,
        1.385227, 1.445354, 1.499564, 1.553922, 1.597805, 1.632375, 1.667319, 1.703708, 1.737959, 1.766179,
        1.788929, 1.804009, 1.826032, 1.849385, 1.875023, 1.896171, 1.918452, 1.925371, 1.935356, 1.949481,
        1.962521, 1.973262, 1.985127, 1.987307, 1.998694, 2.009569, 2.013478, 2.017498, 2.016949, 2.026927,
        2.040457,
    )
    SCORE_BRIERSS = (
        0.963462, 0.991758, 0.884031, 0.947985, 0.918407, 0.831719, 0.834301, 0.825594, 0.668421, 0.787844,
        0.752612, 0.626984, 0.685151, 0.605340, 0.367168, 0.414723, 0.337996, 0.219436, 0.318056, 0.389602,
        0.349882, 0.338671, 0.318519, 0.312773, 0.246534, 0.215684, 0.200619, 0.105743, 0.149834, 0.128113,
        0.093956, 0.162609, 0.192908, 0.112319, 0.099630, 0.060436, 0.071781, 0.067857, -0.010921, 0.061264,
        0.125927, 0.023495, -0.001196, 0.063073, 0.052794, 0.057256, 0.009462, -0.032178, -0.048264, -0.074188,
        -0.072089,
    )

    HORIZON_LABELS = ('ACC < 0.6', 'ACC < 0.5', 'MSE skill < 0', 'RMSE > 0.7 sigma', 'RMSE > 1.0 sigma', 'Brier skill < 0')
    HORIZON_VALUES = (
        1.93744, 2.19804, 2.19732, 1.53980, 2.21757, 3.78614,
    )
    HORIZON_SPREAD = 2.458854
    ACC_THRESHOLDS = (np.float64(0.3), np.float64(0.35), np.float64(0.4), np.float64(0.45), np.float64(0.5), np.float64(0.55), np.float64(0.6), np.float64(0.65), np.float64(0.7), np.float64(0.75), np.float64(0.8), np.float64(0.85), np.float64(0.9), np.float64(0.95))
    ACC_THRESHOLD_HORIZON = (
        2.81437, 2.63913, 2.49648, 2.35162, 2.19804, 2.05943, 1.93744,
        1.81331, 1.67788, 1.54089, 1.40676, 1.26460, 1.09967, 0.85602,
    )

    ANALYSIS_FRACTION = (
        -0.639428, -0.420939, -0.265073, -0.172567, -0.116540, -0.078573, -0.060244, -0.034363, -0.024087, -0.014991,
        -0.013218, -0.007924, -0.004109, -0.003853, -0.002670, -0.001481, -0.002538, -0.004472, -0.002318, -0.003437,
        -0.002215, -0.001413, -0.002478, -0.000693, -0.000941, -0.001131, 0.000771, -0.000179, -0.000342, 0.001207,
        0.001069, -0.000954, 0.000265, 0.000015, -0.000780, -0.000926, -0.001349, -0.001844, -0.001263, -0.001724,
        -0.002845, -0.002039, -0.000666, -0.001250, -0.001259, -0.000511, -0.000396, -0.000743, -0.001251, -0.001403,
    )
    ASSIM_MSE = 0.924880
    ASSIM_CORRECTED = -0.075120
    ANALYSIS_TRUE_MSE = 0.055480
    OBS_VARIANCE = 1.000000
    MSE_TRUTH = (
        0.055480, 0.081342, 0.120839, 0.178032, 0.258636, 0.372186, 0.535624, 0.764434, 1.080397, 1.492409,
        2.004231, 2.618682, 3.366700, 4.222321, 5.155818, 6.131363, 7.077146, 8.024894, 8.988794, 10.035741,
        11.128148, 12.195082, 13.111578, 13.994356, 14.870970, 15.809890, 16.758339, 17.520985, 18.268817, 18.958742,
        19.398984, 19.667887, 20.019451, 20.611177, 21.181520, 21.616754, 22.065246, 22.505164, 22.864571, 23.209916,
        23.470506, 23.706091, 23.968712, 24.161809, 24.175153, 24.152783, 24.138212, 24.133945, 24.314899, 24.460739,
        24.660088,
    )
    ACCV_TRUTH = (
        0.997881, 0.996894, 0.995389, 0.993208, 0.990132, 0.985799, 0.979558, 0.970830, 0.958802, 0.943089,
        0.923495, 0.899921, 0.871215, 0.838373, 0.802542, 0.764950, 0.728391, 0.691934, 0.655280, 0.615601,
        0.573930, 0.533667, 0.499328, 0.466626, 0.434415, 0.398746, 0.361512, 0.332094, 0.303913, 0.276682,
        0.258949, 0.250016, 0.238030, 0.215845, 0.193382, 0.175508, 0.157897, 0.142803, 0.131649, 0.118781,
        0.108988, 0.101234, 0.093597, 0.087840, 0.086231, 0.085717, 0.085747, 0.084432, 0.076396, 0.069958,
        0.062413,
    )
    MSE_INDEP = (
        1.053249, 1.078653, 1.101685, 1.175902, 1.257699, 1.372759, 1.528407, 1.747055, 2.073551, 2.454615,
        3.064379, 3.625337, 4.383741, 5.243307, 6.161733, 7.139784, 8.070856, 8.983932, 9.993321, 11.003657,
        12.088728, 13.264175, 14.179421, 15.086111, 15.844824, 16.781972, 17.780627, 18.390138, 19.334984, 19.963593,
        20.357722, 20.784902, 21.048100, 21.696941, 22.149782, 22.650825, 22.967384, 23.452495, 23.902415, 24.130886,
        24.343572, 24.724180, 24.848526, 25.196660, 25.160332, 25.159611, 25.231751, 25.117332, 25.246624, 25.532482,
        25.693948,
    )
    ACCV_INDEP = (
        0.961873, 0.960912, 0.960359, 0.957245, 0.954421, 0.950305, 0.944921, 0.936676, 0.924392, 0.910038,
        0.888545, 0.867135, 0.839379, 0.806991, 0.773158, 0.738504, 0.701187, 0.667230, 0.630776, 0.593216,
        0.552987, 0.512382, 0.479109, 0.450083, 0.416656, 0.385129, 0.348134, 0.322169, 0.291059, 0.268031,
        0.250851, 0.238712, 0.226565, 0.205123, 0.186465, 0.169634, 0.154948, 0.140173, 0.124199, 0.116891,
        0.107505, 0.096693, 0.090615, 0.082634, 0.084481, 0.080029, 0.078781, 0.086036, 0.073165, 0.063534,
        0.061907,
    )
    MSE_ANALYSIS = (
        0.000000, 0.029330, 0.069973, 0.130840, 0.214004, 0.328811, 0.493538, 0.718382, 1.043272, 1.456461,
        1.974185, 2.584069, 3.340022, 4.204974, 5.135952, 6.114993, 7.066662, 8.004530, 8.948599, 10.012480,
        11.089906, 12.168069, 13.093055, 13.959684, 14.860667, 15.795005, 16.739380, 17.534502, 18.265555, 18.952266,
        19.422408, 19.688920, 20.000361, 20.616640, 21.181840, 21.599899, 22.044805, 22.474812, 22.822412, 23.180604,
        23.430051, 23.638657, 23.919846, 24.145717, 24.144943, 24.122376, 24.125878, 24.124389, 24.296825, 24.430138,
        24.625491,
    )
    ACCV_ANALYSIS = (
        1.000000, 0.998878, 0.997325, 0.995000, 0.991822, 0.987433, 0.981133, 0.972542, 0.960150, 0.944372,
        0.924517, 0.901071, 0.872026, 0.838766, 0.802952, 0.765189, 0.728329, 0.692158, 0.656252, 0.615843,
        0.574639, 0.533953, 0.499203, 0.467006, 0.433889, 0.398292, 0.361098, 0.330472, 0.302839, 0.275619,
        0.256791, 0.247908, 0.237391, 0.214318, 0.191975, 0.174663, 0.157240, 0.142465, 0.131703, 0.118435,
        0.109002, 0.102209, 0.093969, 0.086891, 0.085755, 0.085348, 0.084612, 0.083133, 0.075481, 0.069424,
        0.061961,
    )
    ACC_CORRECTED = (
        0.997472, 0.996475, 0.995902, 0.992672, 0.989744, 0.985476, 0.979892, 0.971341, 0.958603, 0.943718,
        0.921430, 0.899228, 0.870444, 0.836857, 0.801772, 0.765836, 0.727137, 0.691924, 0.654121, 0.615170,
        0.573453, 0.531345, 0.496841, 0.466740, 0.432077, 0.399383, 0.361019, 0.334092, 0.301831, 0.277950,
        0.260135, 0.247546, 0.234950, 0.212715, 0.193366, 0.175913, 0.160682, 0.145361, 0.128796, 0.121217,
        0.111484, 0.100272, 0.093968, 0.085693, 0.087607, 0.082991, 0.081696, 0.089220, 0.075873, 0.065885,
        0.064198,
    )

    MEAN_OFFSET = 0.5
    BUDGET_KEYS = ('PERFECT', 'BIASED', 'OFFSET')
    BIASED_FORCING = 8.4
    UNDAMPED_HORIZON = 2.217329
    DAMPED_HORIZON = float("nan")
    BUDGET_PERFECT_BIAS = (
        0.000009, 0.000003, 0.000001, 0.000003, 0.000010, 0.000011, 0.000004, 0.000015, 0.000090, 0.000188,
        0.000200, 0.000120, 0.000059, 0.000033, 0.000007, 0.000022, 0.000253, 0.000725, 0.000973, 0.001067,
        0.001611, 0.001779, 0.001591, 0.001142, 0.000530, 0.000426, 0.000988, 0.001381, 0.001237, 0.001956,
        0.003193, 0.002623, 0.001942, 0.001930, 0.002735, 0.004724, 0.007008, 0.007190, 0.005881, 0.006261,
        0.006443, 0.005295, 0.002999, 0.001384, 0.001280, 0.001542, 0.001663, 0.002494, 0.003482, 0.004473,
        0.005225,
    )
    BUDGET_PERFECT_AMPLITUDE = (
        0.000035, 0.000033, 0.000027, 0.000016, 0.000007, 0.000005, 0.000008, 0.000002, 0.000002, 0.000006,
        0.000001, 0.000004, 0.000019, 0.000027, 0.000042, 0.000090, 0.000181, 0.000240, 0.000158, 0.000072,
        0.000068, 0.000018, 0.000002, 0.000062, 0.000227, 0.000234, 0.000072, 0.000045, 0.000094, 0.000033,
        0.000004, 0.000091, 0.000252, 0.000277, 0.000160, 0.000033, 0.000006, 0.000076, 0.000347, 0.000406,
        0.000513, 0.000871, 0.001609, 0.002178, 0.001757, 0.001261, 0.000995, 0.000588, 0.000369, 0.000268,
        0.000266,
    )
    BUDGET_PERFECT_PHASE = (
        0.055436, 0.081306, 0.120811, 0.178013, 0.258619, 0.372169, 0.535611, 0.764417, 1.080305, 1.492215,
        2.004030, 2.618558, 3.366622, 4.222261, 5.155769, 6.131251, 7.076712, 8.023930, 8.987664, 10.034602,
        11.126470, 12.193284, 13.109985, 13.993152, 14.870213, 15.809230, 16.757279, 17.519560, 18.267485, 18.956752,
        19.395787, 19.665173, 20.017256, 20.608971, 21.178625, 21.611996, 22.058233, 22.497898, 22.858343, 23.203249,
        23.463550, 23.699925, 23.964104, 24.158247, 24.172115, 24.149981, 24.135554, 24.130863, 24.311048, 24.455999,
        24.654597,
    )
    RATIO_PERFECT_DAMPED = (
        0.004234, 0.006203, 0.009205, 0.013543, 0.019646, 0.028213, 0.040488, 0.057521, 0.080744, 0.110650,
        0.147262, 0.190303, 0.241222, 0.297454, 0.356319, 0.415279, 0.469911, 0.521707, 0.571063, 0.621457,
        0.671032, 0.715652, 0.751082, 0.782605, 0.811627, 0.841385, 0.869747, 0.890238, 0.908114, 0.923742,
        0.933106, 0.937734, 0.943746, 0.953889, 0.963048, 0.969563, 0.975383, 0.979913, 0.982968, 0.986152,
        0.988335, 0.989917, 0.991404, 0.992474, 0.992769, 0.992897, 0.992956, 0.993188, 0.994418, 0.995286,
        0.996229,
    )
    RATIO_PERFECT_UNDAMPED = (
        0.004183, 0.006132, 0.009110, 0.013422, 0.019498, 0.028059, 0.040380, 0.057630, 0.081450, 0.112511,
        0.151097, 0.197419, 0.253812, 0.318316, 0.388691, 0.462237, 0.533538, 0.604988, 0.677655, 0.756583,
        0.838938, 0.919373, 0.988467, 1.055019, 1.121106, 1.191890, 1.263392, 1.320887, 1.377265, 1.429278,
        1.462467, 1.482740, 1.509244, 1.553853, 1.596851, 1.629662, 1.663474, 1.696639, 1.723734, 1.749769,
        1.769415, 1.787175, 1.806974, 1.821531, 1.822537, 1.820851, 1.819752, 1.819431, 1.833073, 1.844067,
        1.859096,
    )
    BUDGET_BIASED_BIAS = (
        0.000009, 0.001221, 0.003810, 0.005689, 0.006198, 0.005845, 0.005077, 0.003734, 0.002391, 0.001728,
        0.001597, 0.001941, 0.002492, 0.003063, 0.004271, 0.004927, 0.004938, 0.005520, 0.006379, 0.007236,
        0.008884, 0.010852, 0.011228, 0.009808, 0.008257, 0.007909, 0.007991, 0.008527, 0.009367, 0.009897,
        0.008641, 0.006755, 0.006673, 0.006906, 0.007846, 0.010771, 0.010888, 0.012448, 0.015676, 0.017250,
        0.019538, 0.021219, 0.020200, 0.014899, 0.011193, 0.010066, 0.009259, 0.007739, 0.007265, 0.008308,
        0.009147,
    )
    BUDGET_BIASED_AMPLITUDE = (
        0.000035, 0.000004, 0.000108, 0.000898, 0.002767, 0.005511, 0.008688, 0.012282, 0.015672, 0.017727,
        0.018532, 0.018355, 0.018074, 0.018058, 0.017522, 0.018083, 0.019319, 0.019762, 0.020004, 0.020400,
        0.020239, 0.020246, 0.021800, 0.024568, 0.026879, 0.027526, 0.027601, 0.027230, 0.026768, 0.026864,
        0.028639, 0.030641, 0.030066, 0.029310, 0.027921, 0.025428, 0.026323, 0.025803, 0.024626, 0.025254,
        0.025523, 0.026476, 0.029158, 0.034547, 0.037616, 0.037478, 0.036959, 0.037255, 0.035967, 0.033297,
        0.031584,
    )
    BUDGET_BIASED_PHASE = (
        0.055436, 0.081503, 0.122967, 0.187639, 0.285959, 0.431544, 0.643415, 0.938715, 1.342735, 1.862951,
        2.488127, 3.207972, 4.066321, 5.025895, 6.133477, 7.308804, 8.410385, 9.434791, 10.474806, 11.617146,
        12.734199, 13.751493, 14.668085, 15.535651, 16.503876, 17.487821, 18.347714, 19.117588, 19.883053, 20.558280,
        21.292414, 21.969412, 22.519368, 23.022342, 23.386197, 23.632457, 23.957726, 24.299979, 24.505628, 24.668281,
        24.852949, 25.082758, 25.366093, 25.667912, 25.819370, 26.088129, 26.258598, 26.177701, 26.123662, 26.062947,
        26.064540,
    )
    RATIO_BIASED_DAMPED = (
        0.004234, 0.006212, 0.009328, 0.014141, 0.021387, 0.032006, 0.047295, 0.068315, 0.096630, 0.132529,
        0.174871, 0.222456, 0.277418, 0.336505, 0.401663, 0.466736, 0.523966, 0.574335, 0.622607, 0.672227,
        0.717527, 0.755841, 0.787328, 0.814415, 0.842727, 0.869767, 0.891486, 0.909479, 0.925806, 0.938578,
        0.950673, 0.960712, 0.968489, 0.974847, 0.979132, 0.982034, 0.984860, 0.987839, 0.989593, 0.990697,
        0.991990, 0.993358, 0.994702, 0.995745, 0.996192, 0.997310, 0.997936, 0.997616, 0.997473, 0.997412,
        0.997505,
    )
    RATIO_BIASED_UNDAMPED = (
        0.004183, 0.006237, 0.009566, 0.014642, 0.022234, 0.033390, 0.049544, 0.071976, 0.102589, 0.141912,
        0.189095, 0.243375, 0.308106, 0.380489, 0.464039, 0.552737, 0.635878, 0.713184, 0.791673, 0.877887,
        0.962212, 1.039054, 1.108300, 1.173807, 1.246857, 1.321058, 1.385897, 1.443949, 1.501685, 1.552637,
        1.608021, 1.659068, 1.700479, 1.738358, 1.765755, 1.784353, 1.808951, 1.834832, 1.850490, 1.862918,
        1.877033, 1.894556, 1.916042, 1.938802, 1.950173, 1.970339, 1.983090, 1.976899, 1.972692, 1.967992,
        1.968047,
    )
    BUDGET_OFFSET_BIAS = (
        0.247076, 0.248327, 0.248958, 0.248294, 0.246907, 0.246632, 0.247883, 0.246168, 0.240598, 0.236473,
        0.236067, 0.239179, 0.242349, 0.244252, 0.247326, 0.254687, 0.266174, 0.277653, 0.282164, 0.283725,
        0.291743, 0.293959, 0.291482, 0.284933, 0.273550, 0.271058, 0.282421, 0.288546, 0.286409, 0.296187,
        0.309703, 0.303841, 0.296014, 0.295859, 0.305031, 0.323457, 0.340719, 0.341987, 0.332571, 0.335386,
        0.336713, 0.328061, 0.307757, 0.288582, 0.287061, 0.290806, 0.292448, 0.302438, 0.312492, 0.321351,
        0.327509,
    )
    BUDGET_OFFSET_AMPLITUDE = (
        0.000035, 0.000033, 0.000027, 0.000016, 0.000007, 0.000005, 0.000008, 0.000002, 0.000002, 0.000006,
        0.000001, 0.000004, 0.000019, 0.000027, 0.000042, 0.000090, 0.000181, 0.000240, 0.000158, 0.000072,
        0.000068, 0.000018, 0.000002, 0.000062, 0.000227, 0.000234, 0.000072, 0.000045, 0.000094, 0.000033,
        0.000004, 0.000091, 0.000252, 0.000277, 0.000160, 0.000033, 0.000006, 0.000076, 0.000347, 0.000406,
        0.000513, 0.000871, 0.001609, 0.002178, 0.001757, 0.001261, 0.000995, 0.000588, 0.000369, 0.000268,
        0.000266,
    )
    BUDGET_OFFSET_PHASE = (
        0.055436, 0.081306, 0.120811, 0.178013, 0.258619, 0.372169, 0.535611, 0.764417, 1.080305, 1.492215,
        2.004030, 2.618558, 3.366622, 4.222261, 5.155769, 6.131251, 7.076712, 8.023930, 8.987664, 10.034602,
        11.126470, 12.193284, 13.109985, 13.993152, 14.870213, 15.809230, 16.757279, 17.519560, 18.267485, 18.956752,
        19.395787, 19.665173, 20.017256, 20.608971, 21.178625, 21.611996, 22.058233, 22.497898, 22.858343, 23.203249,
        23.463550, 23.699925, 23.964104, 24.158247, 24.172115, 24.149981, 24.135554, 24.130863, 24.311048, 24.455999,
        24.654597,
    )
    RATIO_OFFSET_DAMPED = (
        0.004234, 0.006203, 0.009205, 0.013543, 0.019646, 0.028213, 0.040488, 0.057521, 0.080744, 0.110650,
        0.147262, 0.190303, 0.241222, 0.297454, 0.356319, 0.415279, 0.469911, 0.521707, 0.571063, 0.621457,
        0.671032, 0.715652, 0.751082, 0.782605, 0.811627, 0.841385, 0.869747, 0.890238, 0.908114, 0.923742,
        0.933106, 0.937734, 0.943746, 0.953889, 0.963048, 0.969563, 0.975383, 0.979913, 0.982968, 0.986152,
        0.988335, 0.989917, 0.991404, 0.992474, 0.992769, 0.992897, 0.992956, 0.993188, 0.994418, 0.995286,
        0.996229,
    )
    RATIO_OFFSET_UNDAMPED = (
        0.022809, 0.024853, 0.027879, 0.032140, 0.038112, 0.046651, 0.059067, 0.076187, 0.099582, 0.130324,
        0.168878, 0.215442, 0.272078, 0.336727, 0.407336, 0.481435, 0.553585, 0.625865, 0.698854, 0.777892,
        0.860811, 0.941400, 1.010321, 1.076413, 1.141688, 1.212292, 1.284609, 1.342536, 1.398764, 1.451460,
        1.485575, 1.505448, 1.531413, 1.576012, 1.619640, 1.653691, 1.688632, 1.721879, 1.748363, 1.774582,
        1.794313, 1.811508, 1.829949, 1.843183, 1.844082, 1.842658, 1.841674, 1.842043, 1.856368, 1.867956,
        1.883393,
    )

    AMPLITUDES = (1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1)
    LAMBDA1_L96 = 1.67
    ONE_SCALE_CASES = 24
    TWO_SCALE_CASES = 24
    HORIZON_ONE_SCALE = (
        13.01776, 11.58769, 10.30636, 9.07927, 7.57137, 5.88951, 4.76505, 3.58941,
    )
    HORIZON_TWO_SCALE = (
        3.50693, 2.95570, 2.71278, 2.64356, 2.57277, 2.56684, 2.36607, 2.40036,
    )

    return (
        ACCV_ANALYSIS, ACCV_INDEP, ACCV_TRUTH, ACC_CORRECTED,
        ACC_THRESHOLDS, ACC_THRESHOLD_HORIZON, AMPLITUDES,
        ANALYSIS_FRACTION, ANALYSIS_TRUE_MSE, ASSIM_CORRECTED, ASSIM_MSE,
        BIASED_FORCING, BUDGET_BIASED_AMPLITUDE, BUDGET_BIASED_BIAS,
        BUDGET_BIASED_PHASE, BUDGET_KEYS, BUDGET_OFFSET_AMPLITUDE,
        BUDGET_OFFSET_BIAS, BUDGET_OFFSET_PHASE, BUDGET_PERFECT_AMPLITUDE,
        BUDGET_PERFECT_BIAS, BUDGET_PERFECT_PHASE, CLIMATE_SIGMA,
        CLIMATE_VARIANCE, DAMPED_HORIZON, EVENT_THRESHOLD, HORIZON_LABELS,
        HORIZON_ONE_SCALE, HORIZON_SPREAD, HORIZON_TWO_SCALE,
        HORIZON_VALUES, LAMBDA1_L96, LEADS, MEAN_OFFSET, MSE_ANALYSIS,
        MSE_INDEP, MSE_TRUTH, N_ENSEMBLE_CASES, N_FORECAST_CASES,
        OBS_VARIANCE, ONE_SCALE_CASES, RATIO_BIASED_DAMPED,
        RATIO_BIASED_UNDAMPED, RATIO_OFFSET_DAMPED, RATIO_OFFSET_UNDAMPED,
        RATIO_PERFECT_DAMPED, RATIO_PERFECT_UNDAMPED, SCORE_ACC,
        SCORE_BRIERSS, SCORE_CRPS, SCORE_MSESS, SCORE_RMSE,
        TWO_SCALE_CASES, UNDAMPED_HORIZON,
    )

# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 22 · Forecast Verification and the Practical Horizon

    **Part V — The machinery of prediction.**

    **The forecasting question.** "Weather forecasts are useful about a week ahead."
    Every chapter of this book has circled that sentence. This one asks what would have
    to be true for it to *mean* anything.

    Three difficulties, and none of them is about the atmosphere.

    **Useful by what measure?** Anomaly correlation, mean-square error, CRPS and a
    Brier score are different questions, and the same forecasts answer them
    differently. Section 2 measures how differently.

    **Useful past what threshold?** The conventional answer is anomaly correlation 0.6.
    Section 1 shows where that number comes from — it is not a fact about predictability
    — and section 3 shows what it costs to choose a different one.

    **Verified against what?** Every score in this book so far has compared a forecast
    against a *truth*. **No forecast centre has ever had one.** They have observations,
    which are the truth plus an error, and often the very observations that were
    assimilated to make the analysis the forecast started from. Chapters 13, 17, 18, 19
    and 21 each deferred this problem here. Section 5 takes it up.

    Then the question the book opened with: the horizon has advanced by about a day per
    decade for forty years. How much is left?

    ---

    **What you need before this chapter.** **Chapter 12** for the scale-dependent error
    cascade, whose promise — that the practical and intrinsic limits are different
    quantities — section 6 tests. **Chapter 17** for CRPS and the Brier score.
    **Chapter 19** for the LETKF that supplies the analyses.
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
    ## 1 · Where 0.6 comes from

    Anomaly correlation measures whether a forecast got the *pattern* of departures
    from climatology right:

    $$
    \mathrm{ACC} = \frac{\sum (f-c)(t-c)}
                        {\sqrt{\sum (f-c)^2 \sum (t-c)^2}} .
    $$

    It is deliberately blind to a uniform amplitude error — scale every forecast anomaly
    by three and it does not move — which is exactly why it must be read alongside a
    mean-square error rather than instead of one.

    The conventional "useful" threshold of 0.6 is not a convention about the atmosphere.
    It follows from arithmetic. Take a forecast that is **unbiased** and whose anomaly
    variance **matches the truth's** — an undamped forecast, which is what a raw model
    run gives you. Then

    $$
    \mathrm{MSE} = \sigma_f^2 + \sigma_t^2 - 2r\sigma_f\sigma_t = 2\sigma^2(1-r),
    $$

    against $\sigma^2$ for someone who just forecasts the climatology. So its skill
    score is exactly $2r - 1$: it **ties with climatology at $r = 1/2$**, and 0.6 is that
    number with a margin. `chaoslib` asserts the identity as a test.

    Now damp it. The least-squares rescaling of the forecast anomalies gives
    $\mathrm{MSE} = \sigma_t^2(1-r^2)$, which beats climatology for **any** non-zero
    correlation. The threshold for usefulness is not a property of the atmosphere at
    all; it is a property of a decision not to post-process.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    C_ANALYSIS, C_PERT, C_SAT, C_TRUTH, finish_mpl, mpl_panels, np,
    verification,
):
    _r = np.linspace(0.0, 1.0, 201)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Skill against climatology, by correlation",
                "Anomaly correlation ignores amplitude"),
        figsize=(10.0, 4.0),
    )
    _ax0.plot(_r, 2.0 * _r - 1.0, "-", color=C_TRUTH, linewidth=2.2,
              label="undamped: $2r-1$")
    _ax0.plot(_r, _r**2, "-", color=C_ANALYSIS, linewidth=2.2,
              label="optimally damped: $r^2$")
    _ax0.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.3)
    _ax0.axvline(0.5, color=C_PERT, linestyle=":", linewidth=1.5)
    _ax0.axvline(0.6, color=C_SAT, linestyle=":", linewidth=1.5)
    _ax0.text(0.505, -0.9, "$r=1/2$\nundamped ties", fontsize=7.5, color=C_PERT)
    _ax0.text(0.615, -0.55, "0.6\nconvention", fontsize=7.5, color=C_SAT)
    _ax0.fill_between(_r, -1.05, 0.0, where=_r < 0.5, color=C_PERT, alpha=0.08,
                      linewidth=0)
    _ax0.set_xlabel("anomaly correlation $r$")
    _ax0.set_ylabel("MSE skill score against climatology")
    _ax0.set_ylim(-1.05, 1.02)
    _ax0.legend(fontsize=8, framealpha=0.9, loc="upper left")

    # A concrete illustration, computed live: three forecasts differing only in
    # amplitude, all with identical ACC and wildly different MSE.
    _rng = np.random.default_rng(3)
    _truth = _rng.normal(0.0, 1.0, 4000)
    _base = 0.6 * _truth + _rng.normal(0.0, 0.5, 4000)
    _scales = np.linspace(0.3, 2.5, 40)
    _acc = [verification.anomaly_correlation(_s * _base, _truth,
                                             np.zeros_like(_truth))
            for _s in _scales]
    _mse = [float(np.mean((_s * _base - _truth) ** 2)) / float(np.var(_truth))
            for _s in _scales]
    _ax1.plot(_scales, _acc, "-", color=C_TRUTH, linewidth=2.2,
              label="anomaly correlation")
    _ax1.plot(_scales, _mse, "-", color=C_PERT, linewidth=2.2,
              label="MSE / climatological variance")
    _ax1.axhline(1.0, color="#8b8698", linestyle="--", linewidth=1.2)
    _ax1.axvline(_scales[int(np.argmin(_mse))], color=C_ANALYSIS,
                 linestyle=":", linewidth=1.5)
    _ax1.text(_scales[int(np.argmin(_mse))] + 0.05, 2.4, "optimal\ndamping",
              fontsize=7.5, color=C_ANALYSIS)
    _ax1.set_xlabel("amplitude applied to the forecast anomalies")
    _ax1.set_ylabel("score")
    _ax1.set_ylim(0.0, 3.0)
    _ax1.legend(fontsize=8, framealpha=0.9)
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s1_note(mo, verification):
    mo.md(
        rf"""
    The right-hand panel is the same forecast scaled up and down. Its anomaly
    correlation is a **flat line** — the score cannot see amplitude at all — while its
    mean-square error has a clear minimum, at an amplitude well below 1. The forecast
    that scores best is the *damped* one.

    This is not a trick for flattering a score. A least-squares-optimal point forecast
    of a partly unpredictable quantity **should** be closer to climatology than the
    truth is; pushing full-amplitude anomalies is overconfidence in exactly the sense
    chapter 17 measures for ensembles. What damping loses is the realism of the field —
    too little variance, too few extremes — which is why operational centres issue
    undamped deterministic forecasts and take the score penalty knowingly.

    So the two thresholds are
    {verification.acc_threshold_for_climatological_skill():.1f} undamped and
    {verification.acc_threshold_for_climatological_skill(damped=True):.1f} damped, and
    the familiar 0.6 belongs to the first.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(CLIMATE_SIGMA, N_ENSEMBLE_CASES, N_FORECAST_CASES, mo):
    mo.md(
        rf"""
    ## 2 · One set of forecasts, five answers

    Now some real forecasts: {N_FORECAST_CASES} of them, launched from a cycling LETKF
    analysis of Lorenz 96 (chapter 19's configuration), verified against a truth that
    exists only because this is a simulation. Climatological standard deviation
    {CLIMATE_SIGMA:.2f}. The ensemble scores use {N_ENSEMBLE_CASES} of the cases,
    because they cost twenty times as much to compute.

    Five scores, one set of forecasts.
    """
    )
    return


@app.cell(hide_code=True)
def s2_fig(
    CLIMATE_SIGMA, C_ANALYSIS, C_MEAN, C_PERT, C_SAT, C_TRUTH, LEADS,
    SCORE_ACC, SCORE_BRIERSS, SCORE_CRPS, SCORE_MSESS, SCORE_RMSE, finish_mpl,
    mpl_panels, np,
):
    _leads = np.asarray(LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Scores that fall", "RMSE and CRPS, which rise"),
        figsize=(10.2, 4.0),
    )
    for _values, _colour, _label in (
        (SCORE_ACC, C_TRUTH, "anomaly correlation"),
        (SCORE_MSESS, C_ANALYSIS, "MSE skill score"),
        (SCORE_BRIERSS, C_MEAN, "Brier skill score"),
    ):
        _ax0.plot(_leads, np.asarray(_values), "-", color=_colour,
                  linewidth=2.0, label=_label)
    _ax0.axhline(0.6, color=C_SAT, linestyle=":", linewidth=1.4)
    _ax0.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.2)
    _ax0.text(_leads[-1], 0.63, "ACC 0.6", ha="right", fontsize=7.5,
              color=C_SAT)
    _ax0.set_ylabel("score")
    _ax0.legend(fontsize=8, framealpha=0.9)

    _ax1.plot(_leads, np.asarray(SCORE_RMSE), "-", color=C_PERT, linewidth=2.0,
              label="RMSE")
    _ax1.axhline(CLIMATE_SIGMA, color=C_SAT, linestyle="--", linewidth=1.4)
    _ax1.text(_leads[0], CLIMATE_SIGMA * 1.03, "climatological $\\sigma$",
              fontsize=7.5, color=C_SAT)
    _ax1.axhline(np.sqrt(2.0) * CLIMATE_SIGMA, color=C_SAT, linestyle=":",
                 linewidth=1.4)
    _ax1.text(_leads[0], np.sqrt(2.0) * CLIMATE_SIGMA * 1.02,
              "$\\sqrt{2}\\,\\sigma$ — saturation", fontsize=7.5,
              color=C_SAT)
    _ax1.set_ylabel("RMSE")
    # CRPS on its OWN axis. Rescaling it onto the RMSE axis made it appear to
    # saturate at the climatological sigma, which is not a thing CRPS does --
    # the shared axis invented a reference level for it.
    _twin = _ax1.twinx()
    _twin.plot(_leads, np.asarray(SCORE_CRPS), "-", color=C_MEAN,
               linewidth=2.0, label="CRPS (right axis)")
    _twin.set_ylabel("CRPS", color=C_MEAN)
    _twin.tick_params(axis="y", colors=C_MEAN, labelsize=8)
    _twin.grid(False)
    _handles = _ax1.get_legend_handles_labels()
    _twin_handles = _twin.get_legend_handles_labels()
    _ax1.legend(_handles[0] + _twin_handles[0], _handles[1] + _twin_handles[1],
                fontsize=8, framealpha=0.9, loc="lower right")

    for _ax in (_ax0, _ax1):
        _ax.set_xlabel("forecast lead (TU)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s2_horizons(
    C_ANALYSIS, HORIZON_LABELS, HORIZON_SPREAD, HORIZON_VALUES, finish_mpl,
    mpl_panels, np,
):
    _values = np.asarray(HORIZON_VALUES)
    _order = np.argsort(_values)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.4, 3.6))
    _y = np.arange(_values.size)
    _ax.barh(_y, _values[_order], color=C_ANALYSIS, edgecolor="white",
             linewidth=0.8)
    _ax.set_yticks(_y)
    _ax.set_yticklabels([HORIZON_LABELS[i] for i in _order], fontsize=8.5)
    for _yi, _i in enumerate(_order):
        _ax.text(_values[_i] + 0.04, _yi, f"{_values[_i]:.2f}", va="center",
                 fontsize=8, color="#4a4460")
    _ax.set_xlabel("predictability horizon (TU)")
    _ax.set_xlim(0, _values.max() * 1.15)
    finish_mpl(
        _fig,
        f"The same forecasts, {HORIZON_SPREAD:.1f}x apart depending on the "
        f"definition",
    )
    return


@app.cell(hide_code=True)
def s2_note(HORIZON_LABELS, HORIZON_SPREAD, HORIZON_VALUES, mo, np):
    _v = dict(zip(HORIZON_LABELS, np.asarray(HORIZON_VALUES)))
    mo.md(
        rf"""
    **A factor of {HORIZON_SPREAD:.1f} separates the widest definition from the
    narrowest, and none of them is wrong.** Every one is a defensible answer to "how far
    ahead is this forecast useful"; they are answers to different questions.

    Two of the entries agreeing is not a coincidence. **ACC < 0.5** and
    **MSE skill < 0** give {_v["ACC < 0.5"]:.2f} and {_v["MSE skill < 0"]:.2f} — the same
    number, because section 1's identity says they *are* the same condition for an
    undamped forecast. Seeing an algebraic identity survive contact with a cycling
    assimilation system and a nonlinear model is a reassuring check on both.

    The outlier is instructive too. **Brier skill < 0** at {_v["Brier skill < 0"]:.2f} is
    nearly twice the ACC-0.6 horizon: a probabilistic forecast of *one threshold event*
    stays useful long after the deterministic field forecast has stopped resembling the
    truth. This is the same point chapter 17 made with the cost-loss curve, arriving
    from a different direction — the question "will it freeze?" survives much longer
    than the question "what exactly will the field look like?"

    Note also where the RMSE curve is heading. A saturated undamped forecast is an
    independent draw from the climatology, so its mean-square error tends to
    **twice** the climatological variance, not once — an RMSE of $\sqrt2\,\sigma$, and a
    skill score of $-1$ rather than 0. A forecast can be much worse than useless while
    still being a perfectly realistic-looking field.
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(mo):
    mo.md(
        r"""
    ## 3 · The knob: what counts as useful

    The threshold is a choice, and the chapter's knob. Slide it and watch the headline
    number move.
    """
    )
    return


@app.cell(hide_code=True)
def s3_control(ACC_THRESHOLDS, mo):
    threshold_pick = mo.ui.slider(
        steps=[float(v) for v in ACC_THRESHOLDS],
        value=0.6,
        label="anomaly correlation counted as 'useful'",
        show_value=True,
        full_width=True,
    )
    threshold_pick
    return (threshold_pick,)


@app.cell(hide_code=True)
def s3_fig(
    ACC_THRESHOLDS, ACC_THRESHOLD_HORIZON, C_ANALYSIS, C_PERT, C_SAT, LEADS,
    SCORE_ACC, finish_mpl, mpl_panels, np, threshold_pick,
):
    _t = np.asarray(ACC_THRESHOLDS)
    _h = np.asarray(ACC_THRESHOLD_HORIZON)
    _pick = float(threshold_pick.value)
    _index = int(np.argmin(np.abs(_t - _pick)))
    _leads = np.asarray(LEADS)
    _acc = np.asarray(SCORE_ACC)

    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("The skill curve, and where the threshold cuts it",
                "Horizon against threshold"),
        figsize=(10.2, 4.0),
    )
    _ax0.plot(_leads, _acc, "-", color=C_ANALYSIS, linewidth=2.2)
    _ax0.axhline(_pick, color=C_PERT, linestyle="--", linewidth=1.6)
    _ax0.axvline(_h[_index], color=C_PERT, linestyle=":", linewidth=1.6)
    _ax0.plot([_h[_index]], [_pick], "o", color=C_PERT, markersize=9,
              markeredgecolor="white", markeredgewidth=1.0, zorder=5)
    _ax0.set_xlabel("forecast lead (TU)")
    _ax0.set_ylabel("anomaly correlation")
    _ax0.set_ylim(0, 1.02)

    _ax1.plot(_t, _h, "o-", color=C_ANALYSIS, markersize=4.5, linewidth=1.8)
    _ax1.plot([_t[_index]], [_h[_index]], "o", color=C_PERT, markersize=11,
              markerfacecolor="none", markeredgewidth=2.2)
    _ax1.axvline(0.6, color=C_SAT, linestyle=":", linewidth=1.4)
    _ax1.text(0.61, _h.max() * 0.95, "the convention", fontsize=7.5,
              color=C_SAT)
    _ax1.set_xlabel("anomaly correlation counted as useful")
    _ax1.set_ylabel("horizon (TU)")
    finish_mpl(
        _fig,
        f"At a threshold of {_t[_index]:.2f}, the horizon is "
        f"{_h[_index]:.2f} TU",
    )
    return


@app.cell(hide_code=True)
def s3_note(ACC_THRESHOLDS, ACC_THRESHOLD_HORIZON, DAMPED_HORIZON,
            UNDAMPED_HORIZON, mo, np):
    _t = np.asarray(ACC_THRESHOLDS)
    _h = np.asarray(ACC_THRESHOLD_HORIZON)
    _lo = int(np.argmin(np.abs(_t - 0.4)))
    _mid = int(np.argmin(np.abs(_t - 0.6)))
    _hi = int(np.argmin(np.abs(_t - 0.9)))
    mo.md(
        rf"""
    From {_t[_hi]:.2f} to {_t[_lo]:.2f} the horizon runs
    {_h[_hi]:.2f} to {_h[_lo]:.2f} TU — a factor of {_h[_lo] / _h[_hi]:.1f} — and the
    curve is steepest exactly where the convention sits. A tenth on the threshold is
    worth roughly
    {abs(_h[_mid] - _h[min(_mid + 2, _h.size - 1)]):.2f} TU here, which in atmospheric
    terms is the difference between "useful to five days" and "useful to six".

    **And post-processing moves it further than the threshold does.** Section 1's damping
    argument, applied to these forecasts: the undamped mean-square error reaches the
    climatological value at lead {UNDAMPED_HORIZON:.2f} TU, while the optimally damped
    one does not reach it at all within the {np.asarray([5.0])[0]:.0f} TU measured. The
    same forecasts, the same truth, and a horizon that more than doubles depending on
    whether anyone rescaled the output.

    None of this is an argument that verification is arbitrary. It is an argument that
    **a horizon is a statement about a score, a threshold and a post-processing choice**,
    and quoting one without the other three is quoting a number with its units removed.
    """
    )
    return


# ===========================================================================
# Section 4
# ===========================================================================
@app.cell(hide_code=True)
def s4_md(BIASED_FORCING, MEAN_OFFSET, mo):
    mo.md(
        rf"""
    ## 4 · Where the error actually is

    A single number tells you how wrong a forecast was. Murphy's decomposition tells you
    *how* it was wrong, and the three parts are fixed by completely different things:

    $$
    \mathrm{{MSE}} = \underbrace{{(\bar f - \bar t)^2}}_{{\text{{bias}}}}
      + \underbrace{{(\sigma_f - \sigma_t)^2}}_{{\text{{amplitude}}}}
      + \underbrace{{2\sigma_f\sigma_t(1-r)}}_{{\text{{phase}}}} ,
    $$

    an identity, not an approximation. **Bias** is removable by subtraction.
    **Amplitude** is a calibration problem, and section 1's damping is its cure.
    **Phase** is predictive information actually going away, and it is the only one of
    the three that chaos forces on you.

    Three forecast sets below: the perfect twin, a model run with the wrong forcing
    ($F = {{BIASED_FORCING:g}}$ instead of 8), and the twin with a constant
    {{MEAN_OFFSET:g}} added — the canonical model bias that every operational system
    has.
    """
    )
    return


@app.cell(hide_code=True)
def s4_fig(
    BUDGET_BIASED_AMPLITUDE, BUDGET_BIASED_BIAS, BUDGET_BIASED_PHASE,
    BUDGET_OFFSET_AMPLITUDE, BUDGET_OFFSET_BIAS, BUDGET_OFFSET_PHASE,
    BUDGET_PERFECT_AMPLITUDE, BUDGET_PERFECT_BIAS, BUDGET_PERFECT_PHASE,
    C_ANALYSIS, C_MEAN, C_PERT, LEADS, finish_mpl, mpl_panels, np,
):
    _sets = {
        "perfect twin": (BUDGET_PERFECT_BIAS, BUDGET_PERFECT_AMPLITUDE,
                         BUDGET_PERFECT_PHASE),
        "wrong forcing": (BUDGET_BIASED_BIAS, BUDGET_BIASED_AMPLITUDE,
                          BUDGET_BIASED_PHASE),
        "constant offset": (BUDGET_OFFSET_BIAS, BUDGET_OFFSET_AMPLITUDE,
                            BUDGET_OFFSET_PHASE),
    }
    _leads = np.asarray(LEADS)
    _fig, _axes = mpl_panels(
        ncols=3, titles=tuple(_sets), figsize=(12.4, 3.9)
    )
    for _ax, (_name, (_bias, _amp, _phase)) in zip(_axes, _sets.items()):
        # Log axis: phase is four orders of magnitude above the other two, and a
        # linear axis renders bias and amplitude as a flat line on zero in every
        # panel -- including the panels built to make them visible.
        for _values, _colour, _label in (
            (_phase, C_ANALYSIS, "phase"),
            (_bias, C_PERT, "bias$^2$"),
            (_amp, C_MEAN, "amplitude"),
        ):
            _ax.semilogy(_leads, np.maximum(np.asarray(_values), 1e-6), "-",
                         color=_colour, linewidth=2.0, label=_label)
        _ax.set_xlabel("forecast lead (TU)")
        _ax.set_ylim(1e-6, 1e2)
    _axes[0].set_ylabel("contribution to MSE")
    _axes[0].legend(fontsize=7.5, framealpha=0.9, loc="lower right")
    finish_mpl(_fig, "Each fault lights up its own term")
    return


@app.cell(hide_code=True)
def s4_note(
    BUDGET_BIASED_AMPLITUDE, BUDGET_BIASED_BIAS, BUDGET_BIASED_PHASE,
    BUDGET_OFFSET_BIAS, BUDGET_PERFECT_AMPLITUDE, BUDGET_PERFECT_BIAS,
    BUDGET_PERFECT_PHASE, MEAN_OFFSET, mo, np,
):
    _amp_ratio = (np.asarray(BUDGET_BIASED_AMPLITUDE)[-1]
                  / max(np.asarray(BUDGET_PERFECT_AMPLITUDE)[-1], 1e-12))
    _bias_ratio = (np.asarray(BUDGET_OFFSET_BIAS)[-1]
                   / max(np.asarray(BUDGET_PERFECT_BIAS)[-1], 1e-12))
    _phase_share = (np.asarray(BUDGET_PERFECT_PHASE)[-1]
                    / (np.asarray(BUDGET_PERFECT_PHASE)[-1]
                       + np.asarray(BUDGET_PERFECT_BIAS)[-1]
                       + np.asarray(BUDGET_PERFECT_AMPLITUDE)[-1]))
    mo.md(
        rf"""
    **The perfect twin is {100 * _phase_share:.2f} % phase**, which is not a discovery —
    it is what makes it a perfect twin. The forecast is a genuine trajectory of the same
    system, so it has no systematic bias and its variance is right by construction. Any
    other result would have meant a bug.

    The other two panels are the diagnostic working. The wrong-forcing model's
    **amplitude** term is {_amp_ratio:.0f} times the twin's while its phase term is only
    6 % larger; the constant offset's **bias** term is {_bias_ratio:.0f} times the
    twin's while amplitude and phase do not move at all. Each fault appears in its own
    term and nowhere else.

    Notice what that requires you to read correctly. In the wrong-forcing panel, bias and
    amplitude together are still under 0.2 % of the total error — phase dominates
    everywhere, at every lead, in every panel. **The term that grew is the one that
    identifies the fault, not the term that is largest.** A verification report that only
    quoted total MSE would have shown the wrong-forcing model as 6 % worse and said
    nothing about why.
    """
    )
    return


# ===========================================================================
# Section 5
# ===========================================================================
@app.cell(hide_code=True)
def s5v_md(OBS_VARIANCE, mo):
    mo.md(
        rf"""
    ## 5 · Verified against what?

    Everything so far compared forecasts against a truth. Here is the awkward part:
    **no forecast centre has one.** The candidates are observations, which are the truth
    plus an error, and the verifying analysis, which is a model estimate built from
    those observations.

    Two exact results make the first case tractable. If the observation error is
    **independent** of the forecast error,

    $$
    \mathbb{{E}}\,(f-y)^2 = \mathbb{{E}}\,(f-t)^2 + \sigma_o^2,
    \qquad
    \mathrm{{ACC}}_{{\text{{obs}}}} = \mathrm{{ACC}}_{{\text{{true}}}}
      \left(1 + \frac{{\sigma_o^2}}{{\sigma_t^2}}\right)^{{-1/2}} ,
    $$

    the cross term vanishing in the first and the classical attenuation-by-measurement-
    error appearing in the second. Both are correctable. Below, both are tested against
    a real cycling system with $\sigma_o^2 = {OBS_VARIANCE:g}$.

    The whole difficulty is in the word *independent*.
    """
    )
    return


@app.cell(hide_code=True)
def s5v_fig(
    ACCV_INDEP, ACCV_TRUTH, ACC_CORRECTED, C_ANALYSIS, C_OBS, C_PERT, C_TRUTH,
    LEADS, MSE_INDEP, MSE_TRUTH, OBS_VARIANCE, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Mean-square error inflation", "Anomaly correlation attenuation"),
        figsize=(10.2, 4.0),
    )
    _inflation = np.asarray(MSE_INDEP) - np.asarray(MSE_TRUTH)
    _ax0.plot(_leads, _inflation, "o-", color=C_OBS, markersize=3.5,
              linewidth=1.6, label="measured inflation")
    _ax0.axhline(OBS_VARIANCE, color=C_TRUTH, linestyle="--", linewidth=1.8,
                 label=r"$\sigma_o^2$")
    _ax0.set_ylim(0, OBS_VARIANCE * 2.0)
    _ax0.set_ylabel(r"MSE(vs obs) $-$ MSE(vs truth)")
    _ax0.legend(fontsize=8, framealpha=0.9)

    _ax1.plot(_leads, np.asarray(ACCV_TRUTH), "-", color=C_TRUTH,
              linewidth=2.2, label="against the truth")
    _ax1.plot(_leads, np.asarray(ACCV_INDEP), "-", color=C_OBS, linewidth=2.0,
              label="against noisy observations")
    _ax1.plot(_leads, np.asarray(ACC_CORRECTED), "--", color=C_ANALYSIS,
              linewidth=2.0, label="corrected")
    _ax1.set_ylabel("anomaly correlation")
    _ax1.set_ylim(0, 1.02)
    _ax1.legend(fontsize=8, framealpha=0.9)

    for _ax in (_ax0, _ax1):
        _ax.set_xlabel("forecast lead (TU)")
    finish_mpl(_fig, "Independent observation error: predictable, and removable")
    return


@app.cell(hide_code=True)
def s5v_analysis(
    ANALYSIS_FRACTION, C_PERT, LEADS, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(LEADS)[1:]
    _fraction = 100.0 * np.asarray(ANALYSIS_FRACTION)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.4, 3.8))
    _ax.plot(_leads, _fraction, "o-", color=C_PERT, markersize=4,
             linewidth=1.8)
    _ax.axhline(0.0, color="#4a4460", linestyle="--", linewidth=1.3)
    _ax.set_xlabel("forecast lead (TU)")
    _ax.set_ylabel("MSE against the analysis,\nas a % departure from the truth")
    _ax.set_xscale("log")
    finish_mpl(
        _fig,
        "Verifying against your own analysis flatters the forecast, and only "
        "at short lead",
    )
    return


@app.cell(hide_code=True)
def s5v_note(
    ANALYSIS_FRACTION, ANALYSIS_TRUE_MSE, ASSIM_CORRECTED, ASSIM_MSE, LEADS,
    OBS_VARIANCE, mo, np,
):
    _f = 100.0 * np.asarray(ANALYSIS_FRACTION)
    _leads = np.asarray(LEADS)[1:]
    _at = lambda x: _f[int(np.argmin(np.abs(_leads - x)))]  # noqa: E731
    mo.md(
        rf"""
    **The independent case behaves exactly as the algebra says.** The inflation sits on
    $\sigma_o^2 = {OBS_VARIANCE:g}$ at every lead, from a forecast error of 0.06 to one
    of 25 — a factor of 400 in the quantity being measured, with the same additive
    offset throughout. The corrected anomaly correlation lies on the truth-based curve.
    Noisy verification makes a forecast look *worse*, predictably, and you can undo it.

    **Verifying against your own analysis flatters the forecast, and it is a short-lead
    problem.** The analysis is not an independent estimate: it was built from a
    background forecast produced by the same model, so its errors and the forecast's
    share a component. Measured as a fraction of the true mean-square error, the
    flattery is **{_at(0.1):.0f} %** at lead 0.1, {_at(0.3):.0f} % at 0.3,
    {_at(0.5):.0f} % at 0.5 — and under {abs(_at(2.0)):.1f} % by lead 2. Beyond a day or
    so it hardly matters *in this configuration*, which has an unusually good analysis;
    with an analysis error closer to the forecast error the contamination would persist
    much further.

    /// admonition | And the case where the correction returns an impossible number
        type: danger

    Verify the **analysis** against the observations that were assimilated to make it.
    The measured mean-square error is {ASSIM_MSE:.4f}. If the errors were independent it
    would be {ANALYSIS_TRUE_MSE:.4f} + {OBS_VARIANCE:g} =
    {ANALYSIS_TRUE_MSE + OBS_VARIANCE:.4f}. Applying the standard correction gives

    $$
    {ASSIM_MSE:.4f} - {OBS_VARIANCE:g} = {ASSIM_CORRECTED:.4f},
    $$

    a **negative mean-square error**. The assimilation pulled the analysis towards those
    very observations, so the cross term does not vanish, and the score was optimistic
    rather than pessimistic.

    `chaoslib` returns that negative number rather than clipping it to zero, and this is
    why: an impossible answer announces its own failure. A clipped zero would have looked
    like a superb forecast. The operational remedy is to verify against observations
    withheld from the assimilation — which costs exactly the observations you would most
    have liked to use.
    ///
    """
    )
    return


# ===========================================================================
# Section 6
# ===========================================================================
@app.cell(hide_code=True)
def s6v_md(mo):
    mo.md(
        r"""
    ## 6 · What better initial conditions buy

    The book's opening question, and chapter 12's promise. The practical horizon has
    advanced by roughly a day per decade for forty years. The intrinsic one, chapter 12
    argued, is a different quantity that does not move at all. Here is the test.

    Reduce the initial error across **seven decades** and measure the horizon, in two
    systems: single-scale Lorenz 96, which has one characteristic time; and two-scale
    Lorenz 96, which has fast variables whose errors saturate almost immediately and then
    contaminate the slow ones — chapter 12's cascade, in its cheapest form.
    """
    )
    return


@app.cell(hide_code=True)
def s6v_fig(
    AMPLITUDES, C_ANALYSIS, C_PERT, C_TRUTH, HORIZON_ONE_SCALE,
    HORIZON_TWO_SCALE, LAMBDA1_L96, finish_mpl, mpl_panels, np,
):
    _amp = np.asarray(AMPLITUDES)
    _one = np.asarray(HORIZON_ONE_SCALE)
    _two = np.asarray(HORIZON_TWO_SCALE)
    _fig, (_ax,) = mpl_panels(ncols=1, figsize=(8.8, 4.4))
    _ax.semilogx(_amp, _one, "o-", color=C_TRUTH, markersize=6, linewidth=2.2,
                 label="single-scale Lorenz 96")
    _ax.semilogx(_amp, _two, "s-", color=C_PERT, markersize=6, linewidth=2.2,
                 label="two-scale Lorenz 96 (slow variables)")
    # The logarithmic law: ln(10)/lambda1 per decade, anchored at the largest
    # initial error.
    _predicted = _one[-1] + np.log10(_amp[-1] / _amp) * np.log(10.0) / LAMBDA1_L96
    _ax.semilogx(_amp, _predicted, "--", color=C_ANALYSIS, linewidth=1.6,
                 label=r"$\ln 10/\lambda_1$ per decade")
    _ax.set_xlabel(r"initial error amplitude $\delta_0$")
    _ax.set_ylabel("ACC-0.6 horizon (TU)")
    _ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    finish_mpl(_fig, "Seven decades of better initialisation, two systems")
    return


@app.cell(hide_code=True)
def s6v_note(
    AMPLITUDES, HORIZON_ONE_SCALE, HORIZON_TWO_SCALE, LAMBDA1_L96,
    TWO_SCALE_CASES, mo, np,
):
    _amp = np.asarray(AMPLITUDES)
    _one = np.asarray(HORIZON_ONE_SCALE)
    _two = np.asarray(HORIZON_TWO_SCALE)
    _gain_one = (_one[0] - _one[-1]) / 7.0
    _gain_two = (_two[0] - _two[-1]) / 7.0
    mo.md(
        rf"""
    **The single-scale system keeps paying, at exactly the predicted rate.** Seven
    decades of better initialisation buy {_one[0] - _one[-1]:.2f} TU, which is
    {_gain_one:.2f} TU per decade against the predicted
    $\ln 10/\lambda_1 = {np.log(10.0) / LAMBDA1_L96:.2f}$. The dashed line is that law,
    not a fit. Chapter 20 measured the same thing by an entirely different route.

    **The two-scale system very nearly stops paying.** The same seven decades buy
    {_two[0] - _two[-1]:.2f} TU — {_gain_two:.2f} TU per decade, **{_gain_one / _gain_two:.0f}
    times less**. The fast variables' errors saturate almost at once whatever
    $\delta_0$ was, and contaminate the slow variables from below; making the initial
    condition more accurate does very little about an error that regenerates itself.

    ### An honest qualification

    The two-scale curve is **not flat**. It rises from {_two[0]:.2f} to {_two[-1]:.2f} TU
    and is still creeping upward at $\delta_0 = 10^{{-8}}$, with visible scatter from
    {TWO_SCALE_CASES} cases. A *bounded* horizon — one that converges to a finite limit
    as $\delta_0 \to 0$ — is the limit of this behaviour, and this experiment does not
    demonstrate it outright. What it does demonstrate is the thing that matters
    practically: the return on better initial conditions is **{_gain_one / _gain_two:.0f}
    times smaller** in a system with fast scales, and the two curves diverge steadily as
    $\delta_0$ falls.

    Chapter 12 put it as "the two limits are different quantities, and only one of them
    moves". The measurement supports the first half exactly and softens the second: both
    move, but at rates differing by almost an order of magnitude, and the gap widens.
    Chapter 14 showed why the exponent $\alpha=(3-p)/2$ decides whether the limit is
    finite at all, and a two-scale system with 32 fast variables per slow one is a
    caricature of a spectrum rather than a spectrum.
    """
    )
    return


# ===========================================================================
# Section 7
# ===========================================================================
@app.cell(hide_code=True)
def s7v_md(mo):
    mo.md(
        r"""
    ## 7 · What to take away

    **"Useful to about a week" is a statement about four choices**, not one about the
    atmosphere: which score, which threshold, whether the output was post-processed, and
    what it was verified against. The same forecasts here give horizons a factor of 2.5
    apart across defensible definitions, and post-processing alone more than doubles one
    of them.

    **The 0.6 threshold is arithmetic, not meteorology.** An undamped forecast has skill
    score $2r-1$, so it ties with climatology at $r=1/2$. Damp it optimally and it beats
    climatology at any $r>0$. The convention encodes a decision not to rescale.

    **Verification against observations is correctable; verification against your own
    analysis is not.** Independent observation error adds exactly $\sigma_o^2$ to the
    mean-square error and attenuates the anomaly correlation by exactly a known factor.
    Correlated error does neither, and the correction can return a negative mean-square
    error — which is the most useful thing it can do, because it is impossible and says
    so.

    **The practical horizon advances logarithmically, at $\ln 10/\lambda_1$ per decade of
    initial-condition improvement.** That is a real and continuing return, and it is what
    forty years of NWP progress has been buying.

    **In a system with fast scales, that return is nearly an order of magnitude
    smaller** — and this is the sense in which the intrinsic limit is a different
    quantity from the practical one. It is the honest version of chapter 12's claim: not
    that improvement stops, but that it becomes very much more expensive.

    ### Try this

    1. Set the section 3 threshold to 0.5 and confirm the horizon matches the MSE-skill
       horizon in section 2. Why must these agree exactly?
    2. Section 2's RMSE tends to $\sqrt2\,\sigma$, not $\sigma$. Derive that, and say
       what it implies about a "useful" threshold placed at $\mathrm{RMSE} = \sigma$.
    3. In section 5, the analysis flattery is 64 % at lead 0.1 and 0.3 % at lead 2.
       What property of the assimilation sets that decay rate?
    4. Verifying against assimilated observations gave a negative corrected MSE. Work
       out the sign of the cross term that produces it, and construct the observing
       system for which the correction would be exactly right.
    5. Section 6's two-scale curve is still rising at $\delta_0 = 10^{-8}$. Design the
       experiment that would settle whether it converges, and say why it is expensive.

    ### Where this goes next

    Part V ends here. **Part VI** turns to predictability of the second kind — where the
    quantity being forecast is a statistic of the attractor rather than a point on it,
    and where the verification problem of section 5 becomes much harder still, because a
    climate projection cannot be verified against anything at all for decades.

    ### Further reading

    - Jolliffe & Stephenson, *Forecast Verification* *[citation needed: edition]*
    - Murphy (1988), on the decomposition of the mean-square error *[citation needed]*
    - Murphy & Epstein (1989), on skill scores and their reference forecasts
      *[citation needed]*
    - Bauer, Thorpe & Brunet (2015), "The quiet revolution of numerical weather
      prediction", for the historical skill record *[citation needed: figure]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*
      *[citation needed: chapter]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
