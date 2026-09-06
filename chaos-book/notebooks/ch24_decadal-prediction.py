# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 24 -- The ocean's role: interannual-to-decadal prediction.

Where the memory is, what initialising a slow component buys, and the two
practical difficulties that decide whether a decadal forecast is believable:
drift, and how many re-forecasts it takes to remove it.

Part VI of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib`; this file holds the exposition and the figures.
The experiments are precomputed by `scripts/generate_ch24_data.py`.

To edit:   marimo edit notebooks/ch24_decadal-prediction.py
To export: make nb-one NB=ch24_decadal-prediction
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 24: Decadal Prediction")


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
    # Precomputed by scripts/generate_ch24_data.py (~1 min): a 6000-TU
    # coupled run, 300 initialised/uninitialised forecast pairs, the
    # drift of an imperfect model, and the re-forecast sweep.

    FAST_TIME = 0.1400
    SLOW_TIME = 17.8500
    FAST_SD = 8.590141
    SLOW_SD = 2.468433
    OCEAN_TIME = 50.0
    REFERENCE_Z = 23.6
    FEEDBACK = -0.02
    FORCING_STRENGTH = 30.0
    AUTO_LAGS = (
        0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000, 4.5000,
        5.0000, 5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000, 9.0000, 9.5000,
        10.0000, 10.5000, 11.0000, 11.5000, 12.0000, 12.5000, 13.0000, 13.5000, 14.0000, 14.5000,
        15.0000, 15.5000, 16.0000, 16.5000, 17.0000, 17.5000, 18.0000, 18.5000, 19.0000, 19.5000,
        20.0000, 20.5000, 21.0000, 21.5000, 22.0000, 22.5000, 23.0000, 23.5000, 24.0000, 24.5000,
        25.0000, 25.5000, 26.0000, 26.5000, 27.0000, 27.5000, 28.0000, 28.5000, 29.0000, 29.5000,
        30.0000, 30.5000, 31.0000, 31.5000, 32.0000, 32.5000, 33.0000, 33.5000, 34.0000, 34.5000,
        35.0000, 35.5000, 36.0000, 36.5000, 37.0000, 37.5000, 38.0000, 38.5000, 39.0000, 39.5000,
        40.0000, 40.5000, 41.0000, 41.5000, 42.0000, 42.5000, 43.0000, 43.5000, 44.0000, 44.5000,
        45.0000, 45.5000, 46.0000, 46.5000, 47.0000, 47.5000, 48.0000, 48.5000, 49.0000, 49.5000,
        50.0000, 50.5000, 51.0000, 51.5000, 52.0000, 52.5000, 53.0000, 53.5000, 54.0000, 54.5000,
        55.0000, 55.5000, 56.0000, 56.5000, 57.0000, 57.5000, 58.0000, 58.5000, 59.0000, 59.5000,
        60.0000, 60.5000, 61.0000, 61.5000, 62.0000, 62.5000, 63.0000, 63.5000, 64.0000, 64.5000,
        65.0000, 65.5000, 66.0000, 66.5000, 67.0000, 67.5000, 68.0000, 68.5000, 69.0000, 69.5000,
        70.0000, 70.5000, 71.0000, 71.5000, 72.0000, 72.5000, 73.0000, 73.5000, 74.0000, 74.5000,
        75.0000, 75.5000, 76.0000, 76.5000, 77.0000, 77.5000, 78.0000, 78.5000, 79.0000, 79.5000,
        80.0000, 80.5000, 81.0000, 81.5000, 82.0000, 82.5000, 83.0000, 83.5000, 84.0000, 84.5000,
        85.0000, 85.5000, 86.0000, 86.5000, 87.0000, 87.5000, 88.0000, 88.5000, 89.0000, 89.5000,
        90.0000, 90.5000, 91.0000, 91.5000, 92.0000, 92.5000, 93.0000, 93.5000, 94.0000, 94.5000,
        95.0000, 95.5000, 96.0000, 96.5000, 97.0000, 97.5000, 98.0000, 98.5000, 99.0000, 99.5000,
        100.0000, 100.5000, 101.0000, 101.5000, 102.0000, 102.5000, 103.0000, 103.5000, 104.0000, 104.5000,
        105.0000, 105.5000, 106.0000, 106.5000, 107.0000, 107.5000, 108.0000, 108.5000, 109.0000, 109.5000,
        110.0000, 110.5000, 111.0000, 111.5000, 112.0000, 112.5000, 113.0000, 113.5000, 114.0000, 114.5000,
        115.0000, 115.5000, 116.0000, 116.5000, 117.0000, 117.5000, 118.0000, 118.5000, 119.0000, 119.5000,
        120.0000, 120.5000, 121.0000, 121.5000, 122.0000, 122.5000, 123.0000, 123.5000, 124.0000, 124.5000,
        125.0000, 125.5000, 126.0000, 126.5000, 127.0000, 127.5000, 128.0000, 128.5000, 129.0000, 129.5000,
        130.0000, 130.5000, 131.0000, 131.5000, 132.0000, 132.5000, 133.0000, 133.5000, 134.0000, 134.5000,
        135.0000, 135.5000, 136.0000, 136.5000, 137.0000, 137.5000, 138.0000, 138.5000, 139.0000, 139.5000,
        140.0000, 140.5000, 141.0000, 141.5000, 142.0000, 142.5000, 143.0000, 143.5000, 144.0000, 144.5000,
        145.0000, 145.5000, 146.0000, 146.5000, 147.0000, 147.5000, 148.0000, 148.5000, 149.0000, 149.5000,
        150.0000, 150.5000, 151.0000, 151.5000, 152.0000, 152.5000, 153.0000, 153.5000, 154.0000, 154.5000,
        155.0000, 155.5000, 156.0000, 156.5000, 157.0000, 157.5000, 158.0000, 158.5000, 159.0000, 159.5000,
        160.0000, 160.5000, 161.0000, 161.5000, 162.0000, 162.5000, 163.0000, 163.5000, 164.0000, 164.5000,
        165.0000, 165.5000, 166.0000, 166.5000, 167.0000, 167.5000, 168.0000, 168.5000, 169.0000, 169.5000,
        170.0000, 170.5000, 171.0000, 171.5000, 172.0000, 172.5000, 173.0000, 173.5000, 174.0000, 174.5000,
        175.0000, 175.5000, 176.0000, 176.5000, 177.0000, 177.5000, 178.0000, 178.5000, 179.0000, 179.5000,
        180.0000, 180.5000, 181.0000, 181.5000, 182.0000, 182.5000, 183.0000, 183.5000, 184.0000, 184.5000,
        185.0000, 185.5000, 186.0000, 186.5000, 187.0000, 187.5000, 188.0000, 188.5000, 189.0000, 189.5000,
        190.0000, 190.5000, 191.0000, 191.5000, 192.0000, 192.5000, 193.0000, 193.5000, 194.0000, 194.5000,
        195.0000, 195.5000, 196.0000, 196.5000, 197.0000, 197.5000, 198.0000, 198.5000, 199.0000, 199.5000,
    )
    AUTO_FAST = (
        1.00000, -0.36679, -0.23203, 0.43021, -0.29938, -0.07102, 0.37976, -0.33921, -0.04586, 0.33511,
        -0.27449, -0.02444, 0.25662, -0.24917, 0.02336, 0.20763, -0.23721, 0.05047, 0.16177, -0.20296,
        0.06267, 0.12084, -0.17613, 0.07008, 0.08714, -0.14913, 0.06685, 0.06470, -0.13150, 0.08205,
        0.04340, -0.10872, 0.06865, 0.01668, -0.08305, 0.07311, -0.00289, -0.06437, 0.07037, -0.01448,
        -0.04383, 0.05150, -0.01939, -0.01588, 0.03693, -0.02883, -0.01059, 0.03557, -0.01890, -0.00812,
        0.01892, -0.01492, -0.00249, 0.01982, -0.02585, 0.00819, 0.01689, -0.02420, 0.01690, 0.00172,
        -0.01970, 0.01882, 0.00103, -0.01862, 0.01847, -0.00058, -0.01372, 0.02720, -0.02051, -0.00744,
        0.02001, -0.01104, -0.01025, 0.02380, -0.01972, 0.00032, 0.02042, -0.01810, -0.00087, 0.01939,
        -0.01493, 0.00129, 0.01483, -0.02169, 0.00697, 0.01397, -0.01389, 0.00270, 0.00900, -0.00828,
        0.00035, 0.00696, -0.01238, 0.00622, 0.00948, -0.01957, 0.00688, 0.00717, -0.00731, 0.01352,
        -0.00710, -0.01663, 0.01590, 0.00149, -0.01459, 0.01722, -0.01408, -0.00107, 0.01915, -0.02275,
        0.00319, 0.01331, -0.01653, 0.00969, 0.00583, -0.02028, 0.01583, 0.00291, -0.01385, 0.00905,
        -0.00159, -0.00510, 0.00619, -0.00438, -0.00456, 0.01297, -0.01215, 0.01067, -0.00271, -0.00836,
        0.01214, -0.00251, -0.01246, 0.02154, -0.01478, -0.00471, 0.01764, -0.02304, 0.00122, 0.02247,
        -0.02327, 0.00981, 0.01147, -0.02650, 0.01696, 0.00999, -0.02752, 0.01320, 0.00879, -0.02447,
        0.01758, 0.00090, -0.02444, 0.01904, 0.00626, -0.01868, 0.01526, 0.00239, -0.02159, 0.02524,
        -0.00091, -0.02355, 0.01524, 0.00245, -0.01834, 0.02522, -0.00960, -0.01614, 0.02370, -0.00348,
        -0.01886, 0.02080, -0.00836, -0.01229, 0.02584, -0.01704, -0.00481, 0.01451, -0.01043, -0.00063,
        0.01056, -0.01524, 0.00671, 0.00789, -0.00645, 0.00533, -0.00350, -0.00115, 0.01241, -0.00234,
        -0.01042, 0.00977, -0.00456, -0.00725, 0.01013, -0.00888, 0.00262, 0.01414, -0.02063, 0.00211,
        0.02475, -0.02753, 0.01001, 0.01652, -0.02562, 0.01632, 0.01298, -0.03445, 0.01803, 0.01557,
        -0.03285, 0.01813, 0.01485, -0.03318, 0.02871, 0.00143, -0.03081, 0.02505, -0.00176, -0.02372,
        0.02800, -0.00617, -0.02808, 0.03083, -0.00578, -0.01978, 0.02903, -0.01034, -0.02265, 0.03173,
        -0.00646, -0.03351, 0.03910, -0.01369, -0.02531, 0.04505, -0.01550, -0.03026, 0.05202, -0.02199,
        -0.02041, 0.04210, -0.02813, -0.01386, 0.04469, -0.03254, -0.00963, 0.03322, -0.03104, 0.00002,
        0.03134, -0.03568, 0.01489, 0.01889, -0.03168, 0.01546, 0.01063, -0.02442, 0.01522, 0.00390,
        -0.01093, 0.01335, 0.00298, -0.01355, 0.01329, -0.00322, -0.00990, 0.01329, 0.00533, -0.01275,
        0.01353, -0.00824, -0.01268, 0.02163, -0.01415, -0.00533, 0.02268, -0.02151, 0.00787, 0.00837,
        -0.01841, 0.01108, 0.00404, -0.01215, 0.00665, 0.00471, -0.00843, 0.00884, 0.00289, -0.01688,
        0.00900, 0.00568, -0.00985, 0.00406, -0.00302, -0.00937, 0.01082, 0.00178, -0.01025, 0.01205,
        -0.00817, 0.00641, 0.00053, -0.01228, 0.01468, -0.00624, -0.01278, 0.01325, 0.00110, -0.01299,
        0.00967, -0.00243, -0.01037, 0.02452, -0.01426, -0.00933, 0.02195, -0.01306, -0.00094, 0.01322,
        -0.01852, 0.00713, 0.01586, -0.02230, 0.00438, 0.01041, -0.00504, 0.00449, 0.00006, -0.01143,
        0.00586, 0.01176, -0.01487, -0.00047, 0.00422, -0.00553, 0.00305, 0.00024, -0.01074, 0.01434,
        -0.00801, -0.00265, 0.00799, -0.01006, 0.00786, -0.00025, -0.00858, 0.01377, -0.00496, -0.00258,
        0.00644, -0.00452, -0.00931, 0.01567, -0.00793, -0.00077, 0.01866, -0.01999, -0.00074, 0.01373,
        -0.01372, 0.00406, 0.01008, -0.01511, 0.00832, 0.01126, -0.01680, 0.00817, 0.00479, -0.01483,
        0.00972, -0.00369, -0.01222, 0.01158, -0.00019, -0.00401, 0.00545, -0.00155, -0.00550, 0.01071,
        -0.00190, -0.00656, 0.00682, 0.00168, -0.00403, 0.00707, -0.00560, 0.00078, -0.00246, -0.00633,
        0.00464, 0.00148, 0.00230, 0.00119, -0.00256, 0.00522, -0.00384, -0.00178, -0.00145, -0.00191,
    )
    AUTO_SLOW = (
        1.00000, 0.86052, 0.81168, 0.80345, 0.71584, 0.69949, 0.69808, 0.63008, 0.63057, 0.63589,
        0.58277, 0.58810, 0.59322, 0.54841, 0.55437, 0.55473, 0.51554, 0.52485, 0.52284, 0.49249,
        0.50247, 0.49892, 0.47302, 0.48081, 0.47297, 0.44970, 0.45497, 0.44711, 0.42752, 0.43406,
        0.42351, 0.40504, 0.40725, 0.39502, 0.38114, 0.38373, 0.36995, 0.35833, 0.35804, 0.34137,
        0.33017, 0.32690, 0.31482, 0.30718, 0.30108, 0.28888, 0.28467, 0.28307, 0.27493, 0.26997,
        0.26599, 0.25822, 0.25151, 0.24602, 0.23773, 0.23568, 0.23148, 0.22466, 0.22459, 0.21949,
        0.21211, 0.20797, 0.20258, 0.19784, 0.19514, 0.18852, 0.18119, 0.17862, 0.17003, 0.16514,
        0.16129, 0.15615, 0.15332, 0.15233, 0.14623, 0.14663, 0.14673, 0.14219, 0.13973, 0.13624,
        0.12910, 0.12520, 0.12149, 0.11453, 0.11233, 0.10872, 0.10337, 0.10036, 0.09673, 0.09211,
        0.08793, 0.08426, 0.07966, 0.07720, 0.07481, 0.07169, 0.07192, 0.06926, 0.06491, 0.06052,
        0.05416, 0.05020, 0.04887, 0.04532, 0.04176, 0.04194, 0.03790, 0.03728, 0.03706, 0.03169,
        0.03184, 0.03118, 0.02856, 0.02945, 0.02628, 0.02212, 0.02390, 0.02156, 0.01829, 0.01770,
        0.01496, 0.01362, 0.01330, 0.01276, 0.01283, 0.01396, 0.01131, 0.01322, 0.01269, 0.01327,
        0.01417, 0.01325, 0.01182, 0.01338, 0.00975, 0.00992, 0.01205, 0.01138, 0.01514, 0.01801,
        0.01638, 0.02067, 0.02087, 0.01757, 0.02015, 0.01876, 0.01569, 0.01943, 0.01990, 0.01820,
        0.02209, 0.02232, 0.02248, 0.02579, 0.02599, 0.02577, 0.02807, 0.02771, 0.02635, 0.02905,
        0.02698, 0.02699, 0.03033, 0.02892, 0.02620, 0.02766, 0.02346, 0.02114, 0.02009, 0.01639,
        0.01353, 0.01281, 0.00927, 0.00717, 0.00702, 0.00175, 0.00244, 0.00419, 0.00282, 0.00302,
        0.00339, 0.00088, 0.00036, -0.00126, -0.00324, -0.00371, -0.00482, -0.00470, -0.00613, -0.01200,
        -0.01668, -0.01768, -0.02110, -0.02349, -0.02353, -0.02429, -0.02275, -0.02170, -0.02400, -0.02171,
        -0.01803, -0.02119, -0.01817, -0.01779, -0.02023, -0.01748, -0.01915, -0.02275, -0.01883, -0.01789,
        -0.02034, -0.01678, -0.01609, -0.02080, -0.01909, -0.02348, -0.02851, -0.02827, -0.03118, -0.03342,
        -0.03006, -0.03021, -0.03089, -0.02548, -0.02528, -0.02560, -0.02352, -0.02655, -0.02763, -0.02498,
        -0.02700, -0.02914, -0.02470, -0.02679, -0.02503, -0.01798, -0.02059, -0.02219, -0.01705, -0.02433,
        -0.02860, -0.02845, -0.03671, -0.03996, -0.04004, -0.04849, -0.04941, -0.04905, -0.05525, -0.05568,
        -0.05466, -0.06011, -0.05880, -0.06178, -0.06628, -0.06299, -0.06343, -0.06514, -0.06103, -0.06078,
        -0.06039, -0.05969, -0.06276, -0.06641, -0.06850, -0.07256, -0.07514, -0.07663, -0.07987, -0.08770,
        -0.09290, -0.09830, -0.10159, -0.10101, -0.10410, -0.10411, -0.10349, -0.10808, -0.10582, -0.10548,
        -0.10534, -0.10036, -0.09873, -0.09701, -0.09370, -0.09299, -0.09380, -0.09357, -0.09445, -0.09513,
        -0.09322, -0.09411, -0.09512, -0.09343, -0.09210, -0.08977, -0.08491, -0.08011, -0.07831, -0.07608,
        -0.07794, -0.07854, -0.08020, -0.08005, -0.07637, -0.07605, -0.07366, -0.06947, -0.06689, -0.06366,
        -0.05945, -0.05596, -0.05199, -0.04561, -0.04382, -0.03876, -0.03251, -0.03160, -0.02676, -0.02191,
        -0.02099, -0.01656, -0.01344, -0.01423, -0.01064, -0.00876, -0.00913, -0.00914, -0.01060, -0.01321,
        -0.01392, -0.01320, -0.01407, -0.01198, -0.01073, -0.01040, -0.00797, -0.00571, -0.00612, -0.00440,
        -0.00471, -0.00188, 0.00045, 0.00096, 0.00553, 0.00929, 0.01307, 0.01688, 0.01698, 0.01949,
        0.02157, 0.02305, 0.02582, 0.03245, 0.03484, 0.03652, 0.03713, 0.03174, 0.03001, 0.02938,
        0.02546, 0.02450, 0.02323, 0.02065, 0.02016, 0.01728, 0.01232, 0.01151, 0.00874, 0.00655,
        0.00835, 0.00787, 0.00903, 0.01344, 0.01533, 0.01840, 0.02204, 0.02382, 0.02563, 0.02919,
        0.03020, 0.03088, 0.03198, 0.03179, 0.03051, 0.03017, 0.02790, 0.02895, 0.03030, 0.03213,
        0.03617, 0.03807, 0.03973, 0.03976, 0.03757, 0.03640, 0.03356, 0.02952, 0.02713, 0.02764,
    )
    FEEDBACKS = (0.0, -0.01, -0.02, -0.05, -0.1, -0.25)
    TRADEOFF_SWING = (
        0.000000, 0.027925, 0.050261, 0.108206, 0.192536, 0.380627,
    )
    TRADEOFF_TAU = (
        27.5900, 47.7700, 19.1400, 8.6300, 6.3300, 1.8300,
    )
    TRADEOFF_SD = (
        2.630450, 2.792549, 2.513025, 2.164111, 1.925363, 1.522509,
    )

    LEADS = (np.float64(0.0), np.float64(2.0), np.float64(4.0), np.float64(6.0), np.float64(8.0), np.float64(10.0), np.float64(12.0), np.float64(14.0), np.float64(16.0), np.float64(18.0), np.float64(20.0), np.float64(22.0), np.float64(24.0), np.float64(26.0), np.float64(28.0), np.float64(30.0), np.float64(32.0), np.float64(34.0), np.float64(36.0), np.float64(38.0), np.float64(40.0), np.float64(42.0), np.float64(44.0), np.float64(46.0), np.float64(48.0), np.float64(50.0), np.float64(52.0), np.float64(54.0), np.float64(56.0), np.float64(58.0), np.float64(60.0))
    N_CASES = 300
    SLOW_CLIM_SD = 2.468433
    FAST_CLIM_SD = 8.590141
    SLOW_HORIZON_INIT = 8.0000
    SLOW_HORIZON_UNINIT = 0.0000
    FAST_BEST_EARLY = 0.217540
    FAST_BEST_EARLY_LEAD = 2.00
    FAST_STANDARD_ERROR = 0.012092
    FAST_LATE_MEAN = -0.005771
    FAST_LATE_SD = 0.063191
    FAST_SKILFUL_LEADS = 3
    RMSE_INIT_SLOW = (
        0.049532, 0.434144, 0.995292, 1.443056, 1.948092, 2.165001, 2.423992, 2.196676,
        2.680821, 2.740663, 2.720100, 3.029310, 2.970381, 3.086886, 3.196137, 3.278275,
        3.248193, 3.056542, 3.176474, 3.087012, 3.137023, 3.217332, 3.246766, 3.500843,
        3.628787, 3.298727, 3.451529, 3.476572, 3.367404, 3.514879, 3.473852,
    )
    RMSE_INIT_FAST = (
        0.048147, 2.544300, 5.757090, 8.195139, 10.081755, 9.983899, 11.355258, 11.329157,
        11.753246, 11.925517, 12.189937, 12.272855, 12.795632, 12.589433, 11.728759, 11.679087,
        12.716260, 11.731393, 12.628723, 12.173970, 12.224199, 11.981174, 12.553868, 12.562353,
        11.241475, 11.845677, 12.407223, 12.292730, 12.011418, 11.256364, 11.883277,
    )
    RMSE_UNINIT_SLOW = (
        3.523591, 3.366715, 3.271114, 3.306637, 3.363350, 3.690338, 3.586026, 3.443548,
        3.616371, 3.392522, 3.447373, 3.473110, 3.281866, 3.368433, 3.529022, 3.654377,
        3.555748, 3.291415, 3.600526, 3.442618, 3.313898, 3.561750, 3.359756, 3.502340,
        3.616852, 3.339555, 3.460221, 3.493202, 3.339924, 3.387152, 3.426140,
    )
    RMSE_UNINIT_FAST = (
        0.048147, 4.412999, 5.659617, 9.026865, 10.032394, 11.056309, 11.197317, 11.103673,
        11.531933, 11.291018, 12.084017, 11.352061, 12.421068, 11.821709, 11.922353, 12.469534,
        12.669851, 12.316252, 12.338344, 12.052685, 11.678140, 12.504500, 12.210635, 12.723004,
        11.885176, 12.240801, 12.869015, 12.314694, 11.136399, 11.841051, 12.686939,
    )

    DRIFT_TREND = -0.9678
    PERFECT_TREND = -0.5442


    REFORECAST_SIZES = (5, 10, 20, 40, 80, 160)
    MODEL_REFERENCE_Z = 23.700000000000003
    UNCORRECTED_RMSE = 3.093253
    INSAMPLE_RMSE = 2.765727
    DRIFT_BIAS = (
        -0.002465, -0.102061, -0.174868, -0.439546, -0.411234, -0.653121, -0.830309, -0.763734,
        -1.000116, -1.122124, -1.051976, -0.988783, -1.019014, -1.112362, -1.198202, -1.391434,
        -1.465170, -1.412416, -1.598005, -1.746125, -1.688876, -1.784452, -1.818880, -2.018727,
        -2.058945, -1.984435, -2.024642, -1.903353, -2.136268, -2.204883, -1.790312,
    )
    PERFECT_BIAS = (
        -0.002465, 0.014588, 0.051761, -0.113248, 0.030276, -0.119752, -0.186138, -0.220695,
        -0.310751, -0.323045, -0.322608, -0.180179, -0.032623, -0.205405, -0.207920, -0.392783,
        -0.273210, -0.203757, -0.262849, -0.310033, -0.127529, -0.183056, -0.062144, -0.124794,
        -0.262744, -0.455156, -0.161707, -0.156291, -0.364450, -0.588572, -0.272562,
    )
    DRIFT_RMSE = (
        0.050691, 0.522473, 1.100099, 1.613221, 1.847739, 2.306412, 2.415581, 2.509753,
        2.961999, 3.109219, 3.051552, 3.210112, 3.249927, 3.366302, 3.368184, 3.688587,
        3.643859, 3.499749, 3.605877, 3.526962, 3.682397, 3.696173, 3.620916, 3.885251,
        3.992690, 3.990362, 3.974828, 3.864909, 4.071118, 4.028994, 3.606272,
    )
    PERFECT_RMSE = (
        0.050691, 0.511717, 1.085930, 1.571844, 1.772892, 2.286112, 2.282538, 2.400899,
        2.752845, 2.980434, 2.929407, 3.152761, 3.101491, 3.278025, 3.233068, 3.422155,
        3.340487, 3.332607, 3.361574, 3.465729, 3.261931, 3.436119, 3.508645, 3.572643,
        3.588018, 3.550784, 3.483665, 3.576795, 3.438932, 3.714590, 3.588854,
    )
    UNCORRECTED_BY_LEAD = (
        0.049878, 0.588792, 1.188807, 1.720888, 1.803071, 2.347389, 2.443245, 2.832706,
        2.985166, 3.186129, 3.120677, 3.089794, 3.231078, 3.147420, 3.518266, 3.697259,
        3.578033, 3.643267, 3.614752, 3.508298, 3.883037, 3.669925, 3.699261, 3.786860,
        3.884994, 3.968476, 4.128422, 4.020961, 4.010749, 4.037791, 3.505451,
    )
    CORRECTED_CURVE = (
        3.105252, 3.107385, 2.861014, 2.832176, 2.828896, 2.827571,
    )
    CORRECTED_BY_LEAD = (
        0.050281, 0.576845, 1.177592, 1.640702, 1.747810, 2.326233, 2.387808, 2.691805,
        2.821788, 2.981303, 2.901296, 2.993003, 3.079869, 3.034827, 3.302687, 3.406720,
        3.165116, 3.317318, 3.313489, 3.049868, 3.432000, 3.263648, 3.246580, 3.331095,
        3.434559, 3.644337, 3.649420, 3.622013, 3.426524, 3.456347, 3.181818,
    )

    return (
        AUTO_FAST, AUTO_LAGS, AUTO_SLOW, CORRECTED_BY_LEAD,
        CORRECTED_CURVE, DRIFT_BIAS, DRIFT_RMSE, DRIFT_TREND,
        FAST_BEST_EARLY, FAST_BEST_EARLY_LEAD, FAST_CLIM_SD,
        FAST_LATE_MEAN, FAST_LATE_SD, FAST_SD, FAST_SKILFUL_LEADS,
        FAST_STANDARD_ERROR, FAST_TIME, FEEDBACK, FEEDBACKS,
        FORCING_STRENGTH, INSAMPLE_RMSE, LEADS, MODEL_REFERENCE_Z,
        N_CASES, OCEAN_TIME, PERFECT_BIAS, PERFECT_RMSE, PERFECT_TREND,
        REFERENCE_Z, REFORECAST_SIZES, RMSE_INIT_FAST, RMSE_INIT_SLOW,
        RMSE_UNINIT_FAST, RMSE_UNINIT_SLOW, SLOW_CLIM_SD,
        SLOW_HORIZON_INIT, SLOW_HORIZON_UNINIT, SLOW_SD, SLOW_TIME,
        TRADEOFF_SD, TRADEOFF_SWING, TRADEOFF_TAU, UNCORRECTED_BY_LEAD,
        UNCORRECTED_RMSE,
    )

# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(FEEDBACK, FORCING_STRENGTH, OCEAN_TIME, mo):
    mo.md(
        rf"""
    # Chapter 24 · The Ocean's Role: Interannual-to-Decadal Prediction

    **Part VI — Predictability of the second kind.**

    **The forecasting question.** A decadal forecast claims to say something about the
    next ten years. Chapter 23 showed where such a claim can come from when the forcing
    is *prescribed* — the seasonal cycle, an imposed oscillation. But nobody prescribes
    the ocean. It is part of the system, it is driven by the very weather that cannot be
    predicted past a week, and it is barely observed. What exactly is being initialised,
    and how would you know it helped?

    The system here is a fast Lorenz 63 "atmosphere" coupled to a single slow "ocean"
    variable that integrates it:

    $$
    \dot S = \frac{{-S + \lambda\,(z - z_{{\mathrm{{ref}}}})}}{{T}},
    \qquad T = {OCEAN_TIME:g},\ \lambda = {FORCING_STRENGTH:g},
    $$

    with the ocean feeding back on the atmosphere's effective Rayleigh number,
    $\rho \to \rho + \kappa S$, at $\kappa = {FEEDBACK:g}$.

    This is Hasselmann's picture *[citation needed]*: **the ocean's long memory comes
    from integrating fast weather**, not from slow internal dynamics. A first-order
    relaxation driven by a rapidly decorrelating input has an autocorrelation time of
    order $T$ however short the driving is — which section 1 measures.

    ---

    **What you need before this chapter.** **Chapter 23** for the two-climatology
    argument and predictability of the second kind. **Chapter 22** for why a damped
    forecast scores well, which turns up here as a trap.
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
    ## 1 · Where the memory is

    Two variables in the same system, with completely different memories.
    """
    )
    return


@app.cell(hide_code=True)
def s1_fig(
    AUTO_FAST, AUTO_LAGS, AUTO_SLOW, C_ANALYSIS, C_BG, C_PERT, C_TRUTH,
    FAST_TIME, FEEDBACKS, OCEAN_TIME, SLOW_TIME, TRADEOFF_SWING, TRADEOFF_TAU,
    finish_mpl, mpl_panels, np,
):
    _lags = np.asarray(AUTO_LAGS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Autocorrelation", "The trade-off that fixes the coupling"),
        figsize=(10.2, 4.0),
    )
    _ax0.plot(_lags, np.asarray(AUTO_FAST), "-", color=C_PERT, linewidth=2.3,
              label=f"atmosphere $z$  ($\\tau$ = {FAST_TIME:.2f} TU)")
    _ax0.plot(_lags, np.asarray(AUTO_SLOW), "-", color=C_TRUTH, linewidth=2.3,
              label=f"ocean $S$  ($\\tau$ = {SLOW_TIME:.1f} TU)")
    _ax0.axhline(np.exp(-1.0), color=C_BG, linestyle="--", linewidth=1.3)
    _ax0.text(_lags[-1], np.exp(-1.0) + 0.03, "$1/e$", ha="right", fontsize=8,
              color=C_BG)
    _ax0.set_xscale("symlog", linthresh=1.0)
    _ax0.set_xlabel("lag (time units)")
    _ax0.set_ylabel("autocorrelation")
    _ax0.legend(fontsize=8.5, framealpha=0.9)

    _tau = np.asarray(TRADEOFF_TAU)
    _swing = np.asarray(TRADEOFF_SWING)
    _ax1.plot(_swing, _tau, "o-", color=C_ANALYSIS, markersize=6, linewidth=2.0)
    for _k, _s, _t in zip(FEEDBACKS, _swing, _tau):
        _ax1.annotate(f"$\\kappa={_k:g}$", (_s, _t), fontsize=7.5,
                      textcoords="offset points", xytext=(6, 5),
                      color="#4a4460")
    _ax1.axhline(OCEAN_TIME, color=C_BG, linestyle=":", linewidth=1.3)
    _ax1.text(_swing.max(), OCEAN_TIME * 1.03, "$T$", ha="right", fontsize=8,
              color=C_BG)
    _ax1.set_xlabel(r"how much the ocean moves $\rho$")
    _ax1.set_ylabel("ocean memory $\\tau_S$ (TU)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s1_note(FAST_TIME, FEEDBACK, OCEAN_TIME, SLOW_TIME, mo):
    mo.md(
        rf"""
    The atmosphere forgets in **{FAST_TIME:.2f}** time units; the ocean in
    **{SLOW_TIME:.1f}** — {SLOW_TIME / FAST_TIME:.0f} times longer, and of order the
    relaxation time $T = {OCEAN_TIME:g}$. Nothing in the ocean equation is slow except
    that one constant: the *driving* is the same fast, rapidly decorrelating weather. A
    first-order integrator turns white-ish noise into red noise, and that is the whole
    origin of the memory.

    **The right-hand panel is a constraint, not a design choice**, and it decides the
    rest of the chapter. Turning up the coupling $\kappa$ does let the ocean move the
    atmosphere — but the same loop damps the ocean, and its memory goes with it. Both the
    loop gain and the feedback amplitude scale as $\lambda\kappa$, so they cannot be
    separated: **any coupling strong enough to modulate the atmosphere appreciably has
    already destroyed the ocean's memory.**

    The chapter therefore runs at $\kappa = {FEEDBACK:g}$, near the memory end. That is a
    real limitation and it is worth naming: this model can show you what an ocean's
    *memory* buys, and cannot show you what a strongly coupled ocean like ENSO does to
    the atmosphere above it. A positive $\kappa$, incidentally, is not an option at all —
    it is a runaway that collapses the system onto the origin, which `chaoslib` asserts
    as a test.
    """
    )
    return


# ===========================================================================
# Section 2
# ===========================================================================
@app.cell(hide_code=True)
def s2_md(N_CASES, mo):
    mo.md(
        rf"""
    ## 2 · What initialising the ocean buys

    Now the question that defines decadal prediction. Two sets of {N_CASES} forecasts,
    identical in every respect except one:

    * **initialised** — the slow variable is set from the observed ocean state, with a
      small analysis error;
    * **uninitialised** — the slow variable is drawn from the ocean's climatology,
      as though it had never been measured.

    The fast variables get the *same* analysis error in both. So any difference is the
    value of having observed the ocean. This is exactly the contrast between an
    initialised decadal prediction and a free-running projection.
    """
    )
    return


@app.cell(hide_code=True)
def s2_fig(
    C_ANALYSIS, C_BG, C_PERT, C_TRUTH, FAST_CLIM_SD, LEADS, RMSE_INIT_FAST,
    RMSE_INIT_SLOW, RMSE_UNINIT_FAST, RMSE_UNINIT_SLOW, SLOW_CLIM_SD,
    finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("The ocean $S$", "The weather $z$"),
        figsize=(10.2, 4.0),
    )
    for _ax, _init, _uninit, _sd in (
        (_ax0, RMSE_INIT_SLOW, RMSE_UNINIT_SLOW, SLOW_CLIM_SD),
        (_ax1, RMSE_INIT_FAST, RMSE_UNINIT_FAST, FAST_CLIM_SD),
    ):
        _ax.plot(_leads, np.asarray(_init) / _sd, "-", color=C_ANALYSIS,
                 linewidth=2.4, label="ocean initialised")
        _ax.plot(_leads, np.asarray(_uninit) / _sd, "-", color=C_PERT,
                 linewidth=2.4, label="ocean from climatology")
        _ax.axhline(1.0, color=C_BG, linestyle="--", linewidth=1.3)
        _ax.text(_leads[-1], 1.03, "climatological spread", ha="right",
                 fontsize=7.5, color=C_BG)
        _ax.set_xlabel("forecast lead (time units)")
        _ax.set_ylim(0, 1.7)
        _ax.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    _ax0.set_ylabel("RMSE / climatological spread")
    finish_mpl(
        _fig,
        "The same experiment, scored on the slow variable and on the fast one",
    )
    return


@app.cell(hide_code=True)
def s2_note(
    FAST_BEST_EARLY, FAST_BEST_EARLY_LEAD, FAST_LATE_MEAN, FAST_LATE_SD,
    FAST_STANDARD_ERROR, SLOW_HORIZON_INIT, SLOW_HORIZON_UNINIT, mo,
):
    mo.md(
        rf"""
    **For the ocean, initialising is worth about {SLOW_HORIZON_INIT:.0f} time units.**
    The initialised forecast holds its error below 0.7 of a climatological spread out to
    lead {SLOW_HORIZON_INIT:.0f}; the uninitialised one is above that threshold from the
    start, because a draw from climatology is by definition a climatological-sized error.
    That is the whole basis of initialised decadal prediction, in one panel.

    **For the weather, it is worth almost nothing — but not quite nothing.** The
    initialised run is better by {FAST_BEST_EARLY:.2f} of a climatological spread at lead
    {FAST_BEST_EARLY_LEAD:g}, which at this sample size is about
    {FAST_BEST_EARLY / FAST_STANDARD_ERROR:.0f} standard errors and so is real rather
    than noise. By lead 4 it has gone, and thereafter the difference scatters about zero
    (mean {FAST_LATE_MEAN:+.3f}, sd {FAST_LATE_SD:.3f}).

    The honest summary is therefore *a small brief benefit for the weather, against a
    lasting one for the ocean* — and the brief one exists only because the coupling
    $\kappa$ is non-zero. Had the ocean been purely passive the weather panel would show
    exactly nothing, by construction, and would have proved nothing.

    Notice what the right-hand panel does **not** say. It does not say the ocean is
    irrelevant to weather; it says that in *this* model, with the coupling forced to be
    weak by section 1's trade-off, the ocean cannot do much for the atmosphere. In the
    real system ENSO plainly does. The transferable point is the asymmetry — slow
    components buy predictability for slow variables first — not the size of the effect.
    """
    )
    return


# ===========================================================================
# Section 3
# ===========================================================================
@app.cell(hide_code=True)
def s3_md(MODEL_REFERENCE_Z, REFERENCE_Z, mo):
    mo.md(
        rf"""
    ## 3 · Drift

    Everything above used a perfect model. Now break it, gently: the forecast model
    relaxes its ocean towards $z = {MODEL_REFERENCE_Z:g}$ where the truth uses
    ${REFERENCE_Z:g}$. One parameter, wrong by less than half a per cent.

    That small error displaces the model's *climatology* by about one standard deviation
    of $S$. So a forecast initialised from the observed ocean does not stay near it — it
    slides towards the model's own preferred state, at a rate set by the ocean's
    relaxation time. **That sliding is drift**, and it is the central practical problem of
    initialised decadal prediction.
    """
    )
    return


@app.cell(hide_code=True)
def s3_fig(
    C_ANALYSIS, C_BG, C_PERT, C_TRUTH, DRIFT_BIAS, DRIFT_RMSE, LEADS,
    PERFECT_BIAS, PERFECT_RMSE, SLOW_CLIM_SD, finish_mpl, mpl_panels, np,
):
    _leads = np.asarray(LEADS)
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Mean error: the drift itself", "RMSE"),
        figsize=(10.2, 4.0),
    )
    _ax0.plot(_leads, np.asarray(DRIFT_BIAS), "-", color=C_PERT, linewidth=2.4,
              label="imperfect model")
    _ax0.plot(_leads, np.asarray(PERFECT_BIAS), "-", color=C_TRUTH,
              linewidth=2.4, label="perfect model")
    _ax0.axhline(0.0, color=C_BG, linestyle="--", linewidth=1.3)
    _ax0.set_ylabel("mean error in $S$")
    _ax0.legend(fontsize=8.5, framealpha=0.9, loc="lower left")

    _ax1.plot(_leads, np.asarray(DRIFT_RMSE) / SLOW_CLIM_SD, "-", color=C_PERT,
              linewidth=2.4, label="imperfect model")
    _ax1.plot(_leads, np.asarray(PERFECT_RMSE) / SLOW_CLIM_SD, "-",
              color=C_TRUTH, linewidth=2.4, label="perfect model")
    _ax1.axhline(1.0, color=C_BG, linestyle="--", linewidth=1.3)
    _ax1.set_ylabel("RMSE / climatological spread")
    _ax1.legend(fontsize=8.5, framealpha=0.9, loc="lower right")

    for _ax in (_ax0, _ax1):
        _ax.set_xlabel("forecast lead (time units)")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s3_note(
    DRIFT_BIAS, DRIFT_TREND, LEADS, PERFECT_BIAS, PERFECT_TREND, mo, np,
):
    _d = np.asarray(DRIFT_BIAS)
    _p = np.asarray(PERFECT_BIAS)
    mo.md(
        rf"""
    The imperfect model's mean error grows from {_d[0]:+.3f} to **{_d[-1]:+.2f}**, and
    correlates with lead at {DRIFT_TREND:+.2f}. That is the signature of drift: not a
    fixed offset but a systematic, lead-dependent slide.

    /// admonition | The perfect model's mean error is not zero either
        type: warning

    It reaches {_p[-1]:+.2f} by the longest lead and wanders to
    {np.abs(_p).max():.2f} on the way, with a correlation with lead of
    {PERFECT_TREND:+.2f}. Three hundred cases drawn from a single long trajectory is not
    a large *independent* sample — neighbouring cases share weather — so the baseline
    has real scatter.

    What separates drift from that scatter is therefore **not** that one is zero and the
    other is not. It is that the imperfect model's error is
    {abs(_d[-1] / _p[-1]):.0f} times larger at the longest lead and far more strongly
    trended. Quoting a drift curve without its perfect-model control would make the
    scatter look like signal.
    ///

    And drift is **not a bias in the usual sense**. A bias is one number and you subtract
    it. Drift is a function of lead — near zero at initialisation, large at ten years —
    so removing it requires knowing that function, which requires a set of past
    forecasts. That is the subject of section 4, and it is why decadal prediction systems
    are inseparable from their re-forecast archives.
    """
    )
    return


# ===========================================================================
# Section 4
# ===========================================================================
@app.cell(hide_code=True)
def s4_md(N_CASES, mo):
    mo.md(
        rf"""
    ## 4 · How many re-forecasts does it take?

    The correction is simple to state: run the model over many past cases, measure the
    mean error at each lead, and subtract it. The question that decides whether it works
    is how many past cases you need.

    Below, the drift is estimated from the first $n$ cases and applied to the
    {N_CASES // 2} held-out ones — never scored on the cases it was fitted to.
    """
    )
    return


@app.cell(hide_code=True)
def s4_fig(
    CORRECTED_BY_LEAD, CORRECTED_CURVE, C_ANALYSIS, C_BG, C_PERT, C_TRUTH,
    INSAMPLE_RMSE, LEADS, REFORECAST_SIZES, SLOW_CLIM_SD, UNCORRECTED_BY_LEAD,
    UNCORRECTED_RMSE, finish_mpl, mpl_panels, np,
):
    _fig, (_ax0, _ax1) = mpl_panels(
        ncols=2,
        titles=("Correction against re-forecast count",
                "Where the correction acts"),
        figsize=(10.2, 4.0),
    )
    _sizes = np.asarray(REFORECAST_SIZES, dtype=float)
    _curve = np.asarray(CORRECTED_CURVE)
    _ax0.semilogx(_sizes, _curve, "o-", color=C_ANALYSIS, markersize=6,
                  linewidth=2.0, label="drift-corrected")
    _ax0.axhline(UNCORRECTED_RMSE, color=C_PERT, linestyle="--", linewidth=1.8,
                 label="no correction")
    _ax0.axhline(INSAMPLE_RMSE, color=C_BG, linestyle=":", linewidth=1.6,
                 label="in-sample (not achievable)")
    _worse = _curve > UNCORRECTED_RMSE
    if _worse.any():
        _ax0.fill_between(_sizes, UNCORRECTED_RMSE, _curve, where=_worse,
                          color=C_PERT, alpha=0.15, linewidth=0)
        _ax0.text(_sizes[0], UNCORRECTED_RMSE * 1.005,
                  "correction makes it worse", fontsize=8, color=C_PERT)
    _ax0.set_xlabel("re-forecasts used to estimate the drift")
    _ax0.set_ylabel("RMSE in $S$ (held-out cases)")
    _ax0.set_xticks(_sizes)
    _ax0.set_xticklabels([f"{int(v)}" for v in _sizes])
    _ax0.legend(fontsize=8, framealpha=0.9)

    _leads = np.asarray(LEADS)
    _ax1.plot(_leads, np.asarray(UNCORRECTED_BY_LEAD) / SLOW_CLIM_SD, "-",
              color=C_PERT, linewidth=2.4, label="no correction")
    _ax1.plot(_leads, np.asarray(CORRECTED_BY_LEAD) / SLOW_CLIM_SD, "-",
              color=C_ANALYSIS, linewidth=2.4, label="corrected")
    _ax1.axhline(1.0, color=C_BG, linestyle="--", linewidth=1.3)
    _ax1.set_xlabel("forecast lead (time units)")
    _ax1.set_ylabel("RMSE / climatological spread")
    _ax1.legend(fontsize=8.5, framealpha=0.9, loc="lower right")
    finish_mpl(_fig, None)
    return


@app.cell(hide_code=True)
def s4_note(
    CORRECTED_CURVE, INSAMPLE_RMSE, REFORECAST_SIZES, UNCORRECTED_RMSE, mo, np,
):
    _curve = np.asarray(CORRECTED_CURVE)
    _sizes = np.asarray(REFORECAST_SIZES)
    _helps = _sizes[_curve < UNCORRECTED_RMSE]
    _first = int(_helps[0]) if _helps.size else None
    _best = float(_curve.min())
    mo.md(
        rf"""
    **With enough re-forecasts the correction is worth
    {100 * (1 - _best / UNCORRECTED_RMSE):.0f} %** — from {UNCORRECTED_RMSE:.3f} to
    {_best:.3f}. That is a large gain for an operation that requires no new physics and
    no new model, only an archive.

    **With too few, it is worse than not correcting at all.** At
    {int(_sizes[0])} and {int(_sizes[1])} re-forecasts the "corrected" score is *above*
    the uncorrected one: the drift estimate is then mostly sampling noise, and subtracting
    noise adds noise. The correction first pays at about **{_first} re-forecasts** here.

    That is the operational answer to a question that sounds bureaucratic and is not.
    A decadal prediction system's re-forecast set is not documentation — it is a
    *component of the forecast*, and one that must be large enough or it actively harms
    the product. It is also expensive: each re-forecast is a full model integration over
    the whole forecast range.

    ### And why the correction must be cross-validated

    Estimating the drift on the same cases you then score gives {INSAMPLE_RMSE:.3f} —
    {100 * (1 - INSAMPLE_RMSE / _best):.1f} % better than anything actually achievable.
    The estimate has absorbed some of those cases' own noise, and subtracting it removes
    the noise along with the drift.

    That number is small here, and it is small *because* the held-out set is large. In a
    real decadal system, where re-forecasts number in the tens and each is enormously
    expensive, the same effect is far larger — and a skill claim built on an in-sample
    drift correction is measuring its own fitting procedure.
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

    **The ocean's memory is borrowed, not intrinsic.** A first-order integrator driven by
    fast weather has a long autocorrelation time even though nothing driving it is slow.
    That is where interannual-to-decadal predictability comes from.

    **Initialising a slow component buys predictability for slow variables.** Here about
    eight time units for the ocean — against a brief, small benefit for the weather that
    is real but gone within four. Slow components pay off on their own timescale first.

    **Drift is not bias.** It is a lead-dependent slide from the observed state towards
    the model's own climatology, produced here by a parameter error of less than half a
    per cent. A single subtracted number cannot remove it.

    **Removing drift requires an archive, and the archive can be too small.** With too
    few re-forecasts the correction adds more noise than it removes, and a system is
    better off uncorrected. The re-forecast set is part of the forecast system.

    **Cross-validate, or you are scoring your own fitting procedure.** An in-sample drift
    correction flatters itself, by a little here and by much more where re-forecasts are
    scarce.

    **And a limitation worth carrying forward.** In this model, coupling strong enough for
    the ocean to move the atmosphere destroys the ocean's memory — the two cannot be had
    together, and the chapter runs at the memory end. What transfers is the asymmetry
    between fast and slow, not the magnitude of the ocean's influence on the atmosphere.

    ### Try this

    1. Section 1's trade-off curve bends sharply. Estimate the coupling at which the
       ocean's memory drops below ten time units, and say what that implies for a model
       where the ocean genuinely does drive the atmosphere.
    2. In section 2 the uninitialised ocean forecast starts *above* one climatological
       spread. Why is that the correct starting value, and what would it mean if it
       started below?
    3. Section 3's drift comes from a parameter error of under half a per cent. Estimate
       the drift you would expect from an error ten times larger, and say whether the
       correction of section 4 would still work.
    4. The correction first pays at around twenty re-forecasts here. What property of the
       system sets that number, and would a noisier ocean need more or fewer?

    ### Where this goes next

    **Chapter 25** replaces the ocean's internal variability with a one-way *ramp* in the
    forcing, which is climate projection, and separates the forced response from internal
    variability. **Chapter 27** asks what happens when a slow variable is pushed past a
    bifurcation rather than around a loop.

    ### Further reading

    - Hasselmann (1976), stochastic climate models *[citation needed]*
    - Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on decadal
      prediction *[citation needed: chapter]*
    - Meehl et al. (2021), on decadal prediction systems and drift *[citation needed]*
    - Boer et al., on the Decadal Climate Prediction Project protocol *[citation needed]*
    """
    )
    return


if __name__ == "__main__":
    app.run()
