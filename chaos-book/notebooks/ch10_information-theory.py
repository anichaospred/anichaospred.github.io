# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 10 -- Information theory and predictability.

How much does this forecast tell us that climatology did not? The answer is a
number that does not depend on your choice of norm -- and that decays at
lambda_1, not at the entropy rate.

Part III of *An Interactive Chaos and Predictability Textbook*.

To edit:   marimo edit notebooks/ch10_information-theory.py
To export: make nb-one NB=ch10_information-theory
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 10: Information and Predictability")


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

    from chaoslib import information, plotting

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
        finish_mpl,
        information,
        mo,
        mpl_panels,
        np,
    )


# ---------------------------------------------------------------------------
# Precomputed information curves
# ---------------------------------------------------------------------------
@app.cell
def information_data():
    # From scripts/generate_ch10_data.py: 80 ensemble forecasts of 500 members
    # each, for Lorenz 63 and Lorenz 96, with the relative entropy of the
    # forecast distribution against climatology as a function of lead.
    #
    # Scalar observables, not the full state, and Section 3 shows why: a
    # 500-member ensemble on a 2.06-dimensional attractor has a near-singular
    # covariance in three dimensions, so det Sigma_f -- which the dispersion
    # term depends on logarithmically -- is set by the regularisation. Both
    # regularisations are recorded so the reader can watch the answer move.
    L63_LAMBDA1 = 0.9056
    L63_HKS = 0.901
    L63_VARIABLES = (0, 2)
    L63_TIMES = (
        0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 1.2000,
        1.4000, 1.6000, 1.8000, 2.0000, 2.2000, 2.4000, 2.6000,
        2.8000, 3.0000, 3.2000, 3.4000, 3.6000, 3.8000, 4.0000,
        4.2000, 4.4000, 4.6000, 4.8000, 5.0000, 5.2000, 5.4000,
        5.6000, 5.8000, 6.0000, 6.2000, 6.4000, 6.6000, 6.8000,
        7.0000, 7.2000, 7.4000, 7.6000, 7.8000, 8.0000, 8.2000,
        8.4000, 8.6000, 8.8000, 9.0000, 9.2000, 9.4000, 9.6000,
        9.8000, 10.0000, 10.2000, 10.4000, 10.6000, 10.8000, 11.0000,
        11.2000, 11.4000, 11.6000, 11.8000, 12.0000,
    )
    L63_V0_TOTAL = (
        8.986236e+00, 8.987521e+00, 8.524991e+00, 8.528267e+00,
        8.347251e+00, 8.255763e+00, 7.787805e+00, 7.821621e+00,
        7.597992e+00, 7.445948e+00, 7.182958e+00, 7.304079e+00,
        7.180567e+00, 6.917075e+00, 6.980458e+00, 6.654906e+00,
        6.247201e+00, 6.408060e+00, 6.243708e+00, 6.084623e+00,
        5.776902e+00, 5.493290e+00, 5.504489e+00, 5.565924e+00,
        5.025703e+00, 4.733747e+00, 4.459291e+00, 4.382684e+00,
        4.153754e+00, 4.034475e+00, 3.876657e+00, 3.744171e+00,
        3.312471e+00, 3.064950e+00, 2.795850e+00, 2.719291e+00,
        2.280059e+00, 2.256062e+00, 1.962983e+00, 1.691956e+00,
        1.459677e+00, 1.304293e+00, 1.123186e+00, 9.819480e-01,
        7.305385e-01, 6.272995e-01, 5.629938e-01, 5.388172e-01,
        4.395135e-01, 4.148194e-01, 3.618986e-01, 3.011111e-01,
        2.847904e-01, 2.614668e-01, 2.621755e-01, 2.375211e-01,
        2.080344e-01, 2.298518e-01, 1.971171e-01, 1.886813e-01,
        1.582327e-01,
    )
    L63_V0_SIGNAL = (
        5.082669e-01, 4.543347e-01, 4.798124e-01, 5.314877e-01,
        5.356887e-01, 4.241007e-01, 4.685591e-01, 6.217606e-01,
        4.968487e-01, 4.271719e-01, 4.845548e-01, 6.287189e-01,
        4.882946e-01, 4.144087e-01, 5.484887e-01, 5.673909e-01,
        3.651831e-01, 4.909438e-01, 6.490530e-01, 5.045817e-01,
        4.367375e-01, 4.516533e-01, 5.835537e-01, 5.452059e-01,
        3.958259e-01, 4.145616e-01, 5.333439e-01, 5.933253e-01,
        3.858891e-01, 4.238059e-01, 5.678023e-01, 5.001288e-01,
        3.108166e-01, 5.048564e-01, 5.213659e-01, 4.274143e-01,
        3.266941e-01, 4.496867e-01, 4.351178e-01, 3.118355e-01,
        2.972170e-01, 3.060849e-01, 3.325188e-01, 2.336233e-01,
        2.044066e-01, 1.686785e-01, 1.891292e-01, 1.191965e-01,
        8.934894e-02, 8.756811e-02, 9.005722e-02, 2.906555e-02,
        2.753109e-02, 3.708744e-02, 2.783024e-02, 7.215894e-03,
        1.536867e-02, 3.582261e-02, 6.769260e-03, 3.003389e-03,
        1.669670e-02,
    )
    L63_V0_DISPERSION = (
        8.477969e+00, 8.533187e+00, 8.045178e+00, 7.996779e+00,
        7.811563e+00, 7.831662e+00, 7.319246e+00, 7.199860e+00,
        7.101143e+00, 7.018776e+00, 6.698403e+00, 6.675360e+00,
        6.692272e+00, 6.502666e+00, 6.431969e+00, 6.087515e+00,
        5.882018e+00, 5.917117e+00, 5.594655e+00, 5.580041e+00,
        5.340164e+00, 5.041636e+00, 4.920935e+00, 5.020718e+00,
        4.629877e+00, 4.319185e+00, 3.925947e+00, 3.789359e+00,
        3.767865e+00, 3.610669e+00, 3.308855e+00, 3.244042e+00,
        3.001654e+00, 2.560094e+00, 2.274484e+00, 2.291877e+00,
        1.953365e+00, 1.806376e+00, 1.527865e+00, 1.380120e+00,
        1.162460e+00, 9.982086e-01, 7.906670e-01, 7.483247e-01,
        5.261320e-01, 4.586210e-01, 3.738645e-01, 4.196207e-01,
        3.501645e-01, 3.272513e-01, 2.718414e-01, 2.720455e-01,
        2.572594e-01, 2.243794e-01, 2.343452e-01, 2.303052e-01,
        1.926657e-01, 1.940292e-01, 1.903478e-01, 1.856779e-01,
        1.415360e-01,
    )
    L63_V0_SLOPE = -0.92699
    L63_V2_TOTAL = (
        9.069329e+00, 9.005432e+00, 8.588324e+00, 7.928319e+00,
        7.972136e+00, 7.920295e+00, 7.522784e+00, 7.237628e+00,
        7.075125e+00, 7.020155e+00, 6.859355e+00, 6.827820e+00,
        6.434466e+00, 6.623635e+00, 6.269398e+00, 6.184238e+00,
        6.262843e+00, 5.847103e+00, 5.684353e+00, 5.641628e+00,
        5.321155e+00, 5.066288e+00, 4.862910e+00, 4.953636e+00,
        4.797285e+00, 4.444687e+00, 4.276772e+00, 4.087609e+00,
        3.840334e+00, 3.551196e+00, 3.428604e+00, 3.169301e+00,
        3.139386e+00, 2.773975e+00, 2.728031e+00, 2.292400e+00,
        2.250795e+00, 2.064303e+00, 1.754770e+00, 1.703890e+00,
        1.536342e+00, 1.334015e+00, 1.340266e+00, 9.967051e-01,
        1.048028e+00, 7.725789e-01, 7.754464e-01, 6.030178e-01,
        5.910546e-01, 5.385205e-01, 5.081350e-01, 4.725990e-01,
        4.207175e-01, 3.710290e-01, 4.279792e-01, 3.685915e-01,
        3.439440e-01, 3.001818e-01, 3.187456e-01, 2.913901e-01,
        2.579616e-01,
    )
    L63_V2_SIGNAL = (
        5.078750e-01, 5.235885e-01, 5.205666e-01, 4.326429e-01,
        5.028532e-01, 4.903169e-01, 4.709650e-01, 5.456356e-01,
        4.960167e-01, 4.433616e-01, 4.750002e-01, 5.480226e-01,
        3.685327e-01, 4.757876e-01, 4.553117e-01, 5.138811e-01,
        5.217737e-01, 5.020976e-01, 5.066702e-01, 5.009970e-01,
        3.855328e-01, 4.375546e-01, 5.393230e-01, 4.525876e-01,
        4.226569e-01, 4.791506e-01, 6.195096e-01, 5.541372e-01,
        4.452730e-01, 4.247135e-01, 5.493087e-01, 4.085025e-01,
        4.796187e-01, 4.355247e-01, 5.621245e-01, 4.285266e-01,
        4.287088e-01, 4.951298e-01, 3.584557e-01, 3.818870e-01,
        3.853397e-01, 3.327707e-01, 4.529694e-01, 2.701975e-01,
        3.645001e-01, 3.262267e-01, 3.002052e-01, 2.442842e-01,
        2.562734e-01, 2.655074e-01, 2.496363e-01, 2.136423e-01,
        2.093589e-01, 2.012695e-01, 2.143093e-01, 1.673079e-01,
        1.893731e-01, 1.813947e-01, 1.608295e-01, 1.541005e-01,
        1.645863e-01,
    )
    L63_V2_DISPERSION = (
        8.561454e+00, 8.481843e+00, 8.067758e+00, 7.495676e+00,
        7.469283e+00, 7.429978e+00, 7.051819e+00, 6.691993e+00,
        6.579108e+00, 6.576794e+00, 6.384355e+00, 6.279798e+00,
        6.065933e+00, 6.147848e+00, 5.814086e+00, 5.670357e+00,
        5.741069e+00, 5.345006e+00, 5.177682e+00, 5.140631e+00,
        4.935622e+00, 4.628733e+00, 4.323587e+00, 4.501048e+00,
        4.374628e+00, 3.965536e+00, 3.657262e+00, 3.533472e+00,
        3.395061e+00, 3.126482e+00, 2.879295e+00, 2.760799e+00,
        2.659767e+00, 2.338450e+00, 2.165906e+00, 1.863873e+00,
        1.822086e+00, 1.569173e+00, 1.396315e+00, 1.322003e+00,
        1.151002e+00, 1.001245e+00, 8.872969e-01, 7.265076e-01,
        6.835279e-01, 4.463522e-01, 4.752413e-01, 3.587336e-01,
        3.347812e-01, 2.730132e-01, 2.584987e-01, 2.589567e-01,
        2.113586e-01, 1.697595e-01, 2.136699e-01, 2.012836e-01,
        1.545708e-01, 1.187871e-01, 1.579161e-01, 1.372897e-01,
        9.337530e-02,
    )
    L63_V2_SLOPE = -0.89064
    L96_LAMBDA1 = 1.67
    L96_HKS = 10.21
    L96_VARIABLES = (0, 20)
    L96_TIMES = (
        0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 1.2000,
        1.4000, 1.6000, 1.8000, 2.0000, 2.2000, 2.4000, 2.6000,
        2.8000, 3.0000, 3.2000, 3.4000, 3.6000, 3.8000, 4.0000,
        4.2000, 4.4000, 4.6000, 4.8000, 5.0000, 5.2000, 5.4000,
        5.6000, 5.8000, 6.0000, 6.2000, 6.4000, 6.6000, 6.8000,
        7.0000,
    )
    L96_V0_TOTAL = (
        8.292704e+00, 8.035652e+00, 7.609882e+00, 7.369974e+00,
        6.899311e+00, 6.741313e+00, 6.599908e+00, 6.137130e+00,
        5.786670e+00, 5.777545e+00, 5.385202e+00, 5.084865e+00,
        4.575422e+00, 4.099164e+00, 3.691695e+00, 3.554603e+00,
        3.129243e+00, 2.827480e+00, 2.482153e+00, 2.317297e+00,
        2.036818e+00, 1.674816e+00, 1.409453e+00, 1.097966e+00,
        9.339301e-01, 8.175570e-01, 6.436041e-01, 4.842040e-01,
        4.005612e-01, 3.535450e-01, 3.006054e-01, 2.124421e-01,
        1.595783e-01, 1.533728e-01, 1.299651e-01, 8.217937e-02,
    )
    L96_V0_SIGNAL = (
        5.422153e-01, 5.152947e-01, 4.552801e-01, 5.263950e-01,
        4.625999e-01, 5.501250e-01, 5.648427e-01, 3.866052e-01,
        3.845838e-01, 6.170483e-01, 5.717536e-01, 6.110890e-01,
        5.218009e-01, 4.123230e-01, 4.259319e-01, 4.627965e-01,
        4.518713e-01, 5.033107e-01, 3.951903e-01, 4.977816e-01,
        5.235592e-01, 4.070536e-01, 3.313037e-01, 3.155163e-01,
        3.085079e-01, 2.647836e-01, 2.093586e-01, 1.795204e-01,
        1.915637e-01, 1.727624e-01, 1.479807e-01, 1.184455e-01,
        1.029788e-01, 1.010359e-01, 8.172112e-02, 5.420865e-02,
    )
    L96_V0_DISPERSION = (
        7.750489e+00, 7.520357e+00, 7.154602e+00, 6.843579e+00,
        6.436711e+00, 6.191188e+00, 6.035065e+00, 5.750525e+00,
        5.402086e+00, 5.160497e+00, 4.813448e+00, 4.473776e+00,
        4.053621e+00, 3.686841e+00, 3.265763e+00, 3.091807e+00,
        2.677372e+00, 2.324169e+00, 2.086962e+00, 1.819515e+00,
        1.513259e+00, 1.267762e+00, 1.078149e+00, 7.824493e-01,
        6.254222e-01, 5.527734e-01, 4.342455e-01, 3.046836e-01,
        2.089975e-01, 1.807826e-01, 1.526247e-01, 9.399659e-02,
        5.659957e-02, 5.233695e-02, 4.824399e-02, 2.797073e-02,
    )
    L96_V0_SLOPE = -1.59776
    L96_V20_TOTAL = (
        8.167230e+00, 7.966636e+00, 7.646520e+00, 7.396404e+00,
        6.938935e+00, 6.643105e+00, 6.478458e+00, 6.114209e+00,
        5.838708e+00, 5.459799e+00, 5.092646e+00, 4.817706e+00,
        4.340607e+00, 4.082511e+00, 3.738957e+00, 3.267736e+00,
        2.915588e+00, 2.666707e+00, 2.409060e+00, 2.020027e+00,
        1.914969e+00, 1.608278e+00, 1.377976e+00, 1.079999e+00,
        9.329486e-01, 7.057500e-01, 5.937848e-01, 4.507774e-01,
        3.375451e-01, 3.277647e-01, 2.347518e-01, 1.457463e-01,
        1.083915e-01, 1.007827e-01, 9.437409e-02, 9.047495e-02,
    )
    L96_V20_SIGNAL = (
        4.729754e-01, 4.778901e-01, 5.125088e-01, 5.699167e-01,
        5.163788e-01, 4.951525e-01, 5.519331e-01, 4.936112e-01,
        5.193752e-01, 4.470380e-01, 4.611985e-01, 4.589057e-01,
        5.072163e-01, 4.557260e-01, 4.707647e-01, 4.051806e-01,
        3.916832e-01, 4.468916e-01, 4.702882e-01, 3.135254e-01,
        4.400244e-01, 4.334647e-01, 4.146023e-01, 3.040734e-01,
        3.318726e-01, 2.945774e-01, 2.595021e-01, 2.101007e-01,
        1.710140e-01, 1.811063e-01, 1.310713e-01, 9.598166e-02,
        6.634676e-02, 7.180649e-02, 7.007090e-02, 6.702566e-02,
    )
    L96_V20_DISPERSION = (
        7.694254e+00, 7.488746e+00, 7.134011e+00, 6.826487e+00,
        6.422556e+00, 6.147953e+00, 5.926525e+00, 5.620598e+00,
        5.319333e+00, 5.012761e+00, 4.631447e+00, 4.358800e+00,
        3.833390e+00, 3.626785e+00, 3.268193e+00, 2.862556e+00,
        2.523905e+00, 2.219815e+00, 1.938772e+00, 1.706501e+00,
        1.474944e+00, 1.174814e+00, 9.633741e-01, 7.759256e-01,
        6.010761e-01, 4.111726e-01, 3.342827e-01, 2.406767e-01,
        1.665311e-01, 1.466584e-01, 1.036804e-01, 4.976459e-02,
        4.204477e-02, 2.897620e-02, 2.430319e-02, 2.344929e-02,
    )
    L96_V20_SLOPE = -1.62790
    FLOOR_TAGS = ('1EM12', '1EM06')
    FLOOR_VALUES = (1e-12, 1e-06)
    MI_LAGS = (0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
    MI_BINS = (16, 32, 64, 128)
    L63_FULL_1EM12 = (
        2.645396e+01, 2.929458e+01, 3.177406e+01, 3.228902e+01,
        3.213646e+01, 3.170478e+01, 3.125014e+01, 3.115885e+01,
        3.072742e+01, 3.038793e+01, 3.020687e+01, 2.996637e+01,
        2.948802e+01, 2.937713e+01, 2.894249e+01, 2.857364e+01,
        2.802628e+01, 2.742936e+01, 2.676424e+01, 2.640805e+01,
        2.575110e+01, 2.536581e+01, 2.494212e+01, 2.425543e+01,
        2.368957e+01, 2.260688e+01, 2.139251e+01, 2.052153e+01,
        1.945297e+01, 1.865463e+01, 1.781197e+01, 1.681473e+01,
        1.565248e+01, 1.433065e+01, 1.308652e+01, 1.165306e+01,
        1.102775e+01, 9.578442e+00, 8.577415e+00, 7.702544e+00,
        6.612304e+00, 5.885471e+00, 5.137709e+00, 4.227570e+00,
        3.756737e+00, 3.005611e+00, 2.589733e+00, 2.183944e+00,
        2.071836e+00, 1.847957e+00, 1.640419e+00, 1.551862e+00,
        1.380728e+00, 1.268798e+00, 1.235960e+00, 1.151073e+00,
        1.070646e+00, 9.581899e-01, 9.543354e-01, 8.936765e-01,
        8.031240e-01,
    )
    L63_FULL_1EM12_SLOPE = -1.67721
    L63_FULL_1EM06 = (
        2.541153e+01, 2.550473e+01, 2.508944e+01, 2.464422e+01,
        2.462602e+01, 2.419798e+01, 2.394199e+01, 2.395168e+01,
        2.368172e+01, 2.342857e+01, 2.336727e+01, 2.339549e+01,
        2.298875e+01, 2.286969e+01, 2.268519e+01, 2.270210e+01,
        2.236418e+01, 2.190646e+01, 2.178963e+01, 2.164857e+01,
        2.118415e+01, 2.091110e+01, 2.085191e+01, 2.044620e+01,
        1.991859e+01, 1.921781e+01, 1.849803e+01, 1.779837e+01,
        1.705605e+01, 1.651622e+01, 1.600129e+01, 1.518448e+01,
        1.430519e+01, 1.317185e+01, 1.223103e+01, 1.114530e+01,
        1.053923e+01, 9.148292e+00, 8.208508e+00, 7.468166e+00,
        6.444016e+00, 5.748773e+00, 5.046837e+00, 4.164210e+00,
        3.743495e+00, 2.993653e+00, 2.556196e+00, 2.161412e+00,
        2.061920e+00, 1.823547e+00, 1.630825e+00, 1.541260e+00,
        1.362618e+00, 1.246080e+00, 1.221354e+00, 1.121747e+00,
        1.060217e+00, 9.545546e-01, 9.484734e-01, 8.918915e-01,
        8.029113e-01,
    )
    L63_FULL_1EM06_SLOPE = -0.94023
    MI_PLUGIN = (
        2.575813e+00, 7.553438e-01, 3.576372e-01, 2.816984e-01,
        1.914444e-01, 1.665994e-01, 3.637780e-02, 6.664951e-02,
        6.429619e-02, 9.374744e-03, 3.261974e+00, 9.133834e-01,
        4.576415e-01, 3.915137e-01, 2.690538e-01, 1.947163e-01,
        5.821162e-02, 7.454178e-02, 7.188842e-02, 1.296873e-02,
        3.953377e+00, 1.018581e+00, 5.290556e-01, 4.863308e-01,
        3.296516e-01, 2.273966e-01, 7.845013e-02, 8.414255e-02,
        8.009798e-02, 1.954943e-02, 4.645896e+00, 1.091601e+00,
        5.851379e-01, 5.699631e-01, 4.000274e-01, 2.718023e-01,
        1.104684e-01, 1.062950e-01, 1.011668e-01, 3.938007e-02,
    )
    MI_CORRECTED = (
        2.575838e+00, 7.551777e-01, 3.574151e-01, 2.813610e-01,
        1.911018e-01, 1.662211e-01, 3.600781e-02, 6.626737e-02,
        6.391720e-02, 8.993533e-03, 3.262027e+00, 9.127257e-01,
        4.567396e-01, 3.901726e-01, 2.676749e-01, 1.932250e-01,
        5.669941e-02, 7.299627e-02, 7.031126e-02, 1.139284e-02,
        3.953484e+00, 1.016018e+00, 5.254398e-01, 4.812630e-01,
        3.244006e-01, 2.215788e-01, 7.252686e-02, 7.808789e-02,
        7.390834e-02, 1.338540e-02, 4.646111e+00, 1.081658e+00,
        5.706576e-01, 5.509534e-01, 3.805431e-01, 2.495119e-01,
        8.783099e-02, 8.309709e-02, 7.794284e-02, 1.591351e-02,
    )
    return (
        FLOOR_TAGS,
        FLOOR_VALUES,
        L63_FULL_1EM06,
        L63_FULL_1EM06_SLOPE,
        L63_FULL_1EM12,
        L63_FULL_1EM12_SLOPE,
        L63_HKS,
        L63_LAMBDA1,
        L63_TIMES,
        L63_V0_DISPERSION,
        L63_V0_SIGNAL,
        L63_V0_SLOPE,
        L63_V0_TOTAL,
        L63_V2_DISPERSION,
        L63_V2_SIGNAL,
        L63_V2_SLOPE,
        L63_V2_TOTAL,
        L63_VARIABLES,
        L96_HKS,
        L96_LAMBDA1,
        L96_TIMES,
        L96_V0_DISPERSION,
        L96_V0_SIGNAL,
        L96_V0_SLOPE,
        L96_V0_TOTAL,
        L96_V20_DISPERSION,
        L96_V20_SIGNAL,
        L96_V20_SLOPE,
        L96_V20_TOTAL,
        L96_VARIABLES,
        MI_BINS,
        MI_CORRECTED,
        MI_LAGS,
        MI_PLUGIN,
    )


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------
@app.cell
def helpers(
    L63_HKS,
    L63_LAMBDA1,
    L63_TIMES,
    L63_V0_DISPERSION,
    L63_V0_SIGNAL,
    L63_V0_SLOPE,
    L63_V0_TOTAL,
    L63_V2_DISPERSION,
    L63_V2_SIGNAL,
    L63_V2_SLOPE,
    L63_V2_TOTAL,
    L96_HKS,
    L96_LAMBDA1,
    L96_TIMES,
    L96_V0_DISPERSION,
    L96_V0_SIGNAL,
    L96_V0_SLOPE,
    L96_V0_TOTAL,
    L96_V20_DISPERSION,
    L96_V20_SIGNAL,
    L96_V20_SLOPE,
    L96_V20_TOTAL,
    np,
):
    CURVES = {
        ("L63", 0): (L63_V0_TOTAL, L63_V0_SIGNAL, L63_V0_DISPERSION, L63_V0_SLOPE),
        ("L63", 2): (L63_V2_TOTAL, L63_V2_SIGNAL, L63_V2_DISPERSION, L63_V2_SLOPE),
        ("L96", 0): (L96_V0_TOTAL, L96_V0_SIGNAL, L96_V0_DISPERSION, L96_V0_SLOPE),
        ("L96", 20): (
            L96_V20_TOTAL, L96_V20_SIGNAL, L96_V20_DISPERSION, L96_V20_SLOPE
        ),
    }
    SYSTEMS = {
        "L63": dict(times=L63_TIMES, lambda1=L63_LAMBDA1, h_ks=L63_HKS,
                    label="Lorenz 63", unit="MTU", variables=(0, 2),
                    names={0: "x", 2: "z"}),
        "L96": dict(times=L96_TIMES, lambda1=L96_LAMBDA1, h_ks=L96_HKS,
                    label="Lorenz 96", unit="time units", variables=(0, 20),
                    names={0: "$x_1$", 20: "$x_{21}$"}),
    }

    def curve(system, variable):
        total, signal, dispersion, slope = CURVES[(system, variable)]
        spec = SYSTEMS[system]
        return dict(
            t=np.asarray(spec["times"], dtype=float),
            total=np.asarray(total, dtype=float),
            signal=np.asarray(signal, dtype=float),
            dispersion=np.asarray(dispersion, dtype=float),
            slope=slope, lambda1=spec["lambda1"], h_ks=spec["h_ks"],
            label=spec["label"], unit=spec["unit"],
            name=spec["names"][variable],
        )

    return CURVES, SYSTEMS, curve


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 10 · Information Theory and Predictability

    **Part III — Quantifying chaos and predictability.**

    **The forecasting question.** Every measure of predictability so far has
    been an error: a distance between a forecast and the truth, in some norm.
    Chapter 16 showed that the choice of norm is not innocent — it changes which
    perturbation is "fastest-growing" by tens of degrees. Chapter 9 showed that
    an error's growth rate depends on its amplitude. Both are properties of the
    measuring instrument as much as of the atmosphere.

    There is a way of asking the question that has neither problem. A forecast
    is worth something only if it tells you more than climatology already did,
    and "how much more" has a canonical answer:

    $$D\bigl(p_{\rm forecast}\,\|\,p_{\rm climatology}\bigr)
      = \int p_f \ln\frac{p_f}{p_c}\,dx ,$$

    the **relative entropy** of the forecast distribution from the
    climatological one. It is measured in nats, it is zero exactly when the
    forecast says nothing climatology did not, and — the property that earns it
    a chapter — it is **invariant under any invertible change of variables**.
    Rescale your units, rotate your basis, reweight your variables: the number
    does not move. Section 4 verifies that to machine precision alongside an RMS
    error that varies by a factor of 371 under the same transformations.

    For Gaussians it has a closed form that splits in two:

    $$D = \underbrace{\tfrac12(\mu_c-\mu_f)^{\!\top}\Sigma_c^{-1}(\mu_c-\mu_f)}_{\text{signal}}
      + \underbrace{\tfrac12\Bigl[\operatorname{tr}(\Sigma_c^{-1}\Sigma_f) - k
        + \ln\tfrac{\det\Sigma_c}{\det\Sigma_f}\Bigr]}_{\text{dispersion}} .$$

    **Signal** is "my forecast mean differs from climatology". **Dispersion** is
    "my forecast is sharper than climatology". A forecast can be informative
    either way, and it turns out one of them does nearly all the work.

    | Section | The question |
    |---|---|
    | 1 | How much information does a forecast carry, and of which kind? |
    | 2 | How fast does it decay — and is the rate $\lambda_1$ or $h_{KS}$? |
    | 3 | Why not do this on the full state vector? |
    | 4 | Is it really norm-independent? |
    | 5 | Can mutual information be estimated at all? |
    """
    )
    return


# ===========================================================================
# 1. Signal and dispersion
# ===========================================================================
@app.cell(hide_code=True)
def s1_controls(mo):
    which = mo.ui.dropdown(
        options={
            "Lorenz 63 — x": "L63:0",
            "Lorenz 63 — z": "L63:2",
            "Lorenz 96 — site 1": "L96:0",
            "Lorenz 96 — site 21": "L96:20",
        },
        value="Lorenz 63 — x",
        label="system and observable",
    )
    return (which,)


@app.cell(hide_code=True)
def s1_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    curve,
    finish_mpl,
    mo,
    mpl_panels,
    np,
    which,
):
    _key, _var = str(which.value).split(":")
    _c = curve(_key, int(_var))

    _fig, _ax = mpl_panels(
        3,
        titles=("Forecast information vs lead", "The two components",
                "Dispersion's share"),
        height=3.5,
    )
    _ax[0].plot(_c["t"], _c["total"], linewidth=2.0, color=C_TRUTH,
                label="total $D$")
    _ax[0].axhline(0.0, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[0].set_xlabel(f"lead ({_c['unit']})")
    _ax[0].set_ylabel("relative entropy (nats)")
    _ax[0].legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    _ax[1].plot(_c["t"], _c["dispersion"], linewidth=1.8, color=C_SPREAD,
                label="dispersion")
    _ax[1].plot(_c["t"], _c["signal"], linewidth=1.8, color=C_MEAN,
                label="signal")
    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[1].set_xlabel(f"lead ({_c['unit']})")
    _ax[1].set_ylabel("nats")
    _ax[1].legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    _share = _c["dispersion"] / np.maximum(_c["total"], 1e-12)
    _ax[2].plot(_c["t"], 100.0 * _share, linewidth=1.8, color=C_SPREAD)
    _ax[2].axhline(100.0, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[2].set_xlabel(f"lead ({_c['unit']})")
    _ax[2].set_ylabel("dispersion as % of total")
    _ax[2].set_ylim(0, 110)
    finish_mpl(_fig, suptitle=f"{_c['label']}, observable {_c['name']}, "
                              f"80 forecasts × 500 members")

    _i_half = int(np.argmin(np.abs(_c["total"] - 0.5 * _c["total"][0])))
    _rows = "\n".join(
        f"| {_c['t'][i]:.2f} | {_c['total'][i]:.3f} | {_c['signal'][i]:.3f} | "
        f"{_c['dispersion'][i]:.3f} | {100 * _share[i]:.0f}% |"
        for i in (0, _i_half // 2, _i_half, min(len(_c['t']) - 1, 2 * _i_half))
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 1 · Two ways to be informative, and only one of them matters

    Eighty ensemble forecasts, 500 members each, started from a spread of
    $10^{-3}$ at points spread around the attractor. At each lead, fit a
    Gaussian to the forecast distribution of one observable, compare it against
    that observable's climatology, and split the result."""
        ),
        mo.hstack([which], justify="start"),
        _fig,
        mo.md(
            f"""
| lead ({_c['unit']}) | total | signal | dispersion | dispersion share |
|---|---|---|---|---|
{_rows}

**The dispersion term does essentially all the work.** At lead zero the forecast
carries {_c['total'][0]:.2f} nats, of which {_c['dispersion'][0]:.2f} —
{100 * _share[0]:.0f}% — is dispersion. That is the statement "I know where the
system is to within $10^{{-3}}$, and climatology only knows it to within its
whole spread", and it is worth
$\\tfrac12\\ln(v_c/v_f)$ nats.

**The signal term stays nearly constant and nearly small**, hovering around
{np.mean(_c['signal'][:len(_c['signal'])//2]):.2f} nats. That is not a bug: a
forecast started at a random point on the attractor has, on average, no
particular reason for its mean to sit far from the climatological mean in units
of the climatological spread. The signal term earns its keep in *forced*
problems — an ENSO forecast, a seasonal outlook — where the whole point is that
the mean is displaced. For an initial-value problem on a stationary attractor
it is the sharpness that carries the information.

That is a useful thing to know before quoting a single predictability number,
because the two components decay differently and mean different things to a
user. Chapter 17's forecast-value discussion is where that distinction gets
spent.
"""
        ),
    ])
    return


# ===========================================================================
# 2. The decay rate
# ===========================================================================
@app.cell(hide_code=True)
def s2_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    curve,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _cases = (("L63", 0), ("L63", 2), ("L96", 0), ("L96", 20))
    _fig, _ax = mpl_panels(
        3,
        titles=("Lorenz 63: information vs lead", "Lorenz 96: the same",
                "Measured rate against the two candidates"),
        height=3.5,
    )
    for _panel, _key in ((0, "L63"), (1, "L96")):
        for (_k, _v), _colour in zip(
            [c for c in _cases if c[0] == _key], (C_TRUTH, C_MEAN)
        ):
            _c = curve(_k, _v)
            _ax[_panel].plot(_c["t"], _c["total"], linewidth=1.8, color=_colour,
                             label=f"observable {_c['name']}")
            # The -lambda_1 line anchored at the start of the linear stretch.
            _i0 = int(np.argmin(np.abs(_c["total"] - 0.9 * _c["total"][0])))
            _ax[_panel].plot(
                _c["t"], _c["total"][_i0] - _c["lambda1"] * (_c["t"] - _c["t"][_i0]),
                color=C_SAT, linewidth=1.2, linestyle="--",
                label=f"slope $-\\lambda_1$ = {-_c['lambda1']:.2f}" if _v == 0 else None,
            )
        _c = curve(_key, 0)
        _ax[_panel].set_xlabel(f"lead ({_c['unit']})")
        _ax[_panel].set_ylabel("nats")
        _ax[_panel].set_ylim(0, 1.15 * _c["total"][0])
        _ax[_panel].legend(loc="upper right", fontsize=6.0, framealpha=0.9)

    _labels, _measured, _lam, _hks = [], [], [], []
    for _k, _v in _cases:
        _c = curve(_k, _v)
        _labels.append(f"{_k}\n{_c['name']}")
        _measured.append(-_c["slope"])
        _lam.append(_c["lambda1"])
        _hks.append(_c["h_ks"])
    _x = np.arange(len(_cases))
    _ax[2].bar(_x - 0.22, _measured, width=0.2, color=C_TRUTH, label="measured")
    _ax[2].bar(_x, _lam, width=0.2, color=C_MEAN, label=r"$\lambda_1$")
    _ax[2].bar(_x + 0.22, _hks, width=0.2, color=C_PERT, label=r"$h_{KS}$")
    _ax[2].set_yscale("log")
    _ax[2].set_xticks(_x)
    _ax[2].set_xticklabels(_labels, fontsize=6.0)
    _ax[2].set_ylabel("rate (per time unit)")
    _ax[2].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    _rows = "\n".join(
        f"| {k} | {curve(k, v)['name']} | {-curve(k, v)['slope']:.4f} | "
        f"{curve(k, v)['lambda1']:.4f} | {-curve(k, v)['slope'] / curve(k, v)['lambda1']:.3f} | "
        f"{curve(k, v)['h_ks']:.2f} | {-curve(k, v)['slope'] / curve(k, v)['h_ks']:.3f} |"
        for k, v in _cases
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 2 · The rate is $\lambda_1$, and the way to prove it is Lorenz 96

    The curves in Section 1 fall **linearly**, not exponentially, and there is a
    reason. While the forecast is sharp, the dispersion term is
    $\tfrac12\ln(v_c/v_f)$; the forecast variance grows as $e^{2\lambda t}$; so

    $$\frac{dD}{dt} \to -\lambda .$$

    Which $\lambda$? Two candidates have been measured in this book: the leading
    Lyapunov exponent $\lambda_1$, and the Kolmogorov–Sinai entropy
    $h_{KS} = \sum_{\lambda_i>0}\lambda_i$, which chapter 8 introduced as *the*
    rate at which a system destroys information about its own initial state.

    In Lorenz 63 they are nearly the same number — one positive exponent, so
    $\lambda_1 = 0.906$ and $h_{KS} = 0.901$ — and the measurement cannot tell
    them apart. In Lorenz 96 they differ by a factor of **six**."""
        ),
        _fig,
        mo.md(
            f"""
| system | observable | measured $-dD/dt$ | $\\lambda_1$ | ratio | $h_{{KS}}$ | ratio |
|---|---|---|---|---|---|---|
{_rows}

**It is $\\lambda_1$.** Four measurements, all within 4 % of the leading
Lyapunov exponent, and Lorenz 96 rules out the entropy rate by a factor of 6.4.

Which is worth pausing on, because $h_{{KS}}$ *is* the rate at which the system
destroys information — chapter 8 established that, and Pesin's identity is not
in doubt. The resolution is that $h_{{KS}}$ is the rate for the **full state**,
and a scalar observable cannot see it. One projection of a 40-dimensional
forecast reveals the spreading of the fastest direction and nothing about the
other twelve growing ones. To measure 10.21 nats per time unit you would have to
estimate the joint distribution of all forty components — and Section 3 is about
why you cannot.

So there are two honest numbers, and they answer different questions.
$h_{{KS}} = 10.21$ is how fast Lorenz 96 destroys information about itself.
$\\lambda_1 = 1.67$ is how fast a forecast of any one variable stops being
informative. The second is what a forecast user experiences; the first is a
property of the system that no affordable ensemble can measure.
"""
        ),
    ])
    return


# ===========================================================================
# 3. Why not the full state
# ===========================================================================
@app.cell(hide_code=True)
def s3_figure(
    C_PERT,
    C_SAT,
    C_TRUTH,
    FLOOR_VALUES,
    L63_FULL_1EM06,
    L63_FULL_1EM06_SLOPE,
    L63_FULL_1EM12,
    L63_FULL_1EM12_SLOPE,
    L63_LAMBDA1,
    L63_TIMES,
    curve,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _t = np.asarray(L63_TIMES, dtype=float)
    _fine = np.asarray(L63_FULL_1EM12, dtype=float)
    _coarse = np.asarray(L63_FULL_1EM06, dtype=float)

    _fig, _ax = mpl_panels(
        2,
        titles=("Full-state relative entropy, two regularisations",
                "…and the rate it implies"),
        height=3.5,
    )
    _ax[0].plot(_t, _fine, linewidth=1.8, color=C_TRUTH,
                label=f"floor {FLOOR_VALUES[0]:.0e}")
    _ax[0].plot(_t, _coarse, linewidth=1.8, color=C_PERT, linestyle="--",
                label=f"floor {FLOOR_VALUES[1]:.0e}")
    _ax[0].set_xlabel("lead (MTU)")
    _ax[0].set_ylabel("nats")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _rates = [-L63_FULL_1EM12_SLOPE, -L63_FULL_1EM06_SLOPE,
              -curve("L63", 0)["slope"], L63_LAMBDA1]
    _names = [f"full,\nfloor {FLOOR_VALUES[0]:.0e}", f"full,\nfloor {FLOOR_VALUES[1]:.0e}",
              "scalar $x$", r"$\lambda_1$"]
    _ax[1].bar(np.arange(4), _rates,
               color=(C_TRUTH, C_PERT, C_SAT, "#9ca3af"), width=0.55)
    _ax[1].axhline(L63_LAMBDA1, color=C_SAT, linewidth=1.2, linestyle="--")
    _ax[1].set_xticks(np.arange(4))
    _ax[1].set_xticklabels(_names, fontsize=6.0)
    _ax[1].set_ylabel("implied $-dD/dt$ (nats/MTU)")
    finish_mpl(_fig, suptitle="Lorenz 63, 500-member ensembles in 3 dimensions")

    mo.vstack([
        mo.md(
            r"""---
    ## 3 · Why not just do this on the whole state vector?

    Because it does not work, and the way it fails is instructive rather than
    merely annoying.

    The dispersion term contains $\ln\det\Sigma_f$. A 500-member Lorenz 63
    ensemble does not fill three dimensions — it collapses onto the attractor,
    whose dimension chapter 8 measured as **2.06**, and then stretches into a
    filament along the unstable direction. So $\Sigma_f$ is *near-singular*, its
    smallest eigenvalue runs down to $10^{-13}$, and $\ln\det\Sigma_f$ is
    whatever the smallest eigenvalue is — which is to say, whatever floor you
    add to keep the matrix invertible."""
        ),
        _fig,
        mo.md(
            f"""
| regularisation | implied $-dD/dt$ |
|---|---|
| floor {FLOOR_VALUES[0]:.0e} | **{-L63_FULL_1EM12_SLOPE:.4f}** nats/MTU |
| floor {FLOOR_VALUES[1]:.0e} | **{-L63_FULL_1EM06_SLOPE:.4f}** nats/MTU |
| scalar $x$, no regularisation needed | {-curve("L63", 0)['slope']:.4f} |
| $\\lambda_1$ | {L63_LAMBDA1:.4f} |

**The two floors differ by a factor of
{abs(L63_FULL_1EM12_SLOPE / L63_FULL_1EM06_SLOPE):.2f} in the answer**, and
nothing in either computation complains. Both curves look perfectly reasonable.
One of them happens to agree with $\\lambda_1$ and the other does not, and there
was no way to know in advance which — a floor of $10^{{-6}}$ is not more
principled than $10^{{-12}}$, it just happens to sit where the collapsed
directions stop mattering.

This is the same difficulty chapter 8 met from the other side. There, box
counting starved in three dimensions because $N$ points cannot fill
$\\varepsilon^{{-3}}$ boxes. Here, a covariance cannot be estimated in a space
the ensemble does not span. Both are the curse of dimensionality arriving early
— at three dimensions, not at forty — and the response is the same in both
chapters: use a statistic the sample can support. `chaoslib`'s docstring for
`gaussian_information_components` says so directly, because the failure is silent.
"""
        ),
    ])
    return


# ===========================================================================
# 4. Norm invariance
# ===========================================================================
@app.cell(hide_code=True)
def s4_controls(mo):
    stretch = mo.ui.slider(
        start=-2.0, stop=2.0, step=0.25, value=1.0,
        label="log₁₀ of the factor applied to one variable", show_value=True,
    )
    return (stretch,)


@app.cell(hide_code=True)
def s4_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_TRUTH,
    finish_mpl,
    information,
    mo,
    mpl_panels,
    np,
    stretch,
):
    # Live, and instant: this is 4x4 linear algebra.
    _rng = np.random.default_rng(3)
    _k = 4
    _mean_c = _rng.normal(size=_k)
    _root_c = _rng.normal(size=(_k, _k))
    _cov_c = _root_c @ _root_c.T + _k * np.eye(_k)
    _mean_f = _mean_c + 0.5 * _rng.normal(size=_k)
    _root_f = 0.3 * _rng.normal(size=(_k, _k))
    _cov_f = _root_f @ _root_f.T + 0.2 * np.eye(_k)

    _factor = 10.0 ** float(stretch.value)
    _maps = {
        "identity": np.eye(_k),
        f"one axis × {_factor:g}": np.diag([1.0, 1.0, 1.0, _factor]),
        f"all axes × {_factor:g}": _factor * np.eye(_k),
        "random invertible": _rng.normal(size=(_k, _k)) + 3.0 * np.eye(_k),
    }
    _entropies, _errors, _labels = [], [], []
    for _name, _transform in _maps.items():
        _mf = _transform @ _mean_f
        _mc = _transform @ _mean_c
        _cf = _transform @ _cov_f @ _transform.T
        _cc = _transform @ _cov_c @ _transform.T
        _entropies.append(
            information.gaussian_information_components(_mf, _cf, _mc, _cc)[0]
        )
        _errors.append(float(np.sqrt(np.mean((_mf - _mc) ** 2))))
        _labels.append(_name)

    _fig, _ax = mpl_panels(
        2,
        titles=("Relative entropy under four transformations",
                "RMS error under the same four"),
        height=3.4,
    )
    _x = np.arange(len(_labels))
    _ax[0].bar(_x, _entropies, color=C_TRUTH, width=0.55)
    _ax[0].axhline(_entropies[0], color=C_SAT, linewidth=1.2, linestyle="--")
    _ax[0].set_xticks(_x)
    _ax[0].set_xticklabels(_labels, fontsize=6.0, rotation=20)
    _ax[0].set_ylabel("D (nats)")
    _ax[0].set_ylim(0, 1.4 * max(_entropies))

    _ax[1].bar(_x, _errors, color=C_PERT, width=0.55)
    _ax[1].set_yscale("log")
    _ax[1].set_xticks(_x)
    _ax[1].set_xticklabels(_labels, fontsize=6.0, rotation=20)
    _ax[1].set_ylabel("RMS error of the mean")
    finish_mpl(_fig)

    _spread_d = max(_entropies) - min(_entropies)
    _spread_e = max(_errors) / min(_errors)

    mo.vstack([
        mo.md(
            r"""---
    ## 4 · Is it really norm-independent?

    Yes, exactly — and this is the reason the whole apparatus is worth the
    trouble. Relative entropy depends on the distributions only through
    $\Sigma_c^{-1}\Sigma_f$ and
    $(\mu_c-\mu_f)^{\!\top}\Sigma_c^{-1}(\mu_c-\mu_f)$, and both are unchanged
    by $x \mapsto Ax$ for any invertible $A$: the Jacobians cancel between the
    two densities.

    Below, the same forecast and climatology under four transformations. Drag
    the slider to change how violently one variable is rescaled."""
        ),
        mo.hstack([stretch], justify="start"),
        _fig,
        mo.md(
            f"""
| transformation | $D$ (nats) | RMS error of the mean |
|---|---|---|
""" + "\n".join(
                f"| {l} | {d:.8f} | {e:.6f} |"
                for l, d, e in zip(_labels, _entropies, _errors)
            ) + f"""

**The relative entropy is identical to eight decimal places** across all four —
spread {_spread_d:.2e} nats, which is round-off. **The RMS error varies by a
factor of {_spread_e:.0f}.**

That is the whole argument. An RMS error is a statement about the atmosphere
*and* about the units you chose to measure it in; a relative entropy is a
statement about the atmosphere. Chapter 16 found the sharp version of the same
problem on the other side of the ledger — the leading singular vector rotates by
tens of degrees when the norm changes, so "which perturbation grows fastest" has
no answer until a norm is fixed. Information measures simply do not have that
degree of freedom.

The price is that you need a *distribution*, not a single forecast, and Section
5 is about how hard that is to estimate.
"""
        ),
    ])
    return


# ===========================================================================
# 5. Estimating mutual information
# ===========================================================================
@app.cell(hide_code=True)
def s5_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    L63_LAMBDA1,
    MI_BINS,
    MI_CORRECTED,
    MI_LAGS,
    MI_PLUGIN,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _lags = np.asarray(MI_LAGS, dtype=float)
    _bins = np.asarray(MI_BINS, dtype=int)
    _plug = np.asarray(MI_PLUGIN, dtype=float).reshape(_bins.size, _lags.size)
    _corr = np.asarray(MI_CORRECTED, dtype=float).reshape(_bins.size, _lags.size)

    _fig, _ax = mpl_panels(
        3,
        titles=("$I(x_0; x_t)$, plug-in estimator", "With Miller–Madow",
                "The bias floor"),
        height=3.5,
    )
    _cols = (C_TRUTH, C_MEAN, C_SPREAD, C_PERT)
    for _row, _b, _colour in zip(_plug, _bins, _cols):
        _ax[0].semilogy(_lags[1:], np.maximum(_row[1:], 1e-4), marker="o",
                        markersize=3, linewidth=1.4, color=_colour,
                        label=f"{_b} bins")
    _ax[0].set_xlabel("lag (MTU)")
    _ax[0].set_ylabel("I (nats)")
    _ax[0].legend(loc="upper right", fontsize=6.0, framealpha=0.9)

    for _row, _b, _colour in zip(_corr, _bins, _cols):
        _ax[1].semilogy(_lags[1:], np.maximum(_row[1:], 1e-4), marker="o",
                        markersize=3, linewidth=1.4, color=_colour)
    _ax[1].set_xlabel("lag (MTU)")
    _ax[1].set_ylabel("I (nats)")

    _ax[2].loglog(_bins, _plug[:, -1], marker="o", markersize=6,
                  color=C_TRUTH, linewidth=1.8, label="plug-in")
    _ax[2].loglog(_bins, np.maximum(_corr[:, -1], 1e-5), marker="s",
                  markersize=6, color=C_PERT, linewidth=1.8,
                  label="Miller–Madow")
    _ax[2].set_xlabel("bins")
    _ax[2].set_ylabel(f"I at lag {_lags[-1]:.0f} MTU (true ≈ 0)")
    _ax[2].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle="Lorenz 63, x component, 395,000 samples")

    _rows = "\n".join(
        f"| {b} | {p[0]:.3f} | {p[-1]:.4f} | {c[-1]:.4f} |"
        for b, p, c in zip(_bins, _plug, _corr)
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 5 · Mutual information, and whether it can be estimated

    The other natural information measure is the mutual information between the
    state now and the state later,

    $$I\bigl(x(0); x(t)\bigr) = \sum_{ij} p_{ij}\ln\frac{p_{ij}}{p_i p_j},$$

    which is how much knowing the present tells you about the future. It needs
    no ensemble at all — a long enough trajectory suffices — and it makes no
    Gaussian assumption. Both are real advantages.

    The catch is that it must be estimated from a histogram, and the plug-in
    estimator is biased **upward** by roughly the number of occupied bins over
    twice the sample size. Which means the estimate has a floor, and below that
    floor you are measuring the estimator."""
        ),
        _fig,
        mo.md(
            f"""
| bins | $I$ at zero lag | $I$ at lag {_lags[-1]:.0f} MTU (plug-in) | with Miller–Madow |
|---|---|---|---|
{_rows}

**Two things in that table are pure artefact.** The zero-lag column is not a
mutual information at all — $I(x;x)$ is the entropy of $x$, which for a
continuous variable is infinite and for a histogram grows without limit as the
bins shrink. It rises from {_plug[0, 0]:.2f} to {_plug[-1, 0]:.2f} nats purely
because there are more bins.

And the long-lag column should be **zero**: after {_lags[-1]:.0f} MTU, which is
{_lags[-1] * L63_LAMBDA1 / np.log(2):.0f} error-doubling times, the present
tells you nothing about the future. It reads {_plug[0, -1]:.4f} to
{_plug[-1, -1]:.4f} nats instead, rising monotonically with bin count — the
bias, plotted on its own in the third panel. Miller–Madow reduces it by a factor
of two to four, and does not remove it.

**So the practical rule is the one the third panel draws.** Choose the bin count
you can afford, measure the floor at a lag where the true answer is known to be
zero, and treat anything below that floor as unmeasured. That is an unglamorous
conclusion, and it is why relative entropy against a *fitted* distribution —
Sections 1 to 4 — is usually the more practical route, despite needing the
Gaussian assumption that mutual information avoids.
"""
        ),
    ])
    return


# ===========================================================================
# 6. Closing
# ===========================================================================
@app.cell(hide_code=True)
def closing(mo):
    mo.md(
        r"""
    ---
    ## Try this

    1. **Find the signal term doing something.** Section 1's signal component is
       small and flat for every observable offered. Describe a forecast problem
       where it would dominate instead, and say what would have to be true of
       the ensemble mean.
    2. **Rule out $h_{KS}$ yourself.** Section 2's Lorenz 96 panel is the
       decisive one. Before reading the table, predict what the measured rate
       would be if a scalar observable *did* see the full entropy rate, and say
       how far off the figure would look.
    3. **Break the full-state estimate.** In Section 3, the two regularisations
       differ by a factor of two in the answer. Work out from chapter 8's
       $D_2 = 2.06$ how many members you would need for the smallest eigenvalue
       of $\Sigma_f$ to be genuinely resolved, and decide whether that is
       affordable.
    4. **Push the norm.** In Section 4, take the slider to both extremes. The
       relative entropy does not move at all; the RMS error moves by orders of
       magnitude. Now explain why chapter 16 could not simply have used an
       information measure and avoided its norm problem.
    5. **Measure your own floor.** In Section 5, pick a bin count and read the
       long-lag value. Then decide the shortest lag at which you would be
       willing to quote a mutual information for that bin count, and justify it.

    ## What you should have seen

    A forecast's information content, measured as relative entropy against
    climatology, is almost entirely **dispersion** — "I am sharper than
    climatology" — and hardly at all **signal** — "my mean differs from
    climatology's". At lead zero a $10^{-3}$-spread ensemble carries about 9
    nats, of which 94 % is dispersion, and the signal term sits near 0.5 nats
    throughout.

    That information decays **linearly**, and at $\lambda_1$. Four measurements
    across two systems and two observables land within 4 % of the leading
    Lyapunov exponent — and Lorenz 96, where $\lambda_1 = 1.67$ and
    $h_{KS} = 10.21$ differ six-fold, rules out the entropy rate decisively. The
    two numbers answer different questions: $h_{KS}$ is how fast the system
    destroys information about *itself*, and no affordable ensemble can measure
    it; $\lambda_1$ is how fast a forecast of any *one* variable stops being
    informative, which is what a user experiences.

    Doing this on the full state vector fails, quietly. A 500-member ensemble on
    a 2.06-dimensional attractor has a near-singular covariance in three
    dimensions, so $\ln\det\Sigma_f$ is set by the regularisation: two
    defensible floors give decay rates differing by a factor of 1.78, with
    nothing to indicate which is right.

    And the reason to accept all that trouble: relative entropy is **invariant
    under any invertible change of variables**, to eight decimal places, where
    RMS error varies by a factor of hundreds under the same transformations. An
    error norm measures the atmosphere and your units; an information measure
    measures the atmosphere.

    Mutual information avoids the Gaussian assumption and pays for it in
    estimator bias: at a lag where the true value is zero, the plug-in estimate
    reads 0.009 to 0.039 nats depending only on the bin count. Miller–Madow cuts
    that by two to four times. Below the floor you are measuring the estimator.

    ## Further reading

    - Kleeman, R. (2002). Measuring dynamical prediction utility using relative
      entropy. *Journal of the Atmospheric Sciences*, **59**, 2057–2072
      *[citation needed: confirm pages]*.
    - DelSole, T. (2004). Predictability and information theory. *Journal of the
      Atmospheric Sciences*, **61**, 2425–2440 *[citation needed: confirm]* —
      the signal/dispersion decomposition.
    - Schneider, T. and Griffies, S. M. (1999). A conceptual framework for
      predictability studies. *Journal of Climate*, **12**, 3133–3155
      *[citation needed: confirm]*.
    - Cover, T. M. and Thomas, J. A. *Elements of Information Theory*
      *[citation needed: edition and chapter]*.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 14 *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
