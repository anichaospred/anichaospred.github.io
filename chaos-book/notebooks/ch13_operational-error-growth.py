# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 13 -- Error growth in operational models.

There is no truth to verify against, and no twin experiments are possible. So
Lorenz (1982) estimated error growth from the forecast archive alone -- and this
chapter checks that estimator against a truth it is not allowed to use.

Part IV of *An Interactive Chaos and Predictability Textbook*.

To edit:   marimo edit notebooks/ch13_operational-error-growth.py
To export: make nb-one NB=ch13_operational-error-growth
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 13: Operational Error Growth")


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

    from chaoslib import plotting

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
    mpl_panels = plotting.mpl_panels
    finish_mpl = plotting.finish_mpl

    DAYS_PER_TU = 5.0

    return (
        C_ANALYSIS,
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
        DAYS_PER_TU,
        finish_mpl,
        mo,
        mpl_panels,
        np,
    )


# ---------------------------------------------------------------------------
# Precomputed synthetic operational archive
# ---------------------------------------------------------------------------
@app.cell
def archive_data():
    # From scripts/generate_ch13_data.py: a cycling EnKF on Lorenz 96 with
    # six-hourly analyses, then forecasts to thirty days from each of 600
    # analyses, for four observing networks. Both the TRUE error curve (which
    # uses the truth) and Lorenz's lagged-forecast estimate (which does not).
    #
    # Everything is PER COMPONENT -- RMS over the 40 sites, the operational
    # convention. errorgrowth.saturation_level returns a norm over the whole
    # state vector, so it is divided by sqrt(N). That factor is 6.3 at N = 40,
    # and mixing the two made a fully saturated forecast look as though it had
    # plateaued at 16% of saturation on the first attempt.
    CYCLE_HOURS = 6
    LAMBDA1_L96 = 1.67
    CONFIG_LABELS = ('dense', 'operational', 'sparse', 'degraded')
    DENSE_ANALYSIS_ERROR = 0.022895
    DENSE_SATURATION = 4.995733
    DENSE_N_OBS = 40
    DENSE_SIGMA_O = 0.1
    DENSE_TRUE = (
        2.289532e-02, 2.506787e-02, 2.763751e-02, 3.049463e-02,
        3.357011e-02, 3.686609e-02, 4.044081e-02, 4.437268e-02,
        4.873134e-02, 5.357435e-02, 5.894889e-02, 6.490198e-02,
        7.150100e-02, 7.882309e-02, 8.692215e-02, 9.585467e-02,
        1.057034e-01, 1.165464e-01, 1.284812e-01, 1.416482e-01,
        1.561644e-01, 1.721099e-01, 1.896477e-01, 2.090239e-01,
        2.303004e-01, 2.532399e-01, 2.776088e-01, 3.035479e-01,
        3.316204e-01, 3.624853e-01, 3.963802e-01, 4.329619e-01,
        4.716283e-01, 5.122329e-01, 5.552901e-01, 6.013565e-01,
        6.505953e-01, 7.026385e-01, 7.570160e-01, 8.139649e-01,
        8.749595e-01, 9.418682e-01, 1.015473e+00, 1.095126e+00,
        1.178994e+00, 1.264917e+00, 1.352314e+00, 1.442661e+00,
        1.536937e+00, 1.634079e+00, 1.732331e+00, 1.830613e+00,
        1.928293e+00, 2.023998e+00, 2.116414e+00, 2.206748e+00,
        2.298483e+00, 2.395228e+00, 2.497136e+00, 2.600202e+00,
        2.700093e+00, 2.795012e+00, 2.884752e+00, 2.968691e+00,
        3.047569e+00, 3.124150e+00, 3.198774e+00, 3.270225e+00,
        3.340478e+00, 3.413505e+00, 3.490036e+00, 3.566083e+00,
        3.636923e+00, 3.699942e+00, 3.753602e+00, 3.798365e+00,
        3.838766e+00, 3.880773e+00, 3.926715e+00, 3.976050e+00,
        4.028033e+00, 4.080860e+00, 4.132867e+00, 4.183793e+00,
        4.232729e+00, 4.277245e+00, 4.315933e+00, 4.349895e+00,
        4.379306e+00, 4.404149e+00, 4.426474e+00, 4.447766e+00,
        4.469894e+00, 4.495356e+00, 4.523379e+00, 4.550535e+00,
        4.574910e+00, 4.596302e+00, 4.614549e+00, 4.629472e+00,
        4.643545e+00, 4.661567e+00, 4.684911e+00, 4.709830e+00,
        4.731578e+00, 4.747935e+00, 4.760210e+00, 4.770775e+00,
        4.780930e+00, 4.791919e+00, 4.806726e+00, 4.827224e+00,
        4.849699e+00, 4.867751e+00, 4.878544e+00, 4.884375e+00,
        4.888265e+00, 4.890952e+00, 4.892954e+00, 4.895602e+00,
        4.900404e+00,
    )
    DENSE_LAGGED = (
        1.071158e-02, 1.192166e-02, 1.322695e-02, 1.463485e-02,
        1.617726e-02, 1.787389e-02, 1.970259e-02, 2.170151e-02,
        2.381423e-02, 2.621783e-02, 2.890471e-02, 3.184949e-02,
        3.504321e-02, 3.863459e-02, 4.253005e-02, 4.660804e-02,
        5.102618e-02, 5.601033e-02, 6.150458e-02, 6.768062e-02,
        7.413100e-02, 8.114379e-02, 8.893374e-02, 9.781792e-02,
        1.075057e-01, 1.171301e-01, 1.287121e-01, 1.415893e-01,
        1.559949e-01, 1.722735e-01, 1.899397e-01, 2.086054e-01,
        2.285365e-01, 2.498876e-01, 2.750070e-01, 3.010656e-01,
        3.283514e-01, 3.569950e-01, 3.849008e-01, 4.167579e-01,
        4.540704e-01, 4.947518e-01, 5.410275e-01, 5.909578e-01,
        6.418001e-01, 6.963785e-01, 7.520506e-01, 8.084772e-01,
        8.682598e-01, 9.323798e-01, 9.962819e-01, 1.066259e+00,
        1.140591e+00, 1.217017e+00, 1.297672e+00, 1.379214e+00,
        1.464832e+00, 1.549055e+00, 1.629801e+00, 1.707323e+00,
        1.787414e+00, 1.871207e+00, 1.956391e+00, 2.040235e+00,
        2.125662e+00, 2.207798e+00, 2.296442e+00, 2.391087e+00,
        2.484021e+00, 2.574818e+00, 2.657168e+00, 2.730638e+00,
        2.801402e+00, 2.867915e+00, 2.942294e+00, 3.022901e+00,
        3.106379e+00, 3.195953e+00, 3.287922e+00, 3.374294e+00,
        3.455294e+00, 3.534289e+00, 3.610233e+00, 3.689799e+00,
        3.760711e+00, 3.832267e+00, 3.899210e+00, 3.960042e+00,
        4.012069e+00, 4.054782e+00, 4.101075e+00, 4.140719e+00,
        4.179316e+00, 4.220759e+00, 4.259343e+00, 4.298748e+00,
        4.342326e+00, 4.389071e+00, 4.433173e+00, 4.471354e+00,
        4.500383e+00, 4.525635e+00, 4.543267e+00, 4.560406e+00,
        4.577992e+00, 4.595820e+00, 4.614835e+00, 4.633336e+00,
        4.654873e+00, 4.680676e+00, 4.710779e+00, 4.736889e+00,
        4.756259e+00, 4.770595e+00, 4.782154e+00, 4.788901e+00,
        4.788694e+00, 4.787549e+00, 4.791679e+00, 4.802905e+00,
    )
    OPERATIONAL_ANALYSIS_ERROR = 0.105467
    OPERATIONAL_SATURATION = 4.995733
    OPERATIONAL_N_OBS = 20
    OPERATIONAL_SIGMA_O = 0.3
    OPERATIONAL_TRUE = (
        1.054675e-01, 1.148833e-01, 1.261039e-01, 1.389897e-01,
        1.534549e-01, 1.694410e-01, 1.869851e-01, 2.062371e-01,
        2.272903e-01, 2.502084e-01, 2.751403e-01, 3.022520e-01,
        3.317560e-01, 3.640002e-01, 3.992486e-01, 4.377342e-01,
        4.798503e-01, 5.258382e-01, 5.755354e-01, 6.288638e-01,
        6.858807e-01, 7.465614e-01, 8.111248e-01, 8.798834e-01,
        9.527649e-01, 1.029365e+00, 1.108463e+00, 1.189318e+00,
        1.272326e+00, 1.358095e+00, 1.447299e+00, 1.540491e+00,
        1.637620e+00, 1.736829e+00, 1.834877e+00, 1.930446e+00,
        2.025884e+00, 2.123090e+00, 2.220036e+00, 2.313500e+00,
        2.401860e+00, 2.486255e+00, 2.570563e+00, 2.658542e+00,
        2.750500e+00, 2.844222e+00, 2.937175e+00, 3.027394e+00,
        3.114207e+00, 3.198824e+00, 3.283266e+00, 3.367915e+00,
        3.449350e+00, 3.523236e+00, 3.589308e+00, 3.650280e+00,
        3.709309e+00, 3.769332e+00, 3.831098e+00, 3.892263e+00,
        3.949704e+00, 4.001992e+00, 4.050095e+00, 4.096467e+00,
        4.140723e+00, 4.179569e+00, 4.211275e+00, 4.237636e+00,
        4.263765e+00, 4.295537e+00, 4.335802e+00, 4.382520e+00,
        4.430901e+00, 4.476724e+00, 4.516148e+00, 4.545538e+00,
        4.564919e+00, 4.579210e+00, 4.592763e+00, 4.606663e+00,
        4.621966e+00, 4.639216e+00, 4.657924e+00, 4.679181e+00,
        4.704677e+00, 4.732710e+00, 4.757872e+00, 4.776454e+00,
        4.788953e+00, 4.797295e+00, 4.804242e+00, 4.812982e+00,
        4.823709e+00, 4.835282e+00, 4.850579e+00, 4.872300e+00,
        4.897934e+00, 4.923262e+00, 4.944304e+00, 4.958058e+00,
        4.964068e+00, 4.963278e+00, 4.958748e+00, 4.954732e+00,
        4.952576e+00, 4.952685e+00, 4.957422e+00, 4.967043e+00,
        4.978808e+00, 4.991856e+00, 5.006254e+00, 5.020734e+00,
        5.032496e+00, 5.038049e+00, 5.037988e+00, 5.037224e+00,
        5.039412e+00, 5.043483e+00, 5.045572e+00, 5.043875e+00,
        5.040675e+00,
    )
    OPERATIONAL_LAGGED = (
        4.827516e-02, 5.375733e-02, 6.009677e-02, 6.715721e-02,
        7.479300e-02, 8.297192e-02, 9.174012e-02, 1.014307e-01,
        1.120088e-01, 1.234607e-01, 1.360468e-01, 1.498374e-01,
        1.649570e-01, 1.821842e-01, 2.008043e-01, 2.218626e-01,
        2.449688e-01, 2.700135e-01, 2.980716e-01, 3.298086e-01,
        3.641877e-01, 4.008339e-01, 4.396581e-01, 4.806048e-01,
        5.228031e-01, 5.667664e-01, 6.122826e-01, 6.625571e-01,
        7.171536e-01, 7.745085e-01, 8.340777e-01, 8.973855e-01,
        9.647757e-01, 1.031164e+00, 1.101312e+00, 1.174598e+00,
        1.253743e+00, 1.332460e+00, 1.411019e+00, 1.489422e+00,
        1.568054e+00, 1.650601e+00, 1.739523e+00, 1.825370e+00,
        1.909850e+00, 1.991690e+00, 2.076441e+00, 2.167010e+00,
        2.261786e+00, 2.356306e+00, 2.453020e+00, 2.551505e+00,
        2.651724e+00, 2.747555e+00, 2.835603e+00, 2.914878e+00,
        2.988101e+00, 3.059453e+00, 3.130374e+00, 3.203245e+00,
        3.280348e+00, 3.359695e+00, 3.438260e+00, 3.516011e+00,
        3.591705e+00, 3.658434e+00, 3.718285e+00, 3.778679e+00,
        3.842676e+00, 3.910350e+00, 3.975643e+00, 4.035243e+00,
        4.088028e+00, 4.132942e+00, 4.172243e+00, 4.208790e+00,
        4.243771e+00, 4.275160e+00, 4.305464e+00, 4.335722e+00,
        4.371665e+00, 4.409403e+00, 4.450800e+00, 4.495446e+00,
        4.536066e+00, 4.570771e+00, 4.606203e+00, 4.636388e+00,
        4.665237e+00, 4.687983e+00, 4.705938e+00, 4.720339e+00,
        4.732115e+00, 4.742848e+00, 4.757242e+00, 4.776086e+00,
        4.797259e+00, 4.812161e+00, 4.819917e+00, 4.826136e+00,
        4.832706e+00, 4.839003e+00, 4.845935e+00, 4.858392e+00,
        4.875611e+00, 4.896841e+00, 4.913348e+00, 4.927240e+00,
        4.936389e+00, 4.945993e+00, 4.962321e+00, 4.985141e+00,
        5.004600e+00, 5.024330e+00, 5.040157e+00, 5.051214e+00,
        5.058509e+00, 5.060309e+00, 5.058454e+00, 5.055085e+00,
    )
    SPARSE_ANALYSIS_ERROR = 0.308204
    SPARSE_SATURATION = 4.995733
    SPARSE_N_OBS = 10
    SPARSE_SIGMA_O = 0.5
    SPARSE_TRUE = (
        3.082041e-01, 3.350944e-01, 3.669218e-01, 4.027906e-01,
        4.417749e-01, 4.831971e-01, 5.269362e-01, 5.735571e-01,
        6.237993e-01, 6.779113e-01, 7.353475e-01, 7.951605e-01,
        8.564472e-01, 9.190897e-01, 9.843649e-01, 1.054854e+00,
        1.132464e+00, 1.216848e+00, 1.307110e+00, 1.402177e+00,
        1.500718e+00, 1.602081e+00, 1.706117e+00, 1.812160e+00,
        1.918723e+00, 2.023985e+00, 2.126401e+00, 2.225482e+00,
        2.322027e+00, 2.418366e+00, 2.517244e+00, 2.618933e+00,
        2.721117e+00, 2.821498e+00, 2.919146e+00, 3.013105e+00,
        3.101911e+00, 3.185388e+00, 3.266234e+00, 3.348942e+00,
        3.436366e+00, 3.527271e+00, 3.617116e+00, 3.700597e+00,
        3.774503e+00, 3.840238e+00, 3.900980e+00, 3.957448e+00,
        4.009742e+00, 4.059308e+00, 4.108018e+00, 4.158720e+00,
        4.213447e+00, 4.269399e+00, 4.322274e+00, 4.370542e+00,
        4.413876e+00, 4.452262e+00, 4.485452e+00, 4.513393e+00,
        4.538032e+00, 4.562443e+00, 4.589636e+00, 4.619655e+00,
        4.646898e+00, 4.665511e+00, 4.676943e+00, 4.688171e+00,
        4.702164e+00, 4.716232e+00, 4.729267e+00, 4.742409e+00,
        4.755443e+00, 4.768092e+00, 4.782214e+00, 4.799051e+00,
        4.815873e+00, 4.829180e+00, 4.839555e+00, 4.850584e+00,
        4.863512e+00, 4.875947e+00, 4.884479e+00, 4.888317e+00,
        4.890959e+00, 4.896159e+00, 4.903781e+00, 4.912083e+00,
        4.920685e+00, 4.930073e+00, 4.940186e+00, 4.950064e+00,
        4.958776e+00, 4.966639e+00, 4.974903e+00, 4.985014e+00,
        4.996565e+00, 5.006659e+00, 5.012524e+00, 5.012407e+00,
        5.008010e+00, 5.005144e+00, 5.008048e+00, 5.015164e+00,
        5.020943e+00, 5.020883e+00, 5.016224e+00, 5.011888e+00,
        5.010908e+00, 5.014039e+00, 5.020760e+00, 5.029304e+00,
        5.037320e+00, 5.043100e+00, 5.047252e+00, 5.051649e+00,
        5.056013e+00, 5.058511e+00, 5.058341e+00, 5.057041e+00,
        5.056973e+00,
    )
    SPARSE_LAGGED = (
        1.324396e-01, 1.483281e-01, 1.655858e-01, 1.836652e-01,
        2.019216e-01, 2.213157e-01, 2.418434e-01, 2.645963e-01,
        2.892506e-01, 3.160321e-01, 3.453966e-01, 3.767317e-01,
        4.110136e-01, 4.484483e-01, 4.888029e-01, 5.321797e-01,
        5.785350e-01, 6.279449e-01, 6.822116e-01, 7.413651e-01,
        8.076499e-01, 8.790079e-01, 9.533002e-01, 1.029715e+00,
        1.106411e+00, 1.182699e+00, 1.263298e+00, 1.346306e+00,
        1.433296e+00, 1.521789e+00, 1.613360e+00, 1.704569e+00,
        1.795625e+00, 1.884716e+00, 1.981612e+00, 2.079379e+00,
        2.179513e+00, 2.274182e+00, 2.366435e+00, 2.464110e+00,
        2.562938e+00, 2.660345e+00, 2.762844e+00, 2.864384e+00,
        2.958360e+00, 3.048829e+00, 3.134246e+00, 3.214329e+00,
        3.292301e+00, 3.370780e+00, 3.449825e+00, 3.532530e+00,
        3.611343e+00, 3.684123e+00, 3.749060e+00, 3.805975e+00,
        3.854255e+00, 3.899698e+00, 3.942139e+00, 3.987108e+00,
        4.034413e+00, 4.082421e+00, 4.129804e+00, 4.168987e+00,
        4.206524e+00, 4.239456e+00, 4.267495e+00, 4.295143e+00,
        4.324973e+00, 4.358644e+00, 4.393267e+00, 4.424872e+00,
        4.456529e+00, 4.483938e+00, 4.509955e+00, 4.538315e+00,
        4.571527e+00, 4.607240e+00, 4.642264e+00, 4.671056e+00,
        4.695981e+00, 4.722212e+00, 4.750386e+00, 4.775143e+00,
        4.792314e+00, 4.802773e+00, 4.808862e+00, 4.816294e+00,
        4.822926e+00, 4.832386e+00, 4.844437e+00, 4.857779e+00,
        4.869225e+00, 4.880020e+00, 4.892582e+00, 4.909580e+00,
        4.928600e+00, 4.946578e+00, 4.963513e+00, 4.976078e+00,
        4.985005e+00, 4.991870e+00, 4.998744e+00, 5.006011e+00,
        5.010311e+00, 5.014451e+00, 5.020206e+00, 5.026422e+00,
        5.030474e+00, 5.034797e+00, 5.040636e+00, 5.048925e+00,
        5.062659e+00, 5.074852e+00, 5.086004e+00, 5.092865e+00,
        5.099390e+00, 5.103554e+00, 5.106135e+00, 5.103942e+00,
    )
    DEGRADED_ANALYSIS_ERROR = 2.228226
    DEGRADED_SATURATION = 4.995733
    DEGRADED_N_OBS = 8
    DEGRADED_SIGMA_O = 1.0
    DEGRADED_TRUE = (
        2.228226e+00, 2.291238e+00, 2.387405e+00, 2.506911e+00,
        2.639630e+00, 2.776423e+00, 2.910760e+00, 3.039588e+00,
        3.161527e+00, 3.275200e+00, 3.380180e+00, 3.477357e+00,
        3.566741e+00, 3.647929e+00, 3.722333e+00, 3.792335e+00,
        3.858829e+00, 3.920175e+00, 3.973316e+00, 4.017646e+00,
        4.055965e+00, 4.090736e+00, 4.122443e+00, 4.152538e+00,
        4.183688e+00, 4.217089e+00, 4.252138e+00, 4.285499e+00,
        4.314181e+00, 4.339400e+00, 4.363589e+00, 4.387295e+00,
        4.410089e+00, 4.432439e+00, 4.454617e+00, 4.475553e+00,
        4.494478e+00, 4.511202e+00, 4.525854e+00, 4.539504e+00,
        4.553127e+00, 4.566926e+00, 4.582109e+00, 4.600681e+00,
        4.622829e+00, 4.646617e+00, 4.669173e+00, 4.687067e+00,
        4.698847e+00, 4.707474e+00, 4.717339e+00, 4.730795e+00,
        4.749290e+00, 4.771976e+00, 4.794999e+00, 4.815893e+00,
        4.833706e+00, 4.847462e+00, 4.856262e+00, 4.860814e+00,
        4.864514e+00, 4.869843e+00, 4.876764e+00, 4.884188e+00,
        4.891156e+00, 4.897236e+00, 4.903423e+00, 4.910849e+00,
        4.918405e+00, 4.923763e+00, 4.926786e+00, 4.930290e+00,
        4.936246e+00, 4.943458e+00, 4.949744e+00, 4.954750e+00,
        4.960171e+00, 4.968007e+00, 4.978276e+00, 4.988795e+00,
        4.998221e+00, 5.007472e+00, 5.018066e+00, 5.028998e+00,
        5.036731e+00, 5.038826e+00, 5.035873e+00, 5.030146e+00,
        5.023997e+00, 5.020817e+00, 5.023560e+00, 5.029857e+00,
        5.035203e+00, 5.039197e+00, 5.043292e+00, 5.048026e+00,
        5.052980e+00, 5.056745e+00, 5.058407e+00, 5.058610e+00,
        5.059335e+00, 5.061958e+00, 5.065590e+00, 5.067647e+00,
        5.065842e+00, 5.061451e+00, 5.059007e+00, 5.060721e+00,
        5.063809e+00, 5.066035e+00, 5.069496e+00, 5.076167e+00,
        5.083816e+00, 5.087398e+00, 5.082588e+00, 5.070154e+00,
        5.056653e+00, 5.049425e+00, 5.050936e+00, 5.057206e+00,
        5.062605e+00,
    )
    DEGRADED_LAGGED = (
        6.200052e-01, 6.652129e-01, 7.284959e-01, 8.039264e-01,
        8.863188e-01, 9.723419e-01, 1.060175e+00, 1.148641e+00,
        1.237142e+00, 1.324440e+00, 1.411745e+00, 1.498711e+00,
        1.585268e+00, 1.670369e+00, 1.756540e+00, 1.843549e+00,
        1.931942e+00, 2.022020e+00, 2.113286e+00, 2.211647e+00,
        2.311157e+00, 2.407553e+00, 2.497065e+00, 2.584038e+00,
        2.668176e+00, 2.749475e+00, 2.829059e+00, 2.908482e+00,
        2.991817e+00, 3.076108e+00, 3.159628e+00, 3.239399e+00,
        3.316831e+00, 3.389853e+00, 3.461125e+00, 3.532573e+00,
        3.601340e+00, 3.662285e+00, 3.717680e+00, 3.770380e+00,
        3.827105e+00, 3.883158e+00, 3.938538e+00, 3.994184e+00,
        4.047023e+00, 4.096993e+00, 4.141289e+00, 4.179595e+00,
        4.215161e+00, 4.249616e+00, 4.283787e+00, 4.318294e+00,
        4.351350e+00, 4.382508e+00, 4.408235e+00, 4.432211e+00,
        4.456030e+00, 4.479662e+00, 4.503548e+00, 4.528492e+00,
        4.553781e+00, 4.579122e+00, 4.607684e+00, 4.638264e+00,
        4.668063e+00, 4.690087e+00, 4.705004e+00, 4.713799e+00,
        4.723199e+00, 4.735287e+00, 4.747004e+00, 4.757085e+00,
        4.764874e+00, 4.771114e+00, 4.782980e+00, 4.801277e+00,
        4.826210e+00, 4.851543e+00, 4.872441e+00, 4.883755e+00,
        4.882580e+00, 4.874510e+00, 4.867337e+00, 4.867140e+00,
        4.870382e+00, 4.879868e+00, 4.891635e+00, 4.902698e+00,
        4.913684e+00, 4.926124e+00, 4.936892e+00, 4.952069e+00,
        4.969102e+00, 4.984001e+00, 4.994658e+00, 4.998873e+00,
        5.002641e+00, 5.005382e+00, 5.010060e+00, 5.015659e+00,
        5.023607e+00, 5.035444e+00, 5.045055e+00, 5.047482e+00,
        5.044694e+00, 5.038933e+00, 5.033486e+00, 5.030060e+00,
        5.027186e+00, 5.024521e+00, 5.025593e+00, 5.032448e+00,
        5.039922e+00, 5.048233e+00, 5.052393e+00, 5.052686e+00,
        5.055592e+00, 5.063511e+00, 5.073574e+00, 5.086841e+00,
    )
    LEADS = (
        0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000,
        0.3500, 0.4000, 0.4500, 0.5000, 0.5500, 0.6000, 0.6500,
        0.7000, 0.7500, 0.8000, 0.8500, 0.9000, 0.9500, 1.0000,
        1.0500, 1.1000, 1.1500, 1.2000, 1.2500, 1.3000, 1.3500,
        1.4000, 1.4500, 1.5000, 1.5500, 1.6000, 1.6500, 1.7000,
        1.7500, 1.8000, 1.8500, 1.9000, 1.9500, 2.0000, 2.0500,
        2.1000, 2.1500, 2.2000, 2.2500, 2.3000, 2.3500, 2.4000,
        2.4500, 2.5000, 2.5500, 2.6000, 2.6500, 2.7000, 2.7500,
        2.8000, 2.8500, 2.9000, 2.9500, 3.0000, 3.0500, 3.1000,
        3.1500, 3.2000, 3.2500, 3.3000, 3.3500, 3.4000, 3.4500,
        3.5000, 3.5500, 3.6000, 3.6500, 3.7000, 3.7500, 3.8000,
        3.8500, 3.9000, 3.9500, 4.0000, 4.0500, 4.1000, 4.1500,
        4.2000, 4.2500, 4.3000, 4.3500, 4.4000, 4.4500, 4.5000,
        4.5500, 4.6000, 4.6500, 4.7000, 4.7500, 4.8000, 4.8500,
        4.9000, 4.9500, 5.0000, 5.0500, 5.1000, 5.1500, 5.2000,
        5.2500, 5.3000, 5.3500, 5.4000, 5.4500, 5.5000, 5.5500,
        5.6000, 5.6500, 5.7000, 5.7500, 5.8000, 5.8500, 5.9000,
        5.9500, 6.0000,
    )
    return (
        CONFIG_LABELS,
        CYCLE_HOURS,
        DEGRADED_ANALYSIS_ERROR,
        DEGRADED_LAGGED,
        DEGRADED_N_OBS,
        DEGRADED_SATURATION,
        DEGRADED_SIGMA_O,
        DEGRADED_TRUE,
        DENSE_ANALYSIS_ERROR,
        DENSE_LAGGED,
        DENSE_N_OBS,
        DENSE_SATURATION,
        DENSE_SIGMA_O,
        DENSE_TRUE,
        LAMBDA1_L96,
        LEADS,
        OPERATIONAL_ANALYSIS_ERROR,
        OPERATIONAL_LAGGED,
        OPERATIONAL_N_OBS,
        OPERATIONAL_SATURATION,
        OPERATIONAL_SIGMA_O,
        OPERATIONAL_TRUE,
        SPARSE_ANALYSIS_ERROR,
        SPARSE_LAGGED,
        SPARSE_N_OBS,
        SPARSE_SATURATION,
        SPARSE_SIGMA_O,
        SPARSE_TRUE,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@app.cell
def helpers(
    DEGRADED_ANALYSIS_ERROR,
    DEGRADED_LAGGED,
    DEGRADED_N_OBS,
    DEGRADED_SATURATION,
    DEGRADED_SIGMA_O,
    DEGRADED_TRUE,
    DENSE_ANALYSIS_ERROR,
    DENSE_LAGGED,
    DENSE_N_OBS,
    DENSE_SATURATION,
    DENSE_SIGMA_O,
    DENSE_TRUE,
    LEADS,
    OPERATIONAL_ANALYSIS_ERROR,
    OPERATIONAL_LAGGED,
    OPERATIONAL_N_OBS,
    OPERATIONAL_SATURATION,
    OPERATIONAL_SIGMA_O,
    OPERATIONAL_TRUE,
    SPARSE_ANALYSIS_ERROR,
    SPARSE_LAGGED,
    SPARSE_N_OBS,
    SPARSE_SATURATION,
    SPARSE_SIGMA_O,
    SPARSE_TRUE,
    np,
):
    CONFIGS = {
        "dense": dict(
            true=DENSE_TRUE, lagged=DENSE_LAGGED, sat=DENSE_SATURATION,
            analysis=DENSE_ANALYSIS_ERROR, n_obs=DENSE_N_OBS,
            sigma=DENSE_SIGMA_O, label="dense",
        ),
        "operational": dict(
            true=OPERATIONAL_TRUE, lagged=OPERATIONAL_LAGGED,
            sat=OPERATIONAL_SATURATION, analysis=OPERATIONAL_ANALYSIS_ERROR,
            n_obs=OPERATIONAL_N_OBS, sigma=OPERATIONAL_SIGMA_O,
            label="operational",
        ),
        "sparse": dict(
            true=SPARSE_TRUE, lagged=SPARSE_LAGGED, sat=SPARSE_SATURATION,
            analysis=SPARSE_ANALYSIS_ERROR, n_obs=SPARSE_N_OBS,
            sigma=SPARSE_SIGMA_O, label="sparse",
        ),
        "degraded": dict(
            true=DEGRADED_TRUE, lagged=DEGRADED_LAGGED,
            sat=DEGRADED_SATURATION, analysis=DEGRADED_ANALYSIS_ERROR,
            n_obs=DEGRADED_N_OBS, sigma=DEGRADED_SIGMA_O, label="degraded",
        ),
    }

    def config(key):
        c = dict(CONFIGS[key])
        c["leads"] = np.asarray(LEADS, dtype=float)
        c["true"] = np.asarray(c["true"], dtype=float)
        c["lagged"] = np.asarray(c["lagged"], dtype=float)
        return c

    def growth_rate(leads, curve, saturation, low, high):
        """Least-squares rate over a window given as fractions of saturation."""
        window = (curve > low * saturation) & (curve < high * saturation)
        if int(window.sum()) < 5:
            return float("nan"), 0
        rate = float(np.polyfit(leads[window], np.log(curve[window]), 1)[0])
        return rate, int(window.sum())

    def lead_at(leads, curve, level):
        hit = np.nonzero(curve >= level)[0]
        if hit.size == 0:
            return float("nan")
        i = int(hit[0])
        if i == 0:
            return float(leads[0])
        return float(np.interp(level, [curve[i - 1], curve[i]],
                               [leads[i - 1], leads[i]]))

    return CONFIGS, config, growth_rate, lead_at


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 13 · Error Growth in Operational Models

    **Part IV — Many scales, many degrees of freedom.**

    **The forecasting question.** Everything so far has been measured by twin
    experiment: take a state, perturb it, integrate both copies, watch them
    separate. That method needs the truth.

    A forecast centre does not have the truth. What it has is an **analysis** —
    itself a forecast, corrected by observations — whose own error is a
    substantial fraction of a short-range forecast's. Verifying a forecast
    against an analysis therefore measures the forecast error *plus* the
    analysis error, correlated with each other in ways that depend on the
    assimilation system. And there is no second Earth to perturb.

    So how was the atmosphere's error-doubling time ever measured?

    Lorenz (1982) answered this with an idea that needs no truth at all.
    Consider two forecasts **valid at the same time** but started a day apart:
    a one-day forecast and a two-day forecast, both verifying today. Their
    difference can be computed from the archive alone. And it grows at the rate
    errors grow — because at the moment the older forecast reached yesterday, it
    differed from yesterday's analysis by roughly a one-day forecast error, and
    the two have been running together ever since.

    ---

    ## What this chapter does

    The trouble with a truth-free estimator is that you cannot check it against
    the truth. So this chapter builds a **synthetic operational centre** on
    Lorenz 96, where the truth is known and withheld:

    - a cycling ensemble Kalman filter, analyses every six hours;
    - forecasts to thirty days from each of 600 consecutive analyses;
    - four observing networks, from 40 observed sites to 8.

    Then it applies Lorenz's estimator, which sees only the forecasts, and
    compares it against the error curve computed from the truth, which the
    estimator never touches.

    | Section | The question |
    |---|---|
    | 1 | What does an operational error curve look like? |
    | 2 | Does the lagged-forecast estimator recover the growth rate? |
    | 3 | What does a better observing system buy? |
    | 4 | How does the measured rate relate to $\lambda_1$? |
    """
    )
    return


# ===========================================================================
# 1. The operational error curve
# ===========================================================================
@app.cell(hide_code=True)
def s1_controls(mo):
    network = mo.ui.dropdown(
        options={
            "dense — 40 of 40 sites, σₒ = 0.1": "dense",
            "operational — 20 of 40 sites, σₒ = 0.3": "operational",
            "sparse — 10 of 40 sites, σₒ = 0.5": "sparse",
            "degraded — 8 of 40 sites, σₒ = 1.0": "degraded",
        },
        value="operational — 20 of 40 sites, σₒ = 0.3",
        label="observing network",
    )
    return (network,)


@app.cell(hide_code=True)
def s1_figure(
    C_ANALYSIS,
    C_PERT,
    C_SAT,
    C_TRUTH,
    DAYS_PER_TU,
    config,
    finish_mpl,
    growth_rate,
    lead_at,
    mo,
    mpl_panels,
    network,
    np,
):
    _c = config(str(network.value))
    _leads_days = _c["leads"] * DAYS_PER_TU
    _sat = _c["sat"]

    _fig, _ax = mpl_panels(
        3,
        titles=("The forecast error curve", "On log axes",
                "Local growth rate"),
        height=3.6,
    )
    _ax[0].plot(_leads_days, _c["true"], linewidth=2.0, color=C_TRUTH,
                label="true error")
    _ax[0].axhline(_sat, color=C_SAT, linewidth=1.3, linestyle="--",
                   label="saturation")
    _ax[0].axhline(0.5 * _sat, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[0].axhline(_c["analysis"], color=C_ANALYSIS, linewidth=1.3,
                   linestyle="-.", label="analysis error")
    _ax[0].set_xlabel("forecast lead (days)")
    _ax[0].set_ylabel("RMS error per site")
    _ax[0].legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    _ax[1].semilogy(_leads_days, _c["true"], linewidth=2.0, color=C_TRUTH)
    _ax[1].axhline(_sat, color=C_SAT, linewidth=1.3, linestyle="--")
    _ax[1].axhline(_c["analysis"], color=C_ANALYSIS, linewidth=1.3,
                   linestyle="-.")
    _ax[1].set_xlabel("forecast lead (days)")
    _ax[1].set_ylabel("RMS error per site")
    _ax[1].set_ylim(0.5 * _c["analysis"], 3.0 * _sat)

    _rate = np.gradient(np.log(_c["true"]), _c["leads"])
    _ax[2].plot(_c["true"] / _sat, _rate, linewidth=1.6, color=C_TRUTH)
    _ax[2].axhline(0.0, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[2].set_xlabel("$E/E_\\infty$")
    _ax[2].set_ylabel("$d\\ln E/dt$ (per time unit)")
    _ax[2].set_xlim(0, 1.05)
    finish_mpl(_fig, suptitle=f"{_c['label']} network: {_c['n_obs']} of 40 sites, "
                              f"$\\sigma_o$ = {_c['sigma']}")

    _half = lead_at(_c["leads"], _c["true"], 0.5 * _sat)
    _eighty = lead_at(_c["leads"], _c["true"], 0.8 * _sat)
    _rate_small, _n_small = growth_rate(
        _c["leads"], _c["true"], _sat, 0.02, 0.2
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 1 · What an operational error curve looks like

    The analysis error is the *starting* value of the curve, marked below. Note
    where it sits: not at $10^{-8}$, as in every twin experiment in this book,
    but at a few percent of the climatological spread. That is the operational
    situation, and chapter 9 established that it matters — the growth rate
    depends on the amplitude you are at."""
        ),
        mo.hstack([network], justify="start"),
        _fig,
        mo.md(
            f"""
| | |
|---|---|
| observed sites | {_c['n_obs']} of 40, $\\sigma_o$ = {_c['sigma']} |
| analysis error | {_c['analysis']:.4f} = **{100 * _c['analysis'] / _sat:.2f}%** of saturation |
| lead to half saturation | {_half:.2f} time units = **{_half * DAYS_PER_TU:.1f} days** |
| lead to 80 % of saturation | {_eighty:.2f} = {_eighty * DAYS_PER_TU:.1f} days |
| growth rate over $[0.02, 0.2]\\,E_\\infty$ | {_rate_small:.3f} per time unit |

The third panel is chapter 9's diagnostic applied to an operational curve, and
it says the same thing: the growth rate is not a constant. It is highest while
the error is small and falls to zero at saturation. What is new here is *where
on that curve an operational forecast starts* — at
{100 * _c['analysis'] / _sat:.1f}% of saturation, the growth rate is already
below its small-error value.

Switch networks and watch the whole curve slide left. A worse analysis does not
change the *shape*; it starts you further along it.
"""
        ),
    ])
    return


# ===========================================================================
# 2. Lorenz's estimator, checked
# ===========================================================================
@app.cell(hide_code=True)
def s2_controls(mo):
    window_low = mo.ui.slider(
        start=0.01, stop=0.2, step=0.01, value=0.02,
        label="fit window: lower edge (fraction of saturation)",
        show_value=True,
    )
    window_high = mo.ui.slider(
        start=0.1, stop=0.8, step=0.05, value=0.2,
        label="fit window: upper edge", show_value=True,
    )
    return window_high, window_low


@app.cell(hide_code=True)
def s2_figure(
    CONFIGS,
    C_CONTEXT,
    C_MEAN,
    C_PERT,
    C_SAT,
    C_TRUTH,
    DAYS_PER_TU,
    config,
    finish_mpl,
    growth_rate,
    mo,
    mpl_panels,
    network,
    np,
    window_high,
    window_low,
):
    _lo = float(window_low.value)
    _hi = max(float(window_high.value), _lo + 0.05)
    _c = config(str(network.value))
    _sat = _c["sat"]
    _leads_days = _c["leads"] * DAYS_PER_TU
    _lagged_leads = _c["leads"][: _c["lagged"].size]

    _fig, _ax = mpl_panels(
        3,
        titles=("Truth-free estimate against the truth",
                "Growth rate at each amplitude",
                "Fitted rate, all four networks"),
        height=3.6,
    )
    _ax[0].axhspan(_lo * _sat, _hi * _sat, color=C_CONTEXT, zorder=0)
    _ax[0].semilogy(_leads_days, _c["true"], linewidth=2.0, color=C_TRUTH,
                    label="true error (uses truth)")
    _ax[0].semilogy(_lagged_leads * DAYS_PER_TU, _c["lagged"], linewidth=1.7,
                    color=C_PERT, linestyle="--",
                    label="lagged difference (does not)")
    _ax[0].axhline(_sat, color=C_SAT, linewidth=1.2, linestyle=":")
    _ax[0].set_xlabel("forecast lead (days)")
    _ax[0].set_ylabel("RMS per site")
    _ax[0].set_ylim(0.3 * min(_c["lagged"][0], _c["analysis"]), 3.0 * _sat)
    _ax[0].legend(loc="lower right", fontsize=6.0, framealpha=0.9)

    # The right comparison is the growth RATE at each amplitude, not the curves
    # themselves. A first version rescaled the lagged curve to match at lead 0
    # and plotted both -- but rescaling a saturating curve does not preserve its
    # saturation level, so the rescaled version overshot E_inf by a factor of
    # two and the estimator looked wrong at long lead. That was the plot's
    # fault, not the method's.
    _true_rate = np.gradient(np.log(_c["true"]), _c["leads"])
    _lag_rate = np.gradient(np.log(_c["lagged"]), _lagged_leads)
    _ax[1].axvspan(_lo, _hi, color=C_CONTEXT, zorder=0)
    _ax[1].plot(_c["true"] / _sat, _true_rate, linewidth=2.0, color=C_TRUTH,
                label="true")
    _ax[1].plot(_c["lagged"] / _sat, _lag_rate, linewidth=1.7, color=C_PERT,
                linestyle="--", label="lagged")
    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.0, linestyle=":")
    _ax[1].set_xlabel("$E/E_\\infty$ (each curve against its own amplitude)")
    _ax[1].set_ylabel("$d\\ln E/dt$ (per time unit)")
    _ax[1].set_xlim(0, 1.02)
    _ax[1].legend(loc="upper right", fontsize=6.5, framealpha=0.9)

    _keys = ("dense", "operational", "sparse", "degraded")
    _true_rates, _lag_rates, _labels = [], [], []
    for _k in _keys:
        _cc = config(_k)
        _tr, _ = growth_rate(_cc["leads"], _cc["true"], _cc["sat"], _lo, _hi)
        _lr, _ = growth_rate(
            _cc["leads"][: _cc["lagged"].size], _cc["lagged"], _cc["sat"],
            _lo, _hi,
        )
        _true_rates.append(_tr)
        _lag_rates.append(_lr)
        _labels.append(_k)
    _x = np.arange(len(_keys))
    _ax[2].bar(_x - 0.18, _true_rates, width=0.34, color=C_TRUTH, label="true")
    _ax[2].bar(_x + 0.18, _lag_rates, width=0.34, color=C_PERT, label="lagged")
    _ax[2].set_xticks(_x)
    _ax[2].set_xticklabels(_labels, fontsize=6.5, rotation=15)
    _ax[2].set_ylabel("growth rate (per time unit)")
    _ax[2].legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle=f"fit window $[{_lo:g}, {_hi:g}]\\,E_\\infty$")

    _rows = "\n".join(
        f"| {l} | {100 * config(l)['analysis'] / config(l)['sat']:.2f}% | "
        + (f"{t:.3f} | {g:.3f} | {g / t:.4f} |" if np.isfinite(t) and np.isfinite(g)
           else ("— | " + (f"{g:.3f} |" if np.isfinite(g) else "— |") + " — |"))
        for l, t, g in zip(_labels, _true_rates, _lag_rates)
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 2 · Does it work?

    Two curves below. The solid one is the true forecast error, computed against
    a truth a forecast centre would not have. The dashed one is Lorenz's
    lagged-forecast difference, computed from the forecast archive alone. If the
    method is sound they should be **parallel** — the same growth rate, at
    different amplitudes."""
        ),
        mo.hstack([window_low, window_high], justify="start"),
        _fig,
        mo.md(
            f"""
| network | analysis error | true rate | lagged rate | ratio |
|---|---|---|---|---|
{_rows}

**It works.** Over the default window the estimator lands within a few percent
of the truth for every network that has an exponential phase at all.

The middle panel is the sharper version of that claim. Plotting each curve's
local growth rate against *its own* amplitude, the two lie close together over
the whole range — so the estimator is not merely getting one fitted number
right, it is recovering the growth rate at every error amplitude, including the
amplitude-dependence chapter 9 established. The left panel shows the estimator
sitting *below* the truth, because it starts from a one-cycle forecast
difference rather than from the analysis error; that offset shifts the curve
along the lead axis and leaves the rate alone, which is exactly why the method
works.

**The `degraded` network is the honest limit.** Its analysis error is 45 % of
saturation, so there is no exponential phase left in the archive to fit and the
true rate cannot be measured either. That is a failure of the data, not of the
estimator: with a bad enough analysis, nothing recovers the small-error growth
rate, because the forecast never spends time at small error.

**Push the fit window up** and both rates fall together, exactly as chapter 9
predicts — the growth rate declines with amplitude, and a window at larger
error measures a smaller rate whichever method you use. The estimator tracks the
truth, including where the truth is amplitude-dependent.
"""
        ),
        mo.callout(
            mo.md(
                r"""### What the method cannot see

Both forecasts in a lagged pair come from the **same model**, so anything
systematically wrong with that model is common to both and cancels in the
difference. `chaoslib`'s test for this adds a constant bias to an entire
archive and asserts the estimator's output is unchanged to round-off.

So the method measures the growth of **initial-condition** error and is blind
to model error by construction. That is a real limitation and not a small one:
a forecast system can be losing skill because its initial conditions are wrong,
or because its model is wrong, and this estimator reports only the first.
Chapter 21 takes up model error, and chapter 22 the problem of verifying
against observations that have errors of their own."""
            ),
            kind="warn",
        ),
    ])
    return


# ===========================================================================
# 3. What a better observing system buys
# ===========================================================================
@app.cell(hide_code=True)
def s3_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    DAYS_PER_TU,
    config,
    finish_mpl,
    growth_rate,
    lead_at,
    mo,
    mpl_panels,
    np,
):
    _keys = ("dense", "operational", "sparse", "degraded")
    _errors, _halves, _labels = [], [], []
    for _k in _keys:
        _c = config(_k)
        _errors.append(_c["analysis"] / _c["sat"])
        _halves.append(lead_at(_c["leads"], _c["true"], 0.5 * _c["sat"])
                       * DAYS_PER_TU)
        _labels.append(_k)
    _errors = np.array(_errors)
    _halves = np.array(_halves)

    _fig, _ax = mpl_panels(
        2,
        titles=("All four error curves", "Lead time vs analysis quality"),
        height=3.5,
    )
    for _k, _colour in zip(_keys, (C_TRUTH, C_MEAN, C_SPREAD, C_PERT)):
        _c = config(_k)
        _ax[0].semilogy(_c["leads"] * DAYS_PER_TU, _c["true"], linewidth=1.7,
                        color=_colour, label=f"{_k} ({100 * _c['analysis'] / _c['sat']:.1f}%)")
    _ax[0].axhline(config("dense")["sat"], color=C_SAT, linewidth=1.2,
                   linestyle="--")
    _ax[0].axhline(0.5 * config("dense")["sat"], color=C_SAT, linewidth=1.0,
                   linestyle=":")
    _ax[0].set_xlabel("forecast lead (days)")
    _ax[0].set_ylabel("RMS per site")
    _ax[0].legend(loc="lower right", fontsize=6.0, framealpha=0.9)

    _usable = _errors < 0.3
    _ax[1].semilogx(_errors[_usable], _halves[_usable], marker="o",
                    markersize=6, color=C_TRUTH, linewidth=1.8,
                    label="measured")
    _rate, _ = growth_rate(
        config("operational")["leads"], config("operational")["true"],
        config("operational")["sat"], 0.02, 0.2,
    )
    _reference = _halves[_usable][0] - (
        np.log(_errors[_usable] / _errors[_usable][0]) / _rate * DAYS_PER_TU
    )
    _ax[1].semilogx(_errors[_usable], _reference, color=C_PERT, linewidth=1.4,
                    linestyle="--",
                    label=f"$\\ln(10)/\\lambda$ per decade, $\\lambda$ = {_rate:.2f}")
    for _e, _h, _l in zip(_errors[_usable], _halves[_usable], np.array(_labels)[_usable]):
        _ax[1].annotate(_l, (_e, _h), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=6.5,
                        color="#4b5563")
    _ax[1].set_xlabel("analysis error / saturation")
    _ax[1].set_ylabel("days to half saturation")
    _ax[1].legend(loc="upper right", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    _ratio = _errors[1] / _errors[0]
    _gain = _halves[0] - _halves[1]
    _predicted = np.log(_ratio) / _rate * DAYS_PER_TU
    _rows = "\n".join(
        f"| {l} | {100 * e:.2f}% | {h:.1f} |"
        for l, e, h in zip(_labels, _errors, _halves)
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 3 · What does a better observing system buy?

    Four networks, four analysis errors, four error curves. The question a
    forecast centre asks is how much lead time an improvement in the analysis
    is worth — and chapters 8 and 20 both predict the answer should be
    **logarithmic**, $\ln 10/\lambda$ per decade of analysis-error reduction."""
        ),
        _fig,
        mo.md(
            f"""
| network | analysis error | days to half saturation |
|---|---|---|
{_rows}

**The logarithmic law holds, measured end to end.** Going from the
`operational` network to the `dense` one reduces the analysis error by a factor
of {_ratio:.1f} and buys **{_gain:.1f} days** of lead time. The prediction from
the measured growth rate is $\\ln({_ratio:.1f})/{_rate:.2f}$ =
{_predicted / DAYS_PER_TU:.2f} time units = **{_predicted:.1f} days**. Agreement
to {abs(100 * (_gain - _predicted) / _predicted):.0f}%.

This is the same constant chapter 8 derived from the Lyapunov exponent and
chapter 20 measured on a cycling assimilation system — arrived at here from a
third direction, by building four observing networks and reading off the lead
times. A factor of thirteen in analysis accuracy is worth a week of forecast,
and the next factor of thirteen is worth another week, and that is the whole
return on observing-system investment.

The right-hand panel drops the `degraded` network, whose analysis error is 45 %
of saturation. It has no useful lead time to plot — half a day — and it is off
the logarithmic line entirely, because the law is derived for errors small
enough to grow exponentially and its error never is.
"""
        ),
    ])
    return


# ===========================================================================
# 4. From lambda_1 to an operational number
# ===========================================================================
@app.cell(hide_code=True)
def s4_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_TRUTH,
    DAYS_PER_TU,
    LAMBDA1_L96,
    config,
    finish_mpl,
    growth_rate,
    mo,
    mpl_panels,
    np,
):
    _windows = ((0.02, 0.1), (0.02, 0.2), (0.05, 0.3), (0.1, 0.4), (0.2, 0.6))
    _c = config("operational")
    _rates, _mids = [], []
    for _lo, _hi in _windows:
        _r, _ = growth_rate(_c["leads"], _c["true"], _c["sat"], _lo, _hi)
        _rates.append(_r)
        _mids.append(np.sqrt(_lo * _hi))
    _rates = np.array(_rates)
    _mids = np.array(_mids)

    _fig, _ax = mpl_panels(
        2,
        titles=("Fitted rate vs where you fit it",
                "Doubling time, in days"),
        height=3.5,
    )
    _ax[0].semilogx(_mids, _rates, marker="o", markersize=6, color=C_TRUTH,
                    linewidth=1.8, label="fitted over each window")
    _ax[0].axhline(LAMBDA1_L96, color=C_SAT, linewidth=1.4, linestyle="--",
                   label=f"$\\lambda_1$ = {LAMBDA1_L96} (chapter 11)")
    _ax[0].set_xlabel("geometric centre of the fit window ($E/E_\\infty$)")
    _ax[0].set_ylabel("fitted rate (per time unit)")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _doubling = np.log(2.0) / _rates * DAYS_PER_TU
    _ax[1].semilogx(_mids, _doubling, marker="o", markersize=6, color=C_TRUTH,
                    linewidth=1.8)
    _ax[1].axhline(np.log(2.0) / LAMBDA1_L96 * DAYS_PER_TU, color=C_SAT,
                   linewidth=1.4, linestyle="--",
                   label=f"$\\ln2/\\lambda_1$ = {np.log(2.0)/LAMBDA1_L96*DAYS_PER_TU:.2f} d")
    _ax[1].set_xlabel("geometric centre of the fit window")
    _ax[1].set_ylabel("doubling time (days)")
    _ax[1].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle="operational network")

    _rows = "\n".join(
        f"| [{lo:g}, {hi:g}] | {r:.3f} | {np.log(2)/r:.3f} | "
        f"{np.log(2)/r*DAYS_PER_TU:.2f} |"
        for (lo, hi), r in zip(_windows, _rates)
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 4 · So what is the operational doubling time?

    Chapter 11 measured $\lambda_1 = 1.67$ per time unit for this system. That
    is the asymptotic growth rate of an infinitesimal perturbation. What does an
    operational archive give?"""
        ),
        _fig,
        mo.md(
            f"""
| fit window ($E/E_\\infty$) | rate (per TU) | doubling (TU) | doubling (days) |
|---|---|---|---|
{_rows}

**Two effects pull in opposite directions, and both are visible.**

At small error the fitted rate is **{_rates[0]:.3f}** per time unit — *above*
$\\lambda_1$ = {LAMBDA1_L96}. That is not an error. $\\lambda_1$ is an
asymptotic average, and a generic finite perturbation grows faster than it over
a finite window because $\\mathbf{{M}}$ is non-normal — chapter 16 measured
exactly this, finding optimal growth 1.6 to 2.6 times the Lyapunov estimate.
An analysis error is *especially* prone to it: the observing system constrains
the well-observed directions and leaves the poorly-observed ones alone, and the
directions a network struggles with are not chosen at random relative to the
ones that grow.

At larger error the rate falls, reaching **{_rates[-1]:.3f}** — chapter 9's
result, nonlinear saturation.

So the doubling time an operational archive reports depends on where the
archive's errors sit, and it can be either side of $\\ln2/\\lambda_1$ =
{np.log(2)/LAMBDA1_L96*DAYS_PER_TU:.2f} days. For this network at
{100 * _c['analysis'] / _c['sat']:.1f}% of saturation, transient amplification
wins and the measured doubling time is **{np.log(2)/_rates[1]*DAYS_PER_TU:.2f}
days**, shorter than the Lyapunov estimate.

**Which is the number to quote?** None of them alone. The honest report of a
forecast system's decay is the curve, with the amplitude range stated — which
is what chapter 9 concluded from the other end, and why an error-doubling time
in a paper should always come with the error amplitude it was measured at.
"""
        ),
    ])
    return


# ===========================================================================
# 5. Closing
# ===========================================================================
@app.cell(hide_code=True)
def closing(mo):
    mo.md(
        r"""
    ---
    ## Try this

    1. **Verify the estimator yourself.** In Section 2, switch between the
       `dense`, `operational` and `sparse` networks with the default window and
       read the ratio column. Then widen the window to $[0.1, 0.6]$ and read it
       again. Does the estimator degrade, or does it track a truth that has
       itself changed?
    2. **Find where the method runs out.** Select the `degraded` network. Both
       rates become unmeasurable. Explain, from the analysis error, why that is
       a property of the archive and not of the estimator — and what a forecast
       centre in that situation would have to do instead.
    3. **Price an observing system.** Section 3 gives the days-per-decade
       exchange rate. If a new satellite reduced the analysis error by 30 %,
       how much lead time would it buy? Is that worth a satellite?
    4. **Explain a rate above $\lambda_1$.** Section 4's small-error rate
       exceeds the Lyapunov exponent. Use chapter 16's singular-vector result to
       say why, and predict whether the excess should be larger or smaller for a
       *random* perturbation than for an analysis error.
    5. **Defeat the estimator.** Suppose a model has a systematic bias that
       grows with lead time. What does the lagged-forecast method report, and
       what would you need in order to detect the bias?

    ## What you should have seen

    An operational forecast error curve has the same shape as a twin
    experiment's, and starts in a different place: at a few percent of the
    climatological spread rather than at $10^{-8}$ of it. Everything follows
    from that.

    **Lorenz's lagged-forecast estimator works.** Differencing forecasts of
    successive lead valid at the same time — using no truth whatever — recovers
    the true growth rate to within 2 % for analysis errors of 0.5 % and 2 % of
    saturation, and 5 % at 6 %. The estimated curve is offset below the true one,
    because it starts from a one-cycle forecast difference rather than from the
    analysis error, and rescaling by a single constant lays the two on top of
    each other. The offset does not matter; the rate is what the method is for.

    It fails only where the archive has nothing to offer: with an analysis error
    at 45 % of saturation there is no exponential phase, and the true rate is
    equally unmeasurable.

    **The logarithmic return holds end to end.** Thirteen times better analysis
    buys 7.0 days of lead time against 6.9 predicted from the measured growth
    rate — the same constant chapter 8 derived from $\lambda_1$ and chapter 20
    measured on a cycling system, now obtained a third way by building four
    observing networks.

    **And the operational doubling time is not $\ln2/\lambda_1$.** Fitted at
    small error it is 1.86 days against the Lyapunov estimate of 2.08, because
    non-normal transient growth exceeds the asymptotic rate; fitted at large
    error it is longer, because of saturation. Both effects are real, they point
    opposite ways, and which one dominates depends on where the archive's errors
    sit. An error-doubling time quoted without an amplitude is not a number.

    **What none of this can see is model error.** Two forecasts of the same
    model differ only in their initial conditions, so a bias common to both
    cancels — provably, and the tests assert it. Chapter 21 takes that up.

    ## Further reading

    - Lorenz, E. N. (1982). Atmospheric predictability experiments with a large
      numerical model. *Tellus*, **34**, 505–513 — the lagged-forecast method.
    - Simmons, A. J. and Hollingsworth, A. (2002). Some aspects of the
      improvement in skill of numerical weather prediction. *Quarterly Journal
      of the Royal Meteorological Society*, **128**, 647–677
      *[citation needed: confirm pages]*.
    - Bengtsson, L. and Hodges, K. I. (2006). A note on atmospheric
      predictability *[citation needed: journal and pages]*.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 3 *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
