# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 21 -- Model error and the imperfect-model problem.

Every forecast so far assumed a perfect model. Drop that and the return on
better observations stops -- at a level the model's own error decides.

Part V of *An Interactive Chaos and Predictability Textbook*.

To edit:   marimo edit notebooks/ch21_model-error.py
To export: make nb-one NB=ch21_model-error
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 21: Model Error")


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
    mpl_panels = plotting.mpl_panels
    finish_mpl = plotting.finish_mpl

    DAYS_PER_TU = 5.0
    FORCING = 8.0

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
        DAYS_PER_TU,
        FORCING,
        finish_mpl,
        mo,
        mpl_panels,
        np,
    )


# ---------------------------------------------------------------------------
# Precomputed error curves
# ---------------------------------------------------------------------------
@app.cell
def error_data():
    # From scripts/generate_ch21_data.py. Truth is Lorenz 96 at F = 8; three
    # ways of being wrong about it, each measured over 48 start states, plus the
    # combined sweep of initial-condition accuracy against model bias.
    #
    # Per component throughout (RMS over the 40 sites), with the saturation
    # level computed the same way -- saturation_level returns a norm over the
    # whole state, so it is divided by sqrt(N). That factor of 6.3 has caused
    # trouble twice already in this book.
    SATURATION = 5.111441
    LAMBDA1 = 1.67
    IC_AMPLITUDES = (0.01, 0.001, 0.0001, 1e-06, 1e-08)
    BIASES = (0.01, 0.05, 0.2)
    NOISES = (0.02, 0.05, 0.2)
    SWEEP_BIASES = (0.0, 0.01, 0.05, 0.2)
    TIMES = (
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
    IC_CURVES = (
        1.000000e-02, 1.001681e-02, 1.073816e-02, 1.193149e-02,
        1.347915e-02, 1.533123e-02, 1.744739e-02, 1.977071e-02,
        2.227575e-02, 2.499504e-02, 2.797445e-02, 3.122093e-02,
        3.468476e-02, 3.840134e-02, 4.256498e-02, 4.737675e-02,
        5.274785e-02, 5.836367e-02, 6.425748e-02, 7.097303e-02,
        7.913208e-02, 8.895040e-02, 1.001202e-01, 1.121132e-01,
        1.245990e-01, 1.377762e-01, 1.522192e-01, 1.682944e-01,
        1.861793e-01, 2.057525e-01, 2.262777e-01, 2.475639e-01,
        2.714463e-01, 3.005227e-01, 3.344862e-01, 3.703490e-01,
        4.063016e-01, 4.433667e-01, 4.832133e-01, 5.259495e-01,
        5.703947e-01, 6.143716e-01, 6.568558e-01, 7.003164e-01,
        7.472417e-01, 7.966036e-01, 8.454154e-01, 8.941486e-01,
        9.470686e-01, 1.007083e+00, 1.076384e+00, 1.158787e+00,
        1.256457e+00, 1.367332e+00, 1.486089e+00, 1.605675e+00,
        1.719252e+00, 1.819325e+00, 1.898475e+00, 1.958559e+00,
        2.010335e+00, 2.060124e+00, 2.113800e+00, 2.182227e+00,
        2.269350e+00, 2.366229e+00, 2.460824e+00, 2.544544e+00,
        2.613983e+00, 2.673379e+00, 2.734643e+00, 2.801873e+00,
        2.866584e+00, 2.926678e+00, 2.990368e+00, 3.063491e+00,
        3.145309e+00, 3.234067e+00, 3.323639e+00, 3.404187e+00,
        3.473745e+00, 3.537623e+00, 3.595986e+00, 3.641699e+00,
        3.671158e+00, 3.690684e+00, 3.712439e+00, 3.745622e+00,
        3.788901e+00, 3.834613e+00, 3.876556e+00, 3.917487e+00,
        3.969098e+00, 4.034726e+00, 4.103674e+00, 4.165364e+00,
        4.218369e+00, 4.270194e+00, 4.328625e+00, 4.389841e+00,
        4.443243e+00, 4.485825e+00, 4.522312e+00, 4.557203e+00,
        4.589831e+00, 4.616380e+00, 4.636543e+00, 4.653711e+00,
        4.669437e+00, 4.682902e+00, 4.700870e+00, 4.737458e+00,
        4.798627e+00, 4.870333e+00, 4.930094e+00, 4.965662e+00,
        4.981482e+00, 4.991734e+00, 5.005267e+00, 5.016936e+00,
        5.012780e+00, 1.000000e-03, 1.001682e-03, 1.073822e-03,
        1.193167e-03, 1.347958e-03, 1.533202e-03, 1.744843e-03,
        1.977161e-03, 2.227622e-03, 2.499612e-03, 2.797863e-03,
        3.123066e-03, 3.470115e-03, 3.842362e-03, 4.259302e-03,
        4.741770e-03, 5.282188e-03, 5.850977e-03, 6.454584e-03,
        7.151494e-03, 8.009482e-03, 9.058053e-03, 1.027061e-02,
        1.158341e-02, 1.294096e-02, 1.435351e-02, 1.589699e-02,
        1.764306e-02, 1.963173e-02, 2.181736e-02, 2.405341e-02,
        2.630498e-02, 2.880471e-02, 3.182043e-02, 3.537124e-02,
        3.944708e-02, 4.423408e-02, 4.989762e-02, 5.640200e-02,
        6.355342e-02, 7.109878e-02, 7.906315e-02, 8.781370e-02,
        9.701222e-02, 1.059675e-01, 1.147468e-01, 1.239931e-01,
        1.349482e-01, 1.491696e-01, 1.668135e-01, 1.854286e-01,
        2.020796e-01, 2.168237e-01, 2.319015e-01, 2.484854e-01,
        2.677115e-01, 2.923968e-01, 3.244408e-01, 3.624178e-01,
        4.042174e-01, 4.481556e-01, 4.922077e-01, 5.360013e-01,
        5.824032e-01, 6.350030e-01, 6.943952e-01, 7.584852e-01,
        8.234385e-01, 8.875126e-01, 9.533701e-01, 1.023050e+00,
        1.093319e+00, 1.157575e+00, 1.213672e+00, 1.267160e+00,
        1.324741e+00, 1.391100e+00, 1.470697e+00, 1.563903e+00,
        1.663078e+00, 1.754571e+00, 1.831094e+00, 1.897132e+00,
        1.955869e+00, 2.008552e+00, 2.061071e+00, 2.118974e+00,
        2.182366e+00, 2.243803e+00, 2.298463e+00, 2.355507e+00,
        2.423833e+00, 2.497700e+00, 2.568757e+00, 2.635702e+00,
        2.698023e+00, 2.753355e+00, 2.803843e+00, 2.859241e+00,
        2.924978e+00, 2.995641e+00, 3.063581e+00, 3.128561e+00,
        3.199857e+00, 3.282265e+00, 3.367377e+00, 3.445827e+00,
        3.515361e+00, 3.579274e+00, 3.636114e+00, 3.683104e+00,
        3.729666e+00, 3.787825e+00, 3.860990e+00, 3.945609e+00,
        4.031148e+00, 4.101473e+00, 4.145909e+00, 4.171642e+00,
        4.198571e+00, 4.236182e+00, 1.000000e-04, 1.001682e-04,
        1.073822e-04, 1.193169e-04, 1.347962e-04, 1.533209e-04,
        1.744853e-04, 1.977169e-04, 2.227626e-04, 2.499622e-04,
        2.797903e-04, 3.123159e-04, 3.470272e-04, 3.842574e-04,
        4.259561e-04, 4.742144e-04, 5.282872e-04, 5.852342e-04,
        6.457271e-04, 7.156497e-04, 8.018328e-04, 9.073225e-04,
        1.029555e-03, 1.162118e-03, 1.299238e-03, 1.441724e-03,
        1.597235e-03, 1.773403e-03, 1.974911e-03, 2.197280e-03,
        2.424759e-03, 2.652559e-03, 2.903638e-03, 3.204046e-03,
        3.555044e-03, 3.958131e-03, 4.436163e-03, 5.008313e-03,
        5.674255e-03, 6.423164e-03, 7.253751e-03, 8.215687e-03,
        9.384474e-03, 1.072127e-02, 1.208230e-02, 1.328906e-02,
        1.420729e-02, 1.489433e-02, 1.557688e-02, 1.640266e-02,
        1.729798e-02, 1.821841e-02, 1.932215e-02, 2.077368e-02,
        2.261313e-02, 2.486927e-02, 2.769845e-02, 3.122858e-02,
        3.536122e-02, 4.005223e-02, 4.533323e-02, 5.113891e-02,
        5.751308e-02, 6.480661e-02, 7.328940e-02, 8.296637e-02,
        9.368297e-02, 1.052385e-01, 1.180579e-01, 1.327232e-01,
        1.495663e-01, 1.683593e-01, 1.883288e-01, 2.088380e-01,
        2.290741e-01, 2.482591e-01, 2.665558e-01, 2.863711e-01,
        3.099743e-01, 3.365548e-01, 3.645619e-01, 3.927756e-01,
        4.203531e-01, 4.477823e-01, 4.763239e-01, 5.067337e-01,
        5.409088e-01, 5.814285e-01, 6.270681e-01, 6.742000e-01,
        7.214478e-01, 7.703957e-01, 8.244647e-01, 8.852473e-01,
        9.526589e-01, 1.026203e+00, 1.100775e+00, 1.168909e+00,
        1.228434e+00, 1.283376e+00, 1.337825e+00, 1.391968e+00,
        1.444765e+00, 1.500736e+00, 1.566891e+00, 1.641885e+00,
        1.714962e+00, 1.777066e+00, 1.828785e+00, 1.882002e+00,
        1.949663e+00, 2.030875e+00, 2.111335e+00, 2.178336e+00,
        2.236452e+00, 2.300120e+00, 2.376672e+00, 2.463298e+00,
        2.548737e+00, 2.620550e+00, 2.676104e+00, 1.000000e-06,
        1.001682e-06, 1.073823e-06, 1.193169e-06, 1.347963e-06,
        1.533210e-06, 1.744855e-06, 1.977170e-06, 2.227626e-06,
        2.499623e-06, 2.797907e-06, 3.123169e-06, 3.470289e-06,
        3.842597e-06, 4.259590e-06, 4.742185e-06, 5.282947e-06,
        5.852491e-06, 6.457564e-06, 7.157041e-06, 8.019288e-06,
        9.074872e-06, 1.029827e-05, 1.162531e-05, 1.299805e-05,
        1.442430e-05, 1.598070e-05, 1.774410e-05, 1.976211e-05,
        2.199015e-05, 2.426954e-05, 2.655074e-05, 2.906288e-05,
        3.206571e-05, 3.557105e-05, 3.959665e-05, 4.437591e-05,
        5.010398e-05, 5.678282e-05, 6.431786e-05, 7.273132e-05,
        8.258632e-05, 9.471129e-05, 1.087819e-04, 1.233667e-04,
        1.364883e-04, 1.464264e-04, 1.536239e-04, 1.607927e-04,
        1.696331e-04, 1.785405e-04, 1.865718e-04, 1.962658e-04,
        2.103018e-04, 2.293884e-04, 2.538206e-04, 2.854493e-04,
        3.259197e-04, 3.745762e-04, 4.309788e-04, 4.946439e-04,
        5.634767e-04, 6.360523e-04, 7.134575e-04, 7.957035e-04,
        8.816227e-04, 9.706728e-04, 1.064330e-03, 1.171533e-03,
        1.299110e-03, 1.444887e-03, 1.601375e-03, 1.770477e-03,
        1.970535e-03, 2.204530e-03, 2.454073e-03, 2.702365e-03,
        2.942357e-03, 3.185105e-03, 3.432550e-03, 3.678329e-03,
        3.949412e-03, 4.285851e-03, 4.705138e-03, 5.206737e-03,
        5.764537e-03, 6.361841e-03, 7.029270e-03, 7.800145e-03,
        8.671453e-03, 9.639194e-03, 1.071601e-02, 1.188494e-02,
        1.310401e-02, 1.442314e-02, 1.601433e-02, 1.796960e-02,
        2.014500e-02, 2.237569e-02, 2.475280e-02, 2.757060e-02,
        3.102987e-02, 3.508524e-02, 3.957535e-02, 4.434043e-02,
        4.935401e-02, 5.464227e-02, 6.003416e-02, 6.548645e-02,
        7.130440e-02, 7.764990e-02, 8.505029e-02, 9.460198e-02,
        1.068833e-01, 1.215137e-01, 1.373703e-01, 1.531825e-01,
        1.685350e-01, 1.840920e-01, 2.007797e-01, 2.191114e-01,
        1.000000e-08, 1.001682e-08, 1.073823e-08, 1.193169e-08,
        1.347963e-08, 1.533210e-08, 1.744855e-08, 1.977170e-08,
        2.227626e-08, 2.499623e-08, 2.797907e-08, 3.123169e-08,
        3.470289e-08, 3.842597e-08, 4.259590e-08, 4.742185e-08,
        5.282948e-08, 5.852492e-08, 6.457567e-08, 7.157047e-08,
        8.019298e-08, 9.074889e-08, 1.029830e-07, 1.162536e-07,
        1.299810e-07, 1.442437e-07, 1.598079e-07, 1.774420e-07,
        1.976224e-07, 2.199033e-07, 2.426976e-07, 2.655099e-07,
        2.906315e-07, 3.206596e-07, 3.557126e-07, 3.959680e-07,
        4.437605e-07, 5.010419e-07, 5.678322e-07, 6.431873e-07,
        7.273327e-07, 8.259064e-07, 9.472002e-07, 1.087978e-06,
        1.233929e-06, 1.365261e-06, 1.464734e-06, 1.536758e-06,
        1.608503e-06, 1.697006e-06, 1.786123e-06, 1.866341e-06,
        1.963138e-06, 2.103439e-06, 2.294385e-06, 2.538917e-06,
        2.855557e-06, 3.260781e-06, 3.748057e-06, 4.312953e-06,
        4.950499e-06, 5.639537e-06, 6.365567e-06, 7.139183e-06,
        7.960281e-06, 8.817236e-06, 9.705012e-06, 1.063894e-05,
        1.170876e-05, 1.298240e-05, 1.443709e-05, 1.599775e-05,
        1.768569e-05, 1.968625e-05, 2.202782e-05, 2.452692e-05,
        2.701641e-05, 2.942265e-05, 3.185102e-05, 3.432194e-05,
        3.677776e-05, 3.949048e-05, 4.285558e-05, 4.704129e-05,
        5.204177e-05, 5.760331e-05, 6.356932e-05, 7.025253e-05,
        7.798385e-05, 8.672515e-05, 9.642389e-05, 1.071958e-04,
        1.188609e-04, 1.309719e-04, 1.440072e-04, 1.597223e-04,
        1.791365e-04, 2.009019e-04, 2.234026e-04, 2.474858e-04,
        2.760381e-04, 3.112043e-04, 3.526110e-04, 3.982882e-04,
        4.460029e-04, 4.955234e-04, 5.488993e-04, 6.068561e-04,
        6.672157e-04, 7.257137e-04, 7.807973e-04, 8.409728e-04,
        9.197506e-04, 1.020814e-03, 1.134881e-03, 1.252055e-03,
        1.370654e-03, 1.497435e-03, 1.639085e-03, 1.798580e-03,
        1.971704e-03,
    )
    BIAS_CURVES = (
        0.000000e+00, 4.875478e-04, 9.534121e-04, 1.411613e-03,
        1.888889e-03, 2.420044e-03, 3.038750e-03, 3.767609e-03,
        4.610848e-03, 5.550155e-03, 6.559813e-03, 7.628612e-03,
        8.754069e-03, 9.917178e-03, 1.110357e-02, 1.233161e-02,
        1.355376e-02, 1.468363e-02, 1.569743e-02, 1.670165e-02,
        1.783375e-02, 1.906911e-02, 2.032694e-02, 2.162809e-02,
        2.302566e-02, 2.451524e-02, 2.605408e-02, 2.773994e-02,
        2.970529e-02, 3.194830e-02, 3.450903e-02, 3.747887e-02,
        4.083325e-02, 4.443571e-02, 4.823879e-02, 5.246596e-02,
        5.764688e-02, 6.438204e-02, 7.279892e-02, 8.264696e-02,
        9.406660e-02, 1.079102e-01, 1.245569e-01, 1.427814e-01,
        1.607330e-01, 1.769722e-01, 1.912237e-01, 2.044292e-01,
        2.178053e-01, 2.324156e-01, 2.492390e-01, 2.679874e-01,
        2.870310e-01, 3.057993e-01, 3.254973e-01, 3.478796e-01,
        3.742413e-01, 4.045263e-01, 4.373527e-01, 4.743088e-01,
        5.202568e-01, 5.757956e-01, 6.366573e-01, 6.981292e-01,
        7.556117e-01, 8.078955e-01, 8.570404e-01, 9.029574e-01,
        9.456399e-01, 9.874045e-01, 1.030702e+00, 1.079556e+00,
        1.137883e+00, 1.203228e+00, 1.269328e+00, 1.331996e+00,
        1.388618e+00, 1.441075e+00, 1.498026e+00, 1.566354e+00,
        1.642629e+00, 1.720086e+00, 1.794383e+00, 1.862926e+00,
        1.923595e+00, 1.979439e+00, 2.039373e+00, 2.109407e+00,
        2.186664e+00, 2.262094e+00, 2.331704e+00, 2.401245e+00,
        2.478827e+00, 2.568645e+00, 2.669248e+00, 2.778332e+00,
        2.892253e+00, 3.003198e+00, 3.106046e+00, 3.199587e+00,
        3.285984e+00, 3.368609e+00, 3.449280e+00, 3.527237e+00,
        3.598933e+00, 3.660691e+00, 3.711170e+00, 3.751413e+00,
        3.784259e+00, 3.815964e+00, 3.858398e+00, 3.921069e+00,
        4.004305e+00, 4.097536e+00, 4.186883e+00, 4.267293e+00,
        4.337187e+00, 4.391101e+00, 4.428031e+00, 4.457124e+00,
        4.487948e+00, 0.000000e+00, 2.437739e-03, 4.767079e-03,
        7.058203e-03, 9.445020e-03, 1.210190e-02, 1.519758e-02,
        1.884535e-02, 2.306621e-02, 2.776822e-02, 3.282312e-02,
        3.817549e-02, 4.381157e-02, 4.963417e-02, 5.557434e-02,
        6.171999e-02, 6.782220e-02, 7.345518e-02, 7.851990e-02,
        8.357226e-02, 8.928635e-02, 9.551018e-02, 1.018622e-01,
        1.084581e-01, 1.155280e-01, 1.230187e-01, 1.307538e-01,
        1.393275e-01, 1.494748e-01, 1.612436e-01, 1.748217e-01,
        1.905145e-01, 2.081158e-01, 2.269575e-01, 2.468157e-01,
        2.685317e-01, 2.932289e-01, 3.208864e-01, 3.495410e-01,
        3.768286e-01, 4.030973e-01, 4.309063e-01, 4.607955e-01,
        4.924682e-01, 5.275496e-01, 5.683969e-01, 6.147013e-01,
        6.641934e-01, 7.170111e-01, 7.740368e-01, 8.315164e-01,
        8.846212e-01, 9.347053e-01, 9.883034e-01, 1.051377e+00,
        1.123827e+00, 1.199959e+00, 1.273576e+00, 1.341901e+00,
        1.405624e+00, 1.468138e+00, 1.537646e+00, 1.624886e+00,
        1.726735e+00, 1.826376e+00, 1.914517e+00, 1.992434e+00,
        2.063296e+00, 2.133056e+00, 2.212940e+00, 2.308491e+00,
        2.411808e+00, 2.511285e+00, 2.601607e+00, 2.682773e+00,
        2.758792e+00, 2.839385e+00, 2.929487e+00, 3.025838e+00,
        3.119276e+00, 3.202258e+00, 3.276890e+00, 3.350337e+00,
        3.427330e+00, 3.506491e+00, 3.579758e+00, 3.642923e+00,
        3.706308e+00, 3.781862e+00, 3.864313e+00, 3.938121e+00,
        3.994976e+00, 4.033751e+00, 4.058748e+00, 4.077676e+00,
        4.095960e+00, 4.117906e+00, 4.147703e+00, 4.187360e+00,
        4.238577e+00, 4.303169e+00, 4.376316e+00, 4.448044e+00,
        4.510307e+00, 4.555731e+00, 4.580285e+00, 4.592886e+00,
        4.611189e+00, 4.643411e+00, 4.681902e+00, 4.714573e+00,
        4.738190e+00, 4.758537e+00, 4.776404e+00, 4.786679e+00,
        4.794605e+00, 4.814170e+00, 4.849300e+00, 4.893119e+00,
        4.939973e+00, 4.982599e+00, 0.000000e+00, 9.750966e-03,
        1.906859e-02, 2.823491e-02, 3.778873e-02, 4.843272e-02,
        6.084768e-02, 7.549051e-02, 9.244299e-02, 1.113311e-01,
        1.316465e-01, 1.531751e-01, 1.758396e-01, 1.992200e-01,
        2.230827e-01, 2.477067e-01, 2.719459e-01, 2.942107e-01,
        3.143901e-01, 3.350082e-01, 3.584765e-01, 3.838427e-01,
        4.099859e-01, 4.373988e-01, 4.664056e-01, 4.962603e-01,
        5.270290e-01, 5.626967e-01, 6.063294e-01, 6.574383e-01,
        7.145956e-01, 7.749458e-01, 8.358535e-01, 8.971810e-01,
        9.615942e-01, 1.031444e+00, 1.106789e+00, 1.186425e+00,
        1.268791e+00, 1.351693e+00, 1.433951e+00, 1.516275e+00,
        1.599795e+00, 1.685851e+00, 1.776185e+00, 1.866518e+00,
        1.949176e+00, 2.026462e+00, 2.103850e+00, 2.183277e+00,
        2.270693e+00, 2.374649e+00, 2.493910e+00, 2.618541e+00,
        2.741773e+00, 2.864325e+00, 2.988123e+00, 3.109055e+00,
        3.217042e+00, 3.304888e+00, 3.377879e+00, 3.447500e+00,
        3.516853e+00, 3.579716e+00, 3.632093e+00, 3.679533e+00,
        3.727291e+00, 3.773204e+00, 3.821235e+00, 3.875051e+00,
        3.929782e+00, 3.973802e+00, 3.998013e+00, 4.015661e+00,
        4.050110e+00, 4.106358e+00, 4.171169e+00, 4.228785e+00,
        4.272304e+00, 4.305570e+00, 4.334984e+00, 4.363847e+00,
        4.398398e+00, 4.446451e+00, 4.503749e+00, 4.561982e+00,
        4.617743e+00, 4.670579e+00, 4.720198e+00, 4.758275e+00,
        4.775341e+00, 4.775173e+00, 4.774848e+00, 4.789931e+00,
        4.820498e+00, 4.852594e+00, 4.870549e+00, 4.869046e+00,
        4.856265e+00, 4.849878e+00, 4.864743e+00, 4.898990e+00,
        4.937220e+00, 4.967825e+00, 4.991878e+00, 5.018661e+00,
        5.052827e+00, 5.087762e+00, 5.113344e+00, 5.119007e+00,
        5.100347e+00, 5.069443e+00, 5.044571e+00, 5.034196e+00,
        5.037111e+00, 5.049578e+00, 5.068687e+00, 5.094186e+00,
        5.128633e+00, 5.171963e+00, 5.211085e+00,
    )
    NOISE_CURVES = (
        0.000000e+00, 4.355431e-03, 6.291452e-03, 8.236700e-03,
        9.929952e-03, 1.195323e-02, 1.412974e-02, 1.642626e-02,
        1.878775e-02, 2.169950e-02, 2.465081e-02, 2.795233e-02,
        3.115400e-02, 3.459589e-02, 3.897463e-02, 4.352961e-02,
        4.848779e-02, 5.345058e-02, 5.875276e-02, 6.502829e-02,
        7.199723e-02, 8.036760e-02, 9.045926e-02, 1.022425e-01,
        1.158302e-01, 1.306457e-01, 1.465196e-01, 1.631538e-01,
        1.810489e-01, 2.000658e-01, 2.190378e-01, 2.379700e-01,
        2.584936e-01, 2.821989e-01, 3.093061e-01, 3.398489e-01,
        3.742487e-01, 4.141645e-01, 4.591402e-01, 5.040913e-01,
        5.461308e-01, 5.882152e-01, 6.347649e-01, 6.858220e-01,
        7.357657e-01, 7.825202e-01, 8.325800e-01, 8.948123e-01,
        9.742749e-01, 1.066854e+00, 1.164880e+00, 1.259813e+00,
        1.344000e+00, 1.415621e+00, 1.476306e+00, 1.530107e+00,
        1.583796e+00, 1.643967e+00, 1.714368e+00, 1.796290e+00,
        1.886763e+00, 1.980414e+00, 2.073564e+00, 2.165768e+00,
        2.257637e+00, 2.350744e+00, 2.447342e+00, 2.548873e+00,
        2.654261e+00, 2.761211e+00, 2.864719e+00, 2.954426e+00,
        3.023785e+00, 3.080642e+00, 3.143541e+00, 3.228874e+00,
        3.332405e+00, 3.431719e+00, 3.510026e+00, 3.573666e+00,
        3.643947e+00, 3.727632e+00, 3.809236e+00, 3.872594e+00,
        3.913857e+00, 3.941224e+00, 3.966864e+00, 3.996566e+00,
        4.028339e+00, 4.063452e+00, 4.105962e+00, 4.156053e+00,
        4.215721e+00, 4.283367e+00, 4.350765e+00, 4.412879e+00,
        4.474676e+00, 4.539325e+00, 4.600715e+00, 4.652606e+00,
        4.695624e+00, 4.733303e+00, 4.768762e+00, 4.801019e+00,
        4.829286e+00, 4.854678e+00, 4.872670e+00, 4.882493e+00,
        4.893966e+00, 4.910985e+00, 4.927083e+00, 4.943920e+00,
        4.968136e+00, 4.995584e+00, 5.014828e+00, 5.026649e+00,
        5.043241e+00, 5.070968e+00, 5.101225e+00, 5.123278e+00,
        5.133028e+00, 0.000000e+00, 1.088858e-02, 1.572875e-02,
        2.059203e-02, 2.482500e-02, 2.988362e-02, 3.532458e-02,
        4.106306e-02, 4.696289e-02, 5.423830e-02, 6.161617e-02,
        6.987032e-02, 7.788256e-02, 8.650563e-02, 9.748440e-02,
        1.089046e-01, 1.213183e-01, 1.337056e-01, 1.468622e-01,
        1.623566e-01, 1.793668e-01, 1.996668e-01, 2.240931e-01,
        2.526845e-01, 2.857425e-01, 3.215321e-01, 3.594440e-01,
        3.990628e-01, 4.422878e-01, 4.892013e-01, 5.369513e-01,
        5.834379e-01, 6.303722e-01, 6.815642e-01, 7.381367e-01,
        7.997658e-01, 8.667801e-01, 9.413201e-01, 1.024826e+00,
        1.115366e+00, 1.207440e+00, 1.298751e+00, 1.392618e+00,
        1.491754e+00, 1.591256e+00, 1.686268e+00, 1.777852e+00,
        1.869330e+00, 1.958728e+00, 2.040519e+00, 2.117218e+00,
        2.199758e+00, 2.297558e+00, 2.407049e+00, 2.518331e+00,
        2.627823e+00, 2.728707e+00, 2.816318e+00, 2.894661e+00,
        2.967578e+00, 3.032286e+00, 3.087009e+00, 3.137180e+00,
        3.189802e+00, 3.243712e+00, 3.299467e+00, 3.361579e+00,
        3.430953e+00, 3.505753e+00, 3.590054e+00, 3.679450e+00,
        3.761610e+00, 3.830202e+00, 3.889369e+00, 3.949370e+00,
        4.015841e+00, 4.082861e+00, 4.145727e+00, 4.209540e+00,
        4.275614e+00, 4.332570e+00, 4.371085e+00, 4.395959e+00,
        4.417884e+00, 4.444236e+00, 4.477599e+00, 4.507127e+00,
        4.519726e+00, 4.515714e+00, 4.511425e+00, 4.524638e+00,
        4.562222e+00, 4.617146e+00, 4.677489e+00, 4.723888e+00,
        4.739657e+00, 4.733990e+00, 4.731983e+00, 4.749043e+00,
        4.782719e+00, 4.821540e+00, 4.856400e+00, 4.884050e+00,
        4.907332e+00, 4.935610e+00, 4.963111e+00, 4.976599e+00,
        4.980490e+00, 4.984202e+00, 4.987651e+00, 4.986432e+00,
        4.978268e+00, 4.969576e+00, 4.975168e+00, 5.004355e+00,
        5.049023e+00, 5.093344e+00, 5.125617e+00, 5.140764e+00,
        5.143976e+00, 5.142167e+00, 0.000000e+00, 4.355433e-02,
        6.291733e-02, 8.237358e-02, 9.930163e-02, 1.195424e-01,
        1.412952e-01, 1.641841e-01, 1.876915e-01, 2.166899e-01,
        2.461601e-01, 2.791216e-01, 3.112244e-01, 3.459102e-01,
        3.901220e-01, 4.359955e-01, 4.854055e-01, 5.336800e-01,
        5.828345e-01, 6.393108e-01, 6.991124e-01, 7.697677e-01,
        8.543184e-01, 9.515572e-01, 1.058472e+00, 1.165470e+00,
        1.271765e+00, 1.377765e+00, 1.489407e+00, 1.607563e+00,
        1.727697e+00, 1.843281e+00, 1.955663e+00, 2.065703e+00,
        2.171370e+00, 2.270700e+00, 2.357905e+00, 2.438403e+00,
        2.521732e+00, 2.616406e+00, 2.717002e+00, 2.813552e+00,
        2.895887e+00, 2.968964e+00, 3.027961e+00, 3.082743e+00,
        3.147062e+00, 3.225315e+00, 3.316326e+00, 3.407478e+00,
        3.496310e+00, 3.579199e+00, 3.662256e+00, 3.748991e+00,
        3.827288e+00, 3.895846e+00, 3.949326e+00, 3.995258e+00,
        4.037917e+00, 4.083995e+00, 4.135078e+00, 4.187025e+00,
        4.236220e+00, 4.292373e+00, 4.351402e+00, 4.407899e+00,
        4.461009e+00, 4.509494e+00, 4.547884e+00, 4.571321e+00,
        4.574796e+00, 4.571199e+00, 4.569386e+00, 4.577847e+00,
        4.592885e+00, 4.617280e+00, 4.642514e+00, 4.656344e+00,
        4.658603e+00, 4.664331e+00, 4.688360e+00, 4.739005e+00,
        4.807436e+00, 4.868940e+00, 4.907457e+00, 4.917679e+00,
        4.916325e+00, 4.916981e+00, 4.918063e+00, 4.910537e+00,
        4.882821e+00, 4.843510e+00, 4.806289e+00, 4.780837e+00,
        4.775125e+00, 4.788961e+00, 4.818650e+00, 4.846895e+00,
        4.871554e+00, 4.892560e+00, 4.914714e+00, 4.938330e+00,
        4.956524e+00, 4.964734e+00, 4.971232e+00, 4.976911e+00,
        4.987617e+00, 5.005636e+00, 5.038227e+00, 5.072251e+00,
        5.093451e+00, 5.095563e+00, 5.088637e+00, 5.080741e+00,
        5.074315e+00, 5.084057e+00, 5.114387e+00, 5.154362e+00,
        5.181037e+00, 5.181740e+00, 5.163029e+00,
    )
    SWEEP_LEADS = (
        2.71979, 2.72972, 2.63971, 2.03873, 3.88365, 3.74937, 3.04441,
        2.06688, 5.17471, 3.92746, 3.05138, 2.06076, float("nan"),
        3.92595, 3.04702, 2.06028, float("nan"), 3.92591, 3.04697,
        2.06027,
    )
    BIAS_FLOORS = (
        3.92591, 3.04697, 2.06027,
    )
    return (
        BIASES,
        BIAS_CURVES,
        BIAS_FLOORS,
        IC_AMPLITUDES,
        IC_CURVES,
        LAMBDA1,
        NOISES,
        NOISE_CURVES,
        SATURATION,
        SWEEP_BIASES,
        SWEEP_LEADS,
        TIMES,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 21 · Model Error and the Imperfect-Model Problem

    **Part V — The machinery of prediction.**

    **The forecasting question.** Every chapter so far has assumed a perfect
    model and imperfect initial conditions. Chapter 8 priced better
    observations at $\ln 10/\lambda_1$ = 3.8 days per decimal digit; chapter 13
    measured the same constant on a synthetic operational archive and got 7.0
    days per factor of thirteen in analysis error. Both calculations assumed
    that if you knew the state well enough, the forecast would be good.

    No forecast is made with the right model. So: **does the return on better
    observations continue, or does it stop?**

    It stops. This chapter measures where.

    ---

    ## Three ways to be wrong, three growth laws

    An initial-condition error is injected **once**, at $t = 0$, and then grows.
    A model error is injected **continuously**, at every step. That difference
    is not a detail — it changes the functional form of the error growth, and
    the three cases can be worked out from
    $\dot E = \lambda E + (\text{source})$:

    | source | short lead | long lead |
    |---|---|---|
    | initial condition, $\delta_0$ at $t=0$ | $\delta_0 e^{\lambda t}$ | $\delta_0 e^{\lambda t}$ |
    | deterministic bias $b$ | $b\,t$ — **linear** | $(b/\lambda)e^{\lambda t}$ |
    | stochastic forcing $\sigma$ | $\sigma\sqrt t$ — **diffusive** | $(\sigma/\sqrt{2\lambda})e^{\lambda t}$ |

    All three become exponential at the same rate eventually, so the
    distinction is visible only at short lead — which is exactly where an
    operational forecast lives.

    The experiment: truth is Lorenz 96 at $F = 8$. Model error is a wrong $F$,
    either a constant offset or noise added to the tendency, which is the
    crudest possible caricature of a wrong parameterisation and enough to make
    the point.

    | Section | The question |
    |---|---|
    | 1 | Do the three growth laws actually differ as predicted? |
    | 2 | Does better initialisation still buy lead time? |
    | 3 | What does that mean for observations, assimilation, and ensembles? |
    """
    )
    return


# ===========================================================================
# 1. Three growth laws
# ===========================================================================
@app.cell(hide_code=True)
def s1_controls(mo):
    source = mo.ui.dropdown(
        options={
            "all three together": "all",
            "initial-condition error only": "ic",
            "deterministic bias only": "bias",
            "stochastic forcing only": "noise",
        },
        value="all three together",
        label="error source",
    )
    return (source,)


@app.cell(hide_code=True)
def s1_figure(
    BIASES,
    BIAS_CURVES,
    C_CONTEXT,
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    IC_AMPLITUDES,
    IC_CURVES,
    LAMBDA1,
    NOISES,
    NOISE_CURVES,
    SATURATION,
    TIMES,
    finish_mpl,
    mo,
    mpl_panels,
    np,
    source,
):
    _t = np.asarray(TIMES, dtype=float)
    _ic = np.asarray(IC_CURVES, dtype=float).reshape(len(IC_AMPLITUDES), _t.size)
    _bias = np.asarray(BIAS_CURVES, dtype=float).reshape(len(BIASES), _t.size)
    _noise = np.asarray(NOISE_CURVES, dtype=float).reshape(len(NOISES), _t.size)
    _pick = str(source.value)

    _sets = {
        "ic": (_ic, IC_AMPLITUDES, C_TRUTH, r"IC $\delta_0$"),
        "bias": (_bias, BIASES, C_PERT, "bias $b$"),
        "noise": (_noise, NOISES, C_SPREAD, r"noise $\sigma$"),
    }
    _shown = list(_sets) if _pick == "all" else [_pick]

    _fig, _ax = mpl_panels(
        3,
        titles=("Error growth, log axes", "Short lead, log–log",
                "Short-lead power-law slope"),
        height=3.6,
    )
    for _name in _shown:
        _curves, _values, _colour, _label = _sets[_name]
        for _row, _value in zip(_curves, _values):
            _ax[0].semilogy(_t, _row, linewidth=1.5, color=_colour, alpha=0.85)
        _ax[0].semilogy([], [], linewidth=1.5, color=_colour,
                        label=f"{_label}")
    _ax[0].axhline(SATURATION, color=C_SAT, linewidth=1.3, linestyle="--",
                   label="saturation")
    _ax[0].set_xlabel("lead (time units)")
    _ax[0].set_ylabel("RMS error per site")
    _ax[0].set_ylim(1e-9, 4.0 * SATURATION)
    _ax[0].legend(loc="lower right", fontsize=6.0, framealpha=0.9)

    _window = (_t > 0.03) & (_t < 0.5)
    for _name in _shown:
        _curves, _values, _colour, _label = _sets[_name]
        # Normalise each curve by its own imposed amplitude so the SHAPES
        # can be compared on one axis; the point is the slope, not the offset.
        for _row, _value in zip(_curves, _values):
            _ax[1].loglog(_t[_window], _row[_window] / _value, linewidth=1.5,
                          color=_colour, alpha=0.85)
    _ax[1].loglog(_t[_window], 0.3 * _t[_window], color="#6b7280",
                  linewidth=1.1, linestyle=":", label=r"$\propto t$")
    _ax[1].loglog(_t[_window], 0.3 * np.sqrt(_t[_window]), color="#6b7280",
                  linewidth=1.1, linestyle="--", label=r"$\propto \sqrt{t}$")
    _ax[1].set_xlabel("lead (time units)")
    _ax[1].set_ylabel("error / imposed amplitude")
    _ax[1].legend(loc="lower right", fontsize=6.0, framealpha=0.9)

    _slopes, _labels, _colours = [], [], []
    for _name, (_curves, _values, _colour, _label) in _sets.items():
        _s = [
            float(np.polyfit(np.log(_t[_window]), np.log(_row[_window]), 1)[0])
            for _row in _curves
        ]
        _slopes.append(float(np.mean(_s)))
        _labels.append(_label)
        _colours.append(_colour)
    _ax[2].bar(np.arange(3), _slopes, color=_colours, width=0.55)
    _ax[2].axhline(1.0, color=C_SAT, linewidth=1.2, linestyle="--",
                   label="1 (linear)")
    _ax[2].axhline(0.5, color=C_MEAN, linewidth=1.2, linestyle=":",
                   label="½ (diffusive)")
    _ax[2].set_xticks(np.arange(3))
    _ax[2].set_xticklabels(_labels, fontsize=6.5)
    _ax[2].set_ylabel(r"$d\ln E/d\ln t$ over $[0.03, 0.5]$")
    _ax[2].legend(loc="upper left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle="Lorenz 96, F = 8, 48 start states")

    _ratio = _bias[1] / _ic[2]      # bias 0.05 against delta0 = 1e-4

    mo.vstack([
        mo.md(
            r"""---
    ## 1 · Do the three laws differ as predicted?

    The middle panel divides each curve by the amplitude that was imposed, so
    the *shapes* can be compared. Reference lines mark $\propto t$ and
    $\propto\sqrt t$."""
        ),
        mo.hstack([source], justify="start"),
        _fig,
        mo.md(
            f"""
| source | measured $d\\ln E/d\\ln t$ | predicted |
|---|---|---|
| initial condition | {_slopes[0]:+.3f} | $\\lambda t$, small and rising |
| deterministic bias | **{_slopes[1]:+.3f}** | **+1** (linear in $t$) |
| stochastic forcing | {_slopes[2]:+.3f} | +½ (diffusive) |

**The deterministic-bias law comes out exactly right**: {_slopes[1]:.3f} against
a predicted 1, and identical to three decimals across biases spanning a factor
of twenty, which is what a genuine power law looks like.

The other two are contaminated, and it is worth being precise about why rather
than calling them approximate. Each measured slope is the sum of the source's
own power law and the exponential growth that is always superposed: for pure
exponential growth $d\\ln E/d\\ln t = \\lambda t$, which over this window
contributes about 0.3. So the initial-condition case reads
{_slopes[0]:.3f} where a clean exponential would give ~0.3, and the stochastic
case reads {_slopes[2]:.3f} where $\\tfrac12 + \\lambda t$ would give ~0.8. The
bias case is the one that separates cleanly because its power law is the
steepest and therefore dominates longest.

**And the amplitudes are not comparable in the way one might assume.** A bias of
{BIASES[1]:g} — a {100 * BIASES[1] / 8.0:.2f}% error in the forcing — produces
{_ratio[2]:.0f} times more error at a lead of 0.1 than a
$\\delta_0 = 10^{{-4}}$ initial perturbation does, and
{_ratio[20]:.0f} times more at a lead of 1. The two sources are not
competitors of similar size; for any plausible pair of amplitudes one of them
simply dominates, and Section 2 is about which.
"""
        ),
    ])
    return


# ===========================================================================
# 2. The floor
# ===========================================================================
@app.cell(hide_code=True)
def s2_figure(
    BIAS_FLOORS,
    C_MEAN,
    C_PERT,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    DAYS_PER_TU,
    IC_AMPLITUDES,
    SWEEP_BIASES,
    SWEEP_LEADS,
    finish_mpl,
    mo,
    mpl_panels,
    np,
):
    _amps = np.asarray(IC_AMPLITUDES, dtype=float)
    _biases = np.asarray(SWEEP_BIASES, dtype=float)
    _leads = np.asarray(SWEEP_LEADS, dtype=float).reshape(
        _amps.size, _biases.size
    )
    _floors = np.asarray(BIAS_FLOORS, dtype=float)
    _digits = -np.log10(_amps)

    _fig, _ax = mpl_panels(
        2,
        titles=("Lead time vs initial accuracy",
                "What two decades of accuracy buy"),
        height=3.5,
    )
    _cols = (C_TRUTH, C_MEAN, C_SPREAD, C_PERT)
    for _j, (_bias, _colour) in enumerate(zip(_biases, _cols)):
        _column = _leads[:, _j]
        _ok = np.isfinite(_column)
        _ax[0].plot(_digits[_ok], _column[_ok] * DAYS_PER_TU, marker="o",
                    markersize=5, linewidth=1.7, color=_colour,
                    label=f"bias {_bias:g}" if _bias else "perfect model")
        if _bias > 0:
            _ax[0].axhline(_floors[_j - 1] * DAYS_PER_TU, color=_colour,
                           linewidth=0.9, linestyle=":")
    _ax[0].set_xlabel(r"decimal digits of initial accuracy ($-\log_{10}\delta_0$)")
    _ax[0].set_ylabel("days to 30 % of saturation")
    _ax[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)

    # The gain from 1e-2 to 1e-4, which is measurable for ALL four cases.
    #
    # A first version plotted 1e-4 -> 1e-8 instead. Those gains are zero for
    # every biased case (-0.008, -0.022, -0.002 days) and off-archive for the
    # perfect one, so the panel showed three small NEGATIVE bars on an axis
    # running to -0.022 and no bar at all for the perfect model -- which reads
    # as "better initial conditions make the forecast worse", the opposite of
    # the point. Two decades lower down, every case is measurable and the
    # progression is the argument.
    _gain = []
    for _j in range(_biases.size):
        _column = _leads[:, _j]
        _gain.append((_column[2] - _column[0]) * DAYS_PER_TU)
    _labels = ["perfect"] + [f"{b:g}" for b in _biases[1:]]
    _ax[1].bar(np.arange(_biases.size), _gain, color=list(_cols), width=0.5)
    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    for _j, _value in enumerate(_gain):
        _ax[1].annotate(f"{_value:+.1f}", (_j, _value),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=6.5, color="#4b5563")
    _ax[1].set_xticks(np.arange(_biases.size))
    _ax[1].set_xticklabels(_labels, fontsize=6.5)
    _ax[1].set_xlabel("model bias")
    _ax[1].set_ylabel(r"days gained, $\delta_0: 10^{-2}\to10^{-4}$")
    finish_mpl(_fig, suptitle="Lorenz 96; the perfect-model curve runs off the "
                              "top of the 30-day archive")

    _rows = "\n".join(
        "| $10^{" + f"{int(-d)}" + "}$ | "
        + " | ".join(
            "off the chart" if not np.isfinite(v) else f"{v * DAYS_PER_TU:.1f}"
            for v in _leads[i]
        ) + " |"
        for i, d in enumerate(_digits)
    )
    _header = " | ".join(
        "perfect model" if b == 0 else f"bias {b:g}" for b in _biases
    )

    mo.vstack([
        mo.md(
            r"""---
    ## 2 · Does better initialisation still buy lead time?

    Now both error sources at once. Every combination of initial-condition
    accuracy and model bias, with the lead time to 30 % of saturation — the
    point at which the forecast has lost most of its value."""
        ),
        _fig,
        mo.md(
            f"""
| $\\delta_0$ | {_header} |
|---|---|---|---|---|
{_rows}

**With a perfect model the return continues.** Going from $10^{{-2}}$ to
$10^{{-4}}$ buys {(_leads[2, 0] - _leads[0, 0]) * DAYS_PER_TU:.1f} days, and
beyond $10^{{-4}}$ the forecast outlives the thirty-day archive entirely. That
is chapters 8 and 13's logarithmic law, still paying.

**With any model bias at all it stops.** At a bias of {_biases[1]:g} — a
{100 * _biases[1] / 8.0:.3f}% error in the forcing — the lead time is
{_leads[2, 1] * DAYS_PER_TU:.1f} days at $\\delta_0 = 10^{{-4}}$ and
{_leads[-1, 1] * DAYS_PER_TU:.1f} days at $10^{{-8}}$. **Four orders of
magnitude of improvement in the initial state buys
{(_leads[-1, 1] - _leads[2, 1]) * DAYS_PER_TU:+.3f} days** — zero, to the
precision of the measurement.

The right-hand panel makes the same comparison two decades lower down, where
every case is still measurable: the *same* two-decade improvement in
$\\delta_0$, from $10^{{-2}}$ to $10^{{-4}}$, is worth
{(_leads[2, 0] - _leads[0, 0]) * DAYS_PER_TU:.1f} days with a perfect model,
{(_leads[2, 1] - _leads[0, 1]) * DAYS_PER_TU:.1f} at bias {_biases[1]:g},
{(_leads[2, 2] - _leads[0, 2]) * DAYS_PER_TU:.1f} at {_biases[2]:g}, and
{(_leads[2, 3] - _leads[0, 3]) * DAYS_PER_TU:.1f} at {_biases[3]:g}. The return
does not merely stop; it is already being eaten well before it stops.

And the level it stops at is set by the bias alone. With a *perfect* initial
condition, bias {_biases[1]:g} gives {_floors[0] * DAYS_PER_TU:.1f} days,
{_biases[2]:g} gives {_floors[1] * DAYS_PER_TU:.1f}, and {_biases[3]:g} gives
{_floors[2] * DAYS_PER_TU:.1f} — the same numbers the $\\delta_0 \\to 0$ column
converges to, to within two hundredths of a day. The model's own error is a
**ceiling**, and no observing system reaches above it.
"""
        ),
        mo.callout(
            mo.md(
                f"""### What this does to the logarithmic law

Chapters 8, 13 and 20 all measured the same constant: roughly
$\\ln 10/\\lambda_1$ of lead time per decade of analysis-error reduction, about
3.8 days for Lorenz 63 and 7.0 days per factor of thirteen in chapter 13's
synthetic operational archive. That law is real and this chapter does not
contradict it.

What it adds is a **stopping condition**. The logarithmic return holds while the
initial-condition error dominates the model error, and the moment it does not,
the exchange rate goes to zero — not gradually, but over about one decade of
$\\delta_0$. In the table above, the transition for bias {_biases[1]:g} happens
between $\\delta_0 = 10^{{-3}}$ and $10^{{-4}}$, and by $10^{{-4}}$ it is
already complete.

So "how much is a better observing system worth?" has no answer that does not
mention the model. And the corollary is uncomfortable for the way the two are
usually funded: past the crossover, the entire return on observing-system
investment is zero until the model improves."""
            ),
            kind="warn",
        ),
    ])
    return


# ===========================================================================
# 3. Consequences
# ===========================================================================
@app.cell(hide_code=True)
def s3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · Three consequences, and the honest state of the art

    **Data assimilation cannot fix it, and quietly makes it worse.** Every
    scheme in chapters 18–20 assumes the model is right: the background is a
    model forecast, and the background error covariance describes uncertainty
    *given* that the model is correct. Feed a biased model into a cycling
    analysis and the analysis inherits the bias, because the observations are
    pulled toward a background that is systematically wrong and the filter has
    no term to attribute the discrepancy to. Worse, the innovations then look
    larger than the specified observation error, and the usual response — tuning
    $\mathbf{R}$ up, or inflating less — is exactly wrong.

    **And it cannot even be diagnosed from a forecast archive.** Chapter 13's
    lagged-forecast estimator differences two forecasts of the same model, so a
    bias common to both cancels algebraically — `chaoslib` has a test that adds
    a constant bias to an entire archive and asserts the estimator's output is
    unchanged to round-off. Which means the standard truth-free way of measuring
    error growth is, by construction, blind to precisely the error source this
    chapter is about. Model error has to be found by comparison against
    observations, with all the difficulties chapter 22 takes up.

    **Stochastic parameterisation is the standard response, and it is a partial
    one.** The idea is to admit that the unresolved scales are not a
    deterministic function of the resolved ones and to represent them as a
    random process instead — which Section 1 measured as a $\sqrt t$ error
    source rather than a linear one, and which chapter 12 gave the physical
    reason for: the fast variables of a two-scale system saturate almost
    immediately and then force the slow ones at a rate the slow dynamics set.

    What it demonstrably buys is **ensemble reliability**. A deterministic
    parameterisation gives an ensemble that is confidently wrong; adding noise
    of the right amplitude gives spread that matches error, which is what
    chapter 17's scores reward and what makes a probabilistic forecast usable.
    What it does *not* reliably buy is a smaller mean error, and claims that it
    does should be checked against the possibility that the noise is simply
    detuning a bias.

    **What is honestly unsolved.** Nothing in this chapter estimates the real
    atmosphere's model error, because that requires knowing the truth. The
    published approaches — comparing models against each other, against
    reanalyses, or against short-range forecast tendencies — each assume
    something about what they are comparing to
    *[citation needed: on model-error estimation in NWP]*. The imperfect-model
    problem is the reason predictability estimates from perfect-model
    experiments, which is to say almost every number in Parts III and IV, are
    **upper bounds**.
    """
    )
    return


# ===========================================================================
# 4. Closing
# ===========================================================================
@app.cell(hide_code=True)
def closing(mo):
    mo.md(
        r"""
    ---
    ## Try this

    1. **Confirm the linear law.** In Section 1, select the deterministic bias
       alone. The three curves lie on top of each other once divided by $b$, and
       the slope is 1. Explain from $\dot E = \lambda E + b$ why the law is
       $b\,t$ and not $b\,t\,e^{\lambda t}$ at short lead.
    2. **Find the crossover.** In Section 2's table, for each bias find the
       $\delta_0$ at which the lead time stops improving. Compare it with
       $b/\lambda_1$ and say whether that is the right scaling to expect.
    3. **Price an observing system honestly.** A satellite reduces analysis
       error by a factor of three. Using Section 2, state what it buys with a
       perfect model, with a bias of 0.01, and with a bias of 0.2.
    4. **Distinguish the two sources from data.** You are given a single error
       curve and told it comes either from initial-condition error or from a
       deterministic bias. What measurement distinguishes them, and over what
       range of lead times must you have data for it to work?
    5. **Argue against stochastic parameterisation.** Section 3 says it buys
       reliability but not necessarily accuracy. Construct the case where adding
       noise makes a forecast system worse, and say what diagnostic would reveal
       it.

    ## What you should have seen

    Model error and initial-condition error do not grow the same way. An initial
    error is injected once and grows exponentially; a deterministic model bias is
    injected continuously and grows **linearly** in $t$ at short lead —
    measured $d\ln E/d\ln t = 1.087$ against a predicted 1, identical to three
    decimals across biases spanning a factor of twenty. Stochastic model error
    grows **diffusively**, as $\sqrt t$. All three become exponential at the same
    rate eventually, so the distinction lives at short lead, where forecasts do.

    They are also not comparable in size. A 0.6 % error in the forcing produces
    of order a hundred times more error at one time unit than a $10^{-4}$
    initial perturbation.

    **And the model's error is a ceiling.** With a perfect model, better initial
    conditions keep buying lead time and the forecast outlives a thirty-day
    archive. With a bias of 0.01 — a 0.125 % error in $F$ — improving the initial
    state from $10^{-4}$ to $10^{-8}$, four orders of magnitude, buys nothing
    measurable — $-0.005$ days. The lead time is pinned at 19.6 days, which is
    exactly what
    that bias gives with a *perfect* initial condition. The logarithmic return on
    observations of chapters 8, 13 and 20 is real, and it has a stopping
    condition roughly one decade wide.

    None of this can be fixed by assimilation, which assumes the model is right
    and inherits its bias; nor diagnosed by chapter 13's lagged-forecast
    estimator, which cancels it by construction. And it means the predictability
    numbers in Parts III and IV, all measured in perfect-model experiments, are
    upper bounds.

    ## Further reading

    - Palmer, T. N. (2001). A nonlinear dynamical perspective on model error.
      *Quarterly Journal of the Royal Meteorological Society*, **127**, 279–304
      *[citation needed: confirm pages]*.
    - Wilks, D. S. (2005). Effects of stochastic parametrizations in the
      Lorenz '96 system. *QJRMS*, **131**, 389–407.
    - Berner, J. et al. (2017). Stochastic parameterization: toward a new view
      of weather and climate models. *BAMS*, **98**, 565–588
      *[citation needed: confirm]*.
    - Orrell, D., Smith, L., Barkmeijer, J. and Palmer, T. N. (2001). Model
      error in weather forecasting. *Nonlinear Processes in Geophysics*, **8**,
      357–371 *[citation needed: confirm]*.
    - Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and
      Climate*, ch. 10–11 *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
