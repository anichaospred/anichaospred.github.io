# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo==0.23.9",
#   "numpy",
#   "scipy",
#   "matplotlib",
# ]
# ///
"""Chapter 5 -- Maps, bifurcations, and the routes to chaos.

One line of arithmetic, the whole period-doubling cascade, and the discovery
that the numbers describing it belong to no particular map.

Part II of *An Interactive Chaos and Predictability Textbook*.

Numerics come from `chaoslib.maps`; this file holds the exposition and the
figures. Figures are static matplotlib, matching chapters 6 and 7.

To edit:   marimo edit notebooks/ch05_maps-bifurcations.py
To export: make nb-one NB=ch05_maps-bifurcations
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="Chapter 5: Maps and Bifurcations")


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

    from chaoslib import maps, plotting, systems

    C_CONTEXT = plotting.mpl_colour(plotting.C_CONTEXT)
    C_TRUTH = plotting.C_TRUTH
    C_PERT = plotting.C_PERT
    C_SPREAD = plotting.C_SPREAD
    C_MEAN = plotting.C_MEAN
    C_FIXED = plotting.C_FIXED
    C_SAT = plotting.C_SAT
    C_START = plotting.C_START
    C_OBS = plotting.C_OBS
    mpl_panels = plotting.mpl_panels
    mpl_grid = plotting.mpl_grid
    finish_mpl = plotting.finish_mpl

    # The accumulation point of the logistic cascade, and the tangent
    # bifurcation that creates the 3-cycle. Both are needed in several cells.
    R_INFINITY = 3.5699456720
    R_PERIOD3 = maps.period_three_threshold()
    DELTA = maps.FEIGENBAUM_DELTA

    return (
        C_CONTEXT,
        C_FIXED,
        C_MEAN,
        C_OBS,
        C_PERT,
        C_SAT,
        C_SPREAD,
        C_START,
        C_TRUTH,
        DELTA,
        R_INFINITY,
        R_PERIOD3,
        finish_mpl,
        maps,
        mo,
        mpl_grid,
        mpl_panels,
        np,
        systems,
    )


# ---------------------------------------------------------------------------
# Precomputed intermittency scaling
# ---------------------------------------------------------------------------
@app.cell
def laminar_data():
    # Mean laminar-phase length at six distances below r_c, from
    # scripts/generate_ch05_data.py (400,000 iterations per parameter).
    #
    # This is the ONLY precomputed figure in the chapter. Laminar-phase
    # detection is irreducibly serial -- whether iterate n extends a laminar run
    # depends on the run so far -- so unlike the bifurcation diagram and the
    # Lyapunov curve it cannot be vectorised over the parameter axis. Six
    # parameters at 400,000 iterations is ~1.2 s natively, ~12 s in Pyodide, for
    # a curve identical for every reader. Section 4 runs a *single* orbit live at
    # the reader's chosen distance, which costs 6 ms.
    LAMINAR_EPS = (0.002, 0.001, 0.0005, 0.00025, 0.000125, 6e-05)
    LAMINAR_MEAN = (11.7976, 17.0497, 23.7828, 33.7833, 47.9130, 67.1835)
    LAMINAR_COUNT = (8675, 8089, 7440, 6148, 5057, 4055)
    LAMINAR_SLOPE = -0.4965  # theory -1/2
    LAMINAR_INTERCEPT = -0.6044
    return (
        LAMINAR_COUNT,
        LAMINAR_EPS,
        LAMINAR_INTERCEPT,
        LAMINAR_MEAN,
        LAMINAR_SLOPE,
    )


# ===========================================================================
# Title
# ===========================================================================
@app.cell(hide_code=True)
def title(mo):
    mo.md(
        r"""
    # Chapter 5 · Maps, Bifurcations, and the Routes to Chaos

    **Part II — From regular motion to chaos.**

    **The forecasting question.** Every forecast model contains parameters nobody
    can measure exactly: a mixing length, an autoconversion threshold, an
    entrainment rate, a drag coefficient. Suppose one of them is uncertain by half a
    percent. How much does that matter?

    The comfortable answer is *half a percent*, and for most parameter values it is
    right. But there are values where it is catastrophically wrong — where an
    arbitrarily small change in a parameter produces a *qualitative* change in
    behaviour: a steady state becomes an oscillation, an oscillation becomes chaos, a
    single regime becomes two. Those values are called **bifurcations**, and this
    chapter is about what they look like and how they are organised.

    The system used is deliberately absurd in its simplicity:

    $$x_{n+1} = r\,x_n(1 - x_n), \qquad 0 \le x \le 1 .$$

    One variable. One parameter. No time step, no derivatives, no integration
    scheme — just multiplication. A model of a population that grows when small and
    is limited when large. It has no atmosphere in it anywhere.

    And yet it contains the entire period-doubling route to chaos, and the
    **numbers** describing that route are the same numbers as in a dripping tap, a
    convecting fluid, a nonlinear circuit, and a hierarchy of atmospheric models. That
    is the claim chapter 3 made about why a three-variable model can teach something
    true about a $10^9$-variable one. This chapter is where it stops being a claim.

    ---

    ## What is here

    | Section | The question |
    |---|---|
    | 1 | Where does the map settle, and why? Stability, read off a picture |
    | 2 | The bifurcation diagram, and the Lyapunov exponent that makes it precise |
    | 3 | The cascade, and why $\delta = 4.669$ belongs to no particular map |
    | 4 | Intermittency: a regime that ends with no parameter changing |
    | 5 | What any of this has to do with weather and climate |
    """
    )
    return


# ===========================================================================
# 1. The cobweb: stability as a picture
# ===========================================================================
@app.cell(hide_code=True)
def s1_text(mo):
    mo.md(
        r"""
    ---
    ## 1 · Where the map settles

    A fixed point satisfies $x^* = r x^*(1-x^*)$, giving $x^* = 0$ and
    $x^* = 1 - 1/r$. Whether an orbit *goes* there is a separate question, answered by
    the derivative:

    $$f'(x) = r(1 - 2x), \qquad f'(1 - 1/r) = 2 - r .$$

    The fixed point attracts when $|f'(x^*)| < 1$, i.e. for $1 < r < 3$, and repels
    beyond. At $r = 3$ exactly, $f'(x^*) = -1$: the marginal case. The **cobweb**
    construction below makes this visible without any algebra — step vertically to the
    curve, horizontally to the diagonal, repeat. The staircase spirals inward when the
    curve crosses the diagonal shallowly and outward when it crosses steeply, and
    "shallowly" means exactly $|f'| < 1$.

    Push $r$ past 3 and watch the staircase settle into a **square** instead of a
    point: the orbit alternates between two values. Past $r \approx 3.449$ the square
    becomes an eight-vertex circuit, then sixteen. Past $r \approx 3.5699$ it never
    closes.
    """
    )
    return


@app.cell(hide_code=True)
def s1_controls(mo):
    r_slider = mo.ui.slider(
        start=1.5, stop=4.0, step=0.005, value=2.8,
        label="growth parameter r", show_value=True,
    )
    x0_slider = mo.ui.slider(
        start=0.02, stop=0.98, step=0.01, value=0.20,
        label="initial state x₀", show_value=True,
    )
    steps_slider = mo.ui.slider(
        start=10, stop=200, step=10, value=60,
        label="iterations shown", show_value=True,
    )
    return r_slider, steps_slider, x0_slider


@app.cell(hide_code=True)
def s1_figure(
    C_FIXED,
    C_PERT,
    C_START,
    C_TRUTH,
    finish_mpl,
    maps,
    mo,
    mpl_panels,
    np,
    r_slider,
    steps_slider,
    systems,
    x0_slider,
):
    _r = float(r_slider.value)
    _x0 = float(x0_slider.value)
    _n = int(steps_slider.value)

    _grid = np.linspace(0.0, 1.0, 500)
    _curve = systems.logistic_map(_grid, r=_r)
    _cx, _cy = maps.cobweb_path(systems.logistic_map, _x0, _n, r=_r)
    _orbit = maps.map_orbit(systems.logistic_map, _x0, _n, r=_r)

    # The non-trivial fixed point and its multiplier, both in closed form.
    _xstar = 1.0 - 1.0 / _r if _r > 1.0 else 0.0
    _mult = 2.0 - _r if _r > 1.0 else _r

    _fig, _ax = mpl_panels(
        2,
        titles=(f"Cobweb at r = {_r:g}", "The orbit"),
        height=3.7,
    )
    _ax[0].plot(_grid, _curve, color=C_TRUTH, linewidth=1.8, label="f(x) = r x(1−x)")
    _ax[0].plot([0, 1], [0, 1], color="#94a3b8", linewidth=1.0,
                linestyle="--", label="y = x")
    _ax[0].plot(_cx, _cy, color=C_PERT, linewidth=0.9, alpha=0.85, label="cobweb")
    _ax[0].plot([_x0], [_x0], marker="o", markersize=6, color=C_START,
                zorder=5, label="start")
    if _r > 1.0:
        _ax[0].plot([_xstar], [_xstar], marker="*", markersize=13, color=C_FIXED,
                    zorder=6, label="fixed point")
    _ax[0].set_xlabel("$x_n$")
    _ax[0].set_ylabel("$x_{n+1}$")
    _ax[0].set_xlim(0, 1)
    _ax[0].set_ylim(0, 1)
    _ax[0].set_aspect("equal")
    _ax[0].legend(loc="upper left", fontsize=6.5, framealpha=0.9)

    _ax[1].plot(np.arange(_orbit.size), _orbit, marker="o", markersize=2.6,
                color=C_TRUTH, linewidth=1.0)
    if _r > 1.0:
        _ax[1].axhline(_xstar, color=C_FIXED, linewidth=1.2, linestyle="--",
                       label=f"$x^*$ = {_xstar:.4f}")
        _ax[1].legend(loc="upper right", fontsize=7, framealpha=0.9)
    _ax[1].set_xlabel("iteration n")
    _ax[1].set_ylabel("$x_n$")
    _ax[1].set_ylim(-0.02, 1.02)
    finish_mpl(_fig)

    # Classify what the last stretch of the orbit is doing, by counting how many
    # distinct values it visits. A period-p cycle revisits p values to within
    # round-off; a chaotic orbit visits as many values as there are samples.
    _tail = maps.map_orbit(systems.logistic_map, _x0, 200, n_discard=3000, r=_r)
    _distinct = np.unique(np.round(_tail, 5)).size
    if _distinct == 1:
        _verdict = "a **fixed point** — the orbit stops moving"
    elif _distinct <= 64 and _distinct == 2 ** int(round(np.log2(_distinct))):
        _verdict = f"a **{_distinct}-cycle** — period {_distinct} = 2^{int(round(np.log2(_distinct)))}"
    elif _distinct <= 64:
        _verdict = f"a **{_distinct}-cycle**"
    else:
        _verdict = f"**aperiodic** — {_distinct} distinct values in 200 iterates"

    if _r <= 1.0:
        _stab = "everything decays to extinction, $x = 0$"
    elif abs(_mult) < 1.0:
        _stab = f"$|f'(x^*)| = {abs(_mult):.3f} < 1$, so the fixed point **attracts**"
    else:
        _stab = f"$|f'(x^*)| = {abs(_mult):.3f} > 1$, so the fixed point **repels**"

    mo.vstack([
        mo.hstack([r_slider, x0_slider, steps_slider], justify="start"),
        _fig,
        mo.md(
            f"""
| quantity | value |
|---|---|
| fixed point $x^* = 1 - 1/r$ | {_xstar:.6f} |
| multiplier $f'(x^*) = 2 - r$ | {_mult:+.4f} |
| stability | {_stab} |
| what the orbit settles onto | {_verdict} |

Two things are worth doing before moving on. **Change $x_0$** and watch the
verdict *not* change: which cycle the map settles onto is a property of $r$, not
of where you start — a statement that will fail in chapter 27, where two
attractors coexist. And **set $r$ to just below and just above 3**: the fixed
point does not move discontinuously, and nothing about $f$ is singular there.
What changes is only whether $|f'|$ has crossed 1.
"""
        ),
    ])
    return


# ===========================================================================
# 2. The bifurcation diagram and the Lyapunov exponent
# ===========================================================================
@app.cell(hide_code=True)
def s2_text(mo):
    mo.md(
        r"""
    ---
    ## 2 · The whole family at once

    Sweeping $r$ and plotting what each value settles onto gives the
    **bifurcation diagram** — one of the few pictures in nonlinear dynamics that
    deserves its fame. Every vertical slice is the attractor at that parameter.

    Below it, the same information made quantitative. For a one-dimensional map,
    chapter 7's Lyapunov exponent collapses to a single average:

    $$\lambda(r) = \lim_{N\to\infty}\frac{1}{N}\sum_{n=0}^{N-1}\ln\bigl|f'_r(x_n)\bigr|$$

    with no tangent linear model, no QR re-orthonormalisation, and no ambiguity —
    $\ln|f'|$ *is* the local stretching rate when there is only one direction to
    stretch. The sign is the whole diagnosis: $\lambda < 0$ means neighbouring
    states converge and the orbit is periodic; $\lambda > 0$ means they separate and
    it is chaotic; $\lambda = 0$ marks a bifurcation.

    Three exact values to check the curve against: $\lambda = \ln 2 = 0.6931$ at
    $r = 4$, where the map is conjugate to the doubling map $x \mapsto 2x \bmod 1$;
    $\lambda = 0$ at every bifurcation; and $\lambda = -\infty$ at every
    *superstable* parameter, where the orbit lands exactly on $f' = 0$ (the downward
    spikes, clipped to keep the axis readable).
    """
    )
    return


@app.cell(hide_code=True)
def s2_controls(mo):
    window_dropdown = mo.ui.dropdown(
        options={
            "the whole family:  2.5 → 4.0": "2.5,4.0",
            "the first doublings:  2.8 → 3.6": "2.8,3.6",
            "into the accumulation point:  3.54 → 3.5715": "3.54,3.5715",
            "the period-3 window:  3.8284 → 3.857": "3.8284,3.857",
            "its own cascade inside it:  3.841 → 3.85": "3.841,3.85",
        },
        value="the whole family:  2.5 → 4.0",
        label="parameter window",
    )
    return (window_dropdown,)


@app.cell(hide_code=True)
def s2_figure(
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    DELTA,
    R_INFINITY,
    R_PERIOD3,
    finish_mpl,
    maps,
    mo,
    mpl_grid,
    np,
    systems,
    window_dropdown,
):
    _lo, _hi = (float(v) for v in str(window_dropdown.value).split(","))
    _narrow = (_hi - _lo) < 0.3

    # A narrow window needs a longer transient: convergence onto the attractor
    # slows near a bifurcation, and transient smear in a zoomed diagram reads
    # exactly like chaos.
    _n_r = 1400
    _discard = 2500 if _narrow else 700
    _keep = 400 if _narrow else 250
    _rs = np.linspace(_lo, _hi, _n_r)

    _rf, _xf = maps.bifurcation_points(
        systems.logistic_map, _rs, n_discard=_discard, n_keep=_keep
    )
    _lam = maps.map_lyapunov_exponent(
        systems.logistic_map,
        systems.logistic_map_derivative,
        _rs,
        n_discard=_discard,
        n_iter=3000,
        floor=-8.0,
    )

    _fig, _ax = mpl_grid(2, 1, titles=("Bifurcation diagram", "Lyapunov exponent"),
                         panel=(7.8, 2.7))
    _ax[0].plot(_rf, _xf, ",", color=C_TRUTH, alpha=0.35, rasterized=True)
    _ax[0].set_ylabel("attractor of $x$")
    _ax[0].set_xlim(_lo, _hi)
    _ax[0].set_ylim(-0.02, 1.02)

    _ax[1].axhline(0.0, color=C_SAT, linewidth=1.1, linestyle="--")
    _ax[1].plot(_rs, _lam, color=C_SPREAD, linewidth=0.9)
    _ax[1].fill_between(_rs, 0.0, _lam, where=_lam > 0, color=C_SPREAD, alpha=0.25)
    _ax[1].set_xlabel("growth parameter r")
    _ax[1].set_ylabel(r"$\lambda$  (nats/iteration)")
    _ax[1].set_xlim(_lo, _hi)
    _ax[1].set_ylim(max(-2.2, _lam.min() - 0.15), max(0.85, _lam.max() + 0.12))

    for _ax_i in _ax:
        if _lo < R_INFINITY < _hi:
            _ax_i.axvline(R_INFINITY, color="#0f766e", linewidth=1.0, linestyle=":")
        if _lo < R_PERIOD3 < _hi:
            _ax_i.axvline(R_PERIOD3, color="#b45309", linewidth=1.0, linestyle=":")
    finish_mpl(_fig)

    # Measured, not asserted: the fraction of this window that is chaotic, and
    # where lambda first turns positive.
    _chaotic = float(np.mean(_lam > 0.0))
    _positive = np.nonzero(_lam > 0.0)[0]
    _first = f"{_rs[_positive[0]]:.5f}" if _positive.size else "nowhere in this window"
    _lam_at_4 = (
        float(
            maps.map_lyapunov_exponent(
                systems.logistic_map,
                systems.logistic_map_derivative,
                np.array([4.0]),
                n_iter=30000,
            )[0]
        )
    )

    _notes = {
        "2.5,4.0": (
            f"""The dotted lines mark $r_\\infty = {R_INFINITY:.4f}$ (teal), where the
cascade accumulates, and $r_c = 1 + 2\\sqrt{{2}} = {R_PERIOD3:.4f}$ (amber),
where the 3-cycle is born. **{100 * _chaotic:.0f}% of this window has
$\\lambda > 0$** — chaos is common but far from universal, and the white gaps
above $r_\\infty$ are periodic windows, not gaps in the sampling."""
        ),
        "2.8,3.6": (
            """The cascade, level by level: one branch becomes two at $r = 3$, two
become four at $3.449$, four become eight at $3.544$. $\\lambda$ touches zero at
each of those parameters and dips to a deep minimum between them, at the
*superstable* parameter where the cycle contains the critical point. Section 3
measures the spacings."""
        ),
        "3.54,3.5715": (
            f"""Six or seven doublings are resolved before the pixels run out. Note
that $\\lambda$ approaches zero *from below* through the whole cascade and only
crosses at $r_\\infty = {R_INFINITY:.5f}$: the accumulation point is where chaos
begins, and it is a limit of infinitely many bifurcations rather than a
bifurcation itself."""
        ),
        "3.8284,3.857": (
            f"""The period-3 window, from its opening at
$r_c = {R_PERIOD3:.6f}$ to its close. This is **well past
$r_\\infty = {R_INFINITY:.4f}$**, and yet only {100 * _chaotic:.0f}% of it has
$\\lambda > 0$: the attractor is a stable 3-cycle over most of the window, which
then undergoes its own cascade $3 \\to 6 \\to 12 \\to 24$ before returning to
chaos near $r = 3.8496$. Any account of "the transition to chaos" as a single
event at a single parameter is wrong."""
        ),
        "3.841,3.85": (
            """The period-3 window's **own** cascade, in a parameter interval
0.009 wide: 6 becomes 12 at 3.8478, 12 becomes 24 at 3.8492, 24 becomes 48 at
3.8494, accumulating at about 3.84943.

This is the bifurcation diagram again, three branches instead of one and roughly
$10^{-2}$ of the width. And the resemblance is quantitative, not visual: the
superstable parameters of the $3\\cdot 2^n$ cycles are 3.831874, 3.844569,
3.848345, 3.849198, 3.849383, 3.849423, whose spacing ratios run 3.36, 4.42,
4.61, **4.66** — converging to the same $\\delta$ as the main cascade, from a
cascade spanning 0.0175 in $r$ rather than 1.57. `chaoslib` computes it with
`superstable_cascade(..., base_period=3)`, and a test asserts the agreement.
That identity is the renormalisation structure which *makes* $\\delta$
universal, rather than a consequence of it."""
        ),
    }
    _key = f"{_lo:g},{_hi:g}"

    mo.vstack([
        mo.hstack([window_dropdown], justify="start"),
        _fig,
        mo.md(_notes.get(_key, "")),
        mo.md(
            f"""
**Reference values.** Measured $\\lambda(4) = {_lam_at_4:.6f}$ against the exact
$\\ln 2 = {np.log(2.0):.6f}$ — agreement to
{abs(_lam_at_4 - np.log(2.0)):.0e}, which is the sampling error of a
30,000-iterate average and not a discretisation error, because there is no
discretisation. In this window $\\lambda$ first turns positive at
**r = {_first}**, and $\\delta = {DELTA:.6f}$ is the constant Section 3 measures.
"""
        ),
    ])
    return


# ===========================================================================
# 3. The cascade, and universality
# ===========================================================================
@app.cell(hide_code=True)
def s3_text(mo):
    mo.md(
        r"""
    ---
    ## 3 · The cascade, and the number that outlives the map

    The doubling parameters $3,\ 3.449,\ 3.544,\ 3.5645,\ \ldots$ crowd together
    geometrically, and the ratio of successive gaps converges:

    $$\delta = \lim_{n\to\infty}\frac{R_{n-1} - R_{n-2}}{R_n - R_{n-1}}
      = 4.669\,201\,609\ldots$$

    **What is actually located below.** Not the bifurcation parameters $b_n$, where
    a cycle loses stability, but the **superstable** parameters $R_n$, where the
    $2^n$-cycle passes through the critical point $x_c = 1/2$. Since $f'(1/2) = 0$,
    such a cycle has multiplier *exactly* zero, so $R_n$ solves the smooth scalar
    equation $f_r^{2^n}(1/2) = 1/2$ — plain iteration, no cycle-finding. The
    bifurcation parameters would require detecting a marginal condition
    ($|{\rm multiplier}| \to 1$) inside a basin shrinking like $\delta^{-n}$, which
    is ill-conditioned by construction. The two sequences interleave,
    $b_n < R_n < b_{n+1}$, and share both their limit and their $\delta$.

    **Then the real point.** $\delta$ is not a property of $x(1-x)$. Below, the same
    computation runs on three families sharing no algebra at all:

    | family | $f_r(x)$ | critical point | first superstable $R_0$ |
    |---|---|---|---|
    | logistic | $r\,x(1-x)$ | $1/2$ | $2$ |
    | sine | $r\sin(\pi x)$ | $1/2$ | $1/2$ |
    | cubic | $r\,x(1-x^2)$ | $1/\sqrt{3}$ | $3/2$ |

    Polynomial against transcendental; quadratic against cubic; three different
    critical points; three cascades with nothing numerically in common. If $\delta$
    were an artefact of the logistic map's particular quadratic form, this is where
    it would show.
    """
    )
    return


@app.cell(hide_code=True)
def s3_figure(
    C_MEAN,
    C_PERT,
    C_SAT,
    C_TRUTH,
    DELTA,
    R_INFINITY,
    finish_mpl,
    maps,
    mo,
    mpl_panels,
    np,
    systems,
):
    # Computed live, not precomputed: locating R_8 needs 256 map compositions per
    # function evaluation, but each is 256 multiplications, and the bracketing
    # scan converges in one pass. All three families together: ~57 ms natively.
    _families = (
        ("logistic  $r\\,x(1-x)$", systems.logistic_map, 0.5, 1.5, R_INFINITY, 8, C_TRUTH),
        ("sine  $r\\sin(\\pi x)$", systems.sine_map, 0.5, 0.3, 0.8655, 6, C_PERT),
        ("cubic  $r\\,x(1-x^2)$", systems.cubic_map, 1.0 / np.sqrt(3.0), 1.0, 2.3025, 7, C_MEAN),
    )

    _results = []
    for _label, _fmap, _xc, _rlo, _rhi, _nmax, _colour in _families:
        _cascade = maps.superstable_cascade(_fmap, _xc, _rlo, _rhi, n_max=_nmax)
        _ratios = maps.feigenbaum_ratios(_cascade)
        _results.append((_label, _cascade, _ratios, _colour))

    _fig, _ax = mpl_panels(
        2,
        titles=("Three cascades at three scales…", "…shrinking at one rate"),
        height=3.6,
    )
    for _label, _cascade, _ratios, _colour in _results:
        # Raw gaps, NOT rescaled. Rescaling each cascade onto a common axis
        # would make the three curves coincide -- which happens *because* delta
        # is shared, and so hides the very thing the panel is meant to show.
        # On a log axis, different intercepts say the maps are different and
        # parallel slopes say the ratio is not.
        _gaps = np.diff(_cascade)
        _ax[0].semilogy(
            np.arange(1, _gaps.size + 1), _gaps,
            marker="o", markersize=4, linewidth=1.4, color=_colour, label=_label,
        )
        _ax[1].plot(
            np.arange(2, 2 + _ratios.size), _ratios,
            marker="o", markersize=4.5, linewidth=1.4, color=_colour, label=_label,
        )
    # A pure geometric decay at exactly delta, anchored to the logistic map's
    # first gap: the slope every family should end up parallel to.
    _ref_n = np.arange(1, 9)
    _ax[0].semilogy(
        _ref_n, (_results[0][1][1] - _results[0][1][0]) * DELTA ** (1.0 - _ref_n),
        color=C_SAT, linewidth=1.2, linestyle="--", label="decay by δ each level",
    )
    _ax[0].set_xlabel("cascade level n")
    _ax[0].set_ylabel("gap $R_n - R_{n-1}$")
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _ax[1].axhline(DELTA, color=C_SAT, linewidth=1.3, linestyle="--",
                   label=f"δ = {DELTA:.6f}")
    _ax[1].set_xlabel("cascade level n")
    _ax[1].set_ylabel(r"$(R_{n-1}-R_{n-2})/(R_n-R_{n-1})$")
    _ax[1].set_ylim(4.0, 4.95)
    _ax[1].legend(loc="lower right", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig, suptitle="Feigenbaum universality, computed rather than quoted")

    _rows = "\n".join(
        f"| {_label} | {_cascade.size} | {_cascade[0]:.4f} | {_cascade[-1]:.7f} | "
        f"{_ratios[-1]:.6f} | {abs(_ratios[-1] - DELTA):.1e} |"
        for _label, _cascade, _ratios, _colour in _results
    )
    _logistic = _results[0][1]
    _lrows = "\n".join(
        f"| {_n} | {2 ** _n} | {_r:.10f} | "
        + (f"{_logistic[_n] - _logistic[_n - 1]:.3e} |" if _n else "— |")
        for _n, _r in enumerate(_logistic)
    )

    mo.vstack([
        _fig,
        mo.md(
            f"""
| family | levels | $R_0$ | deepest $R_n$ | measured δ | error |
|---|---|---|---|---|---|
{_rows}

**Three maps with nothing in common converge on the same constant.** In the
left panel the three cascades sit at different absolute scales — the logistic
map's first gap is 1.24, the sine map's 0.28 — and every one of them decays
parallel to the dashed reference line, which is a pure geometric decay by
$\\delta$. Parallel on a log axis *is* the shared ratio. The right panel reads it
off directly, and the three land on top of one another.

This is what "universal" means, and it is the licence under which the rest of
this book operates. The Lorenz system is not the atmosphere; Lorenz 96 is not the
atmosphere either. Neither has to be. If the quantitative structure of a
transition is shared across a whole class of systems, then measuring it in the
cheapest member of the class is not a compromise — it is the efficient way to
measure it.

The logistic cascade in full:

| n | period | $R_n$ | gap $R_n - R_{{n-1}}$ |
|---|---|---|---|
{_lrows}

The gaps shrink by a factor of about {DELTA:.3f} each level, which is why only
{_logistic.size} of them fit: by $n = {_logistic.size - 1}$ the spacing is
{_logistic[-1] - _logistic[-2]:.1e}, and locating the next one requires resolving
a gap of $10^{{-8}}$ using {2 ** (_logistic.size)} compositions of the map. Double
precision runs out before the mathematics does — the cascade itself is infinite,
accumulating at $r_\\infty = {R_INFINITY:.7f}$.
"""
        ),
    ])
    return


# ===========================================================================
# 4. Intermittency
# ===========================================================================
@app.cell(hide_code=True)
def s4_text(mo):
    mo.md(
        r"""
    ---
    ## 4 · A regime that ends without anything changing

    Period doubling is not the only route into chaos, and for weather and climate
    it may not be the most relevant one. Consider what happens just *below* the
    parameter where the 3-cycle is born,

    $$r_c = 1 + 2\sqrt{2} = 3.828427\ldots$$

    At $r_c$ the third iterate $f^3$ is **tangent** to the diagonal at three points
    at once: $f^3(x) = x$ with $(f^3)'(x) = +1$. A stable and an unstable 3-cycle
    appear together out of nothing — a tangent, or saddle-node, bifurcation, a
    different mechanism from period doubling.

    Just below $r_c$ the cycle does not exist, but $f^3$ still passes very close to
    the diagonal. An orbit entering that narrow channel is nearly 3-periodic and
    takes many iterations to crawl through it, then escapes, wanders chaotically, and
    eventually re-enters. The result is **type-I intermittency**: long quiet stretches
    punctuated by bursts, with a fixed parameter and no external forcing whatever.

    The local normal form in the channel is $x \mapsto x + a x^2 + \epsilon$ with
    $\epsilon \propto r_c - r$, from which the mean laminar length follows:

    $$\langle L \rangle \sim (r_c - r)^{-1/2} .$$

    Another universal exponent, and the reason this section is in the chapter: it is
    the cleanest available model of a regime that persists for a while and then does
    not, and of a **timescale that diverges as a threshold is approached**.
    """
    )
    return


@app.cell(hide_code=True)
def s4_controls(mo):
    distance_dropdown = mo.ui.dropdown(
        options={
            "far below:  r_c − 2×10⁻³": "0.002",
            "closer:  r_c − 5×10⁻⁴": "0.0005",
            "closer still:  r_c − 1.25×10⁻⁴": "0.000125",
            "very close:  r_c − 6×10⁻⁵": "0.00006",
            "past it:  r_c + 1×10⁻⁴  (the 3-cycle now exists)": "-0.0001",
        },
        value="closer:  r_c − 5×10⁻⁴",
        label="distance below the tangency",
    )
    return (distance_dropdown,)


@app.cell(hide_code=True)
def s4_figure(
    C_CONTEXT,
    C_OBS,
    C_SAT,
    C_SPREAD,
    C_TRUTH,
    LAMINAR_COUNT,
    LAMINAR_EPS,
    LAMINAR_INTERCEPT,
    LAMINAR_MEAN,
    LAMINAR_SLOPE,
    R_PERIOD3,
    distance_dropdown,
    finish_mpl,
    maps,
    mo,
    mpl_panels,
    np,
    systems,
):
    _eps = float(str(distance_dropdown.value))
    _r = R_PERIOD3 - _eps

    # One orbit, live: a few thousand scalar iterations is ~6 ms. The *scaling
    # law* in the right panel is precomputed, being six runs at 400,000 each.
    _lengths = maps.laminar_phases(systems.logistic_map, _r, n_iter=40000)

    # The display window is sized to show roughly a dozen laminar phases. Fixing
    # it at a few thousand iterates makes the panel solid ink -- the orbit
    # revisits three levels hundreds of times and the alternation that is the
    # whole point becomes invisible.
    _typical = _lengths.mean() if _lengths.size else 40.0
    _window = int(min(900, max(300, 13.0 * _typical)))
    _orbit = maps.map_orbit(systems.logistic_map, 0.4, _window, n_discard=2000, r=_r)

    # Mark which iterates are laminar, using the same criterion as the library.
    _f3 = maps.iterate_n(systems.logistic_map, _orbit, 3, r=_r)
    _is_laminar = np.abs(_f3 - _orbit) < 0.02

    _fig, _ax = mpl_panels(
        2,
        titles=(f"The orbit at r = {_r:.6f}  ({_window} iterates)",
                "Mean laminar length vs distance"),
        height=3.6,
    )
    _n = np.arange(_orbit.size)
    _ax[0].fill_between(_n, 0.0, 1.0, where=_is_laminar, color=C_CONTEXT,
                        step="mid", label="nearly 3-periodic")
    # Points, not a line: during a laminar phase the orbit cycles among three
    # levels, and connecting consecutive iterates draws vertical hash over the
    # whole panel instead of showing the three levels.
    _ax[0].plot(_n, _orbit, linestyle="none", marker="o", markersize=1.8,
                color=C_TRUTH, alpha=0.85)
    _ax[0].set_xlabel("iteration n")
    _ax[0].set_ylabel("$x_n$")
    _ax[0].set_xlim(0, _orbit.size - 1)
    _ax[0].set_ylim(0, 1)
    _ax[0].legend(loc="lower left", fontsize=6.5, framealpha=0.9)

    _e = np.asarray(LAMINAR_EPS)
    _m = np.asarray(LAMINAR_MEAN)
    _ax[1].loglog(_e, _m, marker="o", markersize=5, color=C_SPREAD,
                  linewidth=1.5, label="measured")
    _fit = np.exp(LAMINAR_INTERCEPT) * _e**LAMINAR_SLOPE
    _ax[1].loglog(_e, _fit, color=C_SAT, linewidth=1.3, linestyle="--",
                  label=f"slope {LAMINAR_SLOPE:.3f}  (theory −0.5)")
    if _eps > 0:
        _ax[1].plot([_eps], [_lengths.mean() if _lengths.size else np.nan],
                    marker="*", markersize=15, color=C_OBS, zorder=6,
                    label="this run (40k iterates)")
    _ax[1].set_xlabel("$r_c - r$")
    _ax[1].set_ylabel(r"$\langle L \rangle$  (iterations)")
    _ax[1].legend(loc="lower left", fontsize=6.5, framealpha=0.9)
    finish_mpl(_fig)

    if _eps < 0:
        _reading = f"""**Above the tangency.** The 3-cycle now exists and is stable, so
the orbit is periodic and every iterate is "laminar" — the shading fills the
panel. There are no bursts because there is nothing to burst out of. This is the
control case: the intermittency below $r_c$ is not noise or a transient, it is
the signature of a cycle that has not yet been born."""
    else:
        _mean = _lengths.mean() if _lengths.size else float("nan")
        _predicted = np.exp(LAMINAR_INTERCEPT) * _eps**LAMINAR_SLOPE
        _reading = f"""At $r_c - r = {_eps:.2e}$ this 40,000-iterate run found
**{_lengths.size} laminar phases**, mean length **{_mean:.1f}**, longest
**{_lengths.max():.0f}** — against {_predicted:.1f} predicted by the fitted power
law. The star marks it on the right-hand panel.

Notice what the shading shows: the orbit is *not* switching between two
attractors. There is one attractor, and the trajectory spends part of its time in
a region of it where the dynamics are nearly periodic. Nothing external changes
at the moment a burst begins."""

    mo.vstack([
        mo.hstack([distance_dropdown], justify="start"),
        _fig,
        mo.md(_reading),
        mo.callout(
            mo.md(
                f"""### The scaling, and why it matters beyond this map

The precomputed sweep gives an exponent of **{LAMINAR_SLOPE:.4f}** against the
theoretical $-1/2$, measured over a {max(LAMINAR_EPS) / min(LAMINAR_EPS):.0f}-fold
range of $r_c - r$ across which $\\langle L \\rangle$ changes by a factor of
{max(LAMINAR_MEAN) / min(LAMINAR_MEAN):.1f} ({min(LAMINAR_COUNT)}–{max(LAMINAR_COUNT)}
phases sampled per parameter).

The exponent comes from the normal form, not from the logistic map, so it is
another universal number — and it has a practical descendant. A timescale
diverging as a control parameter approaches a threshold is exactly the
**critical slowing down** that early-warning indicators try to detect: rising
autocorrelation and variance in a system approaching a tipping point. Chapter 27
takes that up properly. What this section establishes is that the effect is
real, quantitative, and visible in one line of arithmetic."""
            ),
            kind="info",
        ),
    ])
    return


# ===========================================================================
# 5. Why a one-variable map is in a weather and climate book
# ===========================================================================
@app.cell(hide_code=True)
def s5_text(mo):
    mo.md(
        r"""
    ---
    ## 5 · What this has to do with forecasting

    The logistic map is not a climate model and nothing here is a claim that it is.
    Four transferable lessons, in increasing order of consequence:

    **A parameter's effect need not be smooth.** Return to the opening question — a
    parameter uncertain by half a percent. Over most of the range in Section 2 that
    uncertainty is harmless. Near $r = 3.449$ or $r = 3.828$ it changes the *kind* of
    behaviour the model produces. This is the honest form of "parametric
    uncertainty", and it is why perturbed-parameter ensembles sometimes return
    bimodal distributions rather than a spread around a central estimate.

    **"Chaotic" is a property of a parameter value, not of a system.** Section 2's
    period-3 window sits well above $r_\infty$ and has $\lambda < 0$ throughout, and
    inside it there is a complete cascade, and inside *that* another window. The
    chaotic and periodic parameter sets are interleaved at every scale, so no finite
    sampling of parameter space establishes which side a given model is on.

    **A regime can end with nothing changing.** Section 4's laminar phases begin and
    end with no change in any parameter and no external forcing. When a real
    atmospheric regime — a blocking episode, a stalled jet — breaks down, "what
    caused it?" may have no answer beyond the internal dynamics. And the diverging
    timescale near the threshold is the mechanism behind early-warning indicators.

    **The numbers are portable, and that is the whole design of this book.**
    $\delta = 4.669$ and the $-1/2$ intermittency exponent belong to a *class* of
    systems, not to any member of it. Feigenbaum's constant was subsequently measured
    in convecting mercury, dripping taps and nonlinear circuits
    *[citation needed: Libchaber & Maurer (1980); Linsay (1981)]*. That is the
    licence under which a three-variable Lorenz system earns a place in a book about
    forecasting a $10^9$-variable atmosphere: not because it resembles one, but
    because some quantities do not care about the difference.

    Where this goes next: **chapter 6** builds the Lorenz system, whose transition
    to chaos is a Hopf bifurcation rather than a cascade; **chapter 7** takes
    $\lambda$ into continuous systems with many directions, where it becomes a
    spectrum; **chapter 27** returns to bifurcations as tipping points, with the
    coexisting attractors that Section 1 deliberately did not have.
    """
    )
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

    1. **Find a bifurcation by hand.** In Section 1, put $r$ just below 3.449 and
       then just above. Read the multiplier and the cycle verdict at each. Now do the
       same at 3.544. What is the multiplier doing at the moment the period doubles,
       and why is that the *same* value both times?
    2. **Break the "past $r_\infty$ means chaotic" rule.** Set Section 2's window to
       the period-3 range and read $\lambda$. Then set $r = 3.83$ in Section 1 and
       count the cycle. Explain why the bifurcation diagram alone could mislead you
       here and $\lambda$ could not.
    3. **Watch $\delta$ appear.** In Section 3, read the ratio column down the
       levels. It converges from *below* for the logistic map. Estimate how many more
       levels would be needed for six-figure agreement, then explain why double
       precision will not supply them.
    4. **Approach the tangency.** Step Section 4's distance from $2\times10^{-3}$ to
       $6\times10^{-5}$ and watch the laminar phases lengthen. Predict the mean length
       at $r_c - r = 10^{-5}$ from the scaling law. Then ask what would be needed to
       *measure* it, and why that is the same difficulty a real early-warning
       indicator faces.
    5. **Change the initial condition, in two places.** In Section 1, $x_0$ changes
       nothing about the final state. Convince yourself this is a property of the map
       having one attractor, and predict what Section 1 would look like for a system
       with two.

    ## What you should have seen

    A one-line map has a complete route to chaos: a stable fixed point until
    $r = 3$, a cascade of period doublings accumulating at
    $r_\infty = 3.5699457$, and chaos beyond — interleaved at every scale with
    periodic windows, of which the period-3 window at $r_c = 1 + 2\sqrt{2}$ is the
    widest. The Lyapunov exponent is what tells the two apart, and for a 1-D map it
    is one average of $\ln|f'|$, verified against the exact $\ln 2$ at $r = 4$.

    The cascade spacings shrink by $\delta = 4.669\,201\,609$, and **three map
    families with no algebra in common give the same $\delta$** — measured here to
    $1\times10^{-5}$ for the logistic map, $2\times10^{-4}$ for the cubic. That is
    universality, and it is the reason this book studies low-order models at all.

    Below the period-3 tangency the orbit is intermittent: long nearly periodic
    stretches ended by chaotic bursts, with no parameter changing, and a mean
    laminar length diverging as $(r_c - r)^{-0.4965}$ against a predicted $-1/2$.

    ## Further reading

    - Feigenbaum, M. J. (1978). Quantitative universality for a class of nonlinear
      transformations. *Journal of Statistical Physics*, **19**, 25–52.
    - May, R. M. (1976). Simple mathematical models with very complicated dynamics.
      *Nature*, **261**, 459–467 — the paper that put this map in front of everyone.
    - Pomeau, Y. and Manneville, P. (1980). Intermittent transition to turbulence in
      dissipative dynamical systems. *Communications in Mathematical Physics*,
      **74**, 189–197 — the $-1/2$ law.
    - Strogatz, S. H. *Nonlinear Dynamics and Chaos*, ch. 10 — the clearest textbook
      treatment of this material *[citation needed: edition and section numbers]*.
    - Smith, L. A. (2007). *Chaos: A Very Short Introduction* — ch. 6 on the logistic
      map, written for exactly this book's audience *[citation needed: pages]*.
    """
    )
    return


if __name__ == "__main__":
    app.run()
