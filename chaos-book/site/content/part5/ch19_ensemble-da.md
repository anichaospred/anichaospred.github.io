---
title: "Chapter 19 · Ensemble data assimilation"
weight: 519
part: "Part V — The machinery of prediction"
knob: 'ensemble size, localisation radius'
status: "live"
---

## Overview

[Chapter 18]({{< relref "ch18_variational-da.md" >}}) fixed 3D-Var's structural defect
*inside* the assimilation window, through the propagators in its Hessian. But its
background covariance at the window start is still static, and building the tangent
linear and adjoint models to get that far is years of work for a model that changes
every few months.

So: can you get a flow-dependent background covariance without an adjoint at all? Yes,
and the idea is almost embarrassingly direct — run an ensemble and use its sample
covariance,

$$
\mathbf{P}^f = \frac{1}{k-1}\sum_{i=1}^{k}
  (x^f_i - \bar x^f)(x^f_i - \bar x^f)^{\top}.
$$

Nothing is linearised, nothing is transposed, and the covariance adapts to the flow for
free. This is why ensemble filters reached ocean, land, sea-ice and coupled models long
before variational methods did.

The catch is arithmetic, and it is severe. An operational model has $n \sim 10^8$ state
variables and an ensemble has $k \sim 100$ members, so $\mathbf{P}^f$ has rank at most
$k-1$. Almost everything in this chapter is about living with that.

## The model

Cycling assimilation on Lorenz 96, $N = 40$, $F = 8$: all 40 sites observed every
0.05 TU — the six-hour analogue — with $\sigma_o = 1$, scored over 500 cycles after
spin-up. Climatological spread is 3.64, so a filter that learns nothing scores 3.64 and
a well-configured one about 0.17.

**Every comparison is paired**: one truth trajectory, one array of observation noise and
one pool of initial perturbations, drawn before the sweeps and sliced rather than
redrawn, so a 5-member run sees exactly the same observations as a 40-member one.
**Every scheme is given its own best inflation and its own best localisation radius**
from a reported set — comparing a tuned filter against an untuned one is the commonest
way to make a filter comparison say whatever you wanted.

{{< marimo src="/nb/ch19_ensemble-da.html" >}}

## What the covariance being estimated actually looks like

Not the climatology of the *state* — that is a different object and nearly diagonal for a
homogeneous chaotic system. What a filter estimates is the covariance of **forecast
errors**, and time-averaged over 500 cycles its correlation profile runs $1.000$,
$+0.044$, $\mathbf{-0.183}$, $-0.007$, then flat.

That negative lobe at separation 2 is not noise. The dominant Lorenz 96 mode sits at
$m^* = 8$ — [chapter 11]({{< relref "ch11_lorenz96.md" >}}), from the dispersion
relation — giving $40/8 = 5$ sites per wavelength and an anticorrelation at half of that,
2.5. **The structure localisation must preserve is the shape of the system's leading
wave**, not a smooth blob.

## Five results

**Sampling error scales as $k^{-1/2}$ with a constant of exactly 1.** Sampling a known
correlation matrix whose far field is identically zero, the RMS spurious correlation
times $\sqrt{k}$ runs 1.12, 1.06, 1.03, 1.03, 1.03, 1.02, 1.01, 0.94, 0.99, 1.00 over six
doublings. Read it as economics: halving the noise costs four times the ensemble, and the
ensemble is the most expensive component of a forecasting system. No affordable ensemble
estimates a $10^8\times10^8$ covariance, which makes localisation a **precondition** for
ensemble filtering rather than a refinement of it.

**Localisation's deeper job is rank, not noise.** A global filter's analysis is
$\bar x^f + \mathbf{X}^f w$, so its increment lies in the span of the $k-1$ ensemble
perturbations — verified to $2\times10^{-15}$, an identity rather than an approximation,
and no amount of covariance tapering changes it. A *local* filter solves $N$ separate
$k$-dimensional problems, and at $k=10$ **78 %** of its increment lies outside that span:
directions the ensemble could not represent, corrected anyway. That is what lets five
members assimilate forty variables.

**The cliff.** With no localisation, $k=5$ gives an analysis error of 4.24 — *worse than
climatology*, so the filter has been actively harmed by its own observations. The same at
$k=10$ (3.38). Between $k=10$ and $k=15$ it falls to 0.24 and stays there. That is filter
divergence: a rank-deficient covariance underestimates the background error, so the
observations are underweighted, so the analysis stays too close to the background, so the
next forecast error exceeds what the covariance claims — and the covariance comes from
the same ensemble that is now too narrow.

**The ridge, and an asymmetry worth remembering.** The optimal radius **grows with
ensemble size**: 12 sites at $k=5$, 20 at $k=8$–15, and no localisation at all by
$k=20$. Localisation is a bias–variance trade, and which term dominates depends on how
many members you have. The penalty for getting it wrong is wildly lopsided: too *loose*
costs a factor of **19** at $k=5$; too *tight* costs a factor of 1.5. **If you must guess
a localisation radius, guess low.**

**Localisation and ensemble size are substitutes.** At radius 2 the analysis error is
0.350, 0.342, 0.336, 0.341, 0.333, 0.335, 0.338 for $k = 5 \ldots 40$ — a 3 %
improvement for eight times the members. Tight localisation makes each local problem so
small that five members already suffice, and the other thirty-five buy nothing. This is
why operational ensembles are tens of members and not thousands.

## Deterministic or stochastic

Chapter 20 uses the perturbed-observation EnKF, where each member assimilates
$y + \epsilon_i$. Those $\epsilon_i$ are themselves a finite sample, adding a second
sampling error on top of the one in $\mathbf{P}^f$. The ETKF avoids it with a single
deterministic transform, and its anchor is exact: **the ETKF analysis mean and covariance
are the Kalman filter's for the covariance the ensemble actually has**, to $10^{-12}$ and
$10^{-11}$, at every ensemble size *including $k<n$* where the sample covariance is
singular.

Measured, each scheme at its own best configuration: the deterministic filter is **64 %
better at $k=5$**, falling to 14 % at $k=40$. It shrinks but does not vanish, and it
matters most where ensembles are expensive — which is always.

## Hybrids, and a result that came out small

Blending in a static covariance, $\beta\mathbf{P}^e + (1-\beta)\mathbf{B}$, is full rank
for any $\beta<1$ and is a second, entirely separate cure. Its optimum moves with
ensemble size just as the radius did — best $\beta$ = 0.30, 0.70, 0.95 at $k$ = 5, 10, 20.

But on its own it is much the weaker cure: at $k=5$ the best hybrid (0.44) is only 4 %
better than **discarding the ensemble entirely** and running 3D-Var (0.46), because a
single global $\beta$ cannot supply the missing directions selectively. And adding it on
top of localisation gives 0.36 against 0.39 — **8 %**, a few times this experiment's
noise floor of ±2.5 %, estimated from the flat radius-2 column.

Worth stating plainly, since hybrids demonstrably earn more than 8 % operationally. This
experiment is the friendliest possible case for a localised ensemble: homogeneous system,
perfect model, every variable observed every cycle. Hybrids are reported to help where
the ensemble covariance is worst — sparse observations, model error, flow-dependent
structure a single radius cannot follow *[citation needed]*. **This chapter has not
tested the strong case for hybrids rather than argued against it.**

## A methodological note

The static covariance had to be *tuned*, not merely climatological. Raw Lorenz 96
climatology has a spread of 3.6 against a background error near 0.3; used unscaled it
makes pure 3D-Var 0.94 instead of 0.45, which would have flattered every hybrid measured
against it by a factor of two. The scale used is 0.02 of the climatological covariance,
chosen by sweeping it.

## Exercises

1. Set $k=5$ in the section 4 slider and read the slice left to right. Where is the
   minimum, and how much worse is `inf`? Now do the same at $k=40$.
2. Section 3's localised curves *fall* with $k$ while section 4's optimal radius *grows*
   with $k$. Explain why these are the same fact.
3. The radius-2 column is nearly flat in $k$. Predict what it would look like on Lorenz
   96 with $N=400$, and say which quantity you expect to be unchanged.
4. Inflation's optimum is a bowl whose dangerous side is *too little*; the localisation
   radius' dangerous side is *too loose*. Which would you rather tune in a system that
   must not fail?
5. Show algebraically that a global ensemble filter's increment lies in the span of the
   ensemble perturbations, for **any** gain matrix of the form
   $\mathbf{P}^f\mathbf{H}^{\top}(\cdot)^{-1}$ — including a covariance-localised one.

## Further reading

- Evensen (1994), the original ensemble Kalman filter *[citation needed]*
- Burgers, van Leeuwen & Evensen (1998), on why observations must be perturbed
  *[citation needed]*
- Bishop, Etherton & Majumdar (2001), the ensemble transform Kalman filter
  *[citation needed]*
- Hunt, Kostelich & Szunyogh (2007), the LETKF *[citation needed: pages]*
- Houtekamer & Mitchell (1998, 2001), on localisation and filter divergence
  *[citation needed]*
- Hamill & Snyder (2000), on hybrid ensemble–variational covariances *[citation needed]*
