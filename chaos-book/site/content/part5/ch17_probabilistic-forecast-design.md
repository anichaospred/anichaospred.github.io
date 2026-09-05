---
title: "Chapter 17 · Probabilistic forecast design"
weight: 517
part: "Part V — The machinery of prediction"
knob: 'ensemble size, perturbation strategy'
status: "live"
---

## Overview

[Chapter 19]({{< relref "ch19_ensemble-da.md" >}}) treated an ensemble purely as a
covariance estimator — a device for building $\mathbf{P}^f$. That is one use, and not
the one the public sees. The forecast that goes out is a *probability*: 30 % chance of
frost tonight. A probability makes a claim that can be checked, which a single number
cannot.

Three separate questions follow. **How should the ensemble be built?** The obvious
criterion — pick the fastest-growing directions — turns out to be the wrong one. **What
makes a probability forecast good?** Not accuracy alone; a forecast can be sharp and
wrong or honest and useless, and the standard scores mix these together. **What is it
worth?** This is the question that justifies the expense of running an ensemble, and it
has an answer no accuracy score can give.

## The model

Cycling LETKF on Lorenz 96 ($N=40$, $F=8$, all sites observed every 0.05 TU with
$\sigma_o=1$ — [chapter 19]({{< relref "ch19_ensemble-da.md" >}})'s configuration)
supplies both the analysis and its own analysis ensemble, so **the analysis-error
distribution the ensembles are meant to sample is the one they are scored against.**

Five constructions, all from the same analysis, with the same member count and the same
total perturbation amplitude — so the comparison is about *direction* and nothing else:
isotropic random, bred vectors, bred-and-orthogonalised, singular-vector $\pm$ pairs,
and the EDA analysis ensemble itself.

{{< marimo src="/nb/ch17_probabilistic-forecast-design.html" >}}

## Bred vectors collapse, and worse than that

Breeding (Toth & Kalnay 1993) needs **no adjoint** — perturb, integrate, difference,
rescale, repeat — which is why it was NCEP's operational scheme while ECMWF ran singular
vectors. After a few e-foldings the perturbation points along the locally
fastest-growing direction.

*The* fastest-growing direction: the same one every time. Independently seeded bred
vectors converge onto each other, at a rate set by how well separated the leading
Lyapunov exponent is — about **2 e-foldings on Lorenz 63** (one positive exponent; mean
pairwise angle falling from 52° unbred to 3.8° after 2.3 e-foldings and 0.1° by 5.7)
and about **8 on Lorenz 96**, which has
thirteen positive exponents of similar size and so converges to a subspace rather than
a line, levelling off near 22° from an unbred 84°.

And the degeneracy is sharper than "narrow". Breeding fixes a direction but not a
**sign** — the rescaling normalises the difference and nothing prefers $+v$ to $-v$ — so
a fully bred set collapses into **two antipodal clusters**. Measured on Lorenz 96 after 200 breeding
cycles, **85 % of members** lie within 26° of $\pm$ the leading one — 39 % aligned
with it and 46 % *anti*-aligned. Orthogonalised, that figure is 5 %. An ensemble of two points is not
a sample of a distribution at all, which is why the plain-bred rank histogram is spiked
at the cluster edges rather than merely U-shaped.

Re-orthogonalising after **each** cycle fixes it, and converts breeding into the
Gram–Schmidt construction of [chapter 7]({{< relref "ch07_lyapunov-exponents.md" >}})'s
Lyapunov spectrum. Doing it once at the end would be useless: by then the set is nearly
rank one, and orthogonalising a collapsed set manufactures its extra directions out of
rounding error.

## Four results

**Growing fastest is the wrong objective.** Singular vectors are by construction the
fastest-growing perturbations, and an ensemble of them is **over-dispersed by two thirds** at
medium lead (spread/error 1.67) and scores worse than isotropic noise at short lead.
This is not the method failing; it is the method working as designed. Maximising growth
and sampling the analysis-error distribution are different objectives, and only the
second is what a probability forecast needs.

**The collapse is expensive.** Plain bred vectors are the worst construction at every
lead, and get worse: spread/error falls to **0.67** at lead 2, because every member
grows along one direction while the true error spreads into a thirteen-dimensional
unstable subspace. Orthogonalising recovers most of it.

**Construction beats ensemble size.** CRPS fits $a + b/k$ closely. The $1/k$ term is the
**estimator's** finite-sample bias — $b$ is the same for every construction, costing
about 5 % at $k=20$ — while the asymptote $a$ is where construction shows, and **no
ensemble size closes a gap in $a$**. An infinite ensemble of the wrong directions is
still worse than a small one of the right ones.

**Reliability is repairable; resolution is not.** Murphy's decomposition
$\mathrm{BS} = \mathrm{REL} - \mathrm{RES} + \mathrm{UNC}$ holds *exactly* when the bins
are the distinct forecast values (asserted as a test). The EDA has the **most**
resolution and the **worst** reliability — most information, stated least honestly — and
relabelling its probabilities with the frequencies they actually attained removes the
reliability penalty while leaving resolution unchanged to five decimal places. That
recalibration is in-sample here, which is why reliability comes out at exactly zero; the
structural claim, that resolution survives, does not depend on the sample, but the size
of the gain should be read as an upper bound.

## The knob: what a probability is worth

Every score above measures whether the forecast is *right*. None measures whether it is
*useful*. Put a decision behind it: a user pays $C$ to protect against an event that
would otherwise cost $L$, and protects whenever the forecast probability exceeds
$\alpha = C/L$. The relative value $V(\alpha)$ is 1 for a perfect forecast, 0 for
climatology, and negative for one worse than useless.

**$V$ is a curve, not a number.** A council gritting roads sits at small $\alpha$; a
factory shutting a line sits at large $\alpha$. Measured, with the *same information*
served two ways — the probabilistic forecast against itself pushed through a 50 %
threshold:

| | useful range | peak | worst |
|---|---|---|---|
| probabilistic | 19 / 19 ratios | 0.86 | **+0.55** |
| deterministic | 16 / 19 | 0.80 | **−1.43** |

The deterministic forecast is not merely less useful at the extremes, it is **actively
harmful** — a user at $\alpha = 0.95$ would do better ignoring it and following the
climatology. This is the argument for ensemble forecasting that no accuracy score can
make: the ensemble is not primarily a device for being more often right, it is a device
for letting every user apply their own threshold. A deterministic forecast has already
made that choice on their behalf, and cannot have made it correctly for more than one of
them.

## Exercises

1. Slide the lead in section 2 and watch the singular-vector rank histogram change
   shape. At which lead is it flattest, and what does that say about the window the
   singular vectors were computed over?
2. Bred vectors are under-dispersed and singular vectors over-dispersed. Would an
   ensemble built from both be calibrated? What would its rank histogram look like?
3. Predict $b$ in CRPS $= a + b/k$ for a 100-member ensemble of the *wrong* directions,
   and say which of $a$ and $b$ more computer time can improve.
4. The recalibration in section 4 is in-sample. Design the out-of-sample version and say
   what you expect to happen to reliability and to resolution.
5. Section 5's deterministic forecast thresholds at 50 %. Find the threshold maximising
   its value at $\alpha = 0.1$, and explain why no single choice serves both ends.

## Further reading

- Toth & Kalnay (1993, 1997), breeding and the NCEP ensemble *[citation needed]*
- Molteni et al. (1996), the ECMWF singular-vector ensemble *[citation needed]*
- Buizza & Palmer (1995), on singular vectors and ensemble design *[citation needed]*
- Murphy (1973), the Brier score decomposition *[citation needed]*
- Hersbach (2000), on CRPS and its decomposition *[citation needed]*
- Richardson (2000), on the relative economic value of ensemble forecasts
  *[citation needed]*
- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*
  *[citation needed: chapter number]*
