---
title: "Chapter 28 · Has predictability changed over time?"
weight: 728
part: "Part VII — Frontiers"
knob: 'verification convention, observing-system improvement, record length'
status: "live"
---

## Overview

The useful range of a global forecast has advanced by roughly a day per decade for fifty
years *[citation needed]*. Two entirely different things could have produced that curve:
the forecasting system got better, or the atmosphere itself became easier to forecast.
They have opposite implications — one is an achievement that can be continued, the other
a property of the climate that could reverse — and **a skill record cannot separate
them**, because every point on it is the product of a system and an atmosphere that both
changed.

This chapter builds the separation in a model where both factors are known exactly, and
then asks what it would take to do it from a record.

## A difficulty that comes first

"The predictability of the 1980s" is not a well-posed quantity. A Lyapunov exponent is a
limit as $T \to \infty$; it is a property of an *attractor*, and a system whose forcing
is changing does not have one. Every number in this chapter is therefore a **finite-sample
estimate of a finite-time quantity** — which is not a caveat to be filed at the end, but
the reason the last section exists and the reason its answers are large.

## The model

Lorenz 96 on 40 sites with the forcing ramped from $F = 6$ to $F = 10$ over 4,000 time
units — about 55 years at the conventional five days per unit. The forecasting system is
perfect-model and its analysis error is set by hand, so *the system* and *the atmosphere*
are two independent knobs rather than the confound they are in the real record.

One exact fact makes $F$ the right knob: the trace of the Jacobian is $-N$ for every
state and every $F$, so $\sum_i \lambda_i = -N$ **however far the ramp goes**. The ramp
cannot change the total contraction of the flow, only redistribute it — whatever happens
to $\lambda_1$ is paid for somewhere in the stable spectrum. Two further identities are
asserted as tests rather than assumed: $\langle x^2\rangle = F\langle x\rangle$ holds
exactly on any stationary attractor (worst residual across ten climates:
$1.2\times10^{-4}$), and the ramped right-hand side is bitwise Lorenz 96 at zero ramp
rate.

{{< marimo src="/nb/ch28_nonstationary-predictability.html" >}}

## Which predictability? Six indices, six trends

Over one ramp, with a *frozen* forecasting system, the useful horizon falls from 3.77 to
1.80 time units — 52 %, at $-0.198 \pm 0.022$ per epoch, $t = -9.0$. Nothing about the
forecasting system moved. But that is one of six defensible answers:

| index | change over the ramp | what it governs |
|---|---|---|
| $\lambda_1$ | **+102 %** | the asymptotic error-doubling rate |
| $h_{KS}$ | **+158 %** | the rate at which information is destroyed |
| $D_{KY}$ | **+27 %** | the dimension of the attractor |
| unstable dimension | **+27 %** | the ensemble size that spans the error |
| $\delta_\infty$ | **+48 %** | the amplitude of what is being forecast |
| useful horizon | **−52 %** | how long a forecast is worth using |

**The question is not well posed until the index is named.** Even the *convention*
matters: verifying against each epoch's own climatology gives $-0.198$ per epoch, and a
fixed absolute threshold gives $-0.229$ — 16 % apart, from the same forecasts.

The horizon law $T = \lambda_1^{-1}\ln(f\delta_\infty/\delta_0)$ splits the change
exactly, with no residual and no linearisation, and order-independently. For the frozen
system the two live terms disagree in sign: instability costs **−1.79** time units and
amplitude *returns* **+0.26**. That gift is real and worth naming — a fixed absolute
analysis error is a smaller *relative* error against a larger-amplitude flow, so **a
warming climate hands the forecasting system an improvement that has nothing to do with
the forecasting system and nothing to do with the flow becoming more predictable.**

One entry carries a warning. The unstable dimension runs 11, 12, 12, 13, 13, 13, 14, 13,
14, 14 — **not monotone**. That is not physics: the exponents near zero are unresolved by
1,000 time units of averaging, and the count is a discrete functional of quantities that
are almost exactly zero. [Chapter 29]({{< relref "ch29_ml-prediction.md" >}}) found the
same thing in a learned emulator and called it the fragile entry in the census. It is
fragile in the **perfect model** too — and it is the entry that sets the ensemble size.

## The 2×2, and the return that shrinks

Four experiments, of which the real world runs two. The diagonal is the skill record; the
off-diagonal cells are **reforecasts** — a frozen system re-run on a period it never
operated in. Those two cells are why a reforecast archive is not documentation but an
instrument, and with all four the division

$$
\underbrace{T_{11}-T_{00}}_{\text{observed}}
= \underbrace{(T_{10}-T_{00})}_{\text{system}}
+ \underbrace{(T_{01}-T_{00})}_{\text{climate}}
+ \underbrace{(T_{11}-T_{10}-T_{01}+T_{00})}_{\text{interaction}}
$$

is exact. Measured, at one decade of analysis-error reduction: the climate alone costs
**−1.96** time units, the observing system returns **+2.19**, the interaction is
**−1.21**, and the record moves **−0.98**.

**The record understates the dynamical change by a factor of two, and a flat record
understates it entirely.** At 1.5 decades of observing improvement the skill curve shows
no trend at all while predictability has fallen by half. The direction of the error is
always the same: the observing system masks the climate.

**The interaction is predicted, not fitted.** The horizon law gives
$\ln 10\,(1/\lambda_1^{\rm late} - 1/\lambda_1^{\rm early}) = -1.05$ against $-1.21$
measured — 13 % apart, right sign and magnitude, from a closed form with no fitted
constant. The residual is itself instructive and the next section explains it.

So [chapter 20]({{< relref "../part5/ch20_da-in-practice.md" >}})'s $\ln 10/\lambda_1$
per decade of analysis error is **not a constant of nature**. It falls from 10.4 days to
5.1 days over the ramp — a 51 % decline in the return on observing-system investment. And
holding the horizon merely *level* costs a factor of 30 to 81 in analysis error, one and a
half to two decades, spent entirely on standing still.

## Dynamics or circulation? And the lead time you ask about

Two mechanisms could raise an atmosphere's mean instability: the dynamics changed, or the
circulation changed so that the system now visits the unstable states more often. A
**shift-share** decomposition separates them exactly and symmetrically in the two epochs.
Applied here it reports 100 % "dynamics" — and the honest reason is a limitation of the
model. **No large-scale index predicts local instability in Lorenz 96 at $N = 40$**: the
largest correlation with the finite-time exponent, over seven candidates and both
climates, is $0.13$. At this size the system is spatiotemporally chaotic with a dozen
unstable directions and no regimes, so it has no "kind of day" for occupancy to shift
between. The tool transfers; this model's answer does not, and naming the missing
experiment is more useful than a decomposition that reports 100 % of an effect it could
not have detected.

The section's real finding is elsewhere. **The trend depends on the lead time you ask
about:**

| $\tau$ | change over the ramp |
|---|---|
| 0.5 TU (2.5 d) | **+44 %** |
| 1.0 TU (5 d) | **+52 %** |
| 2.5 TU (12.5 d) | **+69 %** |
| 5.0 TU (25 d) | **+85 %** |
| $\infty$ ($\lambda_1$) | **+102 %** |

At five days — the lead a forecaster cares about most — instability rose 52 % against
102 % for $\lambda_1$. **The asymptotic exponent overstates the forecast-relevant change
by a factor of two.** The mechanism is in the same table: short-lead growth is dominated
by **non-normal** transient amplification
([chapter 16]({{< relref "../part5/ch16_adjoint-sensitivity.md" >}})), whose ratio to
$\lambda_1$ falls from 3.4 to 2.4 over the ramp — the flow becomes *less* non-normal as
it is forced, and the two effects partly cancel. A trend in $\lambda_1$ is an upper bound
on what a forecast office feels.

## How long a record does it take?

An operational centre has one forecast a day from a system that changed continuously, and
a change of a few per cent per decade to find. Two facts set the answer.

**A forecast a day is not a forecast's worth of information a day.** The daily useful
horizon autocorrelates at $+0.52$ at one day and $+0.48$ at two, with an integrated
autocorrelation time of **8 days**, so the effective sample size is **12 %** of the
launches: 365 a year is worth about **46 independent measurements** of the atmosphere's
predictability.

**And length beats density.** With $n$ epochs of $m$ independent cases,
$S_{xx} = m\,n(n^2-1)/12$, so $\operatorname{se}(b) \propto m^{-1/2} n^{-3/2}$ — doubling
the cases per year buys $\sqrt2$, doubling the record buys $2\sqrt2$.

| true trend | record needed | if launches were independent |
|---|---|---|
| 2 % per decade | **31 years** | 16 years |
| 5 % per decade | **17 years** | 9 years |
| 10 % per decade | **11 years** | 6 years |
| 20 % per decade | **7 years** | 4 years |

Treating a daily record as 365 independent cases a year understates the requirement by
roughly a factor of two in every row — the difference between a claim that clears the bar
and one that does not. And a short record does not merely fail: at five years, 0.8 % of
records return a **significant trend of the wrong sign**.

Every year of that is a year of a *frozen* system. The record that exists is not frozen,
so a real trend analysis measures the `observed` column — system, climate and interaction
together. Recovering the climate term needs the two reforecast cells, and those have to be
computed.

## Exercises

1. The two verification conventions give trends of 0.198 and 0.229 time units per epoch.
   Derive the ratio from the decomposition's amplitude term, and predict what it would be
   if the climate's variance had not changed.
2. The break-even calculation is an exponential in $\lambda_1 T$. Show that a 10 % error
   in the starting horizon moves the required accuracy improvement by $e^{0.1\lambda_1T}$,
   evaluate it for this ramp, and say which of the chapter's two break-even numbers you
   would quote.
3. Section 4 found the trend smaller at short lead than at long. Sketch the corresponding
   figure for a system with **no** non-normal growth, and say what measurement would
   distinguish the two cases.
4. Section 5 assumes the case-to-case scatter is stationary. It is not — it falls with the
   horizon. Does a shrinking scatter make the trend easier or harder to detect, and by
   roughly how much?
5. The unstable dimension is non-monotone in the perfect model with 1,000 time units of
   averaging. Estimate the averaging length needed to resolve its trend, and say what that
   implies for reading such a trend off any real record.

## Further reading

- Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on the growth of
  forecast skill *[citation needed: chapter]*
- Lorenz (1982), on error growth estimated from an operational forecast archive
  *[citation needed]*
- On the ECMWF forecast-skill record and the role of reforecasts in interpreting it
  *[citation needed]*
- On observed and projected changes in mid-latitude circulation variability and their
  implications for predictability *[citation needed]*
