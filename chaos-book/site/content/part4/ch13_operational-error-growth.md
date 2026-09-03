---
title: "Chapter 13 · Error growth in operational models"
weight: 413
part: "Part IV — Many scales, many degrees of freedom"
knob: 'the observing network, the fit window'
status: "live"
---

## Overview

Everything so far has been measured by twin experiment: take a state, perturb it,
integrate both copies, watch them separate. That method needs the truth.

A forecast centre does not have the truth. It has an **analysis** — itself a forecast,
corrected by observations — whose own error is a substantial fraction of a short-range
forecast's, so verifying against it measures the forecast error *plus* the analysis
error, correlated in ways that depend on the assimilation system. And there is no second
Earth to perturb.

Lorenz (1982) answered this with an idea needing no truth at all. Two forecasts **valid
at the same time** but started a day apart — a one-day and a two-day forecast, both
verifying today — can be differenced from the archive alone. And their difference grows
at the rate errors grow, because at the moment the older forecast reached yesterday it
differed from yesterday's analysis by roughly a one-day forecast error, and the two have
been running together ever since.

## The model

The trouble with a truth-free estimator is that you cannot check it against the truth.
So this chapter builds a **synthetic operational centre** on Lorenz 96 where the truth
is known and withheld: a cycling ensemble Kalman filter with six-hourly analyses,
forecasts to thirty days from each of 600 consecutive analyses, and four observing
networks from 40 observed sites down to 8. Then it runs Lorenz's estimator — which sees
only the forecasts — against the error curve computed from the truth.

`chaoslib.errorgrowth.lagged_forecast_difference` implements the estimator;
`chaoslib.assimilate` supplies the EnKF.

{{< marimo src="/nb/ch13_operational-error-growth.html" >}}

## Three results

**The estimator works, and not just for one fitted number.** Across the three networks
that have an exponential phase at all, the truth-free estimate recovers the true growth
rate to within 2% at analysis errors of 0.5% and 2% of saturation, and 5% at 6%. More
than that: plotting each curve's local growth rate against its own amplitude, the two lie
close together over the *whole* range — so the method recovers the rate at every error
amplitude, including the amplitude-dependence
[chapter 9]({{< relref "../part3/ch09_nonlinear-error-growth.md" >}}) established. The
estimated curve sits below the true one, because it starts from a one-cycle forecast
difference rather than the analysis error; that offset shifts the curve along the lead
axis and leaves the rate alone, which is exactly why the method works.

It fails only where the archive offers nothing. With the analysis error at 45% of
saturation there is no exponential phase, and the *true* rate is equally unmeasurable —
a failure of the data, not the estimator.

**The logarithmic return holds end to end.** Thirteen times better analysis buys
**7.0 days** of lead time against 6.9 predicted from the measured growth rate — the same
constant [chapter 8]({{< relref "../part3/ch08_attractor-dimension.md" >}}) derived from
$\lambda_1$ and [chapter 20]({{< relref "../part5/ch20_da-in-practice.md" >}}) measured
on a cycling system, obtained here a third way by building four observing networks and
reading off the lead times.

**The operational doubling time is not $\ln2/\lambda_1$, and can be either side of it.**
Fitted at small error the rate is 1.91 per time unit — *above* $\lambda_1 = 1.67$, because
a generic finite perturbation grows faster than the asymptotic rate over a finite window
([chapter 16]({{< relref "../part5/ch16_adjoint-sensitivity.md" >}}) measured optimal
growth at 1.6–2.6× the Lyapunov estimate), and an analysis error is especially prone to
it: the directions an observing network struggles with are not chosen at random relative
to the ones that grow. Fitted at larger error the rate falls, per chapter 9. For this
network the two effects give a doubling time of **1.86 days** against the Lyapunov
estimate's 2.08. Which number to quote? None alone — an error-doubling time without a
stated error amplitude is not a number.

## What the method cannot see

Both forecasts in a lagged pair come from the **same model**, so anything systematically
wrong with that model is common to both and cancels algebraically in the difference. The
test for this adds a constant bias to an entire archive and asserts the estimator's
output is unchanged to round-off.

So the method measures growth of **initial-condition** error and is blind to model error
by construction. A forecast system can be losing skill because its initial conditions are
wrong or because its model is wrong, and this estimator reports only the first.
[Chapter 21]({{< relref "../part5/ch21_model-error.md" >}}) takes up model error, and
[chapter 22]({{< relref "../part5/ch22_verification.md" >}}) the problem of verifying
against observations that have errors of their own.

## Exercises

**Analytic.** Show that if two forecasts valid at time $T$ were started one cycle apart,
their difference at lead $L$ is the growth over $L$ of an initial difference equal to the
one-cycle forecast error. Then explain why the resulting curve is offset from, but
parallel to, the true error curve.

**Computational.** Reproduce the estimator's ratio for all four networks at two fit
windows. Explain why widening the window lowers both rates rather than degrading the
agreement.

**Exploratory.** A model has a bias that grows with lead time. Predict what the
lagged-forecast method reports, and say what additional information would be needed to
detect the bias.

## Further reading

- Lorenz, E. N. (1982). Atmospheric predictability experiments with a large numerical
  model. *Tellus*, **34**, 505–513 — the lagged-forecast method.
- Simmons, A. J. and Hollingsworth, A. (2002). Some aspects of the improvement in skill
  of numerical weather prediction. *QJRMS*, **128**, 647–677
  *[citation needed: confirm pages]*.
- Bengtsson, L. and Hodges, K. I. (2006). A note on atmospheric predictability
  *[citation needed: journal and pages]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 3 *[citation needed: pages]*.
