---
title: "Chapter 8 · Attractors, fractal dimension, and entropy"
weight: 308
part: "Part III — Quantifying chaos and predictability"
knob: 'the scaling window, the Theiler window, the embedding dimension'
status: "live"
---

## Overview

A forecast model has $10^7$ variables. Does an analysis have to constrain $10^7$
numbers?

No — because the trajectory does not visit most of its state space. It settles onto an
attractor whose dimension is generally far smaller than the state dimension and
generally not an integer, and that number is what an observing system actually has to
pin down.

Two ways to measure it, and they share no intermediate quantity:

$$D_{KY} = j + \frac{\sum_{i=1}^{j}\lambda_i}{|\lambda_{j+1}|},
  \qquad C(r) \sim r^{D_2} .$$

The **Kaplan–Yorke dimension** comes from the Lyapunov spectrum — from the dynamics,
via the tangent equations. The **correlation dimension** counts pairs of points closer
than $r$ on a sampled trajectory — from the geometry, using nothing but a cloud of
points. Agreement between them is therefore evidence rather than arithmetic.

Most of the chapter is about how easily both are got wrong. Not as a closing caveat: a
dimension estimate is a slope fitted over a range you chose, and choosing badly returns
a clean fit with a small residual and the wrong answer.

## The model

`chaoslib.dimension` gains box counting and the Rényi dimensions $D_q$, a delay-embedding
helper, and three reference sets built by exact self-similar recursion — Cantor
($\ln2/\ln3$), Koch ($\ln4/\ln3$) and Sierpiński ($\ln3/\ln2$) — whose dimensions are
known in closed form. Those are the calibration: an estimator that cannot return 1.5850
where the answer is known has no business being pointed at an attractor.

Every expensive computation in this chapter is knob-free, and the one thing the reader
chooses — the scaling window — costs a `polyfit` over a stored curve. So the curves are
precomputed and every slider re-fits them, which is both instant and the pedagogical
point: one fixed curve, with the window moving across it.

{{< marimo src="/nb/ch08_attractor-dimension.html" >}}

## Four results

**Two independent routes agree to 0.2%.** For Lorenz 63, $D_2 = 2.0579$ from counting
pairs and $D_{KY} = 2.0618$ from the Lyapunov spectrum. Both estimators work — which
matters given how easily either can be made to return nonsense — and the
**Kaplan–Yorke conjecture** holds here: the dimension implied by the growth rates really
is the dimension of the set. It is a conjecture, not a theorem, and this is what testing
it looks like. The Hénon map is the second case, $D_2 = 1.192$ against a literature 1.22.

**The same curve returns 0.19, 1.92 or 2.51 depending on the window.** Fitted above 30%
of the attractor diameter, where $C \to 1$, the slope collapses to **0.19**. Fitted below
0.2%, where pair counts are quantised, it steepens to **2.51**. Fitted over the whole
range it gives a plausible and wrong **1.92** — the most dangerous of the three, because
nothing in the output distinguishes it from the correct 2.057. So too coarse biases low
and too fine biases high; neither is correctable after the fact.

**The Theiler bias runs opposite to the usual warning.** Temporal correlation puts a bump
in $C(r)$ at the distance the trajectory covers between samples. For Lorenz 63 at
$\Delta t = 0.01$ that step is 1.3% of the diameter — *inside* the usual fit window — so
the bump's rising flank steepens the local slope from 1.92 to 2.22 and biases $D_2$
**high**: 2.139, 2.118, 2.084, 2.039 for Theiler windows of 0, 10, 50, 200 samples. The
standard warning is that temporal correlation makes the estimator report the trajectory's
smooth one-dimensional curve, biasing *low*, which happens when the sampling puts the
step scale *below* the window. Which bias you get depends on where your sampling interval
falls relative to your fit window.

**Box counting works where you can afford it and starves in three dimensions.** It
recovers all three reference dimensions to better than 0.01, but only while there are
more than about ten points per occupied box: on Sierpiński the local slope reads 1.582,
1.568, 1.549, 1.451, 1.127 at occupancies of 186, 63, 21, 7.3, 2.8 against an exact
1.585. On the Lorenz attractor, 19,000 samples do not supply a single clean decade,
because box counting spreads $N$ points over $\varepsilon^{-d}$ boxes while pair counting
uses all $N(N-1)/2$ pairs. That is why every dimension quoted for a real attractor is a
correlation dimension.

## Entropy, and the return on observations

Pesin's identity makes $h_{KS} = \sum_{\lambda_i>0}\lambda_i = 0.901$ nats per MTU
= 1.30 bits per MTU: the rate at which the system destroys information about its own
initial state, and so the rate at which observations must arrive to hold an analysis
steady. Each decimal digit of extra initial precision buys $\ln 10/\lambda_1 = 3.8$ days,
and the next digit buys 3.8 days again — the logarithmic return that
[chapter 20]({{< relref "../part5/ch20_da-in-practice.md" >}}) measures from the
operational end as 6.5 days per decade of analysis-error reduction.

And all of it works from a single observed variable. Takens' theorem says a delay
embedding of one scalar series reconstructs the attractor once $m > 2D$; measured on the
$x$ component of Lorenz 63 alone, $D_2$ reads 1.72, 1.95, 1.99, 2.00, 2.00 for
$m = 2\ldots6$, with increments falling 0.235, 0.045, 0.012, 0.002. Since the criterion
needs the $D$ you are measuring, that **saturation** is the only available diagnostic.

## Exercises

**Analytic.** Show that a set built from $N$ copies of itself at scale $r$ has
$D_0 = \ln N/\ln(1/r)$, and evaluate it for the three reference sets. Then show that
$D_0 \ge D_1 \ge D_2$ for any measure, with equality only when it is uniform.

**Computational.** Reproduce the three wrong answers (0.19, 2.51, 1.92) from the same
$C(r)$ curve, and for each, name the panel that would have warned you.

**Exploratory.** Predict from the sampling interval whether the Theiler bias on the
chapter's curve is high or low *before* changing the Theiler window, then check.

## Further reading

- Grassberger, P. and Procaccia, I. (1983). Characterization of strange attractors.
  *Physical Review Letters*, **50**, 346–349.
- Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited
  time-series data. *Physical Review A*, **34**, 2427–2432.
- Kaplan, J. L. and Yorke, J. A. (1979). Chaotic behavior of multidimensional difference
  equations *[citation needed: pages]*.
- Takens, F. (1981). Detecting strange attractors in turbulence *[citation needed: pages]*.
- Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*,
  ch. 2 *[citation needed: pages]*.
