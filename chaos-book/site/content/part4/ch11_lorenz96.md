---
title: "Chapter 11 · Lorenz 96: a many-variable atmosphere analogue"
weight: 411
part: "Part IV — Many scales, many degrees of freedom"
knob: '$F$, $N$, the time window'
status: "live"
---

## Overview

Everything in Part III was measured on a three-variable system. A forecast model has
$10^9$. Which of those conclusions survive the change of scale?

The question is not rhetorical, and [chapter 16]({{< relref "../part5/ch16_adjoint-sensitivity.md" >}})
already found a casualty: in Lorenz 63 the gradient of a forecast metric points almost
exactly along the fastest-growing direction, and in a forty-variable system the two are
nearly orthogonal. Something has to sit between three variables and a general
circulation model.

$$\frac{dx_k}{dt} = \bigl(x_{k+1} - x_{k-2}\bigr)x_{k-1} - x_k + F,
  \qquad k = 1 \ldots N \quad \text{(cyclic)}$$

Read it as a latitude circle: the quadratic terms conserve $\sum_k x_k^2$ and stand in
for advection, $-x_k$ is dissipation, $F$ is the forcing. Nothing about it is derived
from the equations of motion. It earns its place by being the smallest system that
behaves like a **field** — errors have a wavelength, structures propagate, and the
attractor grows with the domain.

## The model

`chaoslib.systems` supplies the right-hand side and its Jacobian; `chaoslib.spatial`
(new in this chapter, and used again in chapter 12) supplies the diagnostics that only
make sense once a system has a space: power spectra, phase speeds, correlation lengths.

Two exact identities anchor the chapter. Every diagonal entry of the Jacobian is $-1$,
so $\sum_i \lambda_i = \operatorname{tr}\mathbf{J} = -N$ for every $F$, every $N$ and
every trajectory. And the Jacobian at the uniform state $x_k = F$ is **circulant**, so
Fourier modes diagonalise it in closed form:

$$\sigma(\theta) = -1 + F\left(e^{i\theta} - e^{-2i\theta}\right), \qquad \theta = 2\pi m/N.$$

That is not an approximation — it reproduces all $N$ eigenvalues to $5\times10^{-14}$.

{{< marimo src="/nb/ch11_lorenz96.html" >}}

## Four results

**The wavelength is set by a linear instability, and comes out in closed form.** Taking
real parts, $\operatorname{Re}\sigma = -1 + F(\cos\theta - \cos2\theta)$; with
$u = \cos\theta$ the bracket is $1 + u - 2u^2$, maximised at $u = 1/4$ with value $9/8$.
So a long chain destabilises at $F = 8/9$ and a finite one at
$F_{\rm crit} = [\max_m(\cos\theta_m - \cos2\theta_m)]^{-1}$ — exactly $2/\sqrt5 =
0.8944$ at $N = 40$, where the best available mode is $m = 8$. The nonlinear flow peaks
one wavenumber away, at $m = 9$.

**Linear theory gets the scale nearly right and the speed badly wrong.** The measured
phase speed of the dominant mode is $-2.1$ sites per time unit against $-7.3$ predicted
— a factor of 3.5. The prediction is made about a state the system is nowhere near, and
finite-amplitude waves propagate on a flow they have themselves modified. The same
lesson as [chapter 15]({{< relref "../part5/ch15_tangent-linear-adjoint.md" >}})'s window
of validity, from a different direction: the linearisation is quantitatively reliable
for the instability that *creates* a structure, not for that structure's later life.

**There are two thresholds, and they are far apart.** The uniform state loses stability
at $F = 0.894$, but $\lambda_1$ stays within $\pm0.025$ of zero until about $F = 4.25$ —
periodic waves with no error growth at all — and only becomes robustly positive near
$F = 4.5$. At $F = 8$ the spectrum has 13 positive exponents,
$h_{KS} = 10.2$ nats per time unit and $D_{KY} = 27.1$, against Lorenz 63's 1, 0.905 and
2.06. The forty exponents sum to $-40$ to within $2.5\times10^{-5}$, a residual that is
RK4 truncation and not non-convergence: it is independent of the averaging time and
falls as $\Delta t^4$, both asserted by tests.

**The model is extensive, and that is the point.** Because the dynamics are local and
the correlation length is a couple of sites, doubling the ring does not create a faster
instability — it creates more independent copies of the same one. Measured over a
6.7-fold range of domain size: $\lambda_1$ is flat above $N \approx 30$ (mean 1.711),
the spectra collapse onto a single curve under $i \to i/N$, and
$D_{KY} = 0.675\,N$ and $h_{KS} = 0.256\,N$ through the origin, with $D_{KY}/N$ varying
by 2.4% across the whole range. Below $N \approx 30$ $\lambda_1$ is suppressed, for a
visible reason: the preferred wavelength is 4.4 sites, so a 12-site ring holds fewer
than three waves and the instability is cramped by its own periodicity.

The forecasting consequence is a ratio. Lead time is set by $\lambda_1$, an intensive
quantity — a property of the dynamics that no amount of computing changes. The number of
directions an ensemble must span is set by $D_{KY} \approx 0.68N$, an extensive one. At
0.34 growing directions per variable, a model with $10^7$ variables has of order
$3\times10^6$ of them, so fifty members is a rounding error against what needs sampling.
Every ensemble method in Part V is a strategy for living with that ratio, and
[chapter 19]({{< relref "../part5/ch19_ensemble-da.md" >}})'s localisation is a direct
exploitation of the locality that makes the system extensive in the first place.

## Exercises

**Analytic.** Derive $\sigma(\theta)$ from the Jacobian at $x_k = F$, and show that the
maximum of $\cos\theta - \cos2\theta$ is $9/8$ at $\cos\theta = 1/4$. Deduce
$F_{\rm crit} \to 8/9$ in the continuum limit, and explain why a finite ring's threshold
is always *above* that value.

**Computational.** Verify the trace identity $\sum_i\lambda_i = -N$ at two values of
$\Delta t$ and confirm that the residual falls by roughly 16 when $\Delta t$ is halved
while being unchanged by doubling the averaging time. Explain what each of those two
facts rules out.

**Exploratory.** Measure the correlation length at $N = 12$ and at $N = 40$, and use it
to account for the suppression of $\lambda_1$ at small $N$. How many correlation lengths
does a ring need before it counts as large?

## Further reading

- Lorenz, E. N. (1996). Predictability: a problem partly solved. *Proceedings of the
  ECMWF Seminar on Predictability*, vol. 1, 1–18.
- Lorenz, E. N. and Emanuel, K. A. (1998). Optimal sites for supplementary weather
  observations. *Journal of the Atmospheric Sciences*, **55**, 399–414.
- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*, §5.5
  *[citation needed: confirm section]*.
- Grassberger, P. (1989). Information content and predictability of lumped and
  distributed dynamical systems *[citation needed]* — extensivity of the Lyapunov
  spectrum in spatially extended systems.
