---
title: "Chapter 14 · From chaos to turbulence"
weight: 414
part: "Part IV — Many scales, many degrees of freedom"
knob: 'resolution, error-seeding scale'
status: "live"
---

## Overview

[Chapter 12]({{< relref "ch12_scale-dependent-error-growth.md" >}}) established that
whether the predictability horizon is **bounded** depends on one exponent, $\alpha$,
describing how much faster small scales grow than large ones — finite for $\alpha > 0$,
unbounded for $\alpha = 0$. But it *postulated* its octave bands. A real fluid comes with
an energy spectrum instead. So what sets $\alpha$ in an actual flow?

If $E(k) \sim k^{-p}$, an eddy of size $1/k$ has $u_k^2 \sim kE(k)$ and turnover time
$\tau \sim 1/(k u_k)$, so

$$\tau(k) \sim \bigl[k^3 E(k)\bigr]^{-1/2} \sim k^{(p-3)/2}
  \qquad\Longrightarrow\qquad \alpha = \frac{3-p}{2}$$

and the two turbulent cases fall on **opposite sides** of chapter 12's boundary:

| | $p$ | $\alpha$ | chapter 12's verdict |
|---|---|---|---|
| three-dimensional (Kolmogorov) | $5/3$ | $2/3$ | **finite**: horizon 1.4466 |
| two-dimensional (enstrophy range) | $3$ | $0$ | **unbounded**: 0.281 per octave |

## The model

`chaoslib.turbulence` is a two-dimensional pseudospectral solver for Navier–Stokes in
vorticity form, dealiased by the two-thirds rule. Its warrant is that two-dimensional
Euler conserves **two** quantities exactly: after 800 inviscid steps energy has drifted
$1.9\times10^{-7}$ and enstrophy $1.8\times10^{-6}$.

{{< marimo src="/nb/ch14_chaos-to-turbulence.html" >}}

## What the chapter measures

**A fluid differs from a lattice by range, not by having no characteristic scale.** Both
spectra have a peak — a tempting claim to the contrary is simply not what the figure
shows. The fluid's falls **3.55 decades over 3.4 octaves** above its peak; Lorenz 96's
falls **0.71 over 1.2**. And the fluid's peak *moves*: from $k=9$ to $k=4$ over 15 time
units, energy travelling upscale as like-signed vortices merge, with enstrophy falling
from 1.000 to 0.315 while energy falls only 33%. That selective decay — dissipation acts
as $k^2$, energy weights modes by $1/k^2$ — is what permits the upscale transfer.
Lorenz 96's peak cannot move: [chapter 11]({{< relref "ch11_lorenz96.md" >}}) derived it
from a linear instability with $m^*=8$ fixed by the dispersion relation, and a system
whose scale is pinned by its own linear physics cannot have a gradient of growth rates
across scales.

**There is no inertial range at any affordable resolution, and the chapter says so.** The
widest stretch of spectrum whose local slope stays within 0.4 of $-3$ is **0.00, 0.04 and
0.05 octaves** at $64^2$, $128^2$ and $256^2$. Quadrupling the grid does not help. A fit
window chosen after looking at the plot gives $p = 3.011$ over $k\in[6,30]$ — and $1.53$
over $[5,25]$, or $5.52$ over $[10,40]$, which is what makes the 3.011 worthless. This is
the discipline of [chapter 8]({{< relref "../part3/ch08_attractor-dimension.md" >}})
applied to a spectrum: the local slope shows a plateau if there is a power law, and here
there is not. A decade of inertial range needs $N \gtrsim 1024$ with sustained forcing,
which is outside what a browser runs.

**The cascade, though, is real.** Error placed in a single wavenumber shell moves upscale
monotonically: the band $k\in[8,18)$ goes from 0.000 to 0.536 of the total error while
the seeded band falls from 1.000 to 0.394, and the error-weighted mean wavenumber falls
throughout. The total amplitude barely changes, because shell 25 at $96^2$ sits in the
dissipation range — the *redistribution* is what chapter 12's argument needs, and it is
what is shown.

## One input borrowed, and said so

The relation $\alpha = (3-p)/2$ is **algebra**, not a measurement: given this estimate of
$\tau(k)$, its slope against $k$ is fixed by the slope of the spectrum it was computed
from, so agreement between them is a tautology. What this chapter measures is the
cascade; what it takes from theory and observation is the exponent $p$.

## Which is the atmosphere?

Both, at different scales. At synoptic scales the atmosphere is strongly stratified and
rotating, quasi-two-dimensional, with an observed spectrum near $k^{-3}$ — giving
$\alpha \approx 0$ and no intrinsic limit. Below a few hundred kilometres the spectrum
shallows toward $k^{-5/3}$, giving $\alpha \approx 2/3$ and a finite one
*[citation needed: on the observed atmospheric spectral transition]*.

So chapter 12's question has no single answer for the atmosphere: one answer for the
scales carrying most of the energy, another for those carrying most of the enstrophy, and
the intrinsic limit depends on how strongly the second contaminates the first. That is
why the two-week figure has been argued over for fifty years, and why chapter 12's caveat
is the honest position rather than a hedge.

## Exercises

**Analytic.** Derive $\tau(k) \sim [k^3E(k)]^{-1/2}$ from $u_k^2 \sim kE(k)$, obtain
$\alpha = (3-p)/2$, and evaluate it for $p = 1, 5/3, 3, 5$.

**Computational.** Step through the three resolutions watching the *local* slope. At which
wavenumbers is it near $-3$? Then satisfy yourself that a fitted line through a chosen
window would have looked convincing.

**Exploratory.** Section 4's error barely amplifies because shell 25 is dissipative.
Predict what changes at shell 8 and what does not.

## Further reading

- Kraichnan, R. H. (1967). Inertial ranges in two-dimensional turbulence. *Physics of
  Fluids*, **10**, 1417–1423.
- Charney, J. G. (1971). Geostrophic turbulence. *JAS*, **28**, 1087–1095
  *[citation needed: confirm]*.
- Nastrom, G. D. and Gage, K. S. (1985). A climatology of atmospheric wavenumber spectra
  observed by commercial aircraft. *JAS*, **42**, 950–960 *[citation needed: confirm]*.
- Boffetta, G. and Ecke, R. E. (2012). Two-dimensional turbulence. *Annual Review of
  Fluid Mechanics*, **44**, 427–451 *[citation needed: confirm]*.
