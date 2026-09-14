---
title: "Chapter 30 · Ergodic theory and invariant measures"
weight: 830
part: "Part VIII — Structure"
knob: 'averaging window, ensemble size at fixed budget, noise amplitude'
status: "live"
---

## Overview

A modelling centre has two ways to establish what its model's climate is. Run the model
once for five hundred years and average along the run, or run it a hundred times for
five years from scattered initial conditions and average across the runs. Both are done,
the results are quoted interchangeably, and the second is what every climate ensemble
does.

**When are they estimating the same thing, and how long does each have to be?** Not
"when the model is ergodic" — the model is ergodic in every case examined here, and a
single long run still gets one of these climatologies wrong by fifteen times its own
error bar. The answer turns on which observable is averaged, how long the record is compared with the system's own mixing time, and whether
the forcing is moving.

## The object

Every other chapter of this book has treated "the climate" as something one can average
to. This one names it: the **invariant measure** $\nu$, a probability distribution on
state space left unchanged by the dynamics, against which every climatological statement
is an integral, $\langle A\rangle_\nu = \int A\,d\nu$. Climate is a measure; weather is
a trajectory. The **Birkhoff ergodic theorem** licenses using one to estimate the other,

$$
\bar A_T = \frac{1}{T}\int_0^T A(x(t))\,\mathrm{d}t \;\longrightarrow\;
\langle A \rangle_\nu ,
$$

for $\nu$-almost every starting point, provided no part of the attractor is dynamically
invariant on its own.

That theorem is the entire justification for quoting a climatology from one long run,
and it is weaker than it looks. It is a statement about a limit, so it gives **no rate**.
It holds for almost every initial condition *with respect to $\nu$* — not the one you
chose. And it says nothing about how to tell, from inside a run, whether it has got
there. The chapter is those three gaps, in order, and then what happens when there is no
invariant measure to converge to at all.

Lorenz 63 is the reference system. Three others appear because their invariant measures
are known in **closed form**, which is what lets convergence be measured against truth
rather than against a longer run: the logistic map at $r=4$ (the arcsine law), the circle
rotation (uniform), and the noisy double well of
[chapter 27]({{< relref "../part6/ch27_regimes-tipping.md" >}}) (exactly Boltzmann).

{{< marimo src="/nb/ch30_ergodic-theory.html" >}}

## The measure, and why it has no density

One 60,000 time-unit run sampled every time unit, and 4,000 trajectories from scattered
starts read at a single instant, give the same distribution: $W_1 = 0.20$ in $z$, about
2 % of its spread, against $W_1 = 1.02$ for a Gaussian with *exactly the same mean and
variance*. Two estimators, one measure, and neither is more "the climate" than the other.

That measure is **singular**. Count boxes of side $h$ the trajectory ever visits: for a
measure with a density the count scales as $h^{-3}$, and a uniform control put through
the same estimator returns exactly that. Lorenz 63 returns $h^{-1.94}$ — biased low at
finite sample, and to be read against the $D_{KY} = 2.06$ the Lyapunov spectrum gives
independently ([chapter 8]({{< relref "../part3/ch08_attractor-dimension.md" >}}), with the
spectrum summing to the exact trace to eight parts in a million). The invariant measure
assigns probability one to a set of **zero volume**. Every "PDF of the climate" ever
plotted is a coarse-graining, and the coarse-graining is a choice made by whoever chose
the bins.

## Birkhoff gives no rate, and the rate depends on the observable

Split the run into blocks of length $T$, measure the variance across block means, and
divide out to get the correlation time the measurement itself implies,
$\tau_{\rm eff}(T) = T\operatorname{var}(\bar A_T)/\operatorname{var}(A)$. For three
observables of **the same trajectory**:

| observable | $\tau_{\rm int}$, standard estimate | $\tau_{\rm eff}$, measured | overestimate |
|---|---|---|---|
| $x$ | 0.992 | 1.008 | 1.0× |
| $z$ | 0.233 | 0.0098 | 24× |
| $x^2$ | 0.190 | 0.0011 | 176× |

For $x$ the textbook recipe is exactly right and the variance falls like $T^{-0.98}$, the
Monte Carlo law. For $z$ it is wrong by a factor of twenty-four, because $z$ oscillates:
the initial-positive-sequence rule truncates the autocorrelation sum at its first zero
crossing and so discards the negative lobes that do the cancelling. **The error bar is
five times too wide.**

The cause is exact. For any bounded $B(x)$, $\overline{\dot B}_T = (B_T - B_0)/T$ is a
pure boundary term — no ergodicity, no mixing, no Monte Carlo. Applying that to $x^2/2$,
$z$, $z^2/2$ and $y^2/2$ gives four identities the invariant measure satisfies exactly:

$$
\langle xy\rangle = \langle x^2\rangle, \qquad
\langle xy\rangle = \beta\langle z\rangle, \qquad
\langle xyz\rangle = \beta\langle z^2\rangle, \qquad
\rho\langle xy\rangle = \langle y^2\rangle + \langle xyz\rangle .
$$

**With** their boundary terms they hold at every window, exactly, and test the
integrator. **Without** them they hold only in the limit — and their residual is
therefore a **convergence diagnostic that needs no reference run**. Two quantities that
must be equal, computed from one trajectory, disagree by 0.21 after 10 time units and by
0.0001 after 30,000.

And $\langle x\rangle = 0$ *exactly*, by the symmetry $(x,y,z)\to(-x,-y,z)$ that leaves
the equations unchanged. It is the one number known in advance, and the slowest of all of
them to arrive: pinning it to $\pm0.05$ takes 25,000 time units where $\langle z\rangle$
takes 292 — a factor of **87 between two variables of the same run**. The quantity whose
answer we know is the one the model is worst at telling us, which is precisely why it,
and not the well-behaved $\bar z$, is the honest test of run length.

## Ergodicity is not chaos, and chaos is the worse sampler

The circle rotation $x_{k+1} = x_k + \alpha$ with irrational $\alpha$ has $\lambda_1 = 0$
exactly — two nearby points stay exactly as far apart for ever — and yet *every* orbit
equidistributes. It is uniquely ergodic, a stronger statement than Birkhoff's "almost
every". Make $\alpha$ rational and ergodicity fails outright.

| sampler | rms error at $N = 1000$ | measured scaling |
|---|---|---|
| rotation by $(\sqrt5-1)/2$ | $4.5\times10^{-4}$ | $N^{-0.96}$ |
| logistic map, $r = 4$ | $1.0\times10^{-2}$ | $N^{-0.51}$ |
| i.i.d. random | $9.9\times10^{-3}$ | $N^{-0.50}$ |

**The chaotic sampler is indistinguishable from the random one**, and the zero-exponent
rotation beats both by a factor of twenty-two. Mixing buys exactly one thing — effectively
independent successive states — and independence is the Monte Carlo rate, no better. A
quasi-periodic component of a climate (the seasonal cycle, the diurnal cycle) is cheap to
average over; the chaotic part is not, and no amount of chaos makes it cheaper.

## One long run, or many short ones?

At a fixed integration budget $C$ split into $M$ runs, each discarding a spin-up $T_s$,
the usable record is $C - MT_s$ and

$$
\operatorname{var}(\bar A) = \frac{\tau_{\rm int}\operatorname{var}(A)}{C - M T_s},
$$

which rises monotonically with $M$. Measured over 2,000 repetitions at $C = 2000$,
$T_s = 20$: splitting fifty ways inflates the variance by 1.92 against the 1.98 the
formula predicts with nothing fitted. **The spin-up accounting is the whole of the
effect**, to within 6 % at every member count. For an ergodic system the ensemble buys no
accuracy of its own — a state reached by integrating and a state reached by scattering
are samples of the same measure.

## When the ergodic time exceeds the record

Chapter 27's double well with noise **is** ergodic: noise carries it over the barrier
eventually, so there is no invariant subset and no exceptional starting point. Its
invariant measure is exactly Boltzmann, symmetric, so $\langle x\rangle = 0$. The time to
*find* that measure is Kramers', exponential in $2\Delta V/\sigma^2$.

| $T$ (TU) | $T/\tau_{\rm erg}$ | true rms error | error bar the run reports | overconfidence |
|---|---|---|---|---|
| 30 | 0.11 | 0.91 | 0.062 | 15× |
| 100 | 0.36 | 0.85 | 0.115 | 7× |
| 300 | 1.07 | 0.71 | 0.267 | 2.6× |
| 3000 | 10.8 | 0.29 | 0.245 | 1.2× |

**A run of 30 time units gets the answer wrong by 0.91 and says it is accurate to
0.062**, and nothing inside the run gives that away. It looks converged: the trajectory
rattles around one well, its running mean settles, its autocorrelation decays in 2.1 time
units, and every standard check passes. The check measures the within-well wobble, which
is genuinely fast; the quantity it certifies is the occupancy of two wells, which is
genuinely slow — an ergodic time of 279 time units, against Kramers' 263 from a formula
with no fitted constant.

The pooled ensemble is meanwhile right: $\bar x = +0.005$ against an exact zero, because
its members were *started* in both wells. **The previous section's conclusion inverts
exactly here.** When the ergodic time exceeds the run length the long run is not slow but
biased, the ensemble is the only estimator that works — and the choice between them
cannot be made from the run, because it needs the escape time of a barrier the run may
never have crossed. Halving the noise raises the requirement by a factor of 208,000 — the escape time is
exponential in $\Delta V/\sigma^2$, which makes this the one quantity in this book that a
faster computer does not fix.

## No invariant measure at all

If the forcing depends on time, no distribution is left unchanged by the dynamics and
there is no invariant measure. What replaces it is the time-dependent **snapshot** or
**pullback** attractor: the distribution at time $t$ of an ensemble launched in the
indefinite past. It has one estimator, not two.

Ramping Lorenz 63 from $\rho = 28$ to $40$, a trailing time average over a window $T$ is
biased by exactly $-bT/2$ — it returns the value the mean had at the window's *midpoint*.
Measured against $-bT/2$ with $b$ taken from the stationary attractor and nothing fitted
to the ramped run: 3 % at $T = 400$, 2 % at $T = 800$. The bias is independent of the
variability, does not shrink with a longer record, and **gets worse as the window
lengthens** — at the same time as the sampling noise falls, which is what makes the
failure so hard to see.

**The bias does not trade against the record length; it trades against the noise.** A
longer window cuts sampling error as $T^{-1/2}$ and grows bias as $T$, so there is an
optimum, $T^\star = (2\tau\operatorname{var}(A)/b^2)^{1/3}$, at which the total error
does not vanish but sits at a floor proportional to $b^{2/3}$. For this ramp
$T^\star = 55$ time units, with 0.08 of bias and 0.12 of noise for a total of 0.14 — and
no longer run, no finer output interval and no faster computer lowers it. An ensemble has
no floor: its error is $\sigma/\sqrt M$ with no bias, falling for ever.

Under a *slow* trend the time average is therefore both adequate and cheaper, and the
previous section still applies; matching this floor would take some 5,400 members. What
changes is that its error **stops falling**. **This is the structural argument for
ensembles in climate projection**, and it is not the one usually given — the usual one is
that an ensemble samples internal variability. The stronger one is that under a trend a
time average estimates a quantity that does not exist, and the resulting bias is
invisible from inside the estimate: more data along the trajectory makes the error bar
smaller and leaves the bias exactly where it was.
[Chapter 28]({{< relref "../part7/ch28_nonstationary-predictability.md" >}}) measured the
same failure in the language of forecast skill.

## Exercises

**Analytic.**

1. Show that $\overline{xy}_T - \overline{x^2}_T = (x_T^2 - x_0^2)/2\sigma T$ holds for
   *any* trajectory of Lorenz 63, and hence that it tests the integrator rather than
   ergodicity. Find the analogous identity for Lorenz 96 and say what it would diagnose.
2. A trailing average has bias $-bT/2$ and sampling error
   $\sqrt{\tau\operatorname{var}(A)/T}$. Minimise the total mean square error over $T$,
   show the optimum scales as $b^{-2/3}$, and evaluate it for the chapter's ramp.
3. The rotation converges like $\log N/N$ for badly approximable $\alpha$. Show that for
   $\alpha$ with a good rational approximation $p/q$ the error is $O(1)$ for all $N < q$,
   and say what that implies for averaging over a near-resonant orbital cycle.

**Computational.**

4. Reproduce the $\tau_{\rm eff}(T)$ curves for $\langle xz\rangle$ and
   $\langle y^2\rangle$. Predict from the identities which converges fast, then check.
5. Measure the calibration between the ergodic time and Kramers' formula across the noise
   ladder, and say over what range of $2\Delta V/\sigma^2$ the formula is usable as more
   than a scaling.
6. Cross the budget split against the barrier height, and find the barrier at which the
   optimal $M$ jumps from one to as many as possible. Is the transition sharp?

**Exploratory.**

7. Lorenz 63 at $\rho = 28$ has one attractor; at $\rho = 24.2$ it has three coexisting
   ones. Estimate how long a noiseless run must be to discover that its climatology
   depends on where it started, then argue about whether a coupled model's control run
   could contain the same failure undetected.

## Further reading

- Birkhoff's pointwise ergodic theorem, and the distinction between ergodicity, mixing
  and unique ergodicity *[citation needed: a standard ergodic theory text]*
- SRB measures, and the sense in which a physical measure is selected from the many
  invariant measures of a chaotic attractor *[citation needed]*
- Kramers' escape-rate formula and its accuracy at moderate barriers *[citation needed]*
- Snapshot and pullback attractors for time-dependent forcing, and their use in climate
  projection *[citation needed]*
- Palmer & Hagedorn (2006), on the attractor's statistics and predictability of the
  second kind *[citation needed: chapter]*
