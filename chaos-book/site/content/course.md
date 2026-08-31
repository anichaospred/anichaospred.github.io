---
title: "ATOC 4500/5500 — course mapping"
---

This book accompanies **ATOC 4500/5500 *Chaos and Predictability*** at the University of
Colorado Boulder (Tuesday and Thursday, 10:00–11:15, SEEC S126; cross-listed for
undergraduates and graduate students). It is deliberately **broader** than the course: some
chapters develop dynamical-systems material the lectures only touch, and Part VIII goes well
past the syllabus. The table below says which chapters serve which lectures, so chapters can
be assigned directly.

**Canvas remains the authoritative source** for the syllabus, schedule, assignments and
deadlines. Nothing here supersedes it, and the schedule below is the topic sequence only.

## Reference texts

The course is built on:

- **Palmer, T. and Hagedorn, R. (eds), 2006.** *Predictability of Weather and Climate.*
  Cambridge University Press.
- **Kalnay, E., 2003.** *Atmospheric Modeling, Data Assimilation and Predictability.*
  Cambridge University Press.

with these as additional (not required) reading:

- Lorenz, E.N., 1963. Deterministic nonperiodic flow. *J. Atmos. Sci.*, **20**(2), 130–141.
- Lorenz, E.N., 1969. The predictability of a flow which possesses many scales of motion.
  *Tellus*, **21**(3), 289–307.
- Lorenz, E.N., 1982. Atmospheric predictability experiments with a large numerical model.
  *Tellus*, **34**(6), 505–513.
- Smith, L., 2007. *Chaos: A Very Short Introduction.*

Each chapter's "Further reading" points at the corresponding sections. Where a section or
page number has not yet been checked against the text it is marked *[citation needed]*
rather than guessed.

## Lecture → chapter mapping

<div class="wide">

| Lecture | Topic | Chapters |
|---|---|---|
| 1 | Introduction: what is predictability? | [1](/part1/ch01_what-is-predictability/), [3](/part1/ch03_model-hierarchy/), [10](/part3/ch10_information-theory/), [22](/part5/ch22_verification/) |
| 2–3 | History of numerical weather prediction | [2](/part1/ch02_history-of-nwp/) |
| 4–5 | Predictability and error growth | [7](/part3/ch07_lyapunov-exponents/), [9](/part3/ch09_nonlinear-error-growth/), [11](/part4/ch11_lorenz96/), [13](/part4/ch13_operational-error-growth/) |
| 7 | Chaos and predictability limits | **[6](/part2/ch06_lorenz63/)**, [7](/part3/ch07_lyapunov-exponents/), [8](/part3/ch08_attractor-dimension/), [12](/part4/ch12_scale-dependent-error-growth/) |
| 8 | Tangent linear and adjoint models | [15](/part5/ch15_tangent-linear-adjoint/) |
| 9 | Adjoint sensitivity, applications | [16](/part5/ch16_adjoint-sensitivity/) |
| 10 | Probabilistic forecast design | [17](/part5/ch17_probabilistic-forecast-design/) |
| 11 | Variational data assimilation | [18](/part5/ch18_variational-da/) |
| 12 | Ensemble data assimilation | [19](/part5/ch19_ensemble-da/) |
| 14 | Data assimilation applications | [20](/part5/ch20_da-in-practice/), [22](/part5/ch22_verification/) |
| 15 | Frontiers in Earth system predictability | [26](/part6/ch26_earth-system-prediction/) |
| 16 | Subseasonal prediction | [23](/part6/ch23_boundary-forced-s2s/) |
| 17 | Long-term prediction and the role of the oceans | [24](/part6/ch24_decadal-prediction/) |
| 18 | Interannual–decadal prediction | [24](/part6/ch24_decadal-prediction/) |
| 19–20 | Climate prediction | [25](/part6/ch25_climate-prediction/) |
| 21 | Earth system prediction | [26](/part6/ch26_earth-system-prediction/) |
| 22 | Machine learning applications | [29](/part7/ch29_ml-prediction/) |
| 23 | Change in predictability over time | [28](/part7/ch28_nonstationary-predictability/) |
| 26 | Frontiers in weather and climate prediction | [29](/part7/ch29_ml-prediction/) |

</div>

Lectures 6, 13, 24–25 and 27–29 are project work time and presentations.

**Bold** marks a chapter with a live interactive notebook today.

## Chapters beyond the syllabus

These develop material the course does not cover, for readers who want the dynamical-systems
foundations in more depth or the structure behind the results:

[4](/part2/ch04_pendulum-chaos/) (why dimension rather than nonlinearity is the discriminator) ·
[5](/part2/ch05_maps-bifurcations/) (bifurcations and the Feigenbaum constant) ·
[8](/part3/ch08_attractor-dimension/) (fractal dimension and entropy) ·
[14](/part4/ch14_chaos-to-turbulence/) ·
[21](/part5/ch21_model-error/) ·
[27](/part6/ch27_regimes-tipping/) (tipping points and early-warning signals) ·
[30](/part8/ch30_ergodic-theory/) and [31](/part8/ch31_koopman/) (Part VIII).

## How this fits the course's computing

The course's assessed work runs on **CU Research Computing** and **NCAR Derecho**, in
Python, MATLAB or Julia, and culminates in CESM experiments — two homework projects plus a
group project for ATOC 4500 or an individual project for ATOC 5500.

These browser notebooks are not a substitute for any of that, and they deliberately do not
try to be. Their job is the rung of the hierarchy *below* CESM: build the intuition for
error growth, ensemble spread, adjoint sensitivity and assimilation on systems small enough
to explore in a minute, so that when the same diagnostics are computed on a coupled model
you already know what they should look like and what would count as wrong.

See [how these notebooks work]({{< relref "how-to-run.md" >}}) for running them locally, and
the [dependency page]({{< relref "dependencies.md" >}}) for what `chaoslib` provides — much
of it is directly reusable in a CURC or Derecho workflow, since it is plain NumPy and SciPy.
