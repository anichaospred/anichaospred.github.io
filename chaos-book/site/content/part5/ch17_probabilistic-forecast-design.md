---
title: "Chapter 17 · Probabilistic forecast design"
weight: 517
part: "Part V — The machinery of prediction"
knob: 'ensemble size, perturbation strategy'
status: "planned"
---
## Overview

An ensemble is a sample of the forecast distribution, and verification is the discipline of checking whether the sample was honest. Two distinctions organise the chapter: **accuracy versus calibration** (the CRPS rewards both, the rank histogram isolates the second), and **spread versus error** (a reliable ensemble has RMS spread equal to the RMS error of its own mean; under-dispersion is the characteristic failure of operational systems).

## The planned notebook

Build an ensemble three ways — isotropic random perturbations, singular vectors, bred vectors — and score all three with CRPS, the Brier score and a rank histogram. `chaoslib.ensemble`, `chaoslib.adjoint`.

## Further reading

- Palmer & Hagedorn (2006), ch. 6–7; Kalnay (2003), §6.5
