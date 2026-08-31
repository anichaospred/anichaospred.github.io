---
title: "Chapter 16 · Adjoint sensitivity and optimal perturbations"
weight: 516
part: "Part V — The machinery of prediction"
lectures: "L9"
knob: 'optimisation time $\tau$'
status: "planned"
---
## Overview

If a forecast went wrong, where in the initial state did it go wrong? The adjoint answers that directly, and the same machinery identifies the perturbations that grow fastest over a chosen finite window — the **singular vectors**. These are generally *not* the leading Lyapunov vectors, and over short windows they amplify far more than the leading Lyapunov exponent suggests. That gap is the operational reason forecast centres compute them, and it motivates targeted observing.

## The planned notebook

The leading singular vector of Lorenz 63 or 96 over a window the reader chooses, with its achieved amplification compared against $e^{\lambda_1 \tau}$. `chaoslib.adjoint.singular_vectors`.

## Further reading

- Kalnay (2003), §6.4; Palmer & Hagedorn (2006), ch. 5 *[citation needed]*
