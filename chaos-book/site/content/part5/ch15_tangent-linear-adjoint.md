---
title: "Chapter 15 · Tangent linear and adjoint models"
weight: 515
part: "Part V — The machinery of prediction"
knob: '$\tau$, perturbation amplitude'
status: "planned"
---
## Overview

The single most leveraged idea in operational predictability. Linearise the model about a trajectory and you can propagate a small perturbation forward with the tangent linear model $\mathbf{M}$; transpose it and you can propagate a *sensitivity* backwards. The consequence is that the gradient of any scalar forecast metric with respect to the entire initial state costs **one** adjoint integration rather than one model run per degree of freedom. Without that, neither 4D-Var nor singular vectors would be computable at operational size.

## The planned notebook

The finite-difference validation curve: the reader watches the discrepancy between the tangent linear prediction and the true nonlinear difference fall *linearly* with the perturbation amplitude — the defining signature of a correct tangent linear model, and a test that catches errors an eyeball check of the Jacobian will not. `chaoslib.adjoint`.

## Further reading

- Kalnay (2003), §6.2–6.3; Errico (1997), *What is an adjoint model?* *[citation needed]*
