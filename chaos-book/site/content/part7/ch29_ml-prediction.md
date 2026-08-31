---
title: "Chapter 29 · Machine learning and data-driven prediction"
weight: 729
part: "Part VII — Frontiers"
lectures: "L22, L26"
knob: 'rollout length, training-set size'
status: "planned"
---
## Overview

Learned emulators now match or beat physics-based models on standard forecast scores at several-day leads. The predictability question is sharper than the skill question: does a model trained on trajectories inherit the *dynamics* — the Lyapunov spectrum, the error-growth rate, the attractor's statistics — or only the trajectories? A model that scores well at day five and drifts off the attractor at day sixty has learned something, but not the thing that matters for predictability.

## The planned notebook

Train a small surrogate on Lorenz 96 and compare its Lyapunov spectrum, error-doubling time and long-rollout statistics against the truth. Connects to the existing `ai-models-sensitivity` work.

## Further reading

- *[citations needed]* — Lam et al. (GraphCast), Bi et al. (Pangu), and the AI-model predictability literature
