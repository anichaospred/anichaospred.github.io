# Citations — what is still needed, and where

Every quantitative claim in this book that rests on the literature carries a
`*[citation needed]*` marker instead of a reference, because the house rule in
[`../CLAUDE.md`](../CLAUDE.md) is absolute: **never invent a citation, a page number or a
literature value.** The markers are therefore deliberate, not oversights — but they are
the last thing standing between the book and a finished text.

This file is the worklist, and it is **generated**: run
`python3 scripts/citation_inventory.py` from `chaos-book/` to rebuild it after markers
have moved or been filled. Do not edit it by hand. `CLAUDE.md`, `PLAN.md`,
`docs/authoring.md`, the generator and this file are excluded from the count: their
markers state or illustrate the convention rather than await a citation.

## What is actually outstanding

**258 markers.** They are not equally hard, and the split is the point:

| the marker needs | work already named | work not named | total |
|---|---|---|---|
| a page, chapter or section number | 81 | 2 | 83 |
| an existing reference verified | 14 | 0 | 14 |
| a full record for a work named in the marker | 0 | 19 | 19 |
| a full bibliographic record | 97 | 45 | 142 |
| **total** | **192** | **66** | **258** |

**192 of 258 markers already name the work.** Those are lookups, not
research: the reference is identified and what is missing is a locator or the rest of the
bibliographic record. The remaining **66** need a source to be *found*,
and they are the real task.

## The list is shorter than it looks

The 258 markers are only **162 distinct claims**. A chapter's
further-reading list appears twice — once in the notebook and once on the chapter page —
so 76 of those claims are carried by two markers or more, and filling one fills its
twin. **The real worklist is 162 entries**, and the tables below are grouped
that way: one row per claim, with every place it appears.

## Start here: 6 works account for 24 % of the list

One copy on the desk clears every row that names it.

| work | markers | mostly needing |
|---|---|---|
| Palmer & Hagedorn (2006) | 40 | a chapter or page number |
| Lorenz (1975) | 6 | a full record |
| Kalnay (2003) | 4 | a full record |
| Hasselmann (1976) | 4 | a full record |
| Hawkins & Sutton (2009) | 4 | a full record |
| Hunt, Kostelich & Szunyogh (2007) | 3 | a chapter or page number |

**Palmer & Hagedorn (2006) alone is 40 markers** — the book is the course text, and most of those rows want a chapter or page number.

## Highest priority: literature values compiled into the library

10 markers sit in `chaoslib/` or `tests/` rather than in prose. These are
different in kind from a further-reading entry: they are **numbers the test suite asserts**
or physical values the library returns, and the house rule says such a value must be
traceable to a named source. Until these are cited, the suite is checking a number against
a memory of the literature.

| file | needs | what it justifies |
|---|---|---|
| `chaoslib/assimilate.py:507` | locator — page | with the analysis ensemble :math:`\bar x^b + \mathbf{X}^b(\bar w^a\mathbf{1}^{\top} + \mathbf{W}^a)`. Followi… |
| `chaoslib/dimension.py:379` | full record — Takens (1981) | Takens' theorem says that for almost any smooth observable, an embedding of dimension :math:`m > 2D` reconstr… |
| `chaoslib/ensemble.py:127` | **identify a source** | Independently bred vectors collapse onto each other.** They are all converging to the *same* leading directio… |
| `chaoslib/information.py:88` | **identify a source** | The default of one half is the Krichevsky-Trofimov estimator ; the value matters little, but omitting it |
| `chaoslib/information.py:228` | full record — DelSole (2004) | The two answer different questions . |
| `chaoslib/maps.py:68` | full record — Feigenbaum (1978); Briggs (1991) for the high-precision value | The Feigenbaum constant, to the precision quoted in the literature. >>> FEIGENBAUM_DELTA = 4.669201609102990 |
| `chaoslib/systems.py:178` | **identify a source** | That form is Hasselmann's picture of the climate system : |
| `chaoslib/systems.py:590` | full record — Lorenz (1996); Wilks (2005) | The conventional values are :math:`h=1, b=c=10` with :math:`F=20` . |
| `tests/test_chaoslib.py:2169` | locator — Feigenbaum (1978), table 1 | maps: the period-doubling cascade and Feigenbaum universality ===============================================… |
| `tests/test_chaoslib.py:4262` | **identify a source** | This is Hasselmann's mechanism -- the memory comes from |

## By chapter

`N` is the notebook, `P` the chapter page. Line numbers are as of generation.

### Chapter 1 — What is predictability?  (4 claims, 7 markers)

| where | needs | claim |
|---|---|---|
| N 360 | **identify a source** | Lorenz drew a distinction that organises this whole book . |
| N 691, P 130 | full record | Lorenz (1975), on predictability of the first and second kind |
| N 693, P 132 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, introduction |
| N 695, P 134 | full record | Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, ch. 1 |

### Chapter 2 — A short history of numerical weather prediction  (10 claims, 13 markers)

| where | needs | claim |
|---|---|---|
| N 89 | **identify a source** | The forecasting question.** In 1922 Lewis Fry Richardson published a six-hour weather forecast he had computed by hand from the governing equations. … |
| N 675 | **identify a source** | The work took him something like six weeks . |
| N 680 | **identify a source** | It is the most famous image in the subject, and it was |
| N 1107 | **identify a source** | It is what Charney, Fjørtoft and von Neumann integrated on ENIAC in 1950 , and it is why that forecast was |
| N 1265, P 155 | locator — pages | Richardson, L. F. (1922). *Weather Prediction by Numerical Process.* Cambridge University Press . |
| N 1268, P 158 | full record | Lynch, P. (2006). *The Emergence of Numerical Weather Prediction: Richardson's Dream.* Cambridge University Press — the modern reconstruction, includ… |
| N 1273, P 162 | full record | Courant, R., Friedrichs, K. and Lewy, H. (1928), on the stability of difference equations . |
| P 13 | **identify a source** | The observed change was essentially nothing, and nothing |
| P 124 | **identify a source** | It is what Charney, Fjørtoft and von Neumann integrated on ENIAC in 1950 , and |
| P 160 | locator — pages | Charney, J. G., Fjørtoft, R. and von Neumann, J. (1950). Numerical integration of the barotropic vorticity equation. *Tellus*, **2**, 237–254 . |

### Chapter 3 — The hierarchy of models  (4 claims, 8 markers)

| where | needs | claim |
|---|---|---|
| N 1029, P 152 | locator — pages | Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of motion. *Tellus*, **21**, 289–307 . |
| N 1031, P 154 | full record | Held, I. M. (2005). The gap between simulation and understanding in climate modeling. *Bulletin of the American Meteorological Society* — |
| N 1034, P 157 | locator — chapter | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate* . |
| N 1036, P 159 | **identify a source** | Schneider, T. et al., on model hierarchies and the limits of small models . |

### Chapter 4 — Regular motion and why it is predictable  (1 claims)

| where | needs | claim |
|---|---|---|
| P 84 | locator — section | Strogatz, *Nonlinear Dynamics and Chaos* — Poincaré–Bendixson, and the phase-plane methods used here . |

### Chapter 5 — Maps, bifurcations, and the routes to chaos  (4 claims)

| where | needs | claim |
|---|---|---|
| N 910 | full record — Libchaber & Maurer (1980); Linsay (1981) | The numbers are portable, and that is the whole design of this book.** $\delta = 4.669$ and the $-1/2$ intermittency exponent belong to a *class* of … |
| N 985 | locator — edition and section numbers | Strogatz, S. H. *Nonlinear Dynamics and Chaos*, ch. 10 — the clearest textbook treatment of this material . |
| N 987 | locator — pages | Smith, L. A. (2007). *Chaos: A Very Short Introduction* — ch. 6 on the logistic map, written for exactly this book's audience . |
| P 113 | locator — pages | Smith, L. A. (2007). *Chaos: A Very Short Introduction* . |

### Chapter 6 — The Lorenz (1963) system: the butterfly  (1 claims)

| where | needs | claim |
|---|---|---|
| P 120 | locator — pages | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, ch. 2 . |

### Chapter 7 — Lyapunov exponents and doubling times  (3 claims, 5 markers)

| where | needs | claim |
|---|---|---|
| N 863 | full record — Sparrow (1982) | The literature places the birth of the strange attractor near ρ ≈ 24.06, with a chaotic saddle existing from ρ ≈ 13.93 .""" |
| N 937, P 96 | locator — pages | Oseledets, V. I. (1968). A multiplicative ergodic theorem. *Trudy Moskov. Mat. Obšč.*, **19**, 179–210 . |
| N 943, P 101 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 2 . |

### Chapter 8 — Attractors, fractal dimension, and entropy  (7 claims, 10 markers)

| where | needs | claim |
|---|---|---|
| N 1040 | **identify a source** | literature \| ≈ 2.05 and ≈ 2.06 |
| N 1062 | full record — Grassberger and Procaccia (1983) | The Hénon map is the independent second case: $D_2$ = {HENON_D2:.3f} from a two-dimensional map, against a literature $D_2$ near 1.22 . Note that the… |
| N 1249 | full record — Takens (1981) | reconstruct a set diffeomorphic to the attractor once $m > 2D$, so its dimension, its Lyapunov exponents and its topology all survive . Below is that… |
| N 1281 | full record — on dimension estimates from climate records | Attractor dimensions have been estimated for real geophysical records on exactly this basis, and the estimates have been argued over for decades, lar… |
| N 1353, P 118 | locator — pages | Kaplan, J. L. and Yorke, J. A. (1979). Chaotic behavior of multidimensional difference equations . |
| N 1355, P 119 | locator — pages | Takens, F. (1981). Detecting strange attractors in turbulence . |
| N 1360, P 121 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 2 . |

### Chapter 9 — Error growth beyond the linear regime  (2 claims, 3 markers)

| where | needs | claim |
|---|---|---|
| N 2299, P 142 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 3 . |
| N 2301 | locator — journal and pages | Savijärvi, H. (1995). Error growth in a large numerical forecast system . |

### Chapter 10 — Information theory and predictability  (7 claims, 9 markers)

| where | needs | claim |
|---|---|---|
| N 1163 | locator — confirm pages | Kleeman, R. (2002). Measuring dynamical prediction utility using relative entropy. *Journal of the Atmospheric Sciences*, **59**, 2057–2072 . |
| N 1165 | verify — confirm | DelSole, T. (2004). Predictability and information theory. *Journal of the Atmospheric Sciences*, **61**, 2425–2440 — |
| N 1169, P 111 | verify — confirm | Schneider, T. and Griffies, S. M. (1999). A conceptual framework for predictability studies. *Journal of Climate*, **12**, 3133–3155 . |
| N 1171 | locator — edition and chapter | Cover, T. M. and Thomas, J. A. *Elements of Information Theory* . |
| N 1173, P 113 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 14 . |
| P 107 | locator — confirm pages | Kleeman, R. (2002). Measuring dynamical prediction utility using relative entropy. *JAS*, **59**, 2057–2072 . |
| P 109 | verify — confirm | DelSole, T. (2004). Predictability and information theory. *JAS*, **61**, 2425–2440 — the signal/dispersion decomposition. |

### Chapter 11 — Lorenz 96: a many-variable atmosphere analogue  (5 claims, 7 markers)

| where | needs | claim |
|---|---|---|
| N 994 | full record — Lorenz (1996); Lorenz and Emanuel (1998) | What is a convention.** The reading of one time unit as five days rests on the dissipation term: $-x_k$ gives an $e$-folding decay time of one time u… |
| N 1003 | full record — Palmer and Hagedorn (2006) | That distinction is what makes the doubling time worth quoting. Section 3 measured $\lambda_1 = 1.668$ per time unit, so errors double in $\ln 2/\lam… |
| N 1089, P 115 | locator — confirm section | Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*, §5.5 . |
| N 1091, P 117 | full record | Grassberger, P. (1989). Information content and predictability of lumped and distributed dynamical systems — on extensivity of |
| N 1094 | locator — chapter and pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate* . |

### Chapter 12 — Scale-dependent error growth and the intrinsic limit  (2 claims, 3 markers)

| where | needs | claim |
|---|---|---|
| N 1725, P 140 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 3 . |
| P 138 | verify — confirm | Wilks, D. S. (2005). Effects of stochastic parametrizations in the Lorenz '96 system. *Quarterly Journal of the Royal Meteorological Society*, **131*… |

### Chapter 13 — Error growth in operational models  (3 claims, 6 markers)

| where | needs | claim |
|---|---|---|
| N 1169, P 111 | locator — confirm pages | Simmons, A. J. and Hollingsworth, A. (2002). Some aspects of the improvement in skill of numerical weather prediction. *Quarterly Journal of the Roya… |
| N 1171, P 113 | locator — journal and pages | Bengtsson, L. and Hodges, K. I. (2006). A note on atmospheric predictability . |
| N 1173, P 115 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 3 . |

### Chapter 14 — From chaos to turbulence  (8 claims, 10 markers)

| where | needs | claim |
|---|---|---|
| N 2479 | full record — on resolution requirements for 2-D turbulence spectra | Both are outside what this book runs in a browser, and both are routine in the literature . |
| N 2614 | full record — on the observed atmospheric spectral transition | At scales below a few hundred kilometres the observed spectrum shallows toward $k^{{-5/3}}$, giving $\\alpha \\approx {KOLMOGOROV:.2f}$ and a **finit… |
| N 2703 | verify — confirm | Charney, J. G. (1971). Geostrophic turbulence. *Journal of the Atmospheric Sciences*, **28**, 1087–1095 . |
| N 2706, P 110 | verify — confirm | Nastrom, G. D. and Gage, K. S. (1985). A climatology of atmospheric wavenumber spectra observed by commercial aircraft. *JAS*, **42**, 950–960 — the … |
| N 2708, P 112 | verify — confirm | Boffetta, G. and Ecke, R. E. (2012). Two-dimensional turbulence. *Annual Review of Fluid Mechanics*, **44**, 427–451 . |
| N 2710 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 4 . |
| P 83 | full record — on the observed atmospheric spectral transition | Below a few hundred kilometres the spectrum shallows toward $k^{-5/3}$, giving $\alpha \approx 2/3$ and a finite one . |
| P 108 | verify — confirm | Charney, J. G. (1971). Geostrophic turbulence. *JAS*, **28**, 1087–1095 . |

### Chapter 15 — Tangent linear and adjoint models  (3 claims, 5 markers)

| where | needs | claim |
|---|---|---|
| N 641 | **identify a source** | Operational campaigns have flown aircraft into regions chosen exactly this way . |
| N 805, P 118 | locator — pages | Giering, R. and Kaminski, T. (1998). Recipes for adjoint code construction. *ACM TOMS*, **24**, 437–474 . |
| N 807, P 120 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 5 . |

### Chapter 16 — Adjoint sensitivity and optimal perturbations  (2 claims, 3 markers)

| where | needs | claim |
|---|---|---|
| N 665 | **identify a source** | Field campaigns have used both to choose flight tracks , and ensemble-based sensitivity has since |
| N 923, P 107 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 5 . |

### Chapter 17 — Probabilistic forecast design  (10 claims, 16 markers)

| where | needs | claim |
|---|---|---|
| N 1065 | full record | Epstein (1969), *Stochastic dynamic prediction*, the first formal proposal of ensemble forecasting |
| N 1066 | full record | Leith (1974), on Monte Carlo forecasting |
| N 1067, P 139 | full record | Toth & Kalnay (1993, 1997), breeding and the NCEP ensemble |
| N 1068, P 140 | full record | Molteni et al. (1996), the ECMWF singular-vector ensemble |
| N 1070, P 141 | full record | Buizza & Palmer (1995), on singular vectors and ensemble design |
| N 1071, P 142 | full record | Murphy (1973), the Brier score decomposition |
| N 1072, P 143 | full record | Hersbach (2000), on CRPS and its decomposition |
| N 1074, P 145 | full record | Richardson (2000), on the relative economic value of ensemble forecasts |
| N 1076 | locator — chapter number | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, ch. 10 |
| P 147 | locator — chapter number | Palmer & Hagedorn (2006), *Predictability of Weather and Climate |

### Chapter 18 — Variational data assimilation  (6 claims, 8 markers)

| where | needs | claim |
|---|---|---|
| N 1563, P 168 | locator — section numbers | Kalnay (2003), *Atmospheric Modeling, Data Assimilation and Predictability*, ch. 5 . |
| N 1565, P 170 | full record | Courtier, Thépaut & Hollingsworth (1994), on the incremental formulation . |
| N 1567 | full record | Talagrand & Courtier (1987), on the adjoint and variational assimilation . |
| N 1568 | full record — no year given | Fisher & Andersson , on the Hessian as analysis-error |
| P 172 | full record | Talagrand & Courtier (1987), on the adjoint in variational assimilation |
| P 174 | full record — no year given | Fisher & Andersson, on the Hessian as analysis-error covariance in operational practice |

### Chapter 19 — Ensemble data assimilation  (9 claims, 15 markers)

| where | needs | claim |
|---|---|---|
| N 983 | **identify a source** | Real systems do better than a single global constant: adaptive schemes estimate inflation from the innovation statistics, and relaxation methods infl… |
| N 1238 | **identify a source** | None of that is present here, so this chapter |
| N 1313, P 160 | full record | Evensen (1994), the original ensemble Kalman filter . |
| N 1315, P 162 | full record | Burgers, van Leeuwen & Evensen (1998), on why observations must be perturbed . |
| N 1317, P 164 | full record | Bishop, Etherton & Majumdar (2001), the ensemble transform Kalman filter . |
| N 1318, P 165 | locator — pages | Hunt, Kostelich & Szunyogh (2007), the LETKF . |
| N 1320, P 167 | full record | Houtekamer & Mitchell (1998, 2001), on localisation and filter divergence . |
| N 1322, P 168 | full record | Hamill & Snyder (2000), on hybrid ensemble–variational covariances . |
| P 132 | **identify a source** | Worth stating plainly, since hybrids demonstrably earn more than 8 % operationally. This experiment is the friendliest possible case for a localised … |

### Chapter 20 — Data assimilation in practice  (2 claims)

| where | needs | claim |
|---|---|---|
| N 1102 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, chapter 8 . |
| P 112 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 8 . |

### Chapter 21 — Model error and the imperfect-model problem  (6 claims, 8 markers)

| where | needs | claim |
|---|---|---|
| N 919 | full record — on model-error estimation in NWP | What is honestly unsolved.** Nothing in this chapter estimates the real atmosphere's model error, because that requires knowing the truth. The publis… |
| N 991 | locator — confirm pages | Palmer, T. N. (2001). A nonlinear dynamical perspective on model error. *Quarterly Journal of the Royal Meteorological Society*, **127**, 279–304 . |
| N 996 | verify — confirm | Berner, J. et al. (2017). Stochastic parameterization: toward a new view of weather and climate models. *BAMS*, **98**, 565–588 . |
| N 999, P 125 | verify — confirm | Orrell, D., Smith, L., Barkmeijer, J. and Palmer, T. N. (2001). Model error in weather forecasting. *Nonlinear Processes in Geophysics*, **8**, 357–3… |
| N 1001, P 127 | locator — pages | Palmer, T. and Hagedorn, R., eds. (2006). *Predictability of Weather and Climate*, ch. 10–11 . |
| P 120 | locator — confirm pages | Palmer, T. N. (2001). A nonlinear dynamical perspective on model error. *QJRMS*, **127**, 279–304 . |

### Chapter 22 — Forecast verification and the practical horizon  (5 claims, 10 markers)

| where | needs | claim |
|---|---|---|
| N 1178, P 141 | locator — edition | Jolliffe & Stephenson, *Forecast Verification |
| N 1179, P 142 | full record | Murphy (1988), on the decomposition of the mean-square error |
| N 1181, P 144 | full record | Murphy & Epstein (1989), on skill scores and their reference forecasts |
| N 1183, P 146 | locator — figure | Bauer, Thorpe & Brunet (2015), "The quiet revolution of numerical weather prediction", for the historical skill record |
| N 1185, P 148 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate |

### Chapter 23 — Boundary-forced predictability and the S2S window  (5 claims, 8 markers)

| where | needs | claim |
|---|---|---|
| N 685, P 113 | full record | Lorenz (1975), on predictability of the second kind |
| N 687, P 115 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on seasonal prediction |
| N 689 | full record — no year given | Mariotti et al., on the subseasonal predictability gap and windows of opportunity |
| N 690, P 117 | full record | Shukla (1998), on predictability in the midst of chaos |
| P 116 | full record — no year given | Mariotti et al., on the subseasonal gap and windows of opportunity |

### Chapter 24 — The ocean's role: interannual-to-decadal prediction  (7 claims, 11 markers)

| where | needs | claim |
|---|---|---|
| N 357 | **identify a source** | This is Hasselmann's picture : **the ocean's long memory comes |
| N 809, P 123 | full record | Hasselmann (1976), stochastic climate models |
| N 811, P 125 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on decadal prediction |
| N 812, P 126 | full record | Meehl et al. (2021), on decadal prediction systems and drift |
| N 813, P 127 | full record — no year given | Boer et al., on the Decadal Climate Prediction Project protocol |
| P 7 | **identify a source** | The memory comes from integrating fast weather rather than from slow internal dynamics, which is Hasselmann's mechanism . |
| P 21 | **identify a source** | A fast Lorenz 63 "atmosphere" coupled to a single slow "ocean" variable that integrates it, $\dot S = [-S + \lambda(z - z_{\mathrm{ref}})]/T$, with t… |

### Chapter 25 — Climate prediction and projection  (4 claims, 8 markers)

| where | needs | claim |
|---|---|---|
| N 725, P 124 | full record | Hawkins & Sutton (2009), on the sources of uncertainty in climate projections |
| N 727, P 128 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate |
| N 728, P 126 | full record — no year given | Deser et al., on internal variability and large ensembles |
| N 729, P 125 | full record | Lorenz (1975), on predictability of the second kind |

### Chapter 26 — Earth system prediction  (8 claims, 14 markers)

| where | needs | claim |
|---|---|---|
| N 2687 | full record — Cox et al. (2013) | Differentiating the carbon equation gives its |
| N 2891, P 216 | full record | Hasselmann (1976), stochastic climate models |
| N 2893, P 218 | full record | Cox et al. (2013), on constraining the carbon-climate feedback from interannual variability |
| N 2895, P 220 | full record | Hawkins & Sutton (2009), on the sources of uncertainty in climate projections |
| N 2896, P 221 | full record — no year given | Friedlingstein et al., on carbon-cycle feedback intercomparison |
| N 2898, P 223 | full record | Hall et al. (2019), on the promise and pitfalls of emergent constraints |
| N 2899, P 224 | full record — no year given | Boer et al., on the Decadal Climate Prediction Project protocol |
| P 153 | full record — Cox et al. (2013) | If the feedback strength contributes a quarter of the long-lead spread and is not observable, can it be inferred from something that is? That is the … |

### Chapter 27 — Regimes, bistability, and tipping points  (6 claims, 12 markers)

| where | needs | claim |
|---|---|---|
| N 1351, P 199 | full record | Scheffer et al. (2009), on early-warning signals for critical transitions |
| N 1352, P 200 | full record | Lenton et al. (2008), on tipping elements in the Earth system |
| N 1354, P 202 | full record — no year given | Ditlevsen & Johnsen, on noise-induced versus bifurcation-induced tipping |
| N 1355, P 203 | full record — no year given | Ashwin et al., on rate-induced tipping |
| N 1357, P 205 | full record | Boers (2021), on early-warning signals in the observational record and their statistical pitfalls |
| N 1358, P 206 | full record | Kramers (1940), on escape over a potential barrier |

### Chapter 28 — Has predictability changed over time?  (6 claims)

| where | needs | claim |
|---|---|---|
| N 91 | **identify a source** | The forecasting question.** The useful range of a global forecast has advanced by roughly a day per decade for fifty years . Two entirely different |
| P 12 | **identify a source** | The useful range of a global forecast has advanced by roughly a day per decade for fifty years . Two entirely different things could have produced th… |
| P 204 | locator — chapter | Palmer & Hagedorn (2006), *Predictability of Weather and Climate*, on the growth of forecast skill |
| P 206 | full record | Lorenz (1982), on error growth estimated from an operational forecast archive |
| P 208 | **identify a source** | On the ECMWF forecast-skill record and the role of reforecasts in interpreting it |
| P 210 | **identify a source** | On observed and projected changes in mid-latitude circulation variability and their implications for predictability |

### Chapter 29 — Machine learning and data-driven prediction  (7 claims, 12 markers)

| where | needs | claim |
|---|---|---|
| N 88 | **identify a source** | The forecasting question.** Learned weather models now match or beat physics-based forecasts on the scores operational centres publish . The obvious |
| N 1121, P 157 | locator — pages | Pathak, J. et al. (2018). Model-free prediction of large spatiotemporally chaotic systems from data: a reservoir computing approach. *Physical Review… |
| N 1124, P 159 | full record | Jaeger, H. and Haas, H. (2004), on echo state networks and the echo state property . |
| N 1126, P 160 | full record | Lam, R. et al. (2023), on learned medium-range weather forecasting . |
| N 1128, P 162 | full record | Bonavita, M. (2024), on what learned weather models do and do not inherit . |
| N 1130, P 164 | locator — chapter | Brunton, S. L. and Kutz, J. N. (2019). *Data-Driven Science and Engineering.* Cambridge University Press . |
| P 12 | **identify a source** | Learned weather models now match or beat physics-based forecasts on the scores operational centres publish . The standard worry is that they have |

### Chapter 30 — Ergodic theory and invariant measures  (7 claims, 10 markers)

| where | needs | claim |
|---|---|---|
| N 3789, P 248 | full record — a standard ergodic theory text | Birkhoff's pointwise ergodic theorem and the distinction between ergodicity, mixing and unique ergodicity |
| N 3791, P 250 | **identify a source** | SRB measures and the sense in which a physical measure is selected from the many invariant measures of a chaotic attractor |
| N 3793, P 251 | **identify a source** | Kramers' escape-rate formula and its accuracy at moderate barriers |
| N 3795 | **identify a source** | Snapshot and pullback attractors for systems with time-dependent forcing, and their use in climate projection |
| N 3797 | locator — chapter | Palmer & Hagedorn (2006), on the relation between the attractor's statistics and predictability of the second kind |
| P 253 | **identify a source** | Snapshot and pullback attractors for time-dependent forcing, and their use in climate projection |
| P 255 | locator — chapter | Palmer & Hagedorn (2006), on the attractor's statistics and predictability of the second kind |

### Chapter 31 — The Koopman operator  (8 claims, 12 markers)

| where | needs | claim |
|---|---|---|
| N 1618 | **identify a source** | Koopman's original operator-theoretic formulation of Hamiltonian dynamics |
| N 1620 | **identify a source** | Mezić on spectral properties of the Koopman operator and applications to dimension reduction |
| N 1621, P 222 | **identify a source** | Dynamic mode decomposition and its exact formulation |
| N 1623, P 224 | full record — no year given | Williams, Kevrekidis & Rowley on extended DMD and its convergence to the Koopman operator as dictionary and data grow |
| N 1625, P 226 | **identify a source** | Ruelle–Pollicott resonances and the decay of correlations in mixing systems |
| N 1627, P 228 | **identify a source** | Koopman and transfer-operator methods in data assimilation, including the `quantum-koopman-da` line of work |
| P 219 | **identify a source** | Koopman's operator-theoretic formulation of Hamiltonian dynamics |
| P 221 | **identify a source** | Mezić on spectral properties of the Koopman operator and dimension reduction |

### Not chapter-specific  (1)

| where | needs | claim |
|---|---|---|
| `docs/chaoslib.md:418` | **identify a source** | The default half-count is the Krichevsky–Trofimov estimator ; the value matters little, omitting it turns a finite diagnostic into |

## Conventions

- `*[citation needed]*` — a reference belongs here and none is given.
- `*[citation needed: pages]*` and friends — the work is named in the text; only the
  locator is missing.
- `*[citation needed: confirm]*` — a reference is proposed and has not been checked
  against the source. Do not promote one of these to a citation without opening the paper.
- `*[citation needed: Author (Year)]*` — the work is identified in the marker itself and
  needs its full record.

When filling one, delete the marker and write the reference in the style already used by
the surrounding list. Where a claim turns out to be unsupported by the source, change the
claim — do not keep the sentence and attach the nearest citation to it.
