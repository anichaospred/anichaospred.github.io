# Authoring guide — adding a chapter

The checklist for going from "chapter N should exist" to a merged PR. These
conventions are load-bearing: they are what keep 31 notebooks feeling like one book.
See [`architecture.md`](architecture.md) for how the build works and
[`chaoslib.md`](chaoslib.md) for what the library already provides.

## 0. Before writing anything

- Read the chapter's entry in [`PLAN.md`](../PLAN.md). It specifies the **forecasting
  question**, the notebook concept, the **knob** (the 1–3 parameters the reader
  varies).
- Read [`NOTATION.md`](../NOTATION.md). The symbols and sign conventions there are
  mandatory. A new symbol goes into NOTATION.md in the same PR.
- Check [`chaoslib.md`](chaoslib.md) for primitives you can reuse. Chapters share
  machinery heavily — chapters 7, 11, 16 and 19 all lean on the same tangent-linear
  stepper.

## 1. Branch

```bash
git checkout main && git pull
git checkout -b chNN-short-slug
```

One chapter per branch and PR. Never commit to `main` directly.

## 2. New numerics go in `chaoslib` FIRST

If the chapter needs numerics that do not exist yet:

- add them to an existing module or a new `chaoslib/<topic>.py`;
- **NumPy, SciPy and Plotly only** — no other dependency, and nothing that needs
  compiling. Pyodide has to run it;
- docstring states the equation solved and its conventions, in the notation of
  `NOTATION.md`;
- **write tests that check correctness, not execution.** An analytic solution, a
  conservation law held to tolerance, an exact identity, a measured convergence order,
  or a published value with a citation. A test that only asserts "it ran" is worse
  than no test, because it buys false confidence;
- register the module in `chaoslib/__init__.py` (the module list in the docstring, the
  import, and `__all__`).

**This step is not optional and not a formality.** Writing the library first, with
identity-based tests, caught two real bugs during the book's construction that no
amount of looking at plots would have found:

- a tangent linear model that linearised the *continuous* flow instead of the
  *discrete* RK4 map, leaving a 4.6 % error floor that no reduction in perturbation
  amplitude removed. Caught by asserting that the finite-difference discrepancy falls
  *linearly* in the amplitude;
- a correlation-dimension fit window that sat entirely in the saturated regime,
  returning $D_2 = 1.51$ for the Lorenz attractor instead of 2.05. Caught by
  comparing against the Kaplan–Yorke dimension, which is computed from the dynamics
  and is therefore independent.

Prefer identities that hold *exactly*, independent of resolution or integration
length: $\sum_i \lambda_i = \operatorname{tr}\mathbf{J}$,
$\det\mathbf{M} = e^{\tau\operatorname{tr}\mathbf{J}}$,
$\langle \mathbf{M}x,y\rangle = \langle x,\mathbf{M}^{\!\top}y\rangle$, energy
conservation. These test the implementation rather than its convergence, so they fail
loudly and for one reason.

Because every chapter branch appends to the same `__init__.py` import list and the
same test file, two open chapter PRs will conflict there. The conflicts are purely
additive — resolve by keeping both sides.

## 3. Write the notebook

Copy `notebooks/_template.py` to `notebooks/chNN_slug.py`. Standard section order:

1. **Title and the forecasting question.** Open with something a
   forecaster would ask. The dynamical-systems machinery arrives as the answer, never
   as the premise — this book is weather/climate-first throughout.
2. **The equations**, in the notation of `NOTATION.md`. The printed equation must be
   the equation `chaoslib` steps.
3. **The dual-mode import cell**, unchanged from the template.
4. **Interactive controls** — the knob, and nothing more. Three sliders that each
   matter beat eight that do not.
5. **2–4 diagnostics**, using `chaoslib.plotting` for colours and styling so the
   figure reads as part of the book.
6. **"Try this"** — exploratory prompts. Ask the reader to *find a transition*, not to
   admire a picture.
7. **"What you should have seen"** — state the expected result plainly, so a reader
   who saw something else knows to look again.
8. **Further reading** — the matching Palmer & Hagedorn / Kalnay sections. If you do
   not have the section number to hand, write *[citation needed]*. **Never invent a
   citation or a page number.**

### House style

- **Colours are semantic.** Import them from `chaoslib.plotting`: `C_TRUTH` is the
  truth run, `C_PERT` the perturbed forecast, `C_SPREAD` spread and error curves,
  `C_MEAN` the ensemble mean, `C_FIXED` fixed points, `C_SAT` the saturation level,
  `C_START` the start marker. A reader who has learned the convention in chapter 4
  should never have to relearn it.
- Style every figure with `plotting.style2d` / `plotting.style3d`.
- Marimo cell-local names **must** be underscore-prefixed. Only names in a cell's
  `return` cross cells, and a cell's argument list is its dependency declaration —
  if you remove a name from one cell's return, every cell that lists it breaks.
- Respect the performance budget in [`architecture.md`](architecture.md).

## 4. Write the chapter page

Create `site/content/partN/chNN_slug.md`. Front matter matches the neighbours:

```yaml
---
title: "Chapter NN · <Title>"
weight: <partN × 100 + NN>
part: "Part N — <Part title>"
knob: "<the parameters the reader varies>"
status: live        # or: planned
---
```

Standard sections: **Overview** (the forecasting story), **The model** (equations, and
what the `chaoslib` functions do), the embedded notebook via
`{{< marimo src="/nb/chNN_slug.html" >}}`, **Exercises** in three tiers (analytic,
computational, exploratory), and **Further reading**.

Set `status: live` only once the notebook exports cleanly. A `planned` page renders as
a stub with the abstract and the knob, so the table of contents never contains a dead
link.

## 5. Verify

```bash
make test        # chaoslib correctness -- must pass
make nb-one NB=chNN_slug   # export just this chapter while iterating
make notebooks   # full clean re-export; every notebook must succeed
make serve       # eyeball the chapter page and the embedded notebook
```

While iterating on a chapter, the fastest real check is:

```bash
marimo export html --sandbox notebooks/chNN_slug.py -o <scratch>/check.html
echo "exit=$?"                                  # must be 0
grep -c marimo-error <scratch>/check.html       # must be 0
```

because that form *executes every cell*. `html-wasm` only bundles and will happily
produce a page whose cells all raise in the reader's browser.

**Check the exit code, not only the grep.** This matters and cost real time: a
notebook whose cells all raised `MarimoExceptionRaisedError` still produced
`grep -c marimo-error` = **0**, while the command printed
`Error: Export was successful, but some cells failed to execute.` and exited
non-zero. The grep alone will pass a broken chapter. Read stderr, or test the exit
status.

For a chapter with matplotlib figures, also confirm the figures actually exist and
are the size you expect:

```bash
grep -o 'data:image/png;base64' <scratch>/check.html | wc -l   # one per figure
```

and extract one to look at it. "It rendered" is not "it is right": three of this
book's figures rendered perfectly while showing a trajectory that never left a fixed
point.

Export into the repo or a scratch directory — **never bare `/tmp`**, which fails under
the sandbox with `PermissionError`.

### Verify the physics standalone, before the prose

For any non-trivial claim, write a throwaway NumPy/SciPy script and check the number
*before* it goes into a markdown cell. This is how the two bugs in §2 were found, and
how the values quoted in the tests were pinned. A plot that looks plausible is not
evidence.

## 6. PR

Push the branch and open a PR. CI repeats the test suite and exports every notebook.
Merge only with CI green. Expect the additive conflicts from §2 if `main` moved.
