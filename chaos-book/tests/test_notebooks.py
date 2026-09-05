"""Static checks on the chapter notebooks themselves.

The export pipeline is unusually good at hiding a broken notebook. A cell that
raises `NameError` in the reader's browser leaves the exporter's **exit code at
zero**, leaves `grep -c marimo-error` at **zero**, and renders every other
figure perfectly; the only sign is a line on stderr. That has caught this book
out three times:

* a bare `nan` in a generated data cell (chapters 12 and 22) -- valid Python
  only where numpy is in scope under that name, which in a data cell it is not;
* a case mismatch between the name a generator emitted and the name a figure
  cell asked for, `L63_FULL_1eM12` against `L63_FULL_1EM12` (chapter 10);
* a typo inside an f-string, which no amount of reading the figure code reveals.

All three are the same underlying fault: **a cell uses a name that nothing
provides.** This module checks for exactly that, statically, in a second --
rather than by exporting nineteen notebooks and reading stderr carefully.
"""

from __future__ import annotations

import ast
import builtins
import pathlib

import pytest

NOTEBOOKS = sorted(
    (pathlib.Path(__file__).resolve().parent.parent / "notebooks").glob("ch*.py")
)
BUILTIN_NAMES = frozenset(dir(builtins))


def _bound_names(node: ast.AST) -> set[str]:
    """Every name a cell binds locally: parameters, assignments, imports,
    loop and comprehension targets, `with ... as`, `except ... as`, walrus,
    nested definitions and lambda parameters."""
    bound: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        bound |= {a.arg for a in args.args}
        bound |= {a.arg for a in getattr(args, "posonlyargs", [])}
        bound |= {a.arg for a in args.kwonlyargs}
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                bound.add(extra.arg)

    for sub in ast.walk(node):
        if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
            for target in targets:
                bound |= {
                    n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                }
        elif isinstance(sub, ast.NamedExpr):
            bound.add(sub.target.id)
        elif isinstance(sub, (ast.For, ast.AsyncFor, ast.comprehension)):
            bound |= {
                n.id for n in ast.walk(sub.target) if isinstance(n, ast.Name)
            }
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if sub is not node:
                bound.add(sub.name)
                bound |= {a.arg for a in sub.args.args}
                bound |= {a.arg for a in sub.args.kwonlyargs}
                for extra in (sub.args.vararg, sub.args.kwarg):
                    if extra is not None:
                        bound.add(extra.arg)
        elif isinstance(sub, ast.ClassDef):
            bound.add(sub.name)
        elif isinstance(sub, ast.Lambda):
            bound |= {a.arg for a in sub.args.args}
            bound |= {a.arg for a in sub.args.kwonlyargs}
        elif isinstance(sub, ast.Import):
            for alias in sub.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, ast.ImportFrom):
            for alias in sub.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(sub, ast.withitem):
            if sub.optional_vars is not None:
                bound |= {
                    n.id
                    for n in ast.walk(sub.optional_vars)
                    if isinstance(n, ast.Name)
                }
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, (ast.Global, ast.Nonlocal)):
            bound |= set(sub.names)
    return bound


def _provided_names(tree: ast.Module) -> set[str]:
    """Names available to a cell: module-level bindings (`app`, `marimo`) plus
    everything any cell returns.

    Marimo is reactive, so a cell may legitimately consume a name returned by a
    cell defined later in the file; the union over all cells is therefore the
    right set, not a running prefix.
    """
    provided: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                provided |= {
                    n.id for n in ast.walk(target) if isinstance(n, ast.Name)
                }
        elif isinstance(node, ast.Import):
            for alias in node.names:
                provided.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                provided.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in node.body:
                if isinstance(statement, ast.Return) and statement.value:
                    provided |= {
                        n.id
                        for n in ast.walk(statement.value)
                        if isinstance(n, ast.Name)
                    }
    return provided


def unresolved_names(source: str) -> dict[str, list[str]]:
    """Cells that use a name nothing provides, as ``{cell: [names]}``."""
    tree = ast.parse(source)
    provided = _provided_names(tree)
    problems: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        used = {
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        free = used - _bound_names(node) - BUILTIN_NAMES - provided
        if free:
            problems[node.name] = sorted(free)
    return problems


def test_notebooks_were_found():
    """A glob that matches nothing makes every check below vacuously pass.

    This is not hypothetical: the first run of this check used a probe file
    whose name did not match the glob, reported a clean sweep, and would have
    been believed.
    """
    assert len(NOTEBOOKS) >= 19


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_cell_name_resolves(path):
    problems = unresolved_names(path.read_text())
    assert not problems, (
        f"{path.name} uses names nothing provides: {problems}. "
        "A bare `nan`, a case mismatch or a typo in an f-string all look like "
        "this, and all of them export with exit code 0."
    )


@pytest.mark.parametrize(
    "assignment,expected",
    [
        ("SATURATION = nan", ["nan"]),
        ("SATURATION = float(\"nan\")", []),
    ],
)
def test_the_check_itself_detects_a_bare_nan(assignment, expected):
    """The check has to be able to fail, or it is decoration.

    Both spellings are exercised: the guarded one must pass and the bare one
    must be caught.
    """
    source = (
        "import marimo\n"
        "app = marimo.App()\n"
        "@app.cell\n"
        "def data():\n"
        "    " + assignment + "\n"
        "    return (SATURATION,)\n"
    )
    problems = unresolved_names(source)
    assert problems == ({"data": expected} if expected else {})


def test_the_check_detects_a_misspelled_cross_cell_name():
    """The chapter-10 failure: a generator emitted `L63_FULL_1eM12` and a
    figure cell asked for `L63_FULL_1EM12`. Four figures rendered; the fifth
    silently did not exist."""
    source = (
        "import marimo\n"
        "app = marimo.App()\n"
        "@app.cell\n"
        "def data():\n"
        "    L63_FULL_1eM12 = 1.0\n"
        "    return (L63_FULL_1eM12,)\n"
        "@app.cell\n"
        "def figure(L63_FULL_1eM12):\n"
        "    print(L63_FULL_1EM12)\n"
        "    return\n"
    )
    assert unresolved_names(source) == {"figure": ["L63_FULL_1EM12"]}
