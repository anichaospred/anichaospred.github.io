#!/usr/bin/env python3
r"""Regenerate `docs/citations.md` -- the inventory of outstanding citations.

The house rule is that a literature value or reference is never invented; where
one is missing the text carries a `*[citation needed]*` marker instead. This
script walks the tree, extracts every marker with the claim it attaches to, and
writes the worklist.

Three things it does that a `grep` does not.

* **It reads the qualifier.** `[citation needed: pages]` is a lookup against a
  book already named; a bare `[citation needed]` on a claim with no author is a
  search. Sorting one from the other is most of the document's value.
* **It groups duplicates.** A chapter's further-reading list appears twice, in
  the notebook and on the chapter page, so most claims carry two markers and
  filling one fills both. The tables are one row per *claim*.
* **It excludes the rules from the count.** `CLAUDE.md`, `PLAN.md`,
  `docs/authoring.md` and the output file all contain the marker as an
  *example*; counting those would inflate the inventory with its own
  conventions.

Run from chaos-book/:
    python3 scripts/citation_inventory.py
"""

from __future__ import annotations

import collections
import pathlib
import re

ROOT = pathlib.Path(".")
OUT = pathlib.Path("docs/citations.md")


MARK = re.compile(r"\*?\[citations? needed(?::\s*([^\]]*))?\]\*?")
BULLET = re.compile(r"^\s*[-*+]\s")
# author forms: "Lorenz (1963)", "Lorenz, E. N. (1963)", "Bauer, Thorpe & Brunet (2015)",
# "Palmer and Hagedorn, eds. (2006)"
AUTHOR = re.compile(
    r"\b([A-Z][A-Za-zÀ-ɏ'`-]+(?:,\s*[A-Z]\.(?:\s*[A-Z]\.)*)?"
    r"(?:(?:,|\s+&|\s+and)\s+[A-Z][A-Za-zÀ-ɏ'`-]+(?:,\s*[A-Z]\.(?:\s*[A-Z]\.)*)?)*"
    r"(?:,?\s*(?:et al\.?|eds?\.))?)\s*\((\d{4}[a-z]?(?:\s*,\s*\d{4}[a-z]?)*)\)")

def claim_for(lines, i):
    """The bullet, sentence or comment block the marker belongs to."""
    # a marker inside a run of comment lines: take the whole run, plus the
    # line it justifies -- these are hard-coded literature constants
    if lines[i].lstrip().startswith("#"):
        a = i
        while a > 0 and lines[a - 1].lstrip().startswith("#"):
            a -= 1
        b = i
        while b + 1 < len(lines) and lines[b + 1].lstrip().startswith("#"):
            b += 1
        value = lines[b + 1].strip() if b + 1 < len(lines) else ""
        text = " ".join(lines[a:b + 1]) + "  >>> " + value
        text = MARK.sub("", text)
        return re.sub(r"\s+", " ", text.replace("#:", " ").replace("#", " ")).strip(" *`")
    start = i
    # walk back to the start of this bullet, or to a blank line / heading
    while start > 0:
        prev = lines[start - 1]
        if BULLET.match(lines[start]):
            break
        if not prev.strip() or prev.lstrip().startswith(("#", "|")) or BULLET.match(prev):
            if BULLET.match(prev):
                start -= 1
            break
        start -= 1
    end = i
    while end + 1 < len(lines) and lines[end].rstrip().endswith(("-", ",")) is False and \
            not MARK.search(lines[end]):
        end += 1
    text = " ".join(lines[start:i + 1])
    text = MARK.sub("", text)
    text = re.sub(r'^\s*(r?""")', "", text)
    text = re.sub(r"\s+", " ", text).strip(" -*+|\t`")
    # For running prose (not a bullet) keep the sentence the marker sits in,
    # rather than the whole paragraph that led up to it.
    if not BULLET.match(lines[start]) and len(text) > 160:
        tail = re.split(r"(?<=[.;])\s+(?=[A-Z$`*])", text)
        if tail and len(tail[-1]) >= 40:
            text = tail[-1]
    return text.strip(" -*+|\t`")

rows = []
for p in sorted(list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.md"))):
    sp = str(p)
    if any(x in sp for x in ("__marimo__", "site/public", "__pycache__", "/.git", "build/")):
        continue
    # These state or illustrate the convention rather than await a citation --
    # this script's own docstring included, which it scanned on its first run.
    # Counting them would inflate the inventory with its own rules.
    if sp in ("CLAUDE.md", "PLAN.md", "docs/authoring.md", "docs/citations.md",
              "scripts/citation_inventory.py"):
        continue
    lines = p.read_text().split("\n")
    for i, line in enumerate(lines):
        for m in MARK.finditer(line):
            claim = claim_for(lines, i)
            rows.append({
                "file": sp, "line": i + 1,
                "qualifier": (m.group(1) or "").strip(),
                "claim": claim,
                "authors": [f"{a.strip()} ({y})" for a, y in AUTHOR.findall(claim)],
            })


# chapter titles from the page front matter
titles = {}
for p in pathlib.Path("site/content").rglob("ch*.md"):
    m = re.search(r'^title:\s*"(.*?)"', p.read_text(), re.M)
    n = re.match(r"ch(\d+)_", p.name)
    if m and n:
        titles[int(n.group(1))] = m.group(1).split("·", 1)[-1].strip()

def chapter_of(f):
    m = re.search(r"ch(\d\d)_", f)
    return int(m.group(1)) if m else None

def kind(r):
    q = r["qualifier"]
    if re.search(r"page|chapter|section|edition|figure|journal|table", q): return "locator"
    if "confirm" in q: return "confirm"
    if q: return "source in marker"
    return "open"

NEED = {
    "locator": "locator",
    "confirm": "verify",
    "source in marker": "full record",
    "open": "full record" ,
}

NOYEAR = re.compile(r"\b[A-Z][a-z]+(?:\s*(?:&|and)\s*[A-Z][a-z]+|,? et al\.?)")

def names_a_work(r):
    """Authors with a year, or authors without one -- either way the work is
    identified and only its record is missing."""
    return bool(r["authors"]) or bool(NOYEAR.search(r["claim"]))

def need_of(r):
    k = kind(r)
    if k == "locator": return f"locator — {r['qualifier']}"
    if k == "confirm": return f"verify — {r['qualifier'] or 'the proposed reference'}"
    if k == "source in marker": return f"full record — {r['qualifier']}"
    if r["authors"]:
        return "full record"
    if NOYEAR.search(r["claim"]):
        return "full record — no year given"
    return "**identify a source**"

def cell(s, n=150):
    s = re.sub(r"^[=\s]+", "", s)
    s = s.replace("|", r"\|").replace("\n", " ")
    return (s[: n - 1] + "…") if len(s) > n else s

for r in rows:
    r["chapter"], r["kind"], r["need"] = chapter_of(r["file"]), kind(r), need_of(r)
    r["named"] = names_a_work(r)

out = []
w = out.append
w("# Citations — what is still needed, and where\n")
w("""Every quantitative claim in this book that rests on the literature carries a
`*[citation needed]*` marker instead of a reference, because the house rule in
[`../CLAUDE.md`](../CLAUDE.md) is absolute: **never invent a citation, a page number or a
literature value.** The markers are therefore deliberate, not oversights — but they are
the last thing standing between the book and a finished text.

This file is the worklist, and it is **generated**: run
`python3 scripts/citation_inventory.py` from `chaos-book/` to rebuild it after markers
have moved or been filled. Do not edit it by hand. `CLAUDE.md`, `PLAN.md`,
`docs/authoring.md`, the generator and this file are excluded from the count: their
markers state or illustrate the convention rather than await a citation.\n""")

by_kind = collections.Counter((r["kind"], r["named"]) for r in rows)
named = sum(v for (k, n), v in by_kind.items() if n)
w("## What is actually outstanding\n")
w(f"**{len(rows)} markers.** They are not equally hard, and the split is the point:\n")
w("| the marker needs | work already named | work not named | total |")
w("|---|---|---|---|")
for k, label in (("locator", "a page, chapter or section number"),
                 ("confirm", "an existing reference verified"),
                 ("source in marker", "a full record for a work named in the marker"),
                 ("open", "a full bibliographic record")):
    a, b = by_kind[(k, True)], by_kind[(k, False)]
    w(f"| {label} | {a} | {b} | {a + b} |")
w(f"| **total** | **{named}** | **{len(rows) - named}** | **{len(rows)}** |\n")
w(f"""**{named} of {len(rows)} markers already name the work.** Those are lookups, not
research: the reference is identified and what is missing is a locator or the rest of the
bibliographic record. The remaining **{len(rows) - named}** need a source to be *found*,
and they are the real task.\n""")

norm = lambda c: re.sub(r"\W+", " ", c.lower()).strip()[:90]
groups = collections.OrderedDict()
for r in rows:
    groups.setdefault(norm(r["claim"]), []).append(r)
dup = sum(1 for g in groups.values() if len(g) > 1)
w("## The list is shorter than it looks\n")
w(f"""The {len(rows)} markers are only **{len(groups)} distinct claims**. A chapter's
further-reading list appears twice — once in the notebook and once on the chapter page —
so {dup} of those claims are carried by two markers or more, and filling one fills its
twin. **The real worklist is {len(groups)} entries**, and the tables below are grouped
that way: one row per claim, with every place it appears.\n""")

# high-yield
cnt = collections.Counter()
for r in rows:
    for a in r["authors"]:
        key = re.sub(r"Palmer,? (T\.|and) ?H?a?g?e?d?o?r?n?.*\(2006\)",
                     "Palmer & Hagedorn (2006)", a)
        key = "Palmer & Hagedorn (2006)" if "Hagedorn" in key else key
        cnt[key] += 1
top = [(k, v) for k, v in cnt.most_common() if v >= 3]
w(f"## Start here: {len(top)} works account for {round(100*sum(v for _, v in top)/len(rows))} % of the list\n")
w("One copy on the desk clears every row that names it.\n")
w("| work | markers | mostly needing |")
w("|---|---|---|")
for k, v in top:
    kinds = collections.Counter(r["kind"] for r in rows
                                if any(k.split(" (")[0].split(" &")[0] in a for a in r["authors"]))
    label = {"locator": "a chapter or page number", "open": "a full record",
             "confirm": "verification", "source in marker": "a full record"}
    w(f"| {k} | {v} | {label.get(kinds.most_common(1)[0][0], '—') if kinds else '—'} |")
w(f"\n**Palmer & Hagedorn (2006) alone is {cnt['Palmer & Hagedorn (2006)']} markers** — "
  "the book is the course text, and most of those rows want a chapter or page number.\n")

# hard-coded values
hard = [r for r in rows if r["file"].startswith(("chaoslib/", "tests/"))]
w("## Highest priority: literature values compiled into the library\n")
w(f"""{len(hard)} markers sit in `chaoslib/` or `tests/` rather than in prose. These are
different in kind from a further-reading entry: they are **numbers the test suite asserts**
or physical values the library returns, and the house rule says such a value must be
traceable to a named source. Until these are cited, the suite is checking a number against
a memory of the literature.\n""")
w("| file | needs | what it justifies |")
w("|---|---|---|")
for r in sorted(hard, key=lambda r: r["file"]):
    w(f"| `{r['file']}:{r['line']}` | {r['need']} | {cell(r['claim'], 110)} |")
w("")

# by chapter
w("## By chapter\n")
w("`N` is the notebook, `P` the chapter page. Line numbers are as of generation.\n")
chapters = sorted({r["chapter"] for r in rows if r["chapter"]})
for ch in chapters:
    mine = [r for r in rows if r["chapter"] == ch]
    _u = len({norm(r["claim"]) for r in mine})
    w(f"### Chapter {ch} — {titles.get(ch, '')}  ({_u} claims"
      + (f", {len(mine)} markers)\n" if len(mine) != _u else ")\n"))
    seen = collections.OrderedDict()
    for r in sorted(mine, key=lambda r: (r["file"], r["line"])):
        seen.setdefault(norm(r["claim"]), []).append(r)
    w("| where | needs | claim |")
    w("|---|---|---|")
    for g in seen.values():
        where = ", ".join(("N" if x["file"].startswith("notebooks/") else "P") + f" {x['line']}"
                          for x in g)
        w(f"| {where} | {g[0]['need']} | {cell(g[0]['claim'])} |")
    w("")

rest = [r for r in rows if not r["chapter"] and not r["file"].startswith(("chaoslib/", "tests/"))]
if rest:
    w(f"### Not chapter-specific  ({len(rest)})\n")
    w("| where | needs | claim |")
    w("|---|---|---|")
    for r in sorted(rest, key=lambda r: (r["file"], r["line"])):
        w(f"| `{r['file']}:{r['line']}` | {r['need']} | {cell(r['claim'])} |")
    w("")

w("""## Conventions

- `*[citation needed]*` — a reference belongs here and none is given.
- `*[citation needed: pages]*` and friends — the work is named in the text; only the
  locator is missing.
- `*[citation needed: confirm]*` — a reference is proposed and has not been checked
  against the source. Do not promote one of these to a citation without opening the paper.
- `*[citation needed: Author (Year)]*` — the work is identified in the marker itself and
  needs its full record.

When filling one, delete the marker and write the reference in the style already used by
the surrounding list. Where a claim turns out to be unsupported by the source, change the
claim — do not keep the sentence and attach the nearest citation to it.""")

OUT.write_text("\n".join(out) + "\n")
print(f"wrote {OUT}: {len(out)} lines, {len(rows)} markers, "
      f"{len(groups)} distinct claims, {len(chapters)} chapters")
