#!/usr/bin/env python3
"""Inject a chunk-load-error auto-reload handler into marimo WASM export HTML files.

The deploy workflow re-exports every marimo notebook on every push, and each export
gets fresh content-hashed JS chunk filenames (e.g. `run-page-<hash>.js`). Any tab that
has an older export cached (browser HTTP cache, an already-open tab) can end up trying
to dynamically import a chunk hash that the newest deploy has since replaced, producing
a "Failed to fetch dynamically imported module" error. This script patches the exported
HTML to reload once automatically when that happens, so a stale cache self-heals instead
of showing a broken page.

Usage: python3 scripts/patch_chunk_reload.py <file.html> [<file.html> ...]
"""

import sys

MARKER = "marimo-chunk-reload"

SNIPPET = f"""<script>
(function () {{
  var RELOAD_KEY = "{MARKER}";
  function isChunkLoadError(message) {{
    return /fetch dynamically imported module|error loading dynamically imported module/i.test(message || "");
  }}
  function handle(message) {{
    if (!isChunkLoadError(message)) return;
    if (sessionStorage.getItem(RELOAD_KEY)) return;
    sessionStorage.setItem(RELOAD_KEY, "1");
    window.location.reload();
  }}
  window.addEventListener("error", function (event) {{
    handle(event && event.message);
  }});
  window.addEventListener("unhandledrejection", function (event) {{
    var reason = event && event.reason;
    handle(reason && (reason.message || String(reason)));
  }});
  window.addEventListener("load", function () {{
    sessionStorage.removeItem(RELOAD_KEY);
  }});
}})();
</script>
</head>"""


def patch(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    if MARKER in html:
        return False
    if "</head>" not in html:
        print(f"skip {path}: no </head> found", file=sys.stderr)
        return False

    html = html.replace("</head>", SNIPPET, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return True


def main(paths: list[str]) -> None:
    for path in paths:
        if patch(path):
            print(f"patched {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
