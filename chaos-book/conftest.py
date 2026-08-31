"""Put the repo root on sys.path so `import chaoslib` works from a bare checkout.

Mirrors what the notebooks do in their non-Pyodide branch, so tests and notebooks
import the same code the same way without an editable install.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
