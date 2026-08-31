# Build pipeline: marimo notebooks -> ONE shared WASM bundle in the Hugo static tree.
NB_DIR    := notebooks
OUT_DIR   := site/static/nb
SITE_DIR  := site
NOTEBOOKS := $(wildcard $(NB_DIR)/ch*.py)
# chaoslib wheel, shipped in notebooks/public/ so the WASM export carries it
# (notebooks micropip-install it when running under Pyodide).
WHEEL     := $(NB_DIR)/public/chaoslib-0.1.0-py3-none-any.whl

.PHONY: notebooks nb-one ipynb wheel serve test clean

wheel: $(WHEEL)
$(WHEEL): chaoslib/*.py pyproject.toml
	pip wheel --no-deps -q -w $(NB_DIR)/public .

# Every chapter exports into the SAME directory, so marimo emits one shared
# assets/ and one shared public/ (holding the wheel) rather than a ~27 MB
# self-contained copy per chapter. Verified: 1 notebook and 2 notebooks both
# produce 28 MB / 728 asset files.
#
# `clean` runs first because marimo merges into an existing assets/ rather than
# replacing it (shutil.copytree(..., dirs_exist_ok=True)), so chunks from an
# older marimo version would otherwise accumulate forever.
notebooks: clean $(WHEEL)
	@for nb in $(NOTEBOOKS); do \
		name=$$(basename $$nb .py); \
		echo "exporting $$name"; \
		marimo export html-wasm --sandbox --mode run --no-show-code \
			$$nb -o $(OUT_DIR)/$$name.html || exit 1; \
	done
	@rm -f $(OUT_DIR)/CLAUDE.md
	@python3 scripts/patch_chunk_reload.py $(OUT_DIR)/*.html
	@$(MAKE) --no-print-directory ipynb
	@echo "done -> $(OUT_DIR)"

# Downloadable Jupyter versions, linked from each chapter page. These are marimo's
# own `export ipynb` plus a dependency cell and a header -- see scripts/export_ipynb.py.
# Cheap (no execution), so it runs as part of `notebooks`.
IPYNB_DIR := $(SITE_DIR)/static/ipynb

ipynb:
	@mkdir -p $(IPYNB_DIR)
	@python3 scripts/export_ipynb.py $(NOTEBOOKS) \
		--outdir $(IPYNB_DIR) --site-root $(SITE_DIR) \
		--base-url https://anichaospred.github.io

# Iterate on one chapter without a full re-export:  make nb-one NB=ch06_lorenz63
nb-one: $(WHEEL)
	@test -n "$(NB)" || { echo "usage: make nb-one NB=ch06_lorenz63"; exit 1; }
	marimo export html-wasm --sandbox --mode run --no-show-code \
		$(NB_DIR)/$(NB).py -o $(OUT_DIR)/$(NB).html
	@rm -f $(OUT_DIR)/CLAUDE.md
	@python3 scripts/patch_chunk_reload.py $(OUT_DIR)/$(NB).html

serve:
	cd $(SITE_DIR) && hugo server -D

test:
	pytest -q tests/

clean:
	rm -rf $(OUT_DIR) $(IPYNB_DIR)
