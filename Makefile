PYTHON ?= python
CONFIG ?= configs/config.yaml

.PHONY: all quickstart reproduce-from-outputs test query cpu-bench analysis plots validity data clean

# Full end-to-end reproduction from scratch (GPU + CPU inference, analysis, figures)
all: test query cpu-bench analysis validity plots

# Fast (<30s) end-to-end smoke test: 1 model (qwen2.5:0.5b) x 5 items x 1 rep
quickstart: test
	$(PYTHON) -m src.llm.query --config $(CONFIG) --quickstart

# Regenerate all paper tables (tab2-tab5) and 300-DPI figures (fig1-fig4)
# directly from pre-computed data/outputs/responses_v2.jsonl & cpu_bench.jsonl
reproduce-from-outputs:
	$(PYTHON) -m src.analysis.accuracy --config $(CONFIG)
	$(PYTHON) -m src.analysis.gtheory --config $(CONFIG)
	$(PYTHON) -m src.analysis.validity --config $(CONFIG)
	$(PYTHON) scripts/06_cpu_bench.py --config $(CONFIG) --summarize-only
	$(PYTHON) -m src.viz.plots --config $(CONFIG)

# Optional: re-download and preprocess raw SINAN microdata via PySUS
data:
	$(PYTHON) -m src.data.download --config $(CONFIG)
	$(PYTHON) -m src.data.preprocess --config $(CONFIG)
	$(PYTHON) -m src.data.aggregate --config $(CONFIG)

test:
	$(PYTHON) -m pytest tests/ -v

query:
	$(PYTHON) -m src.llm.query --config $(CONFIG)

cpu-bench:
	$(PYTHON) scripts/06_cpu_bench.py --config $(CONFIG)

analysis:
	$(PYTHON) -m src.analysis.accuracy --config $(CONFIG)
	$(PYTHON) -m src.analysis.gtheory --config $(CONFIG)

validity:
	$(PYTHON) -m src.analysis.validity --config $(CONFIG)

plots:
	$(PYTHON) -m src.viz.plots --config $(CONFIG)

clean:
	rm -f results/tables/*.csv results/tables/*.tex results/tables/*.md
	rm -f results/figures/*.png results/figures/*.pdf
