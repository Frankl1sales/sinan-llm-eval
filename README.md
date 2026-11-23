# SINAN LLM Measurement Study

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11.16](https://img.shields.io/badge/Python-3.11.16-3776AB.svg)](https://www.python.org/)
[![Tests: 59 passing](https://img.shields.io/badge/pytest-59%20passed-brightgreen.svg)](tests/)
[![Ollama 0.30.10](https://img.shields.io/badge/Ollama-0.30.10-black.svg)](https://ollama.com/)

A reproducible psychometric and computational measurement study evaluating local Large Language Models (LLMs) on **Epidemiological Tabular Numeracy** using real Brazilian public-health surveillance tables from **SINAN** (Chikungunya 2024, Minas Gerais — UF `31`).

---

## 1. Project Overview

### Research Question & Construct
Can consumer-grade local LLMs reliably answer quantitative epidemiological queries over real-world public-health surveillance tables, and how do **tool execution**, **self-repair**, and **table serialization** affect both accuracy and psychometric reliability?

We operationalize **Epidemiological Tabular Numeracy** across **36 deterministic Pandas-grounded items** spanning four task types (`lookup`, `aggregation`, `comparison`, `abstention`), three SINAN summary tables (`table_monthly.csv`, `table_weekly.csv`, `table_by_sex.csv`), and five typed answer evaluators (`numeric`, `boolean`, `month`, `category`, `abstention`). Real-world data traps present in official SINAN extracts—a late-2023 notification row (`2023,12,21`) preceding 2024 rows and blank/ignored sex codes (`2024,,1` and `I`)—are preserved verbatim.

### Evaluation Regimes
- **N1 (`zero_shot`)**: Direct table QA in both `csv` and `markdown` formats (`temperature=0.3`, 3 repetitions with seeds `42, 43, 44`).
- **N2 (`tool_1shot`)**: Schema-only prompt (`csv` schema + 5-row preview); the LLM generates Pandas code executed on a pre-loaded `df` inside an isolated subprocess sandbox (5 s timeout, no network, no file writes).
- **N3 (`self_repair`)**: If N2 execution fails (`exec_status != "ok"`), the traceback is sent back to the LLM once for repair and re-execution.

### Key Verified Findings (3,024 GPU Trials + 120 CPU-Only Trials)

| Model | N1 (CSV) | N1 (Markdown) | N2 (Tool 1-shot) | N3 (Self-Repair) |
|---|---:|---:|---:|---:|
| `qwen2.5:0.5b` | 23.1% [15.7, 31.5] | 25.0% [17.6, 34.3] | 10.2% [4.6, 16.7] | 11.1% [5.6, 17.6] |
| `qwen2.5:1.5b` | 22.2% [14.8, 30.6] | 31.5% [23.1, 39.8] | 49.1% [38.9, 58.3] | 50.0% [40.7, 59.3] |
| `qwen2.5:3b` | 36.1% [26.9, 45.4] | 27.8% [19.4, 36.1] | 46.3% [37.0, 55.6] | 71.3% [63.0, 79.6] |
| `qwen2.5:7b` | 47.2% [38.0, 56.5] | 50.0% [40.7, 59.3] | 74.1% [65.7, 82.4] | **84.3% [77.8, 90.7]** |
| `qwen2.5-coder:7b-instruct-q4_K_M` | 52.8% [43.5, 62.1] | 55.6% [46.3, 64.8] | **82.4% [75.0, 89.8]** | 82.4% [75.0, 89.8] |
| `llama3.2:3b` | 50.0% [39.8, 60.2] | 50.9% [40.7, 61.1] | 61.1% [51.9, 69.4] | 66.7% [57.4, 75.0] |
| `mistral:7b` | 49.1% [38.9, 58.3] | 44.4% [35.2, 53.7] | 65.7% [56.5, 75.0] | 73.1% [64.8, 81.5] |
| **Overall** | **40.1% [36.5, 44.0]** | **40.7% [37.6, 44.3]** | **55.6% [52.0, 58.9]** | **62.7% [59.1, 66.1]** |

- **Tool execution + self-repair**: Moves overall accuracy from **40.4%** (N1) → **55.6%** (N2) → **62.7%** (N3), cutting noisy execution errors from **24.3%** to **15.1%** while exposing a persistent **16.9%** silent-error rate (code runs cleanly but computes wrong logic).
- **Psychometric reliability (G-theory)**: Table format accounts for only **0.88%** of total variance (`Model×Item` = 14.96%, `Item` = 10.40%, `Model` = 9.98%, `Regime` = 4.98%). Generalizability $E\rho^2$ increases from **0.8255** (N1) → **0.9352** (N2) → **0.9578** (N3) (overall **0.8484**, 95% CI [0.7100, 0.9072]). Cross-condition model rank correlations are $\rho_s = 0.857$ ($p = 0.014$) and $\tau = 0.714$ ($p = 0.030$).
- **CPU-only viability**: On CPU alone (`num_gpu=0`), `qwen2.5:1.5b` (N2) reaches **70%** accuracy in **2.95 s/query** (4.21 s/correct), and `qwen2.5:7b` (N2/N3) reaches **90%** accuracy in **7.59 s/query** (8.43 s/correct).

---

## 2. Repository Structure & File Index

| Path | Role in Pipeline | Description |
|---|---|---|
| [configs/config.yaml](configs/config.yaml) | Configuration | Central YAML config: models, seeds (`42, 43, 44`), temperature (`0.3`), regimes, CPU benchmark subset, and output paths. |
| [Makefile](Makefile) | Orchestration | Reproducible targets: `quickstart`, `reproduce-from-outputs`, `test`, `data`, `query`, `cpu-bench`, `analysis`, `validity`, `plots`, `all`. |
| [src/data/download.py](src/data/download.py) | Data Acquisition | Downloads raw SINAN Chikungunya 2024 microdata via PySUS 2.x to `data/raw/`. |
| [src/data/preprocess.py](src/data/preprocess.py) | Data Cleaning | Filters microdata to Minas Gerais (`SG_UF=31`) confirmed cases (`CLASSI_FIN=13`) and derives notification year, month, and epidemiological week. |
| [src/data/aggregate.py](src/data/aggregate.py) | Table Generation | Aggregates cleaned microdata into `table_monthly.csv`, `table_weekly.csv`, and `table_by_sex.csv`. |
| [src/prompts/formats.py](src/prompts/formats.py) | Serialization | Formats DataFrames into `csv`, `markdown`, or `json` strings for N1 prompts. |
| [src/prompts/questions.py](src/prompts/questions.py) | Benchmark Bank | Defines all 36 deterministic Pandas-grounded `Question` items across monthly, weekly, by-sex, and abstention tasks. |
| [src/llm/client.py](src/llm/client.py) | Inference Client | Wrapper around `ollama.chat` supporting fixed seeds, multi-turn chat, `num_gpu=0` CPU mode, token counting, and `psutil` RAM tracking. |
| [src/llm/sandbox.py](src/llm/sandbox.py) | Code Sandbox | Isolated subprocess executor for N2/N3 Pandas code (5 s timeout, `RLIMIT_FSIZE=0`, blocked file I/O, network, and subprocess imports). |
| [src/llm/parse.py](src/llm/parse.py) | Typed Scoring | Typed answer extractors and exact/tolerance evaluators for `numeric`, `boolean`, `month`, `category`, and `abstention`. |
| [src/llm/query.py](src/llm/query.py) | Experiment Runner | Runs all 7 models × 3 regimes × 36 items × 3 reps with JSONL checkpoint/resume, error taxonomy classification, and `--quickstart` mode. |
| [scripts/06_cpu_bench.py](scripts/06_cpu_bench.py) | CPU Benchmark | Runs CPU-only (`num_gpu=0`) benchmark across 4 Qwen2.5 scales × 10 monthly items × N1/N2/N3 and exports `tab5_cpu_efficiency`. |
| [src/analysis/accuracy.py](src/analysis/accuracy.py) | Accuracy & Errors | Computes 95% percentile bootstrap CIs (1,000 resamples) and exports `tab2_accuracy_regimes` and `tab3_error_taxonomy` (`.tex`/`.md`). |
| [src/analysis/gtheory.py](src/analysis/gtheory.py) | Psychometrics | Random-effects Method-of-Moments (Henderson's Method 1) G-study, item-bootstrap $E\rho^2$ CIs, D-study curves, rank correlations, and `tab4_gtheory_variance`. |
| [src/analysis/validity.py](src/analysis/validity.py) | Validity Checks | Computes empty/unparsed prediction rate (with `NaN` fix) and per-item/per-task diagnostic summaries. |
| [src/viz/plots.py](src/viz/plots.py) | Visualization | Generates all four 300-DPI publication figures (`fig1`–`fig4`) in both `.png` and `.pdf` formats. |
| [tests/test_parse.py](tests/test_parse.py) | Unit Tests | 56 unit tests covering all 5 typed parsers, tolerance checks, edge cases, and `validity.compute_empty_rate`. |
| [tests/test_pipeline.py](tests/test_pipeline.py) | Integration Tests | 3 tests verifying the 36-item bank structure, sandbox security/isolation, and G-theory MoM estimator. |
| [environment.yml](environment.yml) / [requirements.txt](requirements.txt) | Environment | Exact pinned package versions exported from the `sinan-llm` Conda environment. |

---

## 3. Hardware & Computational Requirements

All reported experiments were executed locally on the following verified workstation:

| Component | Verified Specification |
|---|---|
| **CPU** | AMD Ryzen 7 5700X 8-Core Processor (8 physical cores / 16 logical threads, `x86_64`) |
| **System RAM** | 31 GiB (32 GB installed; 8.0 GiB swap) |
| **GPU** | NVIDIA GeForce RTX 4060 (8,188 MiB / 8 GB VRAM), Driver `595.99.02`, CUDA `13.2` |
| **Operating System** | Ubuntu 24.04.2 LTS (Linux `7.0.0-38-generic x86_64`) |
| **Python / Runtime** | Python `3.11.16` (Conda env `sinan-llm`), Ollama `0.30.10` |

### Measured Runtimes

| Workflow | Command | Measured Wall-Clock Time | Hardware Required |
|---|---|---:|---|
| **(a) Quickstart smoke test** (59 tests + 15 trials) | `make quickstart` | **~17 seconds** | Any CPU/GPU + Ollama (`qwen2.5:0.5b`) |
| **(b) Full GPU experiment** (3,024 trials) | `make query` | **~31 min 21 s** | 8 GB VRAM GPU + Ollama (all 7 models) |
| **(c) CPU-only benchmark** (120 trials, `num_gpu=0`) | `make cpu-bench` | **~4 min 45 s** | Multi-core CPU + ≥8 GB RAM |
| **(d) Analysis & figures from pre-computed JSONL** | `make reproduce-from-outputs` | **~33 seconds** | Any CPU (no Ollama needed) |

---

## 4. Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Frankl1sales/sinan-llm-eval.git
cd sinan-llm-eval
```

### 2. Create the Python Environment
**Option A — Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate sinan-llm
```

**Option B — pip (Python 3.11):**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Pull Local Ollama Models (Required Only for Inference)
Install [Ollama](https://ollama.com/) (`v0.30.10` used in this study), ensure `ollama serve` is running, and pull the 7 evaluated models:

```bash
ollama pull qwen2.5:0.5b                     # 397 MB
ollama pull qwen2.5:1.5b                     # 986 MB
ollama pull qwen2.5:3b                       # 1.9 GB
ollama pull qwen2.5:7b                       # 4.7 GB
ollama pull qwen2.5-coder:7b-instruct-q4_K_M # 4.7 GB
ollama pull llama3.2:3b                      # 2.0 GB
ollama pull mistral:7b                       # 4.4 GB
```

---

## 5. Dataset & Data Provenance

- **Primary Data Source**: Brazilian Ministry of Health — **SINAN** (*Sistema de Informação de Agravos de Notificação*), distributed publicly via **DATASUS** ([datasus.saude.gov.br](https://datasus.saude.gov.br/) / [FTP DATASUS](ftp://ftp.datasus.gov.br/dissemin/publicos/SINAN/)).
- **Acquisition Library**: [PySUS](https://github.com/AlertaDengue/PySUS) (`pysus==2.11.6`), developed by the AlertaDengue / Fiocruz team.
- **Filtering Criteria** ([src/data/preprocess.py](src/data/preprocess.py)):
  - Disease: Chikungunya (`CHIK`), notification year `2024`
  - State: Minas Gerais (`SG_UF == "31"`)
  - Final classification: Confirmed cases (`CLASSI_FIN == "13"`)
- **Processed Tables Included**: The three compact, anonymized aggregate CSVs ([table_monthly.csv](data/processed/table_monthly.csv), [table_weekly.csv](data/processed/table_weekly.csv), [table_by_sex.csv](data/processed/table_by_sex.csv)) are committed in `data/processed/` so experiments and analyses work immediately without re-downloading microdata.
- **Regenerating from Raw DATASUS Microdata**:
  ```bash
  make data
  # Equivalent to:
  # python -m src.data.download --config configs/config.yaml
  # python -m src.data.preprocess --config configs/config.yaml
  # python -m src.data.aggregate --config configs/config.yaml
  ```

---

## 6. Step-by-Step Reproduction Guide

### Option A — Fast Quickstart Smoke Test (~17 s)
Runs the full 59-test unit suite followed by a 15-trial N1/N2/N3 smoke test (`qwen2.5:0.5b` × 5 items × 1 repetition) without overwriting pre-computed results:
```bash
make quickstart
```

### Option B — Regenerate All Paper Tables & Figures from Pre-Computed Outputs (~33 s, No GPU/Ollama Needed)
Uses the committed [data/outputs/responses_v2.jsonl](data/outputs/responses_v2.jsonl) (3,024 trials) and [data/outputs/cpu_bench.jsonl](data/outputs/cpu_bench.jsonl) (120 trials) to recompute all bootstrap CIs, G-theory/D-study estimates, validity checks, LaTeX/Markdown tables, and 300-DPI figures:
```bash
make reproduce-from-outputs
```

### Option C — Full End-to-End Reproduction from Scratch (~37 min)
Re-runs tests, all 3,024 GPU inference trials, all 120 CPU-only trials, statistical analyses, and figure generation (remove `data/outputs/*.jsonl` first if you want a clean rerun instead of checkpoint resume):
```bash
make all
```

---

## 7. Paper Artifacts Mapping

| Paper Artifact | Generating Script / Command | Output Files |
|---|---|---|
| **Table 1** (Benchmark & Item Bank Summary) | [src/prompts/questions.py](src/prompts/questions.py) | 36-item specification in code (`monthly` 15, `weekly` 12, `by_sex` 6, `abstention` 3) |
| **Table 2** (Accuracy + 95% CI by Model & Regime) | `python -m src.analysis.accuracy` | [tab2_accuracy_regimes.tex](results/tables/tab2_accuracy_regimes.tex), [tab2_accuracy_regimes.md](results/tables/tab2_accuracy_regimes.md) |
| **Table 3** (Error Taxonomy: N1/N2/N3) | `python -m src.analysis.accuracy` | [tab3_error_taxonomy.tex](results/tables/tab3_error_taxonomy.tex), [tab3_error_taxonomy.md](results/tables/tab3_error_taxonomy.md) |
| **Table 4** (G-Study Variance, $E\rho^2$, Rank Corr.) | `python -m src.analysis.gtheory` | [tab4_gtheory_variance.tex](results/tables/tab4_gtheory_variance.tex), [tab4_gtheory_variance.md](results/tables/tab4_gtheory_variance.md) |
| **Table 5** (CPU-Only Efficiency Benchmark) | `python scripts/06_cpu_bench.py --summarize-only` | [tab5_cpu_efficiency.tex](results/tables/tab5_cpu_efficiency.tex), [tab5_cpu_efficiency.md](results/tables/tab5_cpu_efficiency.md) |
| **Figure 1** (Accuracy by Model, Regime & Task) | `python -m src.viz.plots` | [fig1_accuracy_by_regime_task.png](results/figures/fig1_accuracy_by_regime_task.png), [.pdf](results/figures/fig1_accuracy_by_regime_task.pdf) |
| **Figure 2** (N2 vs N3 Error Taxonomy) | `python -m src.viz.plots` | [fig2_error_taxonomy.png](results/figures/fig2_error_taxonomy.png), [.pdf](results/figures/fig2_error_taxonomy.pdf) |
| **Figure 3** (G-Study Variance & D-Study Curve) | `python -m src.viz.plots` | [fig3_gtheory_dstudy.png](results/figures/fig3_gtheory_dstudy.png), [.pdf](results/figures/fig3_gtheory_dstudy.pdf) |
| **Figure 4** (CPU-Only Pareto Frontier) | `python -m src.viz.plots` | [fig4_cpu_pareto_frontier.png](results/figures/fig4_cpu_pareto_frontier.png), [.pdf](results/figures/fig4_cpu_pareto_frontier.pdf) |

---

## 8. Testing & Validation

Run the automated test suite at any time:
```bash
pytest tests/ -v
```
All **59 unit and integration tests** verify:
- Typed response extraction and scoring across all 5 `answer_type`s (`numeric` 1% tolerance, `boolean` EN/PT normalization, `month` EN/PT names/abbreviations/ints, `category` normalization, `abstention` refusal detection vs. hallucination rejection).
- Empty/NaN prediction rate calculation (`src/analysis/validity.py`).
- Exact 36-item question bank counts and deterministic ground-truth evaluation (`src/prompts/questions.py`).
- Subprocess sandbox execution, timeout enforcement, and blocking of file writes and network/subprocess imports (`src/llm/sandbox.py`).
- Random-effects Method-of-Moments variance component estimation and non-negative clamping (`src/analysis/gtheory.py`).

---

## 9. Code Reuse, Attribution & License

- **License**: Released under the **MIT License** — see [LICENSE](LICENSE).
- **Third-Party Libraries & Tools**: Built with [PySUS](https://github.com/AlertaDengue/PySUS) (DATASUS/SINAN access), [Ollama](https://ollama.com/) (local LLM serving), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [psutil](https://github.com/giampaolo/psutil), and [pytest](https://docs.pytest.org/).
- **Theoretical Reference**: Psychometric measurement design and Generalizability Theory methodology follow Truong & Koyejo, *AI Measurement Science*, and Brennan (2001), *Generalizability Theory*.
