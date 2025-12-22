
````markdown
# Design Analysis Module

Phase 1 uses an LLM to **generate, execute, and (optionally) auto-fix** a Cell Painting analysis notebook.  
Multiple runs are supported, and the **best run is automatically selected and exported** as a reference package.

---

## Quick Start

```bash
python cellscientist_phase_1.py design_analysis_config.json
````

---

## What Phase 1 Does

1. Generate + execute an analysis notebook (LLM-written).
2. **Auto-Fix (optional)**: repair failing cells using a bounded LLM loop.
3. Run **multiple variants in parallel** (if enabled).
4. Score each run with lightweight heuristics.
5. Export the **best run** into a reproducible `reference/` package.

---

## Key Files

| File                         | Purpose                                         |
| ---------------------------- | ----------------------------------------------- |
| `cellscientist_phase_1.py`   | Phase 1 entry point                             |
| `hypergraph_orchestrator.py` | Multi-run scheduling, scoring, reference export |
| `run_llm_nb.py`              | Notebook execution + Auto-Fix logic             |
| `config_loader.py`           | Load config + prompts, resolve paths            |

---

## Minimal Config You Care About (`design_analysis_config.json`)

### (1) Dataset

```json
"dataset_name": "BBBC036"
```

Used in paths via `${dataset_name}`.

### (2) Auto-Fix & Execution

```json
"exec": {
  "max_fix_rounds": 3,
  "timeout_seconds": 3600
}
```

* Set `max_fix_rounds = 0` to **disable Auto-Fix** (save cost).

### (3) Multi-run (most important knobs)

```json
"multi": {
  "enabled": true,
  "num_runs": 3,
  "max_parallel_workers": 2,
  "prompt_variants": [...],
  "seeds": [...]
}
```

### (4) LLM

```json
"llm": {
  "model": "gpt-5",
  "max_tokens": 12000,
  "temperature": 0.5
}
```

> API key is read from config **or** `OPENAI_API_KEY`.

---

## Reference Export (Automatic)

The best run is selected and exported to:

```
../results/<dataset_name>/design_analysis/reference/
```

Contents:

* `BEST_*.ipynb` – best executed notebook
* `REFERENCE_DATA.h5` – processed data (if generated)
* `reference.json` – selection metadata
* `idea.json` – LLM-generated follow-up experiments
* `summary_report.md` – LLM-generated analysis report

---

## Prompts (`Design_Analysis/prompts/`)

Only the important ones:

* `notebook_generation.yml` – notebook content
* `autofix.yml` – repair instructions
* `idea.yml` – experiment ideas
* `report.yml` – summary report (language, structure)

---

## Output Layout

```
results/<dataset>/design_analysis/
  design_analysis_..._RunX/
    CP_llm.ipynb
    CP_llm_executed.ipynb
    preprocessed_data.h5
  hypergraph_runs/hypergraph.json
  reference/
    BEST_*.ipynb
    REFERENCE_DATA.h5
    summary_report.md
```