# Design_Analysis

## Run
```bash
python cellscientist_phase_1.py design_analysis_config.json
```

## Files
| File | Description |
|------|--------------|
| **cellscientist_phase_1.py** | Main entry. Loads config and runs the full analysis pipeline. |
| **run_llm_nb.py** | Calls the LLM to generate analysis notebooks. |
| **llm_notebook_runner.py** | Handles unified LLM API logic. |
| **hypergraph_orchestrator.py** | Manages multi-run orchestration and workflow control. |
| **config.json** | Configuration file for paths, datasets, and LLM parameters. |

## Paths
- Input data: `../data/${dataset_name}/`  
- Output results: `../results/${dataset_name}/`

## Key Config Parameters (`config.json`)

### Top-level

* **dataset_name** — used in all paths.
  Example: `"BBBC036"` → input `../data/BBBC036/`, output `../results/BBBC036/`.

### llm_notebook

* **paths.data / paper / preprocess** — input files.
* **prompt** — what the LLM should do.
* **llm.provider** — selects provider name from `llm_providers.json`.
* **llm.model** — model name for that provider.

### exec

* **timeout_seconds** — notebook execution timeout.
* **allow_errors** — `true` keeps running even if some cells fail.
* **force_json_mode** — if `true`, LLM must return strict JSON; turn `false` for normal text/code.
* **max_fix_rounds** — number of auto-repair cycles after execution errors.

### multi

* **enabled** — run multiple notebooks in parallel for comparison.
* **num_runs** — number of runs.
* **out_dir** — where results are saved (`../results/${dataset_name}/hypergraph_runs`).
* **prompt_variants** — prompt differences for each run.

### review

* **enabled** — enables notebook evaluation.
* **threshold** — score cutoff for accepting a notebook.
* **reference.export_dir** — folder for accepted notebooks.
* **llm.system_prompt / critique_prompt_template** — define review style and scoring rubric.


## Typical Workflow

1. Edit `dataset_name` and input paths.
2. Adjust `prompt` for your analysis goal.
3. (Optional) change LLM provider or model.
4. Run `cellscientist_phase_1.py`.
5. Check outputs in `../results/<dataset_name>/hypergraph_runs/`.
