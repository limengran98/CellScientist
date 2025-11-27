# Design Analysis Module

## Quick Start

```bash
python cellscientist_phase_1.py design_analysis_config.json
```

## Core Files Overview

| File | Function |
| :--- | :--- |
| **cellscientist\_phase\_1.py** | **Entry Point**. Loads configuration, injects API keys into the environment, and enables unbuffered real-time logging. |
| **hypergraph\_orchestrator.py** | **Scheduler**. Manages parallel execution (ThreadPool), timestamped directory creation, heuristic scoring, and mandatory export of the best result. |
| **run\_llm\_nb.py** | **Execution Engine**. Handles robust LLM communication, notebook execution, and the **Auto-Fix Loop** (error recovery). |
| **config\_loader.py** | **Utility**. Safely loads JSON configs and YAML prompts. |

## Configuration Guide (`design_analysis_config.json`)

### 1\. Basic Settings

  * **`dataset_name`**: Dynamic variable used in paths.
      * *Example*: If set to `"BBBC036"`, `${dataset_name}` in paths resolves to `BBBC036`.

### 2\. LLM Settings (`llm`)

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `model` | `gpt-5` / `gemini-2.5-pro` | Strong reasoning models are required for code generation. |
| `max_tokens` | `10000` - `20000` | Context window size. **Do not set too low**, as notebooks generate long outputs. |
| `temperature` | `0.5` | Controls creativity for generation. Note: **Auto-Fix** forces `0.0` internally for precision. |
| `api_key` | `sk-...` | If empty, the system defaults to the `OPENAI_API_KEY` environment variable. |

### 3\. Execution & Auto-Fix (`exec`)

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| **`max_fix_rounds`** | **3 - 5** | **Critical**. The number of times the LLM is allowed to self-correct code errors. Set to `0` to disable repair. |
| `timeout_seconds` | `1800` - `3600` | Max runtime per notebook. Increase for large datasets. |
| `allow_errors` | `false` | If `true`, the notebook continues execution even after a cell fails (not recommended). |

### 4\. Parallel Orchestration (`multi`)

| Parameter | Recommended | Description |
| :--- | :--- | :--- |
| `enabled` | `true` | Enables multi-run mode. |
| **`max_parallel_workers`** | **2 - 4** | **Critical**. Number of concurrent threads. Depends on GPU VRAM and API rate limits. (e.g., 4 for A100, 2 for standard GPUs). |
| `num_runs` | `1` - `5` | Planned runs. Actual runs = `min(num_runs, len(seeds), len(variants))`. |
| `prompt_variants` | List[Str] | Different analysis instructions (e.g., "Focus on Dose-Response"). |
| `seeds` | List[Int] | Fixed random seeds for reproducibility. |

### 5\. Review & Export (`review.reference`)

Uses **Heuristic Scoring** (Rule-based detection of stats, plots, biology terms).

  * `weights`: Adjust importance of Scientific, Novelty, Reproducibility, and Interpretability scores.
  * **Note**: The system **automatically** calculates scores and exports the best performing notebook to the `reference/` directory.

## Output Structure

Results are saved to `../results/${dataset_name}/design_analysis/`:

1.  **Run Directories**: Named `design_analysis_YYYYMMDD_HHMMSS_RunX`.
      * Contains: `CP_llm.ipynb` (Source), `CP_llm_executed.ipynb` (Executed), `preprocessed_data.h5` (Data artifact).
2.  **`hypergraph_runs/hypergraph.json`**: Metadata and scores for all runs.
3.  **`reference/`**: Automatically contains the **Best Notebook** (`BEST_design_analysis_...ipynb`) and its corresponding H5 data file.