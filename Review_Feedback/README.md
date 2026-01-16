# GenerateExecution Module

## Quick Start

### 1\. Standard Optimization Run

```bash
python cellscientist_phase_3.py --config review_feedback_config.json
```

**Key Behaviors**:

  * **Auto-Discovery**: Automatically finds the best `prompt_run` from the Phase 2 output directory.
  * **Baseline Locking**: Establishes the Phase 2 best score as the static baseline for delta comparison.
  * **Atomic Workspaces**: Creates a unique `review_run_{TIMESTAMP}` directory for isolation.

-----

## Core Files Overview

| File | Function |
| :--- | :--- |
| **cellscientist\_phase\_3.py** | **Entry Point**. Orchestrates the optimization loop: loads notebook $\to$ identifies mutable cells $\to$ queries LLM for edits $\to$ applies changes $\to$ executes. |
| **design\_execution/llm\_utils.py** | **LLM Engine**. Centralized utility for robust JSON extraction (supports "Thinking" models), API interaction, and automatic retries. |
| **design\_execution/nb\_autofix.py** | **Execution & Repair**. Runs generated notebooks. Features **Tiered Repair** (Heuristics for NaNs/Types $\to$ LLM for logic) and **Hash-based Verification**. |
| **design\_execution/config\_loader.py** | **Config Manager**. Handles JSON loading, YAML prompt resolution, and `${VAR}` variable expansion. |
| **design\_execution/experiment\_report.py** | **Analyst**. Generates markdown reports with statistical significance if a new best model is found. |

-----

## Configuration Guide (`review_feedback_config.json`)

### 1\. LLM Settings (`llm`)

  * **`provider`**: Selects the active provider configuration.
  * **`timeout`**: Recommended **300s+** to accommodate "Thinking" process models (e.g., Claude-3.5, Gemini-Pro).
  * **`model`**: Use high-reasoning capability models for code critique.

### 2\. Optimization Control (`review`)

  * **`target_metric`**: The metric to maximize (e.g., `PCC`).
  * **`pass_threshold`**: Early stopping criteria (e.g., `0.85`).
  * **`max_iterations`**: Maximum optimization rounds (Recommended: 5-10).
  * **`protected_sections`**: List of keywords/sections the LLM is **forbidden** to modify (e.g., `Data Loading`, `Evaluation`).
  * **`target_sections`**: Hints for the router on where to focus edits (e.g., `Model`, `Innovation`).

### 3\. Execution & Repair (`exec`)

  * **`max_fix_rounds`**: Maximum attempts to fix runtime errors per iteration (Recommended: 3).
  * **`timeout_seconds`**: Max runtime per candidate notebook.

-----

## Output Structure

Results are saved to `../results/${dataset_name}/review_feedback/review_run_YYYYMMDD_.../`:

1.  **`optimization_history.md`**: **Master Log**. Tracks every iteration's critique, code diffs, score comparison (Candidate vs. Baseline), and pass/fail verdict.
2.  **`notebook_base.ipynb`**: The original notebook imported from Phase 2.
3.  **`notebook_iter_N.ipynb`**: The modified notebook for Iteration *N*.
4.  **`metrics_iter_N.json`**: Performance metrics for Iteration *N*.
5.  **`llm_debug/`**: Raw log of LLM inputs/outputs for debugging prompts.