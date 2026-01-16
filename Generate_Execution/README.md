# Generate Execution Module

## Quick Start

### 1\. Standard Run

```bash
python cellscientist_phase_2.py --config generate_execution_config.json run --use-idea
```

**Optional Arguments**:

  * `--use-idea`: Enables **Idea-Driven Mode**. The system locates the `idea.json` generated in Phase 1, synthesizes a research strategy, and then generates code based on that strategy. **Highly Recommended**.
  * `--prompt-file <path>`: Specify a custom YAML prompt file (defaults to `prompts/pipeline_prompt.yaml`).

### 2\. Step-by-Step Debugging

Use subcommands to isolate specific stages of the pipeline:

  * **Generate Only** (Inspect LLM code quality without execution):
    ```bash
    python cellscientist_phase_2.py --config generate_execution_config.json generate --use-idea
    ```
  * **Execute Only** (Run the latest generated notebook with Auto-Fix):
    ```bash
    python cellscientist_phase_2.py --config generate_execution_config.json execute
    ```
  * **Analyze Only** (Re-calculate metrics and generate reports):
    ```bash
    python cellscientist_phase_2.py --config generate_execution_config.json analyze
    ```

-----

## Core Files Overview

| File | Function |
| :--- | :--- |
| **cellscientist\_phase\_2.py** | **Entry Point**. Handles argument parsing, environment setup (injecting API keys/data paths), loop control, and command dispatch. |
| **design\_execution/llm\_utils.py** | **LLM Engine**. Centralizes all LLM interactions, featuring API Key resolution, automatic **Retries**, robust JSON parsing, and Proxy configuration. |
| **design\_execution/prompt\_orchestrator.py** | **Scheduler**. Manages the standard workflow (Generate -\> Execute -\> Analyze) and handles file path transitions. |
| **design\_execution/prompt\_generator.py** | **Code Generator**. Converts Ideas and Specs into a Jupyter Notebook. Includes logic for "Strategy Synthesis". |
| **design\_execution/prompt\_executor.py** | **Execution Engine**. Runs the notebook and manages the **Auto-Fix Loop** (Capture Error -\> LLM Patch -\> Retry). |
| **design\_execution/experiment\_report.py** | **Analyst**. Reads `metrics.json`, computes statistical significance (P-Values), and generates the detailed `experiment_report.md`. |
| **design\_execution/prompt\_viz.py** | **Visualizer**. Parses notebook metadata to generate `hypergraph.md` (Mermaid flowchart). |

-----

## Configuration Guide (`generate_execution_config.json`)

### 1\. LLM Settings (`llm`)

  * **`base_url`**: **Critical**. Must be set to your proxy address to avoid "Network Unreachable" errors.
  * **`api_key`**: Your API key. If left empty, it defaults to the environment variable `OPENAI_API_KEY`.
  * **`model`**: Strong reasoning models are recommended (e.g., `gpt-4o`, `gemini-3-pro`).

### 2\. Experiment Control (`experiment`)

  * **`primary_metric`**: The core metric (e.g., `PCC`, `MSE`) used to determine "Success" and rank models.
  * **`success_threshold`**: The target score. If a run exceeds this value, the loop terminates early.
  * **`max_iterations`**: Maximum number of attempts (prevents infinite loops).

### 3\. Execution & Repair (`exec`)

  * **`enable_llm_autofix`**: Set to `true` to enable automatic code repair.
  * **`max_fix_rounds`**: Maximum attempts to fix code errors (Recommended: 3-5).
  * **`timeout_seconds`**: Max runtime per notebook (Recommended: 18000s+ for model training).

### 4\. Generation Settings (`prompt_branch`)

  * **`idea_file`**: The filename for ideas (default: `idea.json`). The system automatically searches for this file in the Phase 1 output directory.

-----

## Output Structure

Results are saved to `../results/${dataset_name}/generate_execution/prompt/prompt_run_YYYYMMDD_.../`:

1.  **`notebook_prompt.ipynb`**: The raw, LLM-generated source code.
2.  **`notebook_prompt_exec.ipynb`**: The executed notebook (containing outputs and logs).
3.  **`research_strategy.md`**: The research strategy document synthesized by the LLM (only if `--use-idea` is enabled).
4.  **`metrics.json`**: Extracted model performance metrics.
5.  **`experiment_report.md`**: The final analysis report.
6.  **`hypergraph.md`**: A visualization of the code's logical flow.