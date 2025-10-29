# Generate_Execution

## Run

```bash
python cellscientist_phase_2.py --config generate_execution_config.json run
# optional flags:
#   --with-lit                # enable literature retrieval/synthesis (if configured)
#   --use-stage1-ref          # prepend Stage-1 summary as Markdown (default True)
#   --no-stage1-ref           # disable Stage-1 summary
#   --use-baseline            # include baseline notebook context (if present)
#   --no-baseline             # do not include baseline (default)
```

---

## Files

| File                               | Purpose                                                                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **cellscientist_phase_2.py**       | Main entry. CLI (`run / generate / execute / analyze`) for the prompt-only pipeline. Supports `--use-baseline` and `--use-stage1-ref`. |
| **generate_execution_config.json** | Config (dataset paths, save roots, LLM options, exec behavior).                                                                        |
| **prompts/pipeline_prompt.yaml**   | Prompt spec consumed by the LLM to produce the notebook.                                                                               |


| File                                                          | Purpose                                                                                                               |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **design_execution/prompt_pipeline.py**                                        | Orchestrates generate → execute → analyze (prompt mode). Analyze is read-only.                                        |
| **design_execution/prompt_builder.py**                                         | Builds LLM messages, calls LLM, parses STRICT-JSON, constructs notebook; **prepends Stage-1 Markdown** when enabled.  |
| **design_execution/llm_client.py**                                             | Unified OpenAI-compatible client; prints provider/model each call; `resolve_llm_from_cfg` picks the exact model used. |
| **design_execution/nb_autofix.py**                                             | Execute + auto-fix loop (known patches, optional LLM bug-fix); preserves source in exec by default.                   |
| **design_execution/notebook_executor.py**                                      | Thin wrapper helpers for nb execution.                                                                                |
| **design_execution/report_builder.py**                                         | (If needed) builds minimal markdown/JSON summaries.                                                                   |
| **design_execution/prompts.py / design_execution/prompt_viz.py / design_execution/run_llm_nb.py / design_execution/evaluator.py** | Helpers (message building, hypergraph viz, JSON-chat wrapper, metrics recorder).                                      |


---

## Key CLI flags

* `--use-stage1-ref / --no-stage1-ref` – control whether Stage-1 analysis summary is **prepended as Markdown** to the generated notebook (Cell 1).
* `--use-baseline / --no-baseline` – include baseline notebook context in the prompt (does **not** run a separate baseline branch).
* `--prompt-file / --prompt-text` – pick the spec source; inline text overrides file.

---

## Typical Workflow

### **A. Generate Patch (LLM only)**
Generate improved notebook patches **without execution**.

#### Without literature (simpler, faster)

```bash
python cellscientist_phase_2.py --config generate_execution_config.json generate 
```

#### With literature (adds OpenAlex search + LLM summarization)

```bash
python cellscientist_phase_2.py --config generate_execution_config.json generate --with-lit
```

### **B. Execute**
Run baseline and patched notebooks:
```bash
python cellscientist_phase_2.py --config generate_execution_config.json execute
```

### **C. Analyze**
Generate Markdown summary report:
```bash
python cellscientist_phase_2.py --config generate_execution_config.json analyze
```

### **D. Full Pipeline**
Run all three stages in sequence:
```bash
python cellscientist_phase_2.py --config generate_execution_config.json run --with-lit
```


