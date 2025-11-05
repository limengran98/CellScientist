# run_llm_nb.py (utility module)
# [NEW] This file exposes helper functions for auto-fix and LLM calls.
import os, json, re, io, contextlib, importlib

def _load_runner_clean():
    """Import runner quietly (suppress initialization prints/logs)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mod = importlib.import_module("cellscientist.Design_Analysis.llm_notebook_runner")
    # The original function should return a path to the executed notebook
    # [MODIFIED] Update function name to match what is exported
    return getattr(mod, "run_llm_notebook_from_file")

# --- LLM helper ---
def chat_json(messages, *, api_key, base_url, model, temperature=0.2, max_tokens=800):
    """Return JSON object from an OpenAI-compatible /chat/completions endpoint.
    [FIX] Improved robustness: retries, strict JSON parsing, fenced-block extraction, and tool_call fallback.
    """
    import requests, json as _json, time, re

    def _post(payload):
        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(2):  # Two attempts: strict JSON → relaxed instruction
        payload = dict(base_payload)
        if attempt == 1:
            nudged = messages + [
                {"role": "system", "content": "Return only valid MINIFIED JSON. No markdown, no comments, no extra text."},
                {"role": "user", "content": '{"edits": []}'},  # minimal valid JSON example
            ]
            payload["messages"] = nudged

        try:
            data = _post(payload)
        except Exception:
            time.sleep(0.4)
            continue

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip()

        # 1) Direct JSON
        if content:
            try:
                return _json.loads(content)
            except Exception:
                pass

        # 2) ```json fenced block
        if content:
            m = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.S)
            if m:
                try:
                    return _json.loads(m.group(1))
                except Exception:
                    pass
            # 3) Any substring that looks like JSON
            m2 = re.search(r"(\{(?:.|\n)*\})", content)
            if m2:
                try:
                    return _json.loads(m2.group(1))
                except Exception:
                    pass

        # 4) function_call / tool_calls structure
        fc = msg.get("function_call") or {}
        if isinstance(fc, dict) and "arguments" in fc:
            try:
                return _json.loads(fc["arguments"])
            except Exception:
                pass
        tools = msg.get("tool_calls") or []
        if tools:
            args = tools[0].get("function", {}).get("arguments")
            if args:
                try:
                    return _json.loads(args)
                except Exception:
                    pass

        time.sleep(0.4)

    # Fallback: return empty edits to gracefully stop current fix round
    return {"edits": []}


# --- Notebook helpers ---
def collect_cell_errors(nb):
    """Extract all code cell errors from a Jupyter notebook."""
    errs = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []) or []:
            if out.get("output_type") == "error":
                tb = "\n".join(out.get("traceback") or [])
                errs.append({
                    "cell_index": i,
                    "ename": out.get("ename"),
                    "evalue": out.get("evalue"),
                    "traceback": tb,
                    "source": cell.get("source", ""),
                })
    return errs


def apply_edits(nb, edits):
    """Apply edits from LLM response to notebook cells."""
    changed = 0
    for ed in (edits or []):
        try:
            idx = int(ed.get("cell_index"))
            src = ed.get("source") or ""
            if idx < 0 or idx >= len(nb["cells"]):
                continue
            if nb["cells"][idx].get("cell_type") != "code":
                continue
            old = nb["cells"][idx].get("source", "")
            # [NEW] Only count actual changes
            if old != src:
                nb["cells"][idx]["source"] = src
                changed += 1
        except Exception:
            continue
    return changed


def execute_notebook(nb, *, timeout=1800, allow_errors=True):
    """Execute a notebook and return it with optional error summary."""
    import nbformat as nbf
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=allow_errors)
    errors_summary = []
    try:
        client.execute()
    except Exception as e:
        errors_summary.append(f"Notebook execution raised: {type(e).__name__}: {e}")
    return nb, errors_summary


# [MODIFIED] Added 'autofix_prompt_str' to the function definition
def build_fix_messages(
    language, data_path, csv_preview, paper_excerpt, errors, headings,
    autofix_prompt_str: str 
):
    """Build structured chat messages for LLM-based notebook auto-fix."""
    
    # [MODIFIED] Use the passed-in prompt string, with a fallback
    sys_prompt = ( autofix_prompt_str or 
        (
        "You are a senior Python engineer and Jupyter expert.\n"
        "Given Jupyter cell errors, return ONLY a MINIFIED JSON object with key 'edits'.\n"
        "Each edit MUST be {\"cell_index\": int, \"source\": str} replacing the WHOLE code cell.\n"
        "You MUST cover ALL indices listed in 'target_cell_indices' (one edit per index). Do not add new cells.\n"
        "Do not modify markdown cells. No markdown fences or extra text.\n"
        ) # Fallback
    )
    
    target_indices = [e.get("cell_index") for e in (errors or [])]
    user_obj = {
        "language": language,
        "data_path": data_path,
        "csv_preview": csv_preview,
        "paper_excerpt": (paper_excerpt or "")[:1500] if paper_excerpt else "",
        "required_headings": headings,
        "target_cell_indices": target_indices,
        "errors": errors,
        # [NEW] Example covers all listed indices
        "example": {"edits": [{"cell_index": int(i), "source": "# fixed code here"} for i in target_indices] or [{"cell_index": 0, "source": "# fixed"}]}
    }
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
    ]
