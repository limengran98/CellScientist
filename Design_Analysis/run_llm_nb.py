# run_llm_nb.py
import os, json, re, io, contextlib, importlib, time, requests
from pathlib import Path

def _load_runner_clean():
    """Import runner quietly."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        mod = importlib.import_module("cellscientist.Design_Analysis.llm_notebook_runner")
    return getattr(mod, "run_llm_notebook_from_file")

# --- Centralized LLM Config Resolver ---
def resolve_llm_config(llm_cfg: dict) -> dict:
    """Strictly follows Config > Env Var > Default."""
    llm = llm_cfg or {}
    
    api_key = llm.get("api_key")
    if not api_key:
        api_key_env = llm.get("api_key_env") or "OPENAI_API_KEY"
        api_key = os.environ.get(api_key_env)

    model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    base_url = None
    try:
        base_dir = Path(__file__).resolve().parents[1] 
        prov_file = base_dir / "llm_providers.json"
        prov_name = (llm.get("provider") or "").strip() or None
        if prov_name and prov_file.exists():
            data = json.loads(prov_file.read_text(encoding="utf-8"))
            prov = data.get("providers", {}).get(prov_name)
            if prov:
                base_url = prov.get("base_url")
                if not llm.get("model") and prov.get("models"):
                    model = prov.get("models")[0]
    except Exception:
        pass 

    if not base_url:
        base_url_env = llm.get("base_url_env")
        if base_url_env and os.environ.get(base_url_env):
            base_url = os.environ.get(base_url_env)
        else:
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://vip.yi-zhan.top/v1"
            
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)

    # [CRITICAL] Preserve User Hyperparameters
    max_tokens = int(llm.get("max_tokens") or 10240)
    temperature = float(llm.get("temperature", 0.5)) 

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

# --- LLM helper ---
def chat_json(messages, *, api_key, base_url, model, temperature=0.2, max_tokens=2048):
    """Robust JSON chat completion with retries."""
    def _post(payload):
        url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        # print(f"  [chat_json] ➡️  POST {model} (temp={temperature})", flush=True) 
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(3): 
        payload = dict(base_payload)
        if attempt > 0:
            payload["messages"] = messages + [
                {"role": "system", "content": "You MUST return valid JSON only. Check for missing braces."}
            ]

        try:
            data = _post(payload)
            if "error" in data:
                print(f"  [chat_json] ❌ API Error: {data['error']}", flush=True)
                time.sleep(1)
                continue
            
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            
            try: return json.loads(content)
            except: pass
            
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
            if m:
                try: return json.loads(m.group(1))
                except: pass
            
            m2 = re.search(r"(\{.*\})", content, flags=re.S)
            if m2:
                try: return json.loads(m2.group(1))
                except: pass
                
        except Exception as e:
            print(f"  [chat_json] ⚠️ Attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(1)

    print(f"  [chat_json] ❌ Failed to parse JSON after 3 attempts.", flush=True)
    return {"edits": []} 

# --- Notebook helpers ---
def collect_cell_errors(nb):
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
            if old.strip() != src.strip():
                nb["cells"][idx]["source"] = src
                changed += 1
        except Exception:
            continue
    return changed

def execute_notebook(nb, *, timeout=1800, allow_errors=True):
    from nbclient import NotebookClient
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=allow_errors)
    errors_summary = []
    try:
        client.execute()
    except Exception as e:
        errors_summary.append(f"Execution Exception: {str(e)}")
    return nb, errors_summary

def build_fix_messages(language, data_path, csv_preview, paper_excerpt, errors, headings, autofix_prompt_str):
    sys_prompt = autofix_prompt_str or "You are a Python expert. Fix the code errors. Return JSON {'edits': ...}."
    
    cleaned_errors = []
    target_indices = []
    for e in errors:
        idx = e.get("cell_index")
        if idx not in target_indices: target_indices.append(idx)
        
        tb = (e.get("traceback") or "")
        tb_lines = tb.split("\n")
        short_tb = "\n".join(tb_lines[-20:]) 
        
        cleaned_errors.append({
            "cell_index": idx,
            "ename": e.get("ename"),
            "evalue": str(e.get("evalue"))[:500],
            "traceback_tail": short_tb,
            "code": e.get("source")
        })

    user_obj = {
        "context": {"data": data_path, "lang": language},
        "errors": cleaned_errors,
        "instruction": "Fix these cells. Return ONLY JSON.",
        "target_indices": target_indices
    }
    
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)}
    ]

# --- Auto Fix Logic ---
def auto_fix_notebook(executed_path: str, run_cfg: dict) -> str:
    import nbformat as nbf
    
    nb_cfg = (((run_cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    exec_cfg = nb_cfg.get("exec", {}) or {}
    
    llm_resolved = resolve_llm_config(nb_cfg.get("llm", {}))
    if not llm_resolved["api_key"]:
        print("⚠️ [AUTO-FIX] No API Key found. Skipping.", flush=True)
        return executed_path

    nb_path = Path(executed_path)
    if not nb_path.exists():
        return executed_path

    nb = nbf.read(str(nb_path), as_version=4)
    errors = collect_cell_errors(nb)
    max_rounds = int(exec_cfg.get("max_fix_rounds", 0))
    
    if not errors or max_rounds <= 0:
        return str(nb_path)

    print(f"🔧 [AUTO-FIX] Starting fix loop for {nb_path.name} (Errors: {len(errors)}, Rounds: {max_rounds})", flush=True)
    
    autofix_prompt = (run_cfg.get('prompts', {}).get('autofix', {}).get('system_prompt'))
    paths = nb_cfg.get("paths", {})
    
    round_idx = 0
    while errors and round_idx < max_rounds:
        round_idx += 1
        
        messages = build_fix_messages(
            language="python",
            data_path=paths.get("data", ""),
            csv_preview={}, 
            paper_excerpt="",
            errors=errors,
            headings=[],
            autofix_prompt_str=autofix_prompt
        )
        
        resp = chat_json(
            messages,
            api_key=llm_resolved["api_key"],
            base_url=llm_resolved["base_url"],
            model=llm_resolved["model"],
            temperature=0.0, 
            max_tokens=llm_resolved["max_tokens"]
        )
        
        edits = resp.get("edits") or []
        if not edits:
            print(f"  [Round {round_idx}] No edits returned.", flush=True)
            break
            
        changed = apply_edits(nb, edits)
        if changed == 0:
            print(f"  [Round {round_idx}] Edits yielded no changes.", flush=True)
            break
            
        suffix = f"_fixed_r{round_idx}"
        temp_path = nb_path.with_name(nb_path.stem + suffix + ".ipynb")
        
        timeout_sec = int(exec_cfg.get("timeout_seconds", 1800))
        nb, _ = execute_notebook(nb, timeout=timeout_sec)
        nbf.write(nb, str(temp_path))
        
        errors = collect_cell_errors(nb)
        print(f"  [Round {round_idx}] Fixed {changed} cells. Remaining errors: {len(errors)}", flush=True)

    if round_idx > 0:
        final_path = nb_path.with_name(nb_path.stem + f"_fixed_r{round_idx}.ipynb")
        if final_path.exists():
            return str(final_path)
            
    return str(nb_path)