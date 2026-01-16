# run_llm_nb.py
import os
import sys
import json
import re
import io
import contextlib
import importlib
import time
import requests
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# [FIX] Ensure local imports work regardless of folder name
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

def _load_runner_clean():
    """Import runner quietly and robustly."""
    try:
        import llm_notebook_runner as mod
    except ImportError:
        # Fallback: Load explicitly from file path
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_notebook_runner", 
            os.path.join(THIS_DIR, "llm_notebook_runner.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["llm_notebook_runner"] = mod
        spec.loader.exec_module(mod)
        
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
        # [FIX] Resolve relative to THIS_DIR
        base_dir = Path(THIS_DIR).parent
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
        base_url_env_name = llm.get("base_url_env")
        if base_url_env_name and os.environ.get(base_url_env_name):
            base_url = os.environ.get(base_url_env_name)
        else:
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com"
            
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)

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

# --- Adaptive Graph Executor ---

class AdaptiveGraphExecutor(NotebookClient):
    """
    Stateful execution engine.
    Treats each cell as a 'node' in a graph. 
    If a node fails, it invokes an LLM to fix the code, then retries the node
    WITHOUT restarting the kernel or re-running previous nodes.
    """
    def __init__(self, nb, run_cfg, nb_cfg, llm_config, cuda_device_id=None, **kwargs):
        super().__init__(nb, **kwargs)
        self.run_cfg = run_cfg
        self.nb_cfg = nb_cfg
        self.llm_config = llm_config
        self.cuda_device_id = cuda_device_id
        
        exec_cfg = nb_cfg.get("exec", {})
        self.max_retries_per_node = int(exec_cfg.get("max_fix_rounds", 3))
        self.autofix_prompt = (run_cfg.get('prompts', {}).get('autofix', {}).get('system_prompt'))
        
        self.total_fixes_applied = 0

    def run_adaptive(self):
        env_vars = os.environ.copy()
        if self.cuda_device_id is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(self.cuda_device_id)
            print(f"🖥️  [Adaptive] CUDA Device set to: {self.cuda_device_id}", flush=True)

        print("🔌 [Adaptive] Starting Kernel...", flush=True)
        self.create_kernel_manager()
        self.start_new_kernel(env=env_vars) 
        self.start_new_kernel_client()

        try:
            cell_idx = 0
            while cell_idx < len(self.nb.cells):
                cell = self.nb.cells[cell_idx]
                if cell.cell_type != 'code':
                    cell_idx += 1
                    continue
                
                src_snippet = cell.source.strip().split('\n')[0][:70]
                if len(cell.source) > 70: src_snippet += "..."
                print(f"▶️  [Adaptive] Executing Cell {cell_idx}: {src_snippet}", flush=True)
                
                try:
                    self.execute_cell(cell, cell_idx)
                    print(f"   ✅ Success (Cell {cell_idx})", flush=True)
                    cell_idx += 1 
                    
                except CellExecutionError:
                    print(f"   ❌ Failed (Cell {cell_idx}). Initiating Auto-Fix...", flush=True)
                    fixed_and_executed = self._attempt_fix_node(cell, cell_idx)

                    if fixed_and_executed:
                        print(f"   ✅ Fixed and executed (Cell {cell_idx}). Moving on...", flush=True)
                        cell_idx += 1
                        continue
                    else:
                        print(f"   🛑 Failed to fix Cell {cell_idx} after retries. Aborting.", flush=True)
                        raise RuntimeError(f"Adaptive auto-fix failed at cell {cell_idx}")

        finally:
            print("🔌 [Adaptive] Shutting down Kernel.", flush=True)
            self._cleanup_kernel()

        return self.nb

    def _attempt_fix_node(self, cell, cell_idx):
        def _collect_errors() -> list:
            errs = []
            for out in cell.get("outputs", []) or []:
                if out.get("output_type") == "error":
                    tb = "\n".join(out.get("traceback") or [])
                    errs.append({
                        "cell_index": cell_idx,
                        "ename": out.get("ename"),
                        "evalue": str(out.get("evalue"))[:1000],
                        "traceback_tail": "\n".join(tb.split("\n")[-20:]),
                        "code": cell.get("source", "")
                    })
            return errs

        errors = _collect_errors()
        if not errors:
            print("   (No specific traceback found in outputs)", flush=True)
            return False

        for attempt in range(self.max_retries_per_node):
            last = errors[-1] if errors else {}
            ename = last.get("ename")
            evalue = last.get("evalue")
            if ename or evalue:
                print(f"   🔧 Fix Attempt {attempt+1}/{self.max_retries_per_node}...  ({ename}: {evalue})", flush=True)
            else:
                print(f"   🔧 Fix Attempt {attempt+1}/{self.max_retries_per_node}...", flush=True)

            paths = self.nb_cfg.get("paths", {})
            messages = build_fix_messages(
                language="python",
                data_path=paths.get("data", ""),
                csv_preview={},
                paper_excerpt="",
                errors=errors,
                headings=[],
                autofix_prompt_str=self.autofix_prompt
            )

            resp = chat_json(
                messages,
                api_key=self.llm_config["api_key"],
                base_url=self.llm_config["base_url"],
                model=self.llm_config["model"],
                temperature=0.0,
                max_tokens=self.llm_config["max_tokens"],
            )

            edits = (resp or {}).get("edits") or []
            if not edits:
                print("   LLM returned no edits.", flush=True)
                continue

            edit = edits[0]
            new_source = (edit.get("source") or "")
            if not isinstance(new_source, str) or not new_source.strip():
                print("   LLM produced empty/invalid source.", flush=True)
                continue

            if new_source.strip() == (cell.source or "").strip():
                print("   LLM suggested identical code (Skipping).", flush=True)
                continue

            cell.source = new_source
            self.total_fixes_applied += 1

            try:
                cell["outputs"] = []
                cell["execution_count"] = None
            except Exception:
                pass

            try:
                self.execute_cell(cell, cell_idx)
                print("   ✅ Fix validated by successful execution.", flush=True)
                return True
            except CellExecutionError:
                print("   ❌ Still failing after applying fix.", flush=True)
                errors = _collect_errors()
                if not errors:
                    errors = [{"cell_index": cell_idx, "ename": "CellExecutionError", "evalue": "(no traceback)", "traceback_tail": "", "code": cell.get("source", "")}]
                continue
            except Exception as e:
                print(f"   ❌ Unexpected error while validating fix: {e}", flush=True)
                errors = _collect_errors() or errors
                continue

        return False

def build_fix_messages(language, data_path, csv_preview, paper_excerpt, errors, headings, autofix_prompt_str):
    sys_prompt = autofix_prompt_str or "You are a Python expert. Fix the code errors. Return JSON {'edits': ...}."
    
    cleaned_errors = []
    target_indices = []
    for e in errors:
        idx = e.get("cell_index")
        if idx not in target_indices: target_indices.append(idx)
        
        cleaned_errors.append({
            "cell_index": idx,
            "ename": e.get("ename"),
            "evalue": str(e.get("evalue")),
            "traceback_tail": e.get("traceback_tail"),
            "code": e.get("code")
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

# --- Main Entry Point ---

def auto_fix_notebook(executed_path: str, run_cfg: dict) -> str:
    nb_cfg = (((run_cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    exec_cfg = nb_cfg.get("exec", {}) or {}
    
    llm_resolved = resolve_llm_config(nb_cfg.get("llm", {}))
    if not llm_resolved["api_key"]:
        print("⚠️ [AUTO-FIX] No API Key found. Skipping.", flush=True)
        return executed_path

    nb_path = Path(executed_path)
    if not nb_path.exists():
        return executed_path

    print(f"🚀 [AUTO-FIX] Starting Adaptive Execution for {nb_path.name}", flush=True)
    
    nb = nbf.read(str(nb_path), as_version=4)
    
    timeout_sec = int(exec_cfg.get("timeout_seconds", 1800))
    cuda_id = exec_cfg.get("cuda_device_id", None)

    executor = AdaptiveGraphExecutor(
        nb=nb,
        run_cfg=run_cfg,
        nb_cfg=nb_cfg,
        llm_config=llm_resolved,
        cuda_device_id=cuda_id,
        timeout=timeout_sec,
        kernel_name="python3",
        allow_errors=False 
    )
    
    try:
        final_nb = executor.run_adaptive()
    except Exception as e:
        failed_path = nb_path.with_name(nb_path.stem + "_adaptive_failed.ipynb")
        try:
            nbf.write(nb, str(failed_path))
            print(f"💾 [AUTO-FIX] Saved failed state to {failed_path}", flush=True)
        except Exception as w:
            print(f"⚠️ [AUTO-FIX] Failed to save failed notebook: {w}", flush=True)
        raise

    fixed_path = nb_path.with_name(nb_path.stem + "_adaptive_fixed.ipynb")
    nbf.write(final_nb, str(fixed_path))
    
    print(f"💾 [AUTO-FIX] Saved stateful result to {fixed_path}", flush=True)
            
    return str(fixed_path)