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
        # [NEW LOGGING] 打印出它正在尝试连接的地址和模型
        print(f"  [chat_json] ➡️  POST to {url} (model={payload.get('model')})")
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        r.raise_for_status()  # 这将对 4xx/5xx 错误引发异常
        return r.json()

    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    
    content = "" # 在循环外定义 content
    data = {} # 在循环外定义 data

    for attempt in range(2):  # Two attempts: strict JSON → relaxed instruction
        payload = dict(base_payload)
        if attempt == 1:
            nudged = messages + [
                {"role": "system", "content": "Return only valid MINIFIED JSON. No markdown, no comments, no extra text."},
                {"role": "user", "content": '{"edits": []}'},  # minimal valid JSON example
            ]
            payload["messages"] = nudged

        try:
            data = _post(payload) # [MODIFIED] 赋值给外部的 data

            # [START ROBUST-FIX 1]
            # 立即检查 API 是否返回了顶层错误 (例如: model_not_found)
            if "error" in data:
                error_details = data.get("error", {})
                print(f"  [chat_json] ❌ API returned an error object (attempt {attempt+1}):")
                print(f"      Message: {error_details.get('message')}")
                print(f"      Type:    {error_details.get('type')}")
                print(f"      Code:    {error_details.get('code')}")
                time.sleep(0.4)
                continue # 跳到下一次尝试 (如果还有)
            # [END ROBUST-FIX 1]

        except Exception as e:
            # [MODIFIED] 打印出 API 调用的具体错误！
            print(f"  [chat_json] ❌ API call failed (attempt {attempt+1}). Error: {e}")
            time.sleep(0.4)
            continue

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = (msg.get("content") or "").strip() # [MODIFIED] 赋值给外部的 content

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

        # [START ROBUST-FIX 2]
        # 如果即将失败，打印出 LLM 返回的原始（非 JSON）内容
        if content:
             print(f"  [chat_json] ⚠️ Failed to parse. Raw content preview: {content[:500]}...")
        else:
            # [NEW] 如果 content 都是空的，也打印出来
            print(f"  [chat_json] ⚠️ Message content was empty. Full data received (preview): {str(data)[:500]}...")
        # [END ROBUST-FIX 2]

        time.sleep(0.4)

    # Fallback: return empty edits to gracefully stop current fix round
    # [NEW LOGGING] 明确指出 LLM 返回为空
    print(f"  [chat_json] ⚠️ Could not get valid JSON response after 2 attempts. Returning empty edits.")
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
    
    # [START ROBUST-FIX 1] 清理和截断错误信息，避免 LLM 上下文过载
    cleaned_errors = []
    target_indices = []
    for e in (errors or []):
        idx = e.get("cell_index")
        if idx is None: continue
        
        if idx not in target_indices:
            target_indices.append(idx)
        
        # 清理 traceback：只保留最后 15 行（最相关的部分）
        tb_lines = (e.get("traceback") or "").strip().split("\n")
        cleaned_tb = "\n".join(tb_lines[-15:]) 
        
        # 截断可能很长的错误值（evalue）
        evalue = e.get("evalue", "")
        cleaned_evalue = (evalue[:300] + "...") if len(evalue) > 300 else evalue
        
        cleaned_errors.append({
            "cell_index": idx,
            "ename": e.get("ename"),
            "evalue": cleaned_evalue,
            "traceback_summary": cleaned_tb, # 只发送清理后的摘要
            "original_source_code": e.get("source", "") # 发送完整的原始代码
        })
    # [END ROBUST-FIX 1]
    
    user_obj = {
        "language": language,
        "data_path": data_path,
        "csv_preview": csv_preview,
        "paper_excerpt": (paper_excerpt or "")[:1500] if paper_excerpt else "",
        "required_headings": headings,
        "target_cell_indices": sorted(list(set(target_indices))), # 确保索引列表唯一且排序
        "errors": cleaned_errors, # [MODIFIED] 使用清理后的错误列表
        "example": {"edits": [{"cell_index": int(i), "source": "# fixed code here"} for i in target_indices] or [{"cell_index": 0, "source": "# fixed"}]}
    }
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
    ]