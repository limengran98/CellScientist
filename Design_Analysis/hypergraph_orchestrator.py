#!/usr/bin/env python3
# hypergraph_orchestrator.py
# Optimized for stability, concurrency, and semantic naming.

import json, hashlib, os, shutil, time
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config_loader import load_app_config
from cellscientist_phase_1 import run_pipeline_basic

# Import centralized tools
from run_llm_nb import (
    resolve_llm_config, 
    auto_fix_notebook, 
    chat_json
)

# --------------------
# Utilities
# --------------------
def _get_prompts_dir(cfg_path: str) -> str:
    return str(Path(cfg_path).parent / 'prompts')

def _load_cfg(cfg_path: str, prompts_dir_path: Optional[str] = None) -> Dict[str, Any]:
    if prompts_dir_path is None:
        prompts_dir_path = _get_prompts_dir(cfg_path)
    return load_app_config(cfg_path, prompts_dir_path)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _hash_file(p: Path) -> str:
    if not p.exists(): return ""
    h = hashlib.sha256()
    try:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""): h.update(chunk)
        return h.hexdigest()[:16]
    except Exception: return ""

def _is_valid_hdf5(p: Path) -> bool:
    """Best-effort HDF5 validation.

    Some environments may not have h5py installed; in that case we fall back
    to checking the HDF5 file signature so that generated artifacts can still
    be exported to the reference directory.
    """
    if not p.exists():
        return False
    try:
        import h5py  # type: ignore
        with h5py.File(p, 'r') as f:
            _ = list(f.keys())
        return True
    except Exception:
        # Fallback: HDF5 signature is: \x89 H D F \r \n \x1a \n
        try:
            with open(p, 'rb') as fh:
                sig = fh.read(8)
            return sig == b"\x89HDF\r\n\x1a\n"
        except Exception:
            return False

# --------------------
# Per-run logic (Worker Function)
# --------------------
def _process_single_run(idx, variant, seed, base_cfg, num_runs) -> Dict[str, Any]:
    """
    Worker function for ThreadPoolExecutor. 
    Naming: design_analysis_YYYYMMDD_HHMMSS_Run{idx}
    """
    try:
        import copy
        cfg = copy.deepcopy(base_cfg)
        
        nb_cfg = cfg["phases"]["task_analysis"]["llm_notebook"]
        multi = nb_cfg.get("multi", {})
        paths = nb_cfg.get("paths", {})
        
        out_dir = Path(multi.get("out_dir") or Path(paths.get("out")).parent / "hypergraph_runs").resolve()
        
        # [MODIFIED] Semantic Naming Strategy
        t_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"design_analysis_{t_str}_Run{idx+1}"
        
        run_dir = (out_dir / run_name).resolve()
        _ensure_dir(run_dir)

        # Inject Variant
        base_prompt = (cfg.get('prompts', {}).get('notebook_generation', {}).get('user_prompt'))
        nb_cfg["prompt"] = base_prompt
        nb_cfg["focus_instruction"] = variant or "Standard analysis."
        
        # Setup Paths
        nb_cfg.setdefault("paths", {})
        nb_cfg["paths"]["out"] = str((run_dir / "CP_llm.ipynb").resolve())
        nb_cfg["paths"]["out_exec"] = str((run_dir / "CP_llm_executed.ipynb").resolve())
        h5_p = (run_dir / "preprocessed_data.h5").resolve()
        nb_cfg["paths"]["h5_out"] = str(h5_p)
        
        if seed is not None:
            nb_cfg.setdefault("llm", {})["seed"] = seed

        # Write Temp Config
        tmp_cfg_path = run_dir / "config.run.json"
        tmp_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        # [MODIFIED] Added flush=True to ensure visibility
        print(f"🚀 [{idx+1}/{num_runs}] Executing {run_name}...", flush=True)
        
        # 1. Run Pipeline
        executed_path = run_pipeline_basic(str(tmp_cfg_path), phase_name="task_analysis")
        # 2. Auto-Fix (guarded to avoid double-fix)
        exec_cfg = (nb_cfg.get("exec") or {})
        enable_adaptive_fix = bool(exec_cfg.get("enable_adaptive_fix", True))
        final_exec = executed_path
        # If upstream already produced an adaptive-fixed notebook, skip to avoid redundant API calls.
        if enable_adaptive_fix and ("adaptive_fixed" not in Path(executed_path).stem):
            final_exec = auto_fix_notebook(executed_path, cfg)
        
        exec_p = Path(final_exec)
        unexec_p = Path(nb_cfg["paths"]["out"])
        
        # 3. Validate H5
        h5_valid = _is_valid_hdf5(h5_p)
        if not h5_valid and h5_p.exists():
            print(f"⚠️ [{run_name}] H5 file generated but corrupt/invalid.", flush=True)

        print(f"✅ [{idx+1}/{num_runs}] Finished {run_name}", flush=True)

        return {
            "edge_id": run_name,
            "executed_ipynb": str(exec_p),
            "unexecuted_ipynb": str(unexec_p),
            "generated_h5": str(h5_p) if h5_valid else None,
            "executed_sha": _hash_file(exec_p),
            "unexecuted_sha": _hash_file(unexec_p),
            "generated_h5_sha": _hash_file(h5_p) if h5_valid else None,
            "prompt_variant": variant,
            "seed": seed,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"❌ [{idx+1}] Run failed: {e}", flush=True)
        return None

# --------------------
# Orchestrator
# --------------------
def orchestrate(cfg_path: str, prompts_dir_path: Optional[str] = None):
    cfg = _load_cfg(cfg_path, prompts_dir_path)
    nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    multi_cfg = nb_cfg.get("multi") or {}

    if not multi_cfg.get("enabled", False):
        print("Multi-run disabled in config.", flush=True)
        return

    # Adaptive Logic
    prompt_variants = multi_cfg.get("prompt_variants") or []
    seeds = multi_cfg.get("seeds") or []
    num_runs_cfg = int(multi_cfg.get("num_runs", 1))
    
    if not seeds or not prompt_variants:
        num_runs = 0
    else:
        num_runs = min(num_runs_cfg, len(seeds), len(prompt_variants))

    print(f"ℹ️  Starting Orchestration: {num_runs} runs planned.", flush=True)
    
    max_workers = int(multi_cfg.get("max_parallel_workers", 2))
    edges = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_runs):
            variant = prompt_variants[i]
            seed = seeds[i]
            # [MODIFIED] Removed time.sleep() as requested - No artificial limiting
            futures.append(executor.submit(_process_single_run, i, variant, seed, cfg, num_runs))
            
        for f in as_completed(futures):
            res = f.result()
            if res:
                edges.append(res)

    if not edges:
        print("⚠️ No successful runs generated.", flush=True)
        return

    # Emit Hypergraph
    hypergraph = {
        "name": multi_cfg.get("hypergraph_name", "CellScientist"),
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "nodes": [{"node_id": f"step-{k+1}", "name": n} for k, n in enumerate(multi_cfg.get("node_names", []))],
        "hyperedges": [{
            "edge_id": e["edge_id"],
            "connects": [f"step-{k+1}" for k in range(len(multi_cfg.get("node_names", [])))],
            "attrs": e
        } for e in edges]
    }
    
    multi_out_dir = Path(multi_cfg.get("out_dir") or "hypergraph_runs").resolve()
    _ensure_dir(multi_out_dir)
    hypergraph_dir = (multi_out_dir / "hypergraph_runs").resolve()
    _ensure_dir(hypergraph_dir)
    hp_path = hypergraph_dir / "hypergraph.json"
    hp_path.write_text(json.dumps(hypergraph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Hypergraph saved: {hp_path}", flush=True)

    # Mandatory Reference Export (Heuristics + Selection)
    print(f"🧪 [HEURISTICS] Calculating scores for {len(edges)} runs...", flush=True)
    calculate_run_heuristics(str(hp_path), cfg_path=cfg_path, prompts_dir_path=prompts_dir_path)
    
    print(f"🏆 [REFERENCE] Selecting and exporting best result (Mandatory)...", flush=True)
    select_and_export_reference(str(hp_path), cfg_path, prompts_dir_path=prompts_dir_path)

# ============================
# Heuristics & Helpers
# ============================
import nbformat as _nbf
from nbformat.reader import reads as _nb_reads

def _safe_read_text(p):
    try: return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception: return ""

def _extract_nb_cells(ipynb_path: str):
    try:
        nb = _nbf.read(ipynb_path, as_version=4)
        return nb.get("cells", [])
    except Exception:
        s = _safe_read_text(ipynb_path)
        try: return _nb_reads(s, as_version=4).get("cells", [])
        except Exception: return []

def _heuristic_scores_from_cells(cells):
    if not cells:
        return dict(scientific=0.0, novelty=0.0, reproducibility=0.0, interpretability=0.0)
    md_cnt = sum(1 for c in cells if c.get("cell_type")=="markdown")
    code_cnt = sum(1 for c in cells if c.get("cell_type")=="code")
    text = " ".join([c.get("source","") for c in cells if isinstance(c.get("source",""), str)]).lower()

    has_stat = any(k in text for k in ["p-value","anova","t-test","wilcoxon","chi-square","adjusted p","fdr","effect size"])
    has_fig  = any(k in text for k in ["plt.","plot","figure","heatmap","violin","boxplot","volcano"])
    has_seed = any(k in text for k in ["random_state","seed=","np.random.seed","torch.manual_seed"])
    has_method = any(k in text for k in ["methods","methodology","pipeline","workflow","protocol"])
    has_disc  = any(k in text for k in ["discussion","limitations","confound","bias","future work"])
    has_bio   = any(k in text for k in ["gene","rna","protein","pathway","enrichment","marker"])

    def clamp(v): return max(0.0, min(1.0, v))

    scientific = clamp(0.25*(1.0 if has_stat else 0.0) + 0.25*(1.0 if has_fig else 0.0) + 0.25*(1.0 if has_bio else 0.0) + 0.25*min(md_cnt/6.0,1.0))
    novelty = clamp(0.4*(1.0 if ("contrastive" in text or "representation" in text or "causal" in text) else 0.0) + 0.3*min(code_cnt/10.0,1.0) + 0.3*min(md_cnt/10.0,1.0))
    reproducibility = clamp(0.6*(1.0 if has_seed else 0.0) + 0.4*(1.0 if ("requirements.txt" in text or "environment" in text or "versions" in text) else 0.0))
    interpretability = clamp(0.5*min(md_cnt/8.0,1.0) + 0.25*(1.0 if has_method else 0.0) + 0.25*(1.0 if has_disc else 0.0))

    return dict(scientific=scientific, novelty=novelty, reproducibility=reproducibility, interpretability=interpretability)

def calculate_run_heuristics(hypergraph_path: str, cfg_path: Optional[str] = None, prompts_dir_path: Optional[str] = None):
    hp = Path(hypergraph_path)
    if not hp.exists(): return
    data = json.loads(hp.read_text(encoding="utf-8"))
    
    for e in data.get("hyperedges", []):
        attrs = e.get("attrs", {})
        ip = attrs.get("executed_ipynb")
        cells = _extract_nb_cells(ip) if ip else []
        attrs["expert_heuristic_scores"] = _heuristic_scores_from_cells(cells)
        e["attrs"] = attrs

    data["heuristics_calculated_utc"] = datetime.utcnow().isoformat() + "Z"
    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧠 [HEURISTICS] Scores saved.", flush=True)

def generate_experiment_ideas(notebook_path: str, output_dir: Path, llm_cfg: Dict[str, Any], prompts_dir_path: Optional[str] = None):
    resolved = resolve_llm_config(llm_cfg)
    
    nb_cells = _extract_nb_cells(notebook_path)
    context_text = []
    for c in nb_cells:
        if c.get("cell_type") == "markdown": 
            context_text.append(c.get("source",""))
        elif c.get("cell_type") == "code":
            for out in c.get("outputs", []) or []:
                data_obj = out.get("data", {})
                text_content = data_obj.get("text/plain") or data_obj.get("text")
                if text_content:
                    val = "".join(text_content)
                    if len(val) < 2000: context_text.append(f"Output: {val}")

    full_text = "\n\n".join(context_text)
    if len(full_text) > 12000:
        full_text = full_text[:3000] + "\n...[SNIP]...\n" + full_text[-9000:]

    system_prompt = ""
    if prompts_dir_path:
        try:
             with open(Path(prompts_dir_path)/"idea.yml", 'r') as f:
                 import yaml
                 y = yaml.safe_load(f)
                 system_prompt = y.get("system_prompt", "")
        except: pass

    if not system_prompt:
        system_prompt = "You are an expert. Generate innovative experimental ideas based on the analysis."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{full_text}\n\nGenerate JSON with ideas."}
    ]

    print(f"💡 [IDEAS] Generating ideas...", flush=True)
    try:
        resp = chat_json(
            messages,
            api_key=resolved["api_key"],
            base_url=resolved["base_url"],
            model=resolved["model"],
            temperature=0.8,
            max_tokens=resolved["max_tokens"]
        )
        if resp:
            (output_dir / "idea.json").write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✨ [IDEAS] Saved.", flush=True)
    except Exception as e:
        print(f"❌ [IDEAS] Failed: {e}", flush=True)


def _truncate_middle(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = max_chars // 3
    tail = max_chars - head - len("\n...[SNIP]...\n")
    return text[:head] + "\n...[SNIP]...\n" + text[-tail:]


def _summarize_h5_for_report(h5_path: Path, max_items: int = 80) -> str:
    if not h5_path.exists():
        return "(no H5 artifact found)"
    lines = []
    try:
        import h5py  # type: ignore
        with h5py.File(h5_path, "r") as f:
            def _walk(name, obj):
                if isinstance(obj, h5py.Dataset):
                    shape = getattr(obj, "shape", None)
                    dtype = str(getattr(obj, "dtype", ""))
                    lines.append(f"- {name}  shape={shape}  dtype={dtype}")
                else:
                    lines.append(f"- {name}/")
            f.visititems(_walk)
        if len(lines) > max_items:
            lines = lines[:max_items] + [f"... ({len(lines)-max_items} more items)"]
        return "\n".join(lines)
    except Exception as e:
        # Keep it non-fatal; report can still be generated.
        try:
            size = h5_path.stat().st_size
        except Exception:
            size = -1
        return f"(failed to inspect H5: {e}; size_bytes={size})"


def _summarize_notebook_for_report(ipynb_path: Path, max_chars: int = 16000) -> str:
    if not ipynb_path.exists():
        return "(notebook missing)"
    cells = _extract_nb_cells(str(ipynb_path))
    parts: List[str] = []

    for c in cells:
        ctype = c.get("cell_type")
        src = c.get("source", "")
        if isinstance(src, list):
            src = "".join(src)

        if ctype == "markdown":
            if src.strip():
                parts.append("[MARKDOWN]\n" + src.strip())
        elif ctype == "code":
            # Keep code only if short (for context), prioritize outputs and markdown.
            if isinstance(src, str) and src.strip() and len(src) <= 1200:
                parts.append("[CODE]\n" + src.strip())

            for out in c.get("outputs", []) or []:
                otype = out.get("output_type")
                if otype == "stream":
                    txt = out.get("text", "")
                    if isinstance(txt, list):
                        txt = "".join(txt)
                    if isinstance(txt, str) and txt.strip() and len(txt) <= 2000:
                        parts.append("[STDOUT]\n" + txt.strip())
                else:
                    data_obj = out.get("data", {}) or {}
                    txt = data_obj.get("text/plain") or data_obj.get("text")
                    if isinstance(txt, list):
                        txt = "".join(txt)
                    if isinstance(txt, str) and txt.strip() and len(txt) <= 2000:
                        parts.append("[OUTPUT]\n" + txt.strip())

    return _truncate_middle("\n\n".join(parts), max_chars=max_chars)


def generate_summary_report(reference_dir: Path, *, cfg: Dict[str, Any], llm_cfg: Dict[str, Any], prompts_dir_path: Optional[str] = None):
    """Read artifacts in reference_dir and call LLM to generate a Markdown report."""
    reference_dir = Path(reference_dir).resolve()
    reference_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = (cfg.get("dataset_name") or "dataset").strip()

    # Locate artifacts
    best_ipynb = None
    for p in sorted(reference_dir.glob("BEST_*.ipynb")):
        best_ipynb = p
        break

    idea_json = None
    idea_path = reference_dir / "idea.json"
    if idea_path.exists():
        try:
            idea_json = json.loads(idea_path.read_text(encoding="utf-8"))
        except Exception:
            idea_json = None

    ref_meta = None
    ref_meta_path = reference_dir / "reference.json"
    if ref_meta_path.exists():
        try:
            ref_meta = json.loads(ref_meta_path.read_text(encoding="utf-8"))
        except Exception:
            ref_meta = None

    h5_path = reference_dir / "REFERENCE_DATA.h5"
    h5_summary = _summarize_h5_for_report(h5_path)
    nb_summary = _summarize_notebook_for_report(best_ipynb) if best_ipynb else "(no BEST_*.ipynb found)"

    # Defaults (English)
    default_sections = [
        "Objective",
        "Data Overview",
        "Data Characteristics & Quality",
        "Detailed Analysis (with evidence/metrics/figures highlights)",
        "Key Findings & Interpretability",
        "Reproducible Experiments & Validation Plan",
        "Next Experiment Ideas (>= 8 items)",
        "Risks, Biases & Caveats",
        "Reproduction Checklist (files/commands/seeds)"
    ]
    report_language = "en"
    system_prompt = ""

    # Load report.yml if present
    report_yml_path = None
    report_cfg = {}
    if prompts_dir_path:
        report_yml_path = Path(prompts_dir_path) / "report.yml"
        if report_yml_path.exists():
            try:
                with open(report_yml_path, "r", encoding="utf-8") as f:
                    import yaml
                    report_cfg = yaml.safe_load(f) or {}
                    system_prompt = report_cfg.get("system_prompt", "") or ""
                    # Optional overrides from YAML
                    report_language = (report_cfg.get("language") or report_language)
                    sections_from_yml = report_cfg.get("sections")
                    if isinstance(sections_from_yml, list) and sections_from_yml:
                        default_sections = [str(x) for x in sections_from_yml]
                print(f"🧾 [REPORT] Loaded prompt config: {report_yml_path}", flush=True)
            except Exception as e:
                print(f"⚠️ [REPORT] Failed to load {report_yml_path}: {e}", flush=True)
        else:
            print(f"⚠️ [REPORT] Prompt file not found: {report_yml_path}", flush=True)

    if not system_prompt:
        system_prompt = (
            "You are a senior Cell Painting / omics data scientist and a scientific writer. "
            "Based on the provided reference artifacts, write a reproducible, actionable summary report in Markdown."
        )

    resolved = resolve_llm_config(llm_cfg)

    # Keep the config excerpt small
    cfg_excerpt = {
        "dataset_name": cfg.get("dataset_name"),
        "workflow_phases": cfg.get("workflow_phases"),
        "split_strategy": (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {}).get("split_strategy"),
        "data_path": (((((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {}).get("paths") or {}).get("data")),
    }

    user_payload = {
        "dataset_name": dataset_name,
        "reference_dir": str(reference_dir),
        "reference_meta": ref_meta,
        "h5_summary": h5_summary,
        "notebook_summary": nb_summary,
        "idea_json": idea_json,
        "config_excerpt": cfg_excerpt,
        "instructions": {
            "language": report_language,
            "format": "markdown",
            "sections": default_sections
        }
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            "Use the JSON below as your ONLY source of information. "
            "Return a JSON object that MUST contain the key `report_md` (a Markdown string). "
            "Output JSON ONLY (no extra text).\n\n"
            f"INPUT_JSON:\n{json.dumps(user_payload, ensure_ascii=False)}"
        )}
    ]

    print("🧾 [REPORT] Generating summary report...", flush=True)
    try:
        resp = chat_json(
            messages,
            api_key=resolved["api_key"],
            base_url=resolved["base_url"],
            model=resolved["model"],
            temperature=float(resolved.get("temperature", 0.3)),
            max_tokens=int(resolved.get("max_tokens", 8192)),
        )
        if not isinstance(resp, dict):
            raise RuntimeError("LLM did not return a JSON object")
        report_md = resp.get("report_md") or ""
        if not isinstance(report_md, str) or not report_md.strip():
            raise RuntimeError("report_md missing/empty")

        (reference_dir / "summary_report.md").write_text(report_md, encoding="utf-8")
        (reference_dir / "summary_report.json").write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")
        print("✅ [REPORT] Saved: summary_report.md", flush=True)
    except Exception as e:
        print(f"❌ [REPORT] Failed: {e}", flush=True)


def _ref_cfg(cfg_path: str, prompts_dir_path: Optional[str] = None):
    try:
        cfg = _load_cfg(cfg_path, prompts_dir_path)
        nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
        multi = nb_cfg.get("multi") or {}
        review = multi.get("review") or {}
        ref = review.get("reference") or {}
        return multi, review, ref
    except Exception:
        return {}, {}, {}

def _score_edge_for_reference(attrs: dict, weights: dict) -> float:
    s = attrs.get("expert_heuristic_scores") or {}
    w_sci = float(weights.get("scientific", 1.0))
    w_nov = float(weights.get("novelty", 1.0))
    w_rep = float(weights.get("reproducibility", 1.0))
    w_int = float(weights.get("interpretability", 1.0))

    score = (
        float(s.get("scientific", 0.0)) * w_sci +
        float(s.get("novelty", 0.0)) * w_nov +
        float(s.get("reproducibility", 0.0)) * w_rep +
        float(s.get("interpretability", 0.0)) * w_int
    )
    if attrs.get("executed_sha"): score += 0.05
    if attrs.get("generated_h5_sha"): score += 0.05
    return score

def select_and_export_reference(hypergraph_path: str, cfg_path: str, prompts_dir_path: Optional[str] = None):
    hp = Path(hypergraph_path)
    if not hp.exists(): return

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])
    if not edges: return

    multi, review, ref_cfg = _ref_cfg(cfg_path, prompts_dir_path)
    
    weights = (ref_cfg.get("weights") or {
        "scientific": 1.0, "novelty": 1.0, "reproducibility": 1.0, "interpretability": 1.0
    })
    
    base_out = Path(multi.get("out_dir") or "hypergraph_runs").resolve()
    export_dir = Path(ref_cfg.get("export_dir") or (base_out / "reference")).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    best = None
    best_score = -1e18
    for e in edges:
        sc = _score_edge_for_reference(e.get("attrs", {}), weights)
        if sc > best_score:
            best_score = sc
            best = e

    if not best:
        print("⚠️ [REFERENCE] Could not determine best edge.", flush=True)
        return

    attrs = best.get("attrs", {})
    src_ipynb = attrs.get("executed_ipynb")
    
    if not src_ipynb or not Path(src_ipynb).exists():
        print(f"⚠️ [REFERENCE] Best run ({best.get('edge_id')}) missing executed notebook.", flush=True)
        return

    # Export Notebook
    ref_name = f"BEST_{best.get('edge_id')}" 
    dst_ipynb = export_dir / f"{ref_name}.ipynb"
    shutil.copyfile(src_ipynb, dst_ipynb)
    print(f"📌 [REFERENCE] Exported Best Notebook → {dst_ipynb}", flush=True)

    # Export H5
    src_h5 = attrs.get("generated_h5")
    dst_h5 = None
    if src_h5 and Path(src_h5).exists():
        dst_h5 = export_dir / "REFERENCE_DATA.h5"
        shutil.copyfile(src_h5, dst_h5)
        print(f"📌 [REFERENCE] Exported Best Data → {dst_h5}", flush=True)

    # Save Meta
    ref_meta = {
        "edge_id": best.get("edge_id"),
        "score": best_score,
        "exported_utc": datetime.utcnow().isoformat() + "Z"
    }
    (export_dir / "reference.json").write_text(json.dumps(ref_meta, indent=2), encoding="utf-8")

    # Generate Ideas
    full_cfg = _load_cfg(cfg_path, prompts_dir_path)
    nb_cfg_full = (((full_cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    
    if ref_cfg.get("generate_ideas", True):
        generate_experiment_ideas(
            notebook_path=str(dst_ipynb),
            output_dir=export_dir,
            llm_cfg=nb_cfg_full.get("llm", {}),
            prompts_dir_path=prompts_dir_path
        )

    # Generate Markdown summary report
    if ref_cfg.get("generate_report", True):
        generate_summary_report(
            reference_dir=export_dir,
            cfg=full_cfg,
            llm_cfg=nb_cfg_full.get("llm", {}),
            prompts_dir_path=prompts_dir_path,
        )

if __name__ == "__main__":
    import sys
    THIS_DIR = Path(__file__).parent
    cfg = sys.argv[1] if len(sys.argv) > 1 else str(THIS_DIR / "design_analysis_config.json")
    prompts = str(Path(cfg).parent / "prompts")
    orchestrate(cfg, prompts)