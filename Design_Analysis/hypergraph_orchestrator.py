#!/usr/bin/env python3
# hypergraph_orchestrator.py
# Orchestrate multi-notebook generation (hyperedges), review them, and export a best "reference" notebook.
# Everything is driven by config.json under phases.task_analysis.llm_notebook.multi.*

import json, hashlib, os, shutil, re
from typing import Optional
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# [NEW] Use the centralized config loader
from config_loader import load_app_config

import uuid

# Use the single-run entry from your main program (must exist)
from cellscientist_phase_1 import run_pipeline_basic

from run_llm_nb import (
    collect_cell_errors, apply_edits, execute_notebook,
    build_fix_messages, chat_json
)

# --------------------
# Utilities
# --------------------
def _get_prompts_dir(cfg_path: str) -> str:
    """Helper to find the prompts/ dir, assumed sibling to config file."""
    config_dir = Path(cfg_path).parent
    return str(config_dir / 'prompts') # Assumes prompts/ is in the same dir

def _load_cfg(cfg_path: str, prompts_dir_path: Optional[str] = None) -> Dict[str, Any]:
    # [MODIFIED] Use centralized loader.
    if prompts_dir_path is None:
        prompts_dir_path = _get_prompts_dir(cfg_path)
    return load_app_config(cfg_path, prompts_dir_path)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _hash_file(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# [START NEW FUNCTION]
def _resolve_llm_cfg_for_autofix(llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    [ROBUST-FIX] Helper to correctly resolve LLM config for auto-fix.
    This logic is duplicated from llm_notebook_runner.py to avoid import issues
    and ensure the auto-fix loop uses the exact same credentials.
    """
    llm = llm_cfg or {}
    
    # 1. Resolve API Key (Priority: config key > config env var > default env var)
    api_key = llm.get("api_key") # 1. 优先使用 config 中写死的 api_key
    if not api_key:
        api_key_env = llm.get("api_key_env") or "OPENAI_API_KEY" # 2. 其次使用 config 指向的环境变量
        api_key = os.environ.get(api_key_env)
        
    # 2. Resolve Model
    model = llm.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    
    # 3. Resolve Base URL (Priority: provider file > config env var > default env var > hardcoded fallback)
    base_url = None
    
    # 3a. Try provider file (if llm_providers.json exists)
    try:
        # __file__ is hypergraph_orchestrator.py. parents[1] is .../cellscientist
        base_dir = Path(__file__).resolve().parents[1] 
        prov_file = base_dir / "llm_providers.json"
        prov_name = (llm.get("provider") or "").strip() or None
        if prov_name and prov_file.exists():
            data = json.loads(prov_file.read_text(encoding="utf-8"))
            prov = data.get("providers", {}).get(prov_name)
            if prov:
                base_url = prov.get("base_url") or None
                if not llm.get("model") and prov.get("models"): # Only use provider model if not set in config
                    model = prov.get("models")[0]
    except Exception:
        pass # Ignore provider file errors
    
    # 3b. Fallback to env vars
    if not base_url:
        base_url_env = llm.get("base_url_env")
        if base_url_env and os.environ.get(base_url_env):
            base_url = os.environ.get(base_url_env)
        else:
            # Default env var or hardcoded fallback from llm_notebook_runner
            base_url = os.environ.get("OPENAI_BASE_URL") or "[https://vip.yi-zhan.top/v1](https://vip.yi-zhan.top/v1)"
    
    # 3c. Clean up hardcoded markdown link if it's used
    if base_url and base_url.startswith("[") and base_url.endswith(")"):
        base_url = re.sub(r"\[(.*?)\]\((.*?)\)", r"\2", base_url)
        
    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }
# [END NEW FUNCTION]


# --------------------
# Per-run config builder
# --------------------
def _mk_run_prompt(base_cfg: Dict[str, Any], idx: int, prompt_variant: str, seed: Optional[int]):
    import copy
    cfg = copy.deepcopy(base_cfg)
    # [MODIFIED] Access nb_cfg via path
    nb_cfg = cfg["phases"]["task_analysis"]["llm_notebook"]
    paths = nb_cfg.get("paths", {})
    multi = nb_cfg.get("multi", {})

    out_dir = Path(multi.get("out_dir") or Path(paths.get("out") or "/mnt/data/").parent / "hypergraph_runs")
    _ensure_dir(out_dir)

    name_template = multi.get("name_template") or "NB{idx:02d}"
    run_name = name_template.format(idx=idx+1, seed=seed if seed is not None else 0)
    run_dir = out_dir / run_name
    _ensure_dir(run_dir)

    # [MODIFIED] Inject focus instruction instead of appending to prompt
    # base_prompt remains clean
    base_prompt = (cfg.get('prompts', {}).get('notebook_generation', {}).get('user_prompt'))
    nb_cfg["prompt"] = base_prompt # Set the base user prompt
    
    # [NEW] Store the variant as a separate "focus_instruction"
    # This will be read by llm_notebook_runner
    nb_cfg["focus_instruction"] = prompt_variant or "Standard analysis. No special focus."
    
    # distinct outputs
    nb_cfg.setdefault("paths", {})
    nb_cfg["paths"]["out"] = str(run_dir / "CP_llm.ipynb")
    nb_cfg["paths"]["out_exec"] = str(run_dir / "CP_llm_executed.ipynb")
    
    # [NEW] Define unique H5 output path for this run
    nb_cfg["paths"]["h5_out"] = str(run_dir / "preprocessed_data.h5")

    # optional seed passthrough
    llm = nb_cfg.get("llm", {}) or {}
    if seed is not None:
        llm["seed"] = seed
        nb_cfg["llm"] = llm
        
    # [MODIFIED] Ensure the full prompts dict is passed along in the temp config
    cfg["prompts"] = base_cfg.get("prompts", {})

    return cfg, run_name, run_dir

# --------------------
# Orchestrate (multi-run)
# --------------------
def orchestrate(cfg_path: str, prompts_dir_path: Optional[str] = None):
    """Generate/execute multiple notebooks and emit hypergraph.json.
       If multi.review.reference.enabled=true, run calculate_run_heuristics() then select_and_export_reference().
    """
    cfg = _load_cfg(cfg_path, prompts_dir_path) # [MODIFIED] _load_cfg now loads prompts too
    nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    multi_cfg = nb_cfg.get("multi") or {}
    if not multi_cfg.get("enabled", False):
        raise SystemExit("Multi-run disabled. Set phases.task_analysis.llm_notebook.multi.enabled=true")

    # [MODIFIED] ### Adaptive Run Logic (Using MIN as requested) ###
    prompt_variants = multi_cfg.get("prompt_variants") or []
    seeds = multi_cfg.get("seeds") or []
    
    # 运行次数将是 num_runs、seeds 长度、variants 长度三者中的最小值
    # 这实现了您要的 "自适应调节"
    num_runs_config = int(multi_cfg.get("num_runs", 1)) # 1 is fallback if key missing
    
    # [USER REQUEST] Set num_runs to the minimum of the three values
    # Note: If lists are empty, len() is 0, min() will be 0 (if num_runs_config=0) or 1.
    if not seeds or not prompt_variants:
         num_runs = 0 # No runs possible if lists are empty
         print(f"⚠️  [Adaptive] Seeds or Variants list empty. Setting runs to 0.")
    else:
        # Only apply min if lists are populated
        num_runs = min(num_runs_config, len(seeds), len(prompt_variants))
    
    print(f"ℹ️  [Adaptive] Runs logic: min(config_num_runs={num_runs_config}, seeds={len(seeds)}, variants={len(prompt_variants)}) = {num_runs} runs")
    # ### End Adaptive Run Logic ###


    node_names = multi_cfg.get("node_names") or [
        "Data Loading & Initial Exploration",
        "Data Patterns",
        "Hidden Information",
        "Innovation Motivation",
        "Experiment & Validation Suggestions"
    ]
    hypergraph_name = multi_cfg.get("hypergraph_name") or "CellForge-HyperGraph"

    edges: List[Dict[str, Any]] = []
    t0 = datetime.utcnow().isoformat() + "Z"

    for i in range(num_runs):
        # [MODIFIED] Use direct indexing (i) because num_runs is now guaranteed to be <= len
        variant = prompt_variants[i]
        seed = seeds[i]
        
        run_cfg, run_name, run_dir = _mk_run_prompt(cfg, i, variant, seed)

        # write a temp cfg for this run
        tmp_cfg = run_dir / "config.run.json"
        # [MODIFIED] We must write the full config (including injected prompts)
        # to the temp file so the runner can load it.
        tmp_cfg.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"🚀 [{i+1}/{num_runs}] Generating+executing: {run_name} (Seed: {seed})")
        # [MODIFIED] run_pipeline_basic will handle loading config + prompts
        executed_path = run_pipeline_basic(str(tmp_cfg), phase_name="task_analysis")
        
        # [AUTO-FIX per-run]
        # [MODIFIED] Pass the full run_cfg (which contains prompts)
        final_exec = _auto_fix_notebook(executed_path, run_cfg)
        exec_p = Path(final_exec)
        unexec_p = Path(run_cfg["phases"]["task_analysis"]["llm_notebook"]["paths"]["out"])
        
        # [NEW] Get path to the H5 file this run generated
        h5_p = Path(run_cfg["phases"]["task_analysis"]["llm_notebook"]["paths"]["h5_out"])

        edges.append({
            "edge_id": run_name,
            "executed_ipynb": str(exec_p),
            "unexecuted_ipynb": str(unexec_p),
            "generated_h5": str(h5_p) if h5_p.exists() else None, # [NEW] Store H5 path
            "executed_sha": _hash_file(exec_p),
            "unexecuted_sha": _hash_file(unexec_p),
            "generated_h5_sha": _hash_file(h5_p), # [NEW] Store H5 hash
            "prompt_variant": variant,
            "llm_model": (run_cfg["phases"]["task_analysis"]["llm_notebook"].get("llm") or {}).get("model"),
            "seed": (run_cfg["phases"]["task_analysis"]["llm_notebook"].get("llm") or {}).get("seed"),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        })

    nodes = [{"node_id": f"step-{k+1}", "name": n} for k, n in enumerate(node_names)]
    hyperedges = [{
        "edge_id": e["edge_id"],
        "connects": [n["node_id"] for n in nodes],
        "attrs": e
    } for e in edges]

    out_dir = Path((nb_cfg.get("multi") or {}).get("out_dir") or "/mnt/data/hypergraph_runs")
    _ensure_dir(out_dir)
    hypergraph = {
        "name": hypergraph_name,
        "created_utc": t0,
        "nodes": nodes,
        "hyperedges": hyperedges,
        "version": "T0-alpha"
    }
    hp_path = out_dir / "hypergraph.json"
    hp_path.write_text(json.dumps(hypergraph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Hypergraph emitted → {hp_path}")
    print("🎯 Orchestration done.")

    # ---- [MODIFIED] Heuristic calculation then reference export
    review_cfg = (multi_cfg.get("review") or {})
    ref_cfg = (review_cfg.get("reference") or {})

    if ref_cfg.get("enabled", False):
        print(f"🧪 [HEURISTICS] Calculating notebook heuristics...")
        # [MODIFIED] Pass prompts_dir_path
        calculate_run_heuristics(str(hp_path), cfg_path=cfg_path, prompts_dir_path=prompts_dir_path)
        
        print(f"🏆 [REFERENCE] Selecting best notebook...")
        select_and_export_reference(str(hp_path), cfg_path, prompts_dir_path=prompts_dir_path)
    else:
        print(f"ℹ️  Reference selection disabled. Skipping heuristic calculation and export.")


# ============================
# Heuristic Calculation
# ============================
import nbformat as _nbf
from nbformat.reader import reads as _nb_reads

def _safe_read_text(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _extract_nb_cells(ipynb_path: str):
    try:
        nb = _nbf.read(ipynb_path, as_version=4)
        return nb.get("cells", [])
    except Exception:
        s = _safe_read_text(ipynb_path)
        try:
            nb = _nb_reads(s, as_version=4)
            return nb.get("cells", [])
        except Exception:
            return []

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

# [REMOVED] _llm_critique function removed.

def calculate_run_heuristics(
    hypergraph_path: str, 
    cfg_path: Optional[str] = None,
    prompts_dir_path: Optional[str] = None # [MODIFIED]
) -> None:
    """
    [MODIFIED] This function now only calculates heuristic scores based on
    notebook content and saves them to the hypergraph.json.
    It no longer performs LLM-based review or makes decisions.
    """
    hp = Path(hypergraph_path)
    if not hp.exists():
        raise FileNotFoundError(f"hypergraph not found: {hp}")

    # [MODIFIED] No LLM review config loading needed.
    
    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])

    for e in edges:
        attrs = e.get("attrs", {})
        ip = attrs.get("executed_ipynb")
        cells = _extract_nb_cells(ip) if ip else []
        scores = _heuristic_scores_from_cells(cells)
        
        # [MODIFIED] Only store heuristic scores. Removed LLM critique.
        attrs["expert_heuristic_scores"] = scores
        e["attrs"] = attrs

    # [MODIFIED] Removed all aggregation, decision, and threshold logic.
    # Just write the scores back to the file.
    data["heuristics_calculated_utc"] = datetime.utcnow().isoformat() + "Z"

    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧠 [HEURISTICS] Heuristic scores calculated and saved.")
    # [MODIFIED] No return value.

# ============================
# Reference selection & export
# ============================
def _ref_cfg(cfg_path: str, prompts_dir_path: Optional[str] = None): # [MODIFIED]
    try:
        # [MODIFIED] _load_cfg gets the config with prompts
        cfg = _load_cfg(cfg_path, prompts_dir_path)
        nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
        multi = nb_cfg.get("multi") or {}
        review = multi.get("review") or {}
        ref = review.get("reference") or {}
        return multi, review, ref
    except Exception:
        return {}, {}, {}

def _score_edge_for_reference(attrs: dict, weights: dict) -> float:
    # [MODIFIED] Read from new 'expert_heuristic_scores' key
    s = attrs.get("expert_heuristic_scores") or {}
    sci = float(s.get("scientific", 0.0))
    nov = float(s.get("novelty", 0.0))
    rep = float(s.get("reproducibility", 0.0))
    intp = float(s.get("interpretability", 0.0))

    w_sci = float(weights.get("scientific", 1.0))
    w_nov = float(weights.get("novelty", 1.0))
    w_rep = float(weights.get("reproducibility", 1.0))
    w_int = float(weights.get("interpretability", 1.0))

    base = sci*w_sci + nov*w_nov + rep*w_rep + intp*w_int
    if attrs.get("executed_sha"):
        base += 0.05
    if attrs.get("generated_h5_sha"): # [NEW] Bonus if H5 was created
        base += 0.05
    pv = (attrs.get("prompt_variant") or "").lower()
    if "enrichment" in pv or "volcano" in pv:
        base += 0.02
    return base

def select_and_export_reference(
    hypergraph_path: str, 
    cfg_path: str, 
    prompts_dir_path: Optional[str] = None # [MODIFIED]
):
    hp = Path(hypergraph_path)
    if not hp.exists():
        print("[REFERENCE] hypergraph not found; skip")
        return None

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])
    if not edges:
        print("[REFERENCE] no edges; skip")
        return None

    # [MODIFIED]
    multi, review, ref_cfg = _ref_cfg(cfg_path, prompts_dir_path)
    if not (ref_cfg.get("enabled", False)):
        print("[REFERENCE] disabled in config; skip")
        return None

    weights = (ref_cfg.get("weights") or {
        "scientific": 1.0, "novelty": 1.0, "reproducibility": 1.0, "interpretability": 1.0
    })
    export_dir = Path(ref_cfg.get("export_dir") or (Path(multi.get("out_dir") or "/mnt/data/hypergraph_runs") / "reference"))
    export_dir.mkdir(parents=True, exist_ok=True)

    # Pick best edge by score
    best = None
    best_score = -1e18
    for e in edges:
        sc = _score_edge_for_reference(e.get("attrs", {}), weights)
        if sc > best_score:
            best_score = sc
            best = e

    if not best:
        print("[REFERENCE] could not determine best edge")
        return None

    attrs = best.get("attrs", {})
    src_ipynb = attrs.get("executed_ipynb") or attrs.get("unexecuted_ipynb")
    if not src_ipynb or not Path(src_ipynb).exists():
        print("[REFERENCE] best edge has no valid notebook path")
        return None

    # --- [NEW] Copy BEST Notebook ---
    ref_name = f"REFERENCE_{best.get('edge_id')}"
    dst_ipynb = export_dir / f"{ref_name}.ipynb"
    try:
        shutil.copyfile(src_ipynb, dst_ipynb)
        print(f"📌 [REFERENCE] exported Notebook → {dst_ipynb}")
    except Exception as e:
        print("[REFERENCE] Notebook copy failed:", e)
        return None
        
    # --- [NEW] Copy BEST H5 File ---
    src_h5 = attrs.get("generated_h5")
    dst_h5 = None
    if src_h5 and Path(src_h5).exists():
        dst_h5 = export_dir / "REFERENCE_DATA.h5"
        try:
            shutil.copyfile(src_h5, dst_h5)
            print(f"📌 [REFERENCE] exported H5 Data → {dst_h5}")
        except Exception as e:
            print("[REFERENCE] H5 copy failed:", e)
            dst_h5 = None # Failed copy
    else:
        print(f"⚠️ [REFERENCE] No H5 file found for best run at: {src_h5}")


    # write reference metadata
    ref_meta = {
        "edge_id": best.get("edge_id"),
        "score": best_score,
        "executed_ipynb": attrs.get("executed_ipynb"),
        "unexecuted_ipynb": attrs.get("unexecuted_ipynb"),
        "generated_h5": attrs.get("generated_h5"),
        "exported_notebook_path": str(dst_ipynb),
        "exported_h5_path": str(dst_h5) if dst_h5 else None,
        "weights": weights,
        "generated_utc": datetime.utcnow().isoformat() + "Z"
    }
    (export_dir / "reference.json").write_text(json.dumps(ref_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    data["reference"] = ref_meta # [MODIFIED] Store full metadata
    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data["reference"]

# ============================
# Closed loop (optional)
# ============================

# [REMOVED] closed_loop_orchestrate function removed as it depends on review decisions.

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    import sys
    # [MODIFIED] Default config path and prompts path updated to root
    THIS_DIR = Path(__file__).parent
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(THIS_DIR / "design_analysis_config.json")
    prompts_path = str(THIS_DIR / "prompts")
    if len(sys.argv) > 1:
        # Guess prompts dir is relative to config's parent dir
        prompts_path = str(Path(cfg_path).parent / "prompts")
        
    orchestrate(cfg_path, prompts_path)  # single orchestration (auto-review + reference export depend on config)


# =============== [META] helpers ===============
def _jaccard_overlap(a: list, b: list) -> float:
    try:
        sa, sb = set([x.strip().lower() for x in a or []]), set([x.strip().lower() for x in b or []])
        if not sa and not sb: 
            return 1.0
        if not sa and sb: 
            return 0.0
        if sa and not sb: 
            return 0.0
        inter = len(sa & sb)
        uni = len(sa | sb)
        return inter / uni if uni else 1.0
    except Exception:
        return 0.0


# =============== [SEMANTIC] extraction ===============
# [NOTE] This function is uncalled, but harmless. Left as-is.
def _extract_semantic_layer(
    cells,
    export_dir: Optional[Path] = None,
    edge_id: Optional[str] = None,
    enable_dot: bool = True
    ) -> dict:
    """
    Very lightweight semantic relation extraction from markdown/code text.
    Heuristics: detect phrases like "effect of X on Y", "increase/decrease", "correlation between A and B".
    Outputs a graph: {"nodes": [...], "edges": [{"src","tgt","rel","evidence"}]}
    Also writes a .json and .dot file if export_dir provided.
    """
    text_blocks = []
    for i, c in enumerate(cells or []):
        src = c.get("source", "")
        if isinstance(src, str):
            text_blocks.append((i, src))
    nodes = set()
    edges = []

    patt_effect = re.compile(r"(?:effect|impact|influence)\s+of\s+([A-Za-z0-9_./-]+)\s+on\s+([A-Za-z0-9_./-]+)", re.I)
    patt_corr   = re.compile(r"(?:correlation|association)\s+(?:between|of)\s+([A-Za-z0-9_./-]+)\s+(?:and|&)\s+([A-Za-z0-9_./-]+)", re.I)
    patt_arrow  = re.compile(r"([A_Za-z0-9_./-]+)\s*->\s*([A_Za-z0-9_./-]+)")

    for idx, txt in text_blocks:
        for m in patt_effect.finditer(txt):
            a, b = m.group(1), m.group(2)
            nodes.update([a, b]); edges.append({"src": a, "tgt": b, "rel": "effect", "evidence_cell": idx})
        for m in patt_corr.finditer(txt):
            a, b = m.group(1), m.group(2)
            nodes.update([a, b]); edges.append({"src": a, "tgt": b, "rel": "correlates", "evidence_cell": idx})
        for m in patt_arrow.finditer(txt):
            a, b = m.group(1), m.group(2)
            nodes.update([a, b]); edges.append({"src": a, "tgt": b, "rel": "arrow", "evidence_cell": idx})

        # polarity cues
        if "increase" in txt.lower() or "upregulat" in txt.lower():
            edges.append({"src": "UNKNOWN", "tgt": "UNKNOWN", "rel": "increase", "evidence_cell": idx})
        if "decrease" in txt.lower() or "downregulat" in txt.lower():
            edges.append({"src": "UNKNOWN", "tgt": "UNKNOWN", "rel": "decrease", "evidence_cell": idx})

    graph = {
        "nodes": sorted(list(nodes)),
        "edges": edges
    }

    if export_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        eid = edge_id or "graph"
        (export_dir / f"{eid}.semantic.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        if enable_dot:
            lines = ["digraph G {"]
            for n in sorted(list(nodes)):
                lines.append(f'  "{n}";')
            for e in edges:
                lines.append(f'  "{e.get("src")}" -> "{e.get("tgt")}" [label="{e.get("rel")}"];')
            lines.append("}")
            (export_dir / f"{eid}.semantic.dot").write_text("\n".join(lines), encoding="utf-8")
    return graph


def _auto_fix_notebook(executed_path: str, run_cfg: dict) -> str:
    """Run the auto-fix loop for one executed notebook and return the final executed path."""
    import nbformat as nbf
    from pathlib import Path
    nb_cfg = (((run_cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    exec_cfg = nb_cfg.get("exec", {}) or {}

    
    # [START MODIFICATION]
    # [ROBUST-FIX] Correctly load LLM config for auto-fix by
    # replicating the logic from the llm_notebook_runner.
    llm_cfg = nb_cfg.get("llm", {}) or {}
    resolved_llm_cfg = _resolve_llm_cfg_for_autofix(llm_cfg)
    
    api_key_final = resolved_llm_cfg["api_key"]
    base_url_final = resolved_llm_cfg["base_url"]
    model_final = resolved_llm_cfg["model"]
    
    max_tokens_final = int(llm_cfg.get("max_tokens") or 10240)
    
    if not api_key_final:
        print("⚠️ [AUTO-FIX] API key not resolved. Auto-fix will likely fail.")
    else:
        # Mask the key for logging
        print(f"ℹ️ [AUTO-FIX] Config loaded: model={model_final}, base_url={base_url_final}, api_key=...{api_key_final[-4:]}")
    
    print(f"ℹ️ [AUTO-FIX] Using max_tokens for fix loop: {max_tokens_final}")
    # [END MODIFICATION]


    nb_path = Path(executed_path)
    if not nb_path.exists():
        print(f"⚠️ [AUTO-FIX] executed notebook not found: {nb_path}")
        return executed_path

    nb = nbf.read(str(nb_path), as_version=4)
    errors = collect_cell_errors(nb)
    max_fix_rounds = int(exec_cfg.get("max_fix_rounds", 0))
    print(f"🔁 [AUTO-FIX] rounds={max_fix_rounds} | base={nb_path.name} | errors={len(errors)}")

    if not errors or max_fix_rounds <= 0:
        return str(nb_path)

    language = (nb_cfg.get("language") or {}).get("name") or "English"
    paths = nb_cfg.get("paths", {}) or {}
    data_path = paths.get("data") or ""
    csv_preview = None
    paper_excerpt = None
    
    # [MODIFIED] Get the autofix prompt from the full cfg['prompts'] dict
    autofix_prompt = (run_cfg.get('prompts', {}).get('autofix', {}).get('system_prompt'))
    if not autofix_prompt:
        autofix_prompt = (
            "You are a senior Python engineer and Jupyter expert.\n"
            "Given Jupyter cell errors, return ONLY a MINIFIED JSON object with key 'edits'.\n"
            "Each edit MUST be {\"cell_index\": int, \"source\": str} replacing the WHOLE code cell.\n"
            "You MUST cover ALL indices listed in 'target_cell_indices' (one edit per index). Do not add new cells.\n"
            "Do not modify markdown cells. No markdown fences or extra text.\n"
        ) # Fallback just in case
        print("Warning: Autofix prompt not found in config, using fallback.")

    round_idx = 0
    while errors and round_idx < max_fix_rounds:
        round_idx += 1
        print(f"🛠  [AUTO-FIX] round {round_idx}: targets={[e['cell_index'] for e in errors]}")

        # [MODIFIED] Pass the loaded autofix_prompt to build_fix_messages
        messages = build_fix_messages(
            language, data_path, csv_preview, paper_excerpt, errors,
            nb_cfg.get("headings") or ["Introduction", "Methods", "Results"],
            autofix_prompt_str=autofix_prompt
        )
        
        # [MODIFICATION] Call chat_json with the correctly resolved config
        spec = chat_json(
            messages,
            api_key=api_key_final,
            base_url=base_url_final,
            model=model_final,
            temperature=0.0,
            max_tokens=max_tokens_final # [FIXED] 使用 max_tokens_final
        )
        edits = spec.get("edits") or []
        if not edits:
            print("ℹ️  [AUTO-FIX] no edits from LLM; stopping.")
            break

        # apply edits only to error-target cells
        target_set = {e["cell_index"] for e in errors}
        valid_edits = []
        for ed in edits:
            try:
                idx = int(ed.get("cell_index"))
                if idx in target_set and 0 <= idx < len(nb["cells"]) and nb["cells"][idx].get("cell_type") == "code":
                    valid_edits.append({"cell_index": idx, "source": ed.get("source") or ""})
            except Exception:
                continue

        changed = apply_edits(nb, valid_edits)
        print(f"✅ [AUTO-FIX] patched={sorted([v['cell_index'] for v in valid_edits])} | changed={changed}")
        if changed == 0:
            print("ℹ️  [AUTO-FIX] no cell changed; stopping.")
            break

        # save patched & re-execute
        patched_unexec = nb_path.with_name(nb_path.stem + f"_r{round_idx}_patched.ipynb")
        nbf.write(nb, str(patched_unexec))
        nb, _ = execute_notebook(
            nb,
            timeout=int(exec_cfg.get("timeout_seconds", 1800)),
            allow_errors=bool(exec_cfg.get("allow_errors", True))
        )
        round_exec = nb_path.with_name(nb_path.stem + f"_r{round_idx}_executed.ipynb")
        nbf.write(nb, str(round_exec))

        errors = collect_cell_errors(nb)
        print(f"🔎 [AUTO-FIX] post-exec errors={len(errors)}")

    final = str(nb_path if round_idx == 0 else nb_path.with_name(nb_path.stem + f"_r{round_idx}_executed.ipynb"))
    print(f"🏁 [AUTO-FIX] final={final}")
    
    # 🔧 OVERWRITE ORIGINAL executed notebook with the fixed one (T0 <- final)
    try:
        if final != str(nb_path) and Path(final).exists():
            from shutil import copyfile
            copyfile(final, str(nb_path))
            print(f"✅ [AUTO-FIX] Overwrote original executed notebook: {nb_path.name}")
    except Exception as e:
        print(f"⚠️ [AUTO-FIX] Failed to overwrite original executed notebook: {e}")
    return str(nb_path)
