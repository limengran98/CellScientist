#!/usr/bin/env python3
# hypergraph_orchestrator.py
# Orchestrate multi-notebook generation (hyperedges), review them, and export a best "reference" notebook.
# Everything is driven by config.json under phases.task_analysis.llm_notebook.multi.*

import json, hashlib, os, shutil
from typing import Optional
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

def _resolve_placeholders(cfg: dict) -> dict:
    import re as _re
    ds = cfg.get("dataset_name", "default_dataset")
    def _subst(v):
        if isinstance(v, str):
            return _re.sub(r"\$\{dataset_name\}", ds, v)
        if isinstance(v, dict):
            return {k: _subst(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_subst(x) for x in v]
        return v
    return _subst(cfg)

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
def _load_cfg(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return _resolve_placeholders(cfg)
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

# --------------------
# Per-run config builder
# --------------------
def _mk_run_prompt(base_cfg: Dict[str, Any], idx: int, prompt_variant: str):
    import copy
    cfg = copy.deepcopy(base_cfg)
    nb_cfg = cfg["phases"]["task_analysis"]["llm_notebook"]
    paths = nb_cfg.get("paths", {})
    multi = nb_cfg.get("multi", {})

    out_dir = Path(multi.get("out_dir") or Path(paths.get("out") or "/mnt/data/").parent / "hypergraph_runs")
    _ensure_dir(out_dir)

    seeds = multi.get("seeds") or []
    seed = seeds[idx] if idx < len(seeds) else None

    name_template = multi.get("name_template") or "NB{idx:02d}"
    run_name = name_template.format(idx=idx+1, seed=seed if seed is not None else 0)
    run_dir = out_dir / run_name
    _ensure_dir(run_dir)

    # augment prompt
    base_prompt = nb_cfg.get("prompt") or ""
    extra = f"\\n\\n[Hyperedge Variant {idx+1}] {prompt_variant}".strip() if prompt_variant else ""
    nb_cfg["prompt"] = (base_prompt + extra).strip()

    # distinct outputs
    nb_cfg.setdefault("paths", {})
    nb_cfg["paths"]["out"] = str(run_dir / "CP_llm.ipynb")
    nb_cfg["paths"]["out_exec"] = str(run_dir / "CP_llm_executed.ipynb")

    # optional seed passthrough
    llm = nb_cfg.get("llm", {}) or {}
    if seed is not None:
        llm["seed"] = seed
        nb_cfg["llm"] = llm

    return cfg, run_name, run_dir

# --------------------
# Orchestrate (multi-run)
# --------------------
def orchestrate(cfg_path: str):
    """Generate/execute multiple notebooks and emit hypergraph.json.
       If multi.review.enabled=true, run expert_review_hypergraph() then select_and_export_reference().
    """
    cfg = _load_cfg(cfg_path)
    nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    multi = nb_cfg.get("multi") or {}
    if not multi.get("enabled", False):
        raise SystemExit("Multi-run disabled. Set phases.task_analysis.llm_notebook.multi.enabled=true")

    prompt_variants = multi.get("prompt_variants") or []
    num_runs = int(multi.get("num_runs", max(1, len(prompt_variants) or 1)))
    node_names = multi.get("node_names") or [
        "Data Loading & Initial Exploration",
        "Data Patterns",
        "Hidden Information",
        "Innovation Motivation",
        "Experiment & Validation Suggestions"
    ]
    hypergraph_name = multi.get("hypergraph_name") or "CellForge-HyperGraph"

    edges: List[Dict[str, Any]] = []
    t0 = datetime.utcnow().isoformat() + "Z"

    for i in range(num_runs):
        variant = prompt_variants[i] if i < len(prompt_variants) else ""
        run_cfg, run_name, run_dir = _mk_run_prompt(cfg, i, variant)

        # write a temp cfg for this run
        tmp_cfg = run_dir / "config.run.json"
        tmp_cfg.write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"🚀 [{i+1}/{num_runs}] Generating+executing: {run_name}")
        executed_path = run_pipeline_basic(str(tmp_cfg), phase_name="task_analysis")
        # [AUTO-FIX per-run]
        final_exec = _auto_fix_notebook(executed_path, run_cfg)
        exec_p = Path(final_exec)
        unexec_p = Path(run_cfg["phases"]["task_analysis"]["llm_notebook"]["paths"]["out"])

        edges.append({
            "edge_id": run_name,
            "executed_ipynb": str(exec_p),
            "unexecuted_ipynb": str(unexec_p),
            "executed_sha": _hash_file(exec_p),
            "unexecuted_sha": _hash_file(unexec_p),
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

    # ---- Auto-review then reference export
    review_cfg = (multi.get("review") or {})
    if review_cfg.get("enabled"):
        threshold = float(review_cfg.get("threshold", 0.7))
        print(f"🧪 [REVIEW] Auto-review enabled. threshold={threshold}")
        expert_review_hypergraph(str(hp_path), threshold=threshold, cfg_path=cfg_path)
        # Then reference export if configured
        select_and_export_reference(str(hp_path), cfg_path)

# ============================
# Expert Agent (review)
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

def _llm_critique(edge_meta: dict, cells, llm_cfg: Optional[dict] = None) -> dict:
    # Placeholder (plug your own client using llm_cfg)
    return {
        "pros": ["Clear pipeline structure detected"] if cells else [],
        "cons": [] if cells else ["Notebook empty or failed to load"],
        "risks": ["Potential data leakage if splits are not stratified"],
        "suggestions": [
            "Report effect sizes alongside p-values",
            "Add seed control and environment versions for reproducibility",
            "Include volcano/box/violin plots where appropriate",
            "Discuss limitations and potential confounders"
        ],
        "llm_used": bool(llm_cfg),
        "llm_cfg": llm_cfg or {}
    }

def expert_review_hypergraph(hypergraph_path: str, threshold: float = 0.7, cfg_path: Optional[str] = None) -> dict:
    hp = Path(hypergraph_path)
    if not hp.exists():
        raise FileNotFoundError(f"hypergraph not found: {hp}")

    # Load review LLM cfg if present
    review_llm_cfg = None
    if cfg_path:
        try:
            _cfg_all = _load_cfg(cfg_path)
            _nb_cfg = (((_cfg_all.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
            _review = (_nb_cfg.get("multi") or {}).get("review") or {}
            review_llm_cfg = _review.get("llm") or None
        except Exception:
            review_llm_cfg = None

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])

    study_scores = []
    for e in edges:
        attrs = e.get("attrs", {})
        ip = attrs.get("executed_ipynb")
        cells = _extract_nb_cells(ip) if ip else []
        scores = _heuristic_scores_from_cells(cells)
        critique = _llm_critique(attrs, cells, review_llm_cfg)

        attrs["expert"] = {"scores": scores, "critique": critique}
        e["attrs"] = attrs
        study_scores.append(scores)

    # aggregate
    def avg(key):
        vals = [s.get(key,0.0) for s in study_scores] or [0.0]
        return sum(vals)/len(vals)

    summary = {
        "mean_scores": {
            "scientific": avg("scientific"),
            "novelty": avg("novelty"),
            "reproducibility": avg("reproducibility"),
            "interpretability": avg("interpretability")
        }
    }
    overall = sum(summary["mean_scores"].values())/4.0
    decision = "stop" if overall >= threshold else "continue"
    directives = [] if decision == "stop" else [
        "Increase statistical rigor (ANOVA / nonparam tests)",
        "Add seed controls and record library versions",
        "Enrich visualizations (volcano, violin, heatmap)",
        "Write a concise Discussion section (limitations/confounders)"
    ]

    data["expert_review"] = {
        "threshold": threshold,
        "overall": overall,
        "decision": decision,
        "directives": directives,
        "reviewed_utc": datetime.utcnow().isoformat() + "Z"
    }

    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧠 [REVIEW] Review complete. Decision: {decision}, overall={overall:.3f}")
    return data["expert_review"]

# ============================
# Reference selection & export
# ============================
def _ref_cfg(cfg_path: str):
    try:
        cfg = _load_cfg(cfg_path)
        nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
        multi = nb_cfg.get("multi") or {}
        review = multi.get("review") or {}
        ref = review.get("reference") or {}
        return multi, review, ref
    except Exception:
        return {}, {}, {}

def _score_edge_for_reference(attrs: dict, weights: dict) -> float:
    expert = attrs.get("expert") or {}
    s = (expert.get("scores") or {})
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
    pv = (attrs.get("prompt_variant") or "").lower()
    if "enrichment" in pv or "volcano" in pv:
        base += 0.02
    return base

def select_and_export_reference(hypergraph_path: str, cfg_path: str):
    hp = Path(hypergraph_path)
    if not hp.exists():
        print("[REFERENCE] hypergraph not found; skip")
        return None

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])
    if not edges:
        print("[REFERENCE] no edges; skip")
        return None

    multi, review, ref_cfg = _ref_cfg(cfg_path)
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

    ref_name = f"REFERENCE_{best.get('edge_id')}"
    dst_ipynb = export_dir / f"{ref_name}.ipynb"
    try:
        shutil.copyfile(src_ipynb, dst_ipynb)
        print(f"📌 [REFERENCE] exported → {dst_ipynb}")
    except Exception as e:
        print("[REFERENCE] copy failed:", e)
        return None

    # write reference metadata
    ref_meta = {
        "edge_id": best.get("edge_id"),
        "score": best_score,
        "executed_ipynb": attrs.get("executed_ipynb"),
        "unexecuted_ipynb": attrs.get("unexecuted_ipynb"),
        "weights": weights,
        "generated_utc": datetime.utcnow().isoformat() + "Z"
    }
    (export_dir / "reference.json").write_text(json.dumps(ref_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    data["reference"] = {"edge_id": best.get("edge_id"), "score": best_score, "path": str(dst_ipynb), "export_dir": str(export_dir)}
    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data["reference"]

# ============================
# Closed loop (optional)
# ============================
def closed_loop_orchestrate(cfg_path: str, max_cycles: int = 1):
    """Run orchestrate → review → if continue, append prompt variants from directives and repeat;
       After final review, export a reference notebook if configured.
    """
    cfg = _load_cfg(cfg_path)
    nb_cfg = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    multi = nb_cfg.get("multi") or {}
    review = (multi.get("review") or {})
    threshold = float(review.get("threshold", 0.7))
    per_cycle_runs = int(review.get("per_cycle_runs", 2))

    # First batch
    orchestrate(cfg_path)
    out_dir = (nb_cfg.get("multi") or {}).get("out_dir") or "/mnt/data/hypergraph_runs"
    hp = Path(out_dir) / "hypergraph.json"
    rv = expert_review_hypergraph(str(hp), threshold=threshold, cfg_path=cfg_path)

    cycles = 0
    while rv.get("decision") == "continue" and cycles < max_cycles:
        cycles += 1
        directives = rv.get("directives") or []
        add_variants = [f"[ExpertCycle{cycles}] {d}" for d in directives][:per_cycle_runs]
        nb_cfg.setdefault("multi", {}).setdefault("prompt_variants", [])
        nb_cfg["multi"]["prompt_variants"].extend(add_variants)
        nb_cfg["multi"]["num_runs"] = len(nb_cfg["multi"]["prompt_variants"])

        # write a temp cfg (in out_dir) and run again
        tmp_cfg = Path(out_dir) / f"config.cycle{cycles}.json"
        cfg["phases"]["task_analysis"]["llm_notebook"] = nb_cfg
        tmp_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🔁 [CLOSED_LOOP] Cycle {cycles}: adding {len(add_variants)} variants and re-running")
        orchestrate(str(tmp_cfg))
        rv = expert_review_hypergraph(str(hp), threshold=threshold, cfg_path=cfg_path)

    # final export
    select_and_export_reference(str(hp), cfg_path)
    print(f"🏁 [CLOSED_LOOP] Finished after {cycles} cycle(s). Decision={rv.get('decision')}")
    return rv

# --------------------
# CLI
# --------------------
if __name__ == "__main__":
    import sys
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("config.json"))
    orchestrate(cfg_path)  # single orchestration (auto-review + reference export depend on config)


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
    patt_arrow  = re.compile(r"([A-Za-z0-9_./-]+)\s*->\s*([A-Za-z0-9_./-]+)")

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
    llm_cfg = nb_cfg.get("llm", {}) or {}
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    base_url_env = llm_cfg.get("base_url_env", "OPENAI_BASE_URL")
    model = llm_cfg.get("model") or os.environ.get("OPENAI_MODEL", "gpt-4o")

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

    round_idx = 0
    while errors and round_idx < max_fix_rounds:
        round_idx += 1
        print(f"🛠  [AUTO-FIX] round {round_idx}: targets={[e['cell_index'] for e in errors]}")

        messages = build_fix_messages(
            language, data_path, csv_preview, paper_excerpt, errors,
            nb_cfg.get("headings") or ["Introduction", "Methods", "Results"]
        )
        spec = chat_json(
            messages,
            api_key=os.environ.get(api_key_env),
            base_url=os.environ.get(base_url_env, "https://api.openai.com/v1"),
            model=model,
            temperature=0.0,
            max_tokens=1024
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
    return final

