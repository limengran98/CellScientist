#!/usr/bin/env python3
# review_agent.py
# This module contains the "gentic" agents for the Review_Feedback phase.
# It reads a hypergraph manifest, scores the artifacts,
# and exports the best-performing one.

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import nbformat as _nbf
from nbformat.reader import reads as _nb_reads

# [MODIFIED] Import from new analysis_helpers for LLM call
# We assume it's available in the python path
try:
    from Design_Analysis.analysis_helpers import chat_json
except ImportError:
    print("Warning: Could not import chat_json. LLM critique will be disabled.")
    print("Ensure Design_Analysis is in your PYTHONPATH.")
    # Define a fallback
    def chat_json(*args, **kwargs):
        return {"pros": ["Fallback: chat_json not found"], "cons": [], "risks": [], "suggestions": []}


# ============================
# Expert Agent (review)
# ============================

def _safe_read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _extract_nb_cells(ipynb_path: str) -> List[Dict[str, Any]]:
    try:
        nb = _nbf.read(ipynb_path, as_version=4)
        return nb.get("cells", [])
    except Exception:
        s = _safe_read_text(Path(ipynb_path))
        try:
            nb = _nb_reads(s, as_version=4)
            return nb.get("cells", [])
        except Exception:
            return []

def _heuristic_scores_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate heuristic scores based on notebook content."""
    if not cells:
        return dict(scientific=0.0, novelty=0.0, reproducibility=0.0, interpretability=0.0, completeness=0.0)
    
    md_cnt = sum(1 for c in cells if c.get("cell_type")=="markdown")
    code_cnt = sum(1 for c in cells if c.get("cell_type")=="code" and c.get("source"))
    text_all = " ".join([c.get("source","") for c in cells if isinstance(c.get("source",""), str)]).lower()
    code_all = " ".join([c.get("source","") for c in cells if c.get("cell_type")=="code" and isinstance(c.get("source",""), str)]).lower()

    def has(keys): return any(k in text_all for k in keys)
    def has_code(keys): return any(k in code_all for k in keys)
    def clamp(v): return max(0.0, min(1.0, v))

    # Scientific Rigor
    has_stat = has(["p-value","anova","t-test","wilcoxon","chi-square","fdr","effect size"])
    has_val = has(["validation set", "test set", "holdout", "cross-validation", "cv="])
    has_h5 = has_code(["h5py", "'.h5'"])
    scientific = clamp(0.4*(1.0 if has_stat else 0.0) + 0.4*(1.0 if has_val else 0.0) + 0.2*(1.0 if has_h5 else 0.0))

    # Novelty / Depth
    has_adv = has(["contrastive", "representation", "causal", "embedding", "deep learning", "umap"])
    has_bio = has(["gene","pathway","enrichment","marker", "compound", "dose"])
    novelty = clamp(0.5*(1.0 if has_adv else 0.0) + 0.3*(1.0 if has_bio else 0.0) + 0.2*min(code_cnt/12.0, 1.0))

    # Reproducibility
    has_seed = has_code(["random_state","seed=","np.random.seed","torch.manual_seed"])
    has_env = has(["requirements.txt", "environment.yml", "pip install"])
    reproducibility = clamp(0.6*(1.0 if has_seed else 0.0) + 0.2*(1.0 if has_env else 0.0) + 0.2*min(md_cnt/5.0, 1.0))

    # Interpretability
    has_fig  = has_code(["plt.","plot(","figure(","heatmap","violin","boxplot","volcano"])
    has_disc  = has(["discussion","limitations","confound","bias","future work", "conclusion"])
    interpretability = clamp(0.4*(1.0 if has_fig else 0.0) + 0.4*(1.0 if has_disc else 0.0) + 0.2*min(md_cnt/8.0, 1.0))

    # Completeness (all nodes present)
    completeness = clamp( (md_cnt + code_cnt) / 15.0 ) # Simple heuristic for now

    return dict(
        scientific=scientific, 
        novelty=novelty, 
        reproducibility=reproducibility, 
        interpretability=interpretability,
        completeness=completeness
    )

def _llm_critique(
    edge_meta: dict, 
    cells: List[Dict[str, Any]], 
    review_config: Optional[dict] = None, # [MODIFIED] Pass full review_config
    prompts: Optional[Dict[str, Any]] = None
) -> dict:
    """Uses an LLM to provide a qualitative critique."""
    llm_cfg = (review_config or {}).get("llm", {})
    if not llm_cfg or not llm_cfg.get("enabled", False):
        return {"llm_critique_enabled": False, "message": "LLM critique disabled in config."}

    # Get prompts from the loaded prompt data
    review_prompts = (prompts or {}).get("review", {})
    system_prompt = review_prompts.get("system_prompt", "You are a reviewer.")
    critique_template = review_prompts.get("critique_template", "Review this: {code}")

    # Prepare context
    code_summary = "\n".join([c.get("source", "") for c in cells if c.get("cell_type") == "code"])[:5000]
    md_summary = "\n".join([c.get("source", "") for c in cells if c.get("cell_type") == "markdown"])[:3000]

    user_prompt = critique_template.format(
        edge_id=edge_meta.get("edge_id"),
        prompt_variant=edge_meta.get("prompt_variant"),
        code_summary=code_summary,
        markdown_summary=md_summary
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        # --- [NEW] Resolve LLM config from providers ---
        providers = (review_config or {}).get("providers", {}) # Get providers from review_config
        prov_name = llm_cfg.get("provider")
        prov = providers.get(prov_name, {}) if prov_name else {}
        
        model = llm_cfg.get("model") or prov.get("model") or (prov.get("models") or ["gpt-4o-mini"])[0]
        api_key = prov.get("api_key") or os.environ.get(llm_cfg.get("api_key_env", "OPENAI_API_KEY"))
        base_url = prov.get("base_url") or os.environ.get(llm_cfg.get("base_url_env", "OPENAI_BASE_URL"))
        # --- End Resolve ---

        response = chat_json(
            messages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(llm_cfg.get("temperature", 0.2)),
            max_tokens=int(llm_cfg.get("max_tokens", 1024))
        )
        response["llm_critique_enabled"] = True
        return response
    except Exception as e:
        print(f"LLM Critique failed: {e}")
        return {"llm_critique_enabled": False, "error": str(e)}


def expert_review_hypergraph(
    hypergraph_path: str, 
    review_config: Dict[str, Any],
    prompts: Dict[str, Any]
) -> None:
    """
    Reads a hypergraph manifest, scores each hyperedge (notebook),
    and updates the manifest file with an "expert_review" section.
    """
    hp = Path(hypergraph_path)
    if not hp.exists():
        raise FileNotFoundError(f"hypergraph not found: {hp}")

    threshold = float(review_config.get("threshold", 0.7))
    # [MODIFIED] Use 'review_weights' for heuristic scoring
    weights = review_config.get("review_weights", {}) 

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])

    study_scores = []
    for e in edges:
        attrs = e.get("attrs", {})
        ip = attrs.get("executed_ipynb")
        cells = _extract_nb_cells(ip) if ip and Path(ip).exists() else []
        
        # 1. Heuristic scores
        scores = _heuristic_scores_from_cells(cells)
        
        # 2. LLM critique
        # [MODIFIED] Pass the full review_config
        critique = _llm_critique(attrs, cells, review_config, prompts)
        
        # 3. Calculate overall *heuristic* weighted score
        overall_score = 0.0
        total_weight = 0.0
        for key, w in weights.items():
            overall_score += float(scores.get(key, 0.0)) * float(w)
            total_weight += float(w)
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        scores["overall_heuristic_weighted"] = overall_score # [MODIFIED] Renamed for clarity
        
        # Store results in the edge attributes
        attrs["expert_review"] = {
            "heuristic_scores": scores, 
            "llm_critique": critique,
            "review_utc": datetime.utcnow().isoformat() + "Z"
        }
        e["attrs"] = attrs
        study_scores.append(overall_score)

    # Add top-level summary to the hypergraph
    overall_avg = sum(study_scores)/len(study_scores) if study_scores else 0.0
    decision = "pass" if overall_avg >= threshold else "fail"
    
    data["expert_review_summary"] = {
        "threshold": threshold,
        "average_heuristic_score": overall_avg,
        "decision": decision,
        "review_utc": datetime.utcnow().isoformat() + "Z",
        "review_config": {k: v for k, v in review_config.items() if k != 'providers'} # Don't log keys
    }

    # Write the updated data back to the same file
    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"🧠 [REVIEW] Heuristic review complete. Wrote scores to {hp}. Decision: {decision} (Avg={overall_avg:.3f})")

# ============================
# Reference selection & export
# ============================

def _score_edge_for_reference(attrs: dict, weights: dict) -> float:
    """
    [MODIFIED] Scores an edge based on OBJECTIVE criteria for final selection.
    """
    # Priority 1: Check for bugs.
    # 0 errors is good, any errors is bad.
    error_count = attrs.get("execution_error_count", 99)
    if error_count > 0:
        return -999.0 # Fail immediately if there are bugs.

    score = 0.0
    
    # Priority 2: Apply objective weights
    # Add base score for execution success
    score += float(weights.get("execution_success", 1.0))
    
    # Add score if H5 file was generated
    if attrs.get("generated_h5_sha"):
        score += float(weights.get("has_h5_output", 1.0))
    
    # Priority 3: Apply experimental metrics
    metrics = attrs.get("experimental_metrics") # This is a dict, e.g., {"accuracy": 0.85}
    metric_weights = weights.get("experimental_metrics", {}) # e.g., {"accuracy": 10.0}
    
    if isinstance(metrics, dict) and isinstance(metric_weights, dict):
        for key, weight in metric_weights.items():
            if key in metrics:
                try:
                    # Add (metric_value * metric_weight)
                    score += float(metrics[key]) * float(weight)
                except (TypeError, ValueError):
                    pass # Ignore non-numeric metrics
    
    return score

def select_and_export_reference(
    hypergraph_path: str, 
    review_config: Dict[str, Any]
) -> None:
    """
    Finds the best-scoring hyperedge from the manifest and copies its
    notebook and H5 file to the 'reference' directory.
    [MODIFIED] Uses new objective scoring.
    """
    hp = Path(hypergraph_path)
    if not hp.exists():
        print("[REFERENCE] hypergraph not found; skip")
        return

    data = json.loads(hp.read_text(encoding="utf-8"))
    edges = data.get("hyperedges", [])
    if not edges:
        print("[REFERENCE] no edges found in hypergraph; skip")
        return

    ref_cfg = review_config.get("reference", {})
    if not (ref_cfg.get("enabled", False)):
        print("[REFERENCE] disabled in review_config.json; skip")
        return

    # [MODIFIED] Read the new 'selection_weights' from the 'reference' block
    weights = (ref_cfg.get("selection_weights") or {})
    
    export_dir_str = ref_cfg.get("export_dir", "../results/reference")
    # Resolve export_dir relative to the hypergraph file's location
    export_dir = (hp.parent / export_dir_str).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    # Pick best edge by score
    best = None
    best_score = -1e18
    for e in edges:
        # [MODIFIED] Call the new objective scoring function
        sc = _score_edge_for_reference(e.get("attrs", {}), weights)
        if sc > best_score:
            best_score = sc
            best = e

    if not best:
        print("[REFERENCE] could not determine best edge (all may have failed execution)")
        return
    
    if best_score == -999.0:
        print(f"[REFERENCE] No bug-free notebook found. Best score was {best_score}. Aborting export.")
        return

    attrs = best.get("attrs", {})
    src_ipynb_path = attrs.get("executed_ipynb") or attrs.get("unexecuted_ipynb")
    
    if not src_ipynb_path:
        print("[REFERENCE] best edge has no notebook path; skip")
        return
        
    src_ipynb = Path(src_ipynb_path)
    if not src_ipynb.exists():
        print(f"[REFERENCE] best notebook file not found at: {src_ipynb}; skip")
        return

    # --- Copy BEST Notebook ---
    ref_name = f"REFERENCE_obj_score_{best_score:.2f}_{best.get('edge_id')}"
    dst_ipynb = export_dir / f"{ref_name}.ipynb"
    try:
        shutil.copyfile(src_ipynb, dst_ipynb)
        print(f"📌 [REFERENCE] exported Notebook → {dst_ipynb}")
    except Exception as e:
        print(f"[REFERENCE] Notebook copy failed: {e}")
        return
        
    # --- Copy BEST H5 File ---
    src_h5_path = attrs.get("generated_h5")
    dst_h5 = None
    if src_h5_path:
        src_h5 = Path(src_h5_path)
        if src_h5.exists():
            dst_h5 = export_dir / f"{ref_name}_DATA.h5"
            try:
                shutil.copyfile(src_h5, dst_h5)
                print(f"📌 [REFERENCE] exported H5 Data → {dst_h5}")
            except Exception as e:
                print(f"[REFERENCE] H5 copy failed: {e}")
                dst_h5 = None
        else:
             print(f"⚠️ [REFERENCE] H5 file path found but file is missing: {src_h5}")
    else:
        print("⚠️ [REFERENCE] No H5 file path recorded for best run.")

    # --- Write reference metadata ---
    ref_meta = {
        "source_hypergraph": str(hp),
        "best_edge_id": best.get("edge_id"),
        "objective_score": best_score,
        "exported_notebook_path": str(dst_ipynb),
        "exported_h5_path": str(dst_h5) if dst_h5 else None,
        "source_attrs": attrs,
        "selection_weights": weights,
        "exported_utc": datetime.utcnow().isoformat() + "Z"
    }
    (export_dir / "reference_manifest.json").write_text(json.dumps(ref_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Also, update the main hypergraph with this info
    data["reference_export"] = ref_meta
    hp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"✅ [REFERENCE] Export complete. Best score: {best_score:.3f}. Manifest written to {export_dir.name}/")
    return