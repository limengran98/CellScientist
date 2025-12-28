#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Metrics Analysis Module.

Strategy: "Comprehensive Finall_Results Analysis"
Evaluates Mechanism Diversity (GED) and Code Complexity by digesting ALL 
artifacts preserved in the 'finall_results' folder.

1. Mechanism Diversity: Derived from mining 'phase2.log' (Breadth) and 'phase3_history' (Depth).
2. Code Complexity: Derived from 'best_code.py' (AST + LLM).

Outputs to: logs_dir/advanced_metrics/
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional

from runner_utils import read_text, safe_read_json, read_text_limited
from runner_report import resolve_report_llm_cfg, chat_text

# =============================================================================
# 1. AST Metrics (Hard Code Complexity)
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 1
    def visit_If(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_For(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_While(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_Try(self, node): self.complexity += 1; self.generic_visit(node)
    def visit_BoolOp(self, node): self.complexity += len(node.values) - 1; self.generic_visit(node)

def calculate_ast_metrics(code_text: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return {"cyclomatic_complexity": -1, "sloc": len(code_text.splitlines()), "note": "SyntaxError"}
    
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    lines = code_text.splitlines()
    sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    return {
        "cyclomatic_complexity": visitor.complexity,
        "sloc": sloc,
        "complexity_density": round(visitor.complexity / max(1, sloc), 3),
        "vocabulary_size": len(set(code_text.split()))
    }

# =============================================================================
# 2. Comprehensive Data Ingestion (From finall_results only)
# =============================================================================

def _extract_p2_breadth_from_log(log_path: str) -> str:
    """
    Minings phase2.log to reconstruct the 'Hypothesis Space'.
    It looks for lines where the system tried different prompts/focuses.
    """
    text = read_text(log_path)
    if not text: return "(Phase 2 log missing in finall_results)"
    
    hits = []
    # Capture lines showing distinct strategies or focuses
    for line in text.splitlines():
        # Heuristics to find strategy changes or run starts
        if any(k in line for k in ["focus=", "prompt_variants", "Executing Run", "Score:", "metrics:"]):
             # Clean up timestamp noise [2025-...]
             clean = re.sub(r"^.*?\[.*?\]", "", line).strip()
             hits.append(clean)
    
    # Return a digest to represent breadth
    return "P2_Log_Digest (All Attempts):\n" + "\n".join(hits[:80])

def ingest_finall_results_comprehensive(finall_dir: str) -> Dict[str, str]:
    """
    Reads ALL relevant files in finall_results to build a full context.
    """
    context = {}
    
    # --- Phase 1 (Design Logic) ---
    p1_report = os.path.join(finall_dir, "phase1_summary_report.md")
    context["p1_logic"] = read_text_limited(p1_report, max_chars=12000) or "(P1 Report Missing)"
    
    # --- Phase 2 (Generation Breadth & Best Strategy) ---
    # 1. The Winner (Depth)
    p2_report = os.path.join(finall_dir, "phase2_best_experiment_report.md")
    context["p2_best_report"] = read_text_limited(p2_report, max_chars=8000) or "(P2 Best Report Missing)"
    
    # 2. The Alternatives (Breadth from Log)
    # Checks for both naming conventions
    p2_log = os.path.join(finall_dir, "phase2.log")
    if not os.path.exists(p2_log): 
        p2_log = os.path.join(finall_dir, "Phase 2.log")
    context["p2_breadth_log"] = _extract_p2_breadth_from_log(p2_log)
    
    # --- Phase 3 (Optimization Depth) ---
    # 1. History JSON (Preferred)
    p3_json = os.path.join(finall_dir, "phase3_history_state.json")
    hist_data = safe_read_json(p3_json)
    if hist_data:
        digest = []
        for e in hist_data:
            if isinstance(e, dict):
                digest.append(f"Iter {e.get('iter')}: {e.get('decision')} | Focus: {e.get('focus')} | Score: {e.get('score')}")
        context["p3_trajectory"] = "\n".join(digest)
    else:
        # Fallback to MD
        p3_md = os.path.join(finall_dir, "phase3_optimization_history.md")
        context["p3_trajectory"] = read_text(p3_md) or "(P3 History Missing)"

    # --- Code (Result) ---
    code_path = os.path.join(finall_dir, "best_code.py")
    if not os.path.exists(code_path):
        # Fallback to ipynb conversion if py missing
        nb_path = os.path.join(finall_dir, "best_code.ipynb")
        if os.path.exists(nb_path):
            from runner_utils import export_notebook_as_py
            export_notebook_as_py(nb_path, code_path)
    
    context["final_code"] = read_text(code_path)
    
    return context

# =============================================================================
# 3. LLM Logic
# =============================================================================

def analyze_via_llm(content: str, system_prompt: str, user_prompt_template: str, llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    user_msg = user_prompt_template.replace("${content}", content)
    system_prompt += "\nOutput MUST be valid JSON only."
    
    resp = chat_text([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}], llm_cfg)
    
    try:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, re.DOTALL)
        if m: return json.loads(m.group(1))
        m2 = re.search(r"(\{.*\})", resp, re.DOTALL)
        if m2: return json.loads(m2.group(1))
        return json.loads(resp)
    except:
        return {"error": "JSON Parse Failed", "raw": resp}

# =============================================================================
# 4. Main Entry Point
# =============================================================================

def perform_advanced_analysis(
    *,
    dataset_name: str,
    logs_dir: str,  # The base dir where 'finall_results' is located
    pipe_cfg: Optional[Dict[str, Any]],
) -> None:
    
    finall_dir = os.path.join(logs_dir, "finall_results")
    out_dir = os.path.join(logs_dir, "advanced_metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n🔍 [Advanced Analysis] Reading comprehensive artifacts from: {finall_dir}")
    if not os.path.exists(finall_dir):
        print(f"[WARN] finall_results not found at {finall_dir}. Skipping advanced metrics.")
        return

    # 1. Ingest EVERYTHING relevant from finall_results
    data = ingest_finall_results_comprehensive(finall_dir)
    llm_cfg = resolve_report_llm_cfg(pipe_cfg)
    
    # -------------------------------------------------------
    # A. Mechanism Diversity (GED) - Full Lifecycle
    # -------------------------------------------------------
    ged_context = f"""
    === Phase 1: Design Logic (The Origin) ===
    {data['p1_logic'][:4000]}...
    
    === Phase 2: Generation Space (The Breadth) ===
    -- Alternative Attempts (Log Digest) --
    {data['p2_breadth_log']}
    
    -- Best Strategy Details --
    {data['p2_best_report'][:4000]}...
    
    === Phase 3: Optimization Space (The Depth) ===
    {data['p3_trajectory']}
    """
    
    ged_sys = (
        "You are a Meta-Scientist. Evaluate the 'Mechanism Diversity' (GED) and 'Exploration Quality' "
        "of this scientific discovery pipeline. Use the logs to judge breadth and the reports to judge depth."
    )
    ged_user = """
    Analyze the full exploration lifecycle:
    ${content}
    
    Evaluate (Score 0-10):
    1. **Hypothesis Diversity (Phase 2)**: Based on the Log Digest, did the system try DISTINCT strategies (High GED) or just random variations (Low GED)?
    2. **Optimization Logic (Phase 3)**: Did the trajectory show logical refinement (Scientific Method) vs random guessing?
    3. **Global Semantic Span**: The conceptual distance from the P1 Design to the P3 Final State.
    
    Return JSON:
    {
        "hypothesis_diversity_score": float,
        "optimization_logic_score": float,
        "global_semantic_span_score": float,
        "diversity_summary": "Concise analysis of the exploration breadth vs depth",
        "detected_strategy_types": ["list", "of", "strategies"]
    }
    """
    
    print("   ... Calculating Mechanism Diversity (GED)...")
    ged_res = analyze_via_llm(ged_context, ged_sys, ged_user, llm_cfg)
    
    # -------------------------------------------------------
    # B. Code Complexity (Scientific Parsimony)
    # -------------------------------------------------------
    comp_context = f"""
    === Final Scientific Code ===
    {data['final_code'][:20000]}
    """
    
    comp_sys = "You are a Code Scientist. Evaluate the 'Scientific Complexity' and 'Parsimony' of this solution."
    comp_user = """
    Review the code:
    ${content}
    
    Evaluate (Score 0-10):
    1. **Parsimony (Ockham's Razor)**: Is the solution simple/elegant (10) or bloated/over-engineered (0)?
    2. **Interpretability**: Can a domain scientist understand the logic?
    3. **Modularity**: Is the code structure robust?
    
    Return JSON:
    {
        "parsimony_score": float,
        "interpretability_score": float,
        "modularity_score": float,
        "complexity_analysis": "string"
    }
    """
    
    print("   ... Calculating Code Complexity...")
    comp_res_sem = analyze_via_llm(comp_context, comp_sys, comp_user, llm_cfg)
    comp_res_hard = calculate_ast_metrics(data['final_code'] or "")
    
    # -------------------------------------------------------
    # Save Results
    # -------------------------------------------------------
    full_report = {
        "dataset": dataset_name,
        "metrics": {
            "mechanism_diversity": ged_res,
            "code_complexity": {
                "algorithmic": comp_res_hard,
                "semantic": comp_res_sem
            }
        },
        "timestamp": _now_iso()
    }
    
    json_path = os.path.join(out_dir, "advanced_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    md_lines = [
        f"# Advanced Scientific Evaluation: {dataset_name}",
        f"**Date**: {_now_iso()}",
        "",
        "## 1. Mechanism Diversity (GED)",
        f"- **Hypothesis Diversity (P2)**: {ged_res.get('hypothesis_diversity_score', '-')}/10",
        f"- **Optimization Logic (P3)**: {ged_res.get('optimization_logic_score', '-')}/10",
        f"- **Global Semantic Span**: {ged_res.get('global_semantic_span_score', '-')}/10",
        "",
        "### Strategies Detected:",
        ", ".join(ged_res.get('detected_strategy_types', [])),
        f"> {ged_res.get('diversity_summary', '-')}",
        "",
        "## 2. Code Parsimony & Complexity",
        f"- **Algorithmic Density**: {comp_res_hard.get('complexity_density', '-')}",
        f"- **Parsimony (Ockham's Razor)**: {comp_res_sem.get('parsimony_score', '-')}/10",
        f"- **Interpretability**: {comp_res_sem.get('interpretability_score', '-')}/10",
        f"> {comp_res_sem.get('complexity_analysis', '-')}"
    ]
    
    md_path = os.path.join(out_dir, "advanced_analysis_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    print(f"✅ Advanced metrics saved to: {json_path}")
    print(f"✅ Advanced report saved to: {md_path}")

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat()

if __name__ == "__main__":
    print("Run via run_cellscientist.py")