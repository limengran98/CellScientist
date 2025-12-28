#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Metrics Analysis Module.

Performs post-hoc analysis on the 'finall_results' AND raw artifacts to evaluate:
1. Mechanism Diversity (GED): The breadth of the hypothesis space explored across ALL PHASES.
2. Code Complexity: The simplicity and elegance of the final scientific explanation.

Outputs to: results/${dataset}/advanced_metrics/
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional

from runner_utils import read_text, safe_read_json, read_text_limited
from runner_report import resolve_report_llm_cfg, chat_text

# =============================================================================
# 1. AST-based Code Complexity (Algorithmic)
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """Calculates Cyclomatic Complexity (McCabe) via AST."""
    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

def calculate_ast_metrics(code_text: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return {"cyclomatic_complexity": -1, "loc": len(code_text.splitlines()), "notes": "Syntax Error"}

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    
    lines = code_text.splitlines()
    sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    vocab_size = len(set(code_text.split()))

    return {
        "cyclomatic_complexity": visitor.complexity,
        "sloc": sloc,
        "complexity_density": round(visitor.complexity / max(1, sloc), 3),
        "vocabulary_size": vocab_size
    }

# =============================================================================
# 2. LLM Helper
# =============================================================================

def analyze_via_llm(content: str, system_prompt: str, user_prompt_template: str, llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    user_msg = user_prompt_template.replace("${content}", content)
    system_prompt += "\nOutput MUST be valid JSON only. No markdown formatting."
    
    resp = chat_text(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
        llm_cfg
    )
    
    try:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, re.DOTALL)
        if m: return json.loads(m.group(1))
        m2 = re.search(r"(\{.*\})", resp, re.DOTALL)
        if m2: return json.loads(m2.group(1))
        return json.loads(resp)
    except Exception:
        return {"error": "JSON Parse Failed", "raw_response": resp}

# =============================================================================
# 3. Smart Data Ingestion (The Optimization)
# =============================================================================

def _ingest_phase1_design(final_results_dir: str) -> str:
    """Intelligently ingest Phase 1: Try JSON summary first, then Markdown headers."""
    # 1. Try structured JSON (Best Case)
    json_path = os.path.join(final_results_dir, "phase1_summary_report.json") # Often saved alongside MD
    if os.path.exists(json_path):
        data = safe_read_json(json_path)
        if data:
            # Extract key fields only to save tokens
            return f"Structured Design:\n{json.dumps(data.get('report_content', data), indent=2)[:5000]}"

    # 2. Fallback to Markdown with smart truncation
    md_path = os.path.join(final_results_dir, "phase1_summary_report.md")
    text = read_text_limited(md_path, max_chars=15000)
    if not text:
        return "(Phase 1 report missing)"
    
    # Simple heuristic to extract Objective/Hypothesis sections
    # (Assuming standard report format)
    lines = []
    capture = True
    for line in text.splitlines():
        if line.startswith("#"):
            if any(k in line.lower() for k in ["code", "appendix", "log"]):
                capture = False
            else:
                capture = True
        if capture:
            lines.append(line)
    
    return "\n".join(lines)[:6000] # Return filtered text

def _ingest_phase2_broad_search(results_root: str, final_results_dir: str) -> str:
    """
    Ingest Phase 2 by scanning the ACTUAL artifacts folders (Generate_Execution),
    not just the logs. This captures the true diversity of prompts tried.
    """
    # 1. Locate Generate_Execution folder
    # results_root is usually ../results/dataset
    ge_dir = os.path.join(results_root, "generate_execution", "prompt")
    
    runs_digest = []
    
    if os.path.exists(ge_dir):
        # Scan run folders
        run_dirs = sorted(glob.glob(os.path.join(ge_dir, "prompt_run_*")))
        for i, rdir in enumerate(run_dirs):
            # Read Config (What was the strategy?)
            cfg_path = os.path.join(rdir, "config.json")
            cfg = safe_read_json(cfg_path)
            
            # Extract prompt focus/variant
            # Structure depends on exact config shape, attempting robust retrieval
            focus = "(Unknown)"
            if cfg:
                focus = (
                    cfg.get("phases", {}).get("task_analysis", {}).get("llm_notebook", {}).get("focus_instruction") or
                    cfg.get("prompt_variant") or 
                    "Standard"
                )
            
            # Read Metrics (Did it work?)
            met_path = os.path.join(rdir, "metrics.json")
            met = safe_read_json(met_path)
            score = "N/A"
            if met:
                # Try to grab a primary score
                score = met.get("winner_score", met.get("aggregate", {}).get("PCC", "N/A"))

            runs_digest.append(f"Run {i+1}: Strategy/Focus='{focus}', Outcome_Score={score}")

    if runs_digest:
        return "Artifact Scan Results:\n" + "\n".join(runs_digest)
    
    # 2. Fallback to Log Parsing (if artifacts deleted)
    log_path = os.path.join(final_results_dir, "phase2.log")
    text = read_text(log_path)
    if not text: return "(Phase 2 data missing)"
    
    # Regex fallback
    hits = []
    for line in text.splitlines():
        if "focus=" in line or "prompt_variants" in line:
            hits.append(line.strip())
    return "Log Scan Results:\n" + "\n".join(hits[:50])

def _ingest_phase3_depth(final_results_dir: str) -> str:
    """Ingest Phase 3 trajectory with calculated optimization delta."""
    p3_path = os.path.join(final_results_dir, "phase3_history_state.json")
    data = safe_read_json(p3_path)
    if not data or not isinstance(data, list):
        return "(No Phase 3 optimization history)"
    
    digest = []
    start_score = None
    best_score = -999.0
    
    for entry in data:
        if not isinstance(entry, dict): continue
        sc = entry.get('score')
        if isinstance(sc, (int, float)):
            if start_score is None: start_score = sc
            if sc > best_score: best_score = sc
            
        digest.append(
            f"Iter {entry.get('iter')}: Action='{entry.get('decision')}', "
            f"Focus='{entry.get('focus')}', Score={sc}"
        )
    
    delta = 0.0
    if start_score is not None and best_score > -999:
        delta = best_score - start_score
        
    header = f"Optimization Stats: Start={start_score}, Best={best_score}, Delta={delta:+.4f}\nTrajectory:\n"
    return header + "\n".join(digest)

# =============================================================================
# 4. Main Entry Point
# =============================================================================

def perform_advanced_analysis(
    *,
    dataset_name: str,
    logs_dir: str,
    pipe_cfg: Optional[Dict[str, Any]],
) -> None:
    """
    Main entry point. 
    """
    final_results_dir = os.path.join(logs_dir, "finall_results")
    # Infer results_root from logs_dir (logs_dir is usually .../results/dataset/logs)
    results_root = os.path.dirname(logs_dir)
    
    out_dir = os.path.join(logs_dir, "advanced_metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n🔍 [Advanced Analysis] Smart Ingestion active. Output: {out_dir}")

    # --- 1. Smart Ingestion ---
    p1_context = _ingest_phase1_design(final_results_dir)
    p2_context = _ingest_phase2_broad_search(results_root, final_results_dir)
    p3_context = _ingest_phase3_depth(final_results_dir)
    
    # Get Code
    code_path = os.path.join(final_results_dir, "best_code.py")
    if not os.path.exists(code_path):
        nb_path = os.path.join(final_results_dir, "best_code.ipynb")
        if os.path.exists(nb_path):
            from runner_utils import export_notebook_as_py
            export_notebook_as_py(nb_path, code_path)
    code_text = read_text(code_path)
    
    llm_cfg = resolve_report_llm_cfg(pipe_cfg)
    
    # -------------------------------------------------------
    # A. Mechanism Diversity (GED)
    # -------------------------------------------------------
    full_history_context = f"""
    === Phase 1: Design Space (Problem Framing) ===
    {p1_context}
    
    === Phase 2: Generation Space (Broad Search / Multi-Hypothesis) ===
    {p2_context}
    
    === Phase 3: Optimization Space (Deep Search / Refinement) ===
    {p3_context}
    """
    
    div_sys = (
        "You are a Meta-Scientist evaluating the 'Mechanism Diversity' (GED) of an AI discovery agent. "
        "Assess how well the agent explored the scientific hypothesis space. "
        "Did it try distinct strategies (High Diversity) or just repeat trivial variations (Low Diversity)?"
    )
    div_user = """
    Analyze the exploration trajectory:
    
    ${content}
    
    Evaluate (Score 0-10):
    1. **Hypothesis Independence (Phase 2)**: Look at the 'Artifact Scan Results'. Did the agent actually execute DISTINCT strategies (e.g., different algorithms, features), or just different random seeds?
    2. **Optimization Efficacy (Phase 3)**: Look at the 'Optimization Stats'. Did the agent improve the score via logical steps?
    3. **Global Exploration Distance (GED)**: The overall semantic distance covered from P1 design to P3 final.
    
    Return JSON:
    {
        "hypothesis_independence_score": float,
        "optimization_efficacy_score": float,
        "global_exploration_distance_score": float,
        "diversity_summary": "Concise analysis string",
        "strategies_tried": ["list", "of", "distinct", "strategies"]
    }
    """
    
    print("   ... Analyzing Diversity (GED) with Artifact Data...")
    diversity_metrics = analyze_via_llm(full_history_context, div_sys, div_user, llm_cfg)
    diversity_metrics["valid_data"] = True

    # -------------------------------------------------------
    # B. Code Complexity
    # -------------------------------------------------------
    complexity_metrics = {"valid_data": False}
    if code_text:
        print("   ... Analyzing Code Complexity...")
        ast_mets = calculate_ast_metrics(code_text)
        
        comp_sys = "Evaluate Scientific Code Parsimony (Ockham's Razor)."
        comp_user = """
        Code (Truncated):
        ```python
        ${content}
        ```
        
        Evaluate (0-10):
        1. **Parsimony**: Simplicity relative to function.
        2. **Modularity**: Good scientific engineering?
        
        Return JSON:
        {
            "parsimony_score": float,
            "modularity_score": float,
            "complexity_summary": "string"
        }
        """
        trunc_code = code_text[:15000]
        llm_comp = analyze_via_llm(trunc_code, comp_sys, comp_user, llm_cfg)
        
        complexity_metrics = {
            "valid_data": True,
            "algorithmic": ast_mets,
            "semantic": llm_comp
        }

    # -------------------------------------------------------
    # C. Save
    # -------------------------------------------------------
    full_report = {
        "dataset": dataset_name,
        "diversity_ged": diversity_metrics,
        "complexity": complexity_metrics,
        "timestamp": _now_iso()
    }
    
    json_path = os.path.join(out_dir, "advanced_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    # Markdown Report
    md = [
        f"# Advanced Scientific Metrics: {dataset_name}",
        f"Generated: {_now_iso()}",
        "",
        "## 1. Mechanism Diversity (GED)",
        "**Context**: Full lifecycle analysis (Design -> Gen -> Opt)",
        f"- **Hypothesis Independence (P2)**: {diversity_metrics.get('hypothesis_independence_score', '-')}/10",
        f"- **Optimization Efficacy (P3)**: {diversity_metrics.get('optimization_efficacy_score', '-')}/10",
        f"- **Global GED Score**: {diversity_metrics.get('global_exploration_distance_score', '-')}/10",
        "",
        "### Detected Strategies:",
        ", ".join(diversity_metrics.get('strategies_tried', [])),
        f"> {diversity_metrics.get('diversity_summary', '-')}",
        "",
        "## 2. Code Parsimony",
        f"- **Algorithmic Density**: {complexity_metrics.get('algorithmic', {}).get('complexity_density', '-')}",
        f"- **Parsimony Score**: {complexity_metrics.get('semantic', {}).get('parsimony_score', '-')}/10",
        f"> {complexity_metrics.get('semantic', {}).get('complexity_summary', '-')}"
    ]
    
    with open(os.path.join(out_dir, "advanced_analysis_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        
    print(f"✅ Advanced metrics saved to: {json_path}")

def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat()