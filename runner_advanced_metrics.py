#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Metrics Analysis Module.

Performs post-hoc analysis on the 'finall_results' artifacts to evaluate:
1. Mechanism Diversity (GED): The breadth of the hypothesis space explored across ALL PHASES (1, 2, 3).
2. Code Complexity: The simplicity and elegance of the final scientific explanation (Code).

Outputs to: results/${dataset}/advanced_metrics/
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, Optional

from runner_utils import read_text, safe_read_json, read_text_limited
from runner_report import resolve_report_llm_cfg, chat_text

# =============================================================================
# 1. AST-based Code Complexity (Algorithmic)
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """Calculates Cyclomatic Complexity (McCabe) via AST."""
    def __init__(self):
        self.complexity = 1  # Base complexity

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

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

def calculate_ast_metrics(code_text: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return {"cyclomatic_complexity": -1, "loc": len(code_text.splitlines()), "notes": "Syntax Error in AST parse"}

    visitor = ComplexityVisitor()
    visitor.visit(tree)
    
    lines = code_text.splitlines()
    loc = len(lines)
    sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    tokens = set(code_text.split())
    vocab_size = len(tokens)

    return {
        "cyclomatic_complexity": visitor.complexity,
        "loc": loc,
        "sloc": sloc,
        "vocabulary_size": vocab_size,
        "complexity_density": round(visitor.complexity / max(1, sloc), 3)
    }

# =============================================================================
# 2. LLM-based Evaluation Helpers
# =============================================================================

def analyze_via_llm(
    content: str,
    system_prompt: str,
    user_prompt_template: str,
    llm_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Generic helper to get JSON metrics from LLM."""
    user_msg = user_prompt_template.replace("${content}", content)
    system_prompt += "\nOutput MUST be valid JSON only. No markdown formatting."
    
    resp = chat_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        llm_cfg
    )
    
    try:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", resp, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        m2 = re.search(r"(\{.*\})", resp, re.DOTALL)
        if m2:
            return json.loads(m2.group(1))
        return json.loads(resp)
    except Exception:
        print(f"[WARN] Failed to parse LLM JSON response. Raw: {resp[:100]}...")
        return {"error": "JSON Parse Failed", "raw_response": resp}

def _extract_phase2_digest(log_path: str) -> str:
    """Extracts hypothesis generation attempts from Phase 2 log."""
    text = read_text(log_path)
    if not text:
        return "(Phase 2 log missing)"
    hits = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "focus=" in line or "prompt_variants" in line or "Executing Run" in line or "ITERATION" in line:
            hits.append(f"P2_Log_L{i}: {line.strip()}")
    return "\n".join(hits[:50])

# =============================================================================
# 3. Core Logic
# =============================================================================

def perform_advanced_analysis(
    *,
    dataset_name: str,
    logs_dir: str,
    pipe_cfg: Optional[Dict[str, Any]],
) -> None:
    """
    Main entry point for advanced metrics.
    Reads artifacts from finall_results, analyzes Phase 1 + 2 + 3.
    """
    final_results_dir = os.path.join(logs_dir, "finall_results")
    out_dir = os.path.join(logs_dir, "advanced_metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n🔍 [Advanced Analysis] Starting deep eval in: {out_dir}")

    # --- 1. Load Artifacts (All 3 Phases) ---
    # Phase 1: Design thinking (Summary Report)
    p1_report_path = os.path.join(final_results_dir, "phase1_summary_report.md")
    # Phase 2: Generation attempts (Log)
    p2_log_path = os.path.join(final_results_dir, "phase2.log")
    # Phase 3: Optimization trajectory (History JSON)
    p3_history_path = os.path.join(final_results_dir, "phase3_history_state.json")
    
    # Code: Final Result
    code_path = os.path.join(final_results_dir, "best_code.py")
    if not os.path.exists(code_path):
        nb_path = os.path.join(final_results_dir, "best_code.ipynb")
        if os.path.exists(nb_path):
            from runner_utils import export_notebook_as_py
            export_notebook_as_py(nb_path, code_path)

    # Read Content
    p1_text = read_text_limited(p1_report_path, max_chars=10000) or "(Phase 1 report missing)"
    p2_digest = _extract_phase2_digest(p2_log_path)
    p3_data = safe_read_json(p3_history_path) or []
    code_text = read_text(code_path)
    
    llm_cfg = resolve_report_llm_cfg(pipe_cfg)
    
    # -------------------------------------------------------
    # A. Mechanism Diversity (Full Trajectory: P1 -> P2 -> P3)
    # -------------------------------------------------------
    
    # Digest Phase 3
    p3_digest = []
    if isinstance(p3_data, list):
        for entry in p3_data:
            if isinstance(entry, dict):
                p3_digest.append(
                    f"P3_Iter_{entry.get('iter')}: Strategy='{entry.get('strategy')}', "
                    f"Focus='{entry.get('focus')}', Decision='{entry.get('decision')}'"
                )
    p3_str = "\n".join(p3_digest) if p3_digest else "(No Phase 3 history)"

    full_history_context = f"""
    === Phase 1: Initial Design Space (Conceptualization) ===
    {p1_text[:3000]}... [truncated]
    
    === Phase 2: Hypothesis Generation Space (Broad Search) ===
    {p2_digest}
    
    === Phase 3: Hypothesis Optimization Space (Deep Search) ===
    {p3_str}
    """
    
    div_sys = (
        "You are a meta-researcher evaluating the 'Mechanism Diversity' (GED) of a scientific AI agent. "
        "You must evaluate the evolution of hypotheses across THREE phases: Design (P1), Generation (P2), and Optimization (P3)."
    )
    div_user = """
    Analyze the full scientific exploration trajectory below:
    
    ${content}
    
    Evaluate 'Mechanism Diversity' on these dimensions (Score 0-10):
    1. **Design Originality (Phase 1)**: Did the initial analysis (P1) propose a novel or logical problem framing, or was it generic?
    2. **Hypothesis Breadth (Phase 2)**: Did Phase 2 explore distinct, non-overlapping strategies (High GED), or just minor variations?
    3. **Optimization Efficiency (Phase 3)**: Did Phase 3 effectively prune the space and converge, or randomly drift?
    4. **Overall Semantic Span**: The total conceptual distance covered from P1 start to P3 end.
    
    Return JSON format:
    {
        "design_originality_score": float,
        "hypothesis_breadth_score": float,
        "optimization_efficiency_score": float,
        "overall_semantic_span_score": float,
        "diversity_analysis_summary": "string summary (max 150 words)",
        "key_conceptual_shifts": ["list", "of", "major", "shifts"]
    }
    """
    
    print("   ... Analyzing Mechanism Diversity (P1 + P2 + P3)...")
    diversity_metrics = analyze_via_llm(full_history_context, div_sys, div_user, llm_cfg)
    diversity_metrics["valid_data"] = True

    # -------------------------------------------------------
    # B. Code Complexity (Scientific Parsimony)
    # -------------------------------------------------------
    
    complexity_metrics = {"valid_data": False}
    
    if code_text:
        print("   ... Analyzing Code Complexity (Final Result)...")
        # 1. Hard Metrics
        ast_mets = calculate_ast_metrics(code_text)
        
        # 2. Soft Metrics (LLM)
        comp_sys = (
            "You are a senior research engineer. Evaluate the 'Scientific Complexity' of this code. "
            "We value Ockham's Razor: the simplest sufficient explanation."
        )
        comp_user = """
        Review the final best code (truncated):
        
        ```python
        ${content}
        ```
        
        Evaluate (Score 0-10):
        1. **Parsimony Score**: Is the logic simple and direct? (10=Simple/Elegant, 0=Bloated).
        2. **Interpretability**: Can a scientist easily map this code to biological meaning?
        3. **Structural Robustness**: Is it modular and safe?
        
        Return JSON format:
        {
            "parsimony_score": float,
            "interpretability_score": float,
            "structural_robustness_score": float,
            "complexity_analysis_summary": "string summary"
        }
        """
        trunc_code = code_text[:20000]
        llm_comp = analyze_via_llm(trunc_code, comp_sys, comp_user, llm_cfg)
        
        complexity_metrics = {
            "valid_data": True,
            "algorithmic_metrics": ast_mets,
            "semantic_metrics": llm_comp
        }
    else:
        print("[WARN] No code found for Complexity analysis.")

    # -------------------------------------------------------
    # C. Save Results
    # -------------------------------------------------------
    full_report = {
        "dataset": dataset_name,
        "mechanism_diversity": diversity_metrics,
        "code_complexity": complexity_metrics,
        "generated_at": _now_iso()
    }
    
    json_path = os.path.join(out_dir, "advanced_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
        
    # Generate Markdown Summary
    md_lines = [
        f"# Advanced Analysis Report: {dataset_name}",
        f"**Date**: {_now_iso()}",
        "",
        "## 1. Mechanism Diversity (GED & Exploration)",
        "**Scope**: Phase 1 (Design) -> Phase 2 (Generation) -> Phase 3 (Optimization)",
        "",
        "### Scores (0-10)",
        f"- **Design Originality (P1)**: {diversity_metrics.get('design_originality_score', 'N/A')}",
        f"- **Hypothesis Breadth (P2)**: {diversity_metrics.get('hypothesis_breadth_score', 'N/A')}",
        f"- **Optimization Efficiency (P3)**: {diversity_metrics.get('optimization_efficiency_score', 'N/A')}",
        f"- **Overall Semantic Span**: {diversity_metrics.get('overall_semantic_span_score', 'N/A')}",
        "",
        "### Key Conceptual Shifts:",
        "\n".join([f"- {s}" for s in diversity_metrics.get('key_conceptual_shifts', [])]),
        "",
        f"> **Analysis**: {diversity_metrics.get('diversity_analysis_summary', 'N/A')}",
        "",
        "## 2. Code Complexity (Scientific Parsimony)",
        "**Target**: Final Best Code Solution",
        "",
        "### Algorithmic Metrics (AST)",
        f"- **Cyclomatic Complexity**: {complexity_metrics.get('algorithmic_metrics', {}).get('cyclomatic_complexity', 'N/A')}",
        f"- **Vocabulary Size**: {complexity_metrics.get('algorithmic_metrics', {}).get('vocabulary_size', 'N/A')}",
        "",
        "### Semantic Metrics (LLM)",
        f"- **Parsimony (Ockham's Razor)**: {complexity_metrics.get('semantic_metrics', {}).get('parsimony_score', 'N/A')}/10",
        f"- **Interpretability**: {complexity_metrics.get('semantic_metrics', {}).get('interpretability_score', 'N/A')}/10",
        "",
        f"> **Analysis**: {complexity_metrics.get('semantic_metrics', {}).get('complexity_analysis_summary', 'N/A')}"
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