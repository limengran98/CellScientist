#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Metrics Analysis Module.

Performs post-hoc analysis on the 'finall_results' artifacts to evaluate:
1. Mechanism Diversity (GED): The breadth of the hypothesis space explored.
2. Code Complexity: The simplicity and elegance of the scientific explanation (Code).

Outputs to: results/${dataset}/advanced_metrics/
"""

from __future__ import annotations

import ast
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from runner_utils import read_text, safe_read_json, safe_copy
from runner_report import resolve_report_llm_cfg, chat_text

# =============================================================================
# 1. AST-based Code Complexity (Algorithmic)
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """Calculates Cyclomatic Complexity (McCabe) via AST."""
    def __init__(self):
        self.complexity = 1  # Base complexity for the code block itself

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
        # We don't increment for def, but we visit body
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        # Count 'and' / 'or' as decision points
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

def calculate_ast_metrics(code_text: str) -> Dict[str, Any]:
    try:
        tree = ast.parse(code_text)
    except SyntaxError:
        return {"cyclomatic_complexity": -1, "loc": len(code_text.splitlines()), "notes": "Syntax Error in AST parse"}

    # 1. Cyclomatic Complexity
    visitor = ComplexityVisitor()
    visitor.visit(tree)
    
    # 2. Raw LOC & Logical LOC (rough proxy)
    lines = code_text.splitlines()
    loc = len(lines)
    sloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    
    # 3. Halstead-ish proxies (Vocabulary size)
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
# 2. LLM-based Evaluation
# =============================================================================

def analyze_via_llm(
    content: str,
    system_prompt: str,
    user_prompt_template: str,
    llm_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Generic helper to get JSON metrics from LLM."""
    
    user_msg = user_prompt_template.replace("${content}", content)
    
    # Force JSON instruction
    system_prompt += "\nOutput MUST be valid JSON only."
    
    resp = chat_text(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        llm_cfg
    )
    
    # Robust JSON parsing
    try:
        # Try to find JSON block if wrapped in markdown
        import re
        m = re.search(r"```json\s*(\{.*?\})\s*```", resp, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        return json.loads(resp)
    except Exception:
        print(f"[WARN] Failed to parse LLM JSON response. Raw: {resp[:100]}...")
        return {"error": "JSON Parse Failed", "raw_response": resp}

# =============================================================================
# 3. Core Logic
# =============================================================================

def perform_advanced_analysis(
    *,
    dataset_name: str,
    logs_dir: str,  # This acts as the 'output_base_dir' where 'finall_results' lives
    pipe_cfg: Optional[Dict[str, Any]],
) -> None:
    """
    Main entry point for advanced metrics.
    Reads from logs_dir/finall_results, writes to logs_dir/advanced_metrics.
    """
    final_results_dir = os.path.join(logs_dir, "finall_results")
    out_dir = os.path.join(logs_dir, "advanced_metrics")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n🔍 [Advanced Analysis] Starting deep eval in: {out_dir}")

    # Load artifacts
    history_path = os.path.join(final_results_dir, "phase3_history_state.json")
    code_path = os.path.join(final_results_dir, "best_code.py")
    
    # If .py doesn't exist, try converting .ipynb
    if not os.path.exists(code_path):
        nb_path = os.path.join(final_results_dir, "best_code.ipynb")
        if os.path.exists(nb_path):
            from runner_utils import export_notebook_as_py
            export_notebook_as_py(nb_path, code_path)

    history_data = safe_read_json(history_path) or []
    code_text = read_text(code_path)
    
    llm_cfg = resolve_report_llm_cfg(pipe_cfg)
    
    # -------------------------------------------------------
    # A. Mechanism Diversity (Hypothesis Space Breadth)
    # -------------------------------------------------------
    # Dimensions: 
    # 1. Semantic Trajectory (LLM): Did we circle or explore?
    # 2. Strategy Coverage (LLM): Categorical diversity.
    # 3. Iteration Count (Stat): Raw count.
    
    diversity_metrics = {"valid_data": False}
    
    if history_data and isinstance(history_data, list):
        # Extract digest for LLM
        history_digest = []
        for i, entry in enumerate(history_data):
            if not isinstance(entry, dict): continue
            history_digest.append(
                f"Iter {entry.get('iter')}: Strategy='{entry.get('strategy')}', "
                f"Focus='{entry.get('focus')}', Decision='{entry.get('decision')}'"
            )
        
        history_str = "\n".join(history_digest)
        
        div_sys = (
            "You are a meta-researcher evaluating the 'Mechanism Diversity' (Exploration Breadth) of a scientific discovery agent. "
            "Evaluate how diverse the hypotheses and strategies were across the iterations."
        )
        div_user = """
        Analyze the following optimization history log:
        
        ${content}
        
        Evaluate 'Mechanism Diversity' on these dimensions (Score 0-10):
        1. **Conceptual Diversity**: Did the agent try fundamentally different scientific approaches (e.g., data prep vs model arch vs loss function)?
        2. **Trajectory Efficiency**: Did the agent avoid redundant loops (low GED) and exploring new areas? (10 = highly efficient exploration, 0 = stuck in loop).
        3. **Hypothesis Depth**: Did the changes go beyond parameter tuning into structural/logic changes?
        
        Return JSON format:
        {
            "conceptual_diversity_score": float,
            "trajectory_efficiency_score": float,
            "hypothesis_depth_score": float,
            "mechanism_diversity_summary": "string summary",
            "detected_strategy_categories": ["list", "of", "categories"]
        }
        """
        
        llm_div = analyze_via_llm(history_str, div_sys, div_user, llm_cfg)
        diversity_metrics = {
            "valid_data": True,
            "iteration_count": len(history_data),
            **llm_div
        }
    else:
        print("[WARN] No history found for Mechanism Diversity analysis.")

    # -------------------------------------------------------
    # B. Code Complexity (Scientific Parsimony)
    # -------------------------------------------------------
    # Dimensions:
    # 1. Cyclomatic Complexity (AST)
    # 2. Vocabulary/Volume (AST)
    # 3. Readability (LLM)
    # 4. Occam's Score (LLM): Simplicity of explanation relative to function.
    
    complexity_metrics = {"valid_data": False}
    
    if code_text:
        # 1. Algorithmic
        ast_mets = calculate_ast_metrics(code_text)
        
        # 2. LLM Semantic Eval
        # We define "Code Complexity" here not just as "hard to read", but "Scientific Complexity".
        # A simpler code that explains data well is better (Occam's Razor).
        comp_sys = (
            "You are a senior research engineer evaluating scientific code. "
            "Assess the 'Code Complexity' and 'Scientific Parsimony' (Occam's Razor)."
        )
        comp_user = """
        Review the following scientific code (truncated if necessary):
        
        ```python
        ${content}
        ```
        
        Evaluate on these dimensions (Score 0-10, where 10 is Good/Simple/Parsimonious, 0 is Bad/Bloated/Complex):
        1. **Readability Score**: How easy is it to understand the logic?
        2. **Occam's Razor Score**: Is the solution as simple as possible to solve the problem, or over-engineered?
        3. **Modularity Score**: How well structured is the code?
        
        Return JSON format:
        {
            "readability_score": float,
            "occams_razor_score": float,
            "modularity_score": float,
            "complexity_analysis_summary": "string summary"
        }
        """
        # Truncate code for LLM context window safety
        trunc_code = code_text[:25000] 
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
        "",
        "## 1. Mechanism Diversity (Hypothesis Space Breadth)",
        f"- **Iteration Count**: {diversity_metrics.get('iteration_count', 'N/A')}",
        f"- **Conceptual Diversity**: {diversity_metrics.get('conceptual_diversity_score', 'N/A')}/10",
        f"- **Trajectory Efficiency**: {diversity_metrics.get('trajectory_efficiency_score', 'N/A')}/10",
        f"- **Hypothesis Depth**: {diversity_metrics.get('hypothesis_depth_score', 'N/A')}/10",
        "",
        "### Strategy Categories Detected:",
        ", ".join(diversity_metrics.get('detected_strategy_categories', [])),
        "",
        f"> **Summary**: {diversity_metrics.get('mechanism_diversity_summary', 'N/A')}",
        "",
        "## 2. Code Complexity (Scientific Simplicity)",
        "### Algorithmic Metrics (AST)",
        f"- **Cyclomatic Complexity**: {complexity_metrics.get('algorithmic_metrics', {}).get('cyclomatic_complexity', 'N/A')}",
        f"- **SLOC**: {complexity_metrics.get('algorithmic_metrics', {}).get('sloc', 'N/A')}",
        f"- **Vocabulary Size**: {complexity_metrics.get('algorithmic_metrics', {}).get('vocabulary_size', 'N/A')}",
        "",
        "### Semantic Metrics (LLM)",
        f"- **Occam's Razor Score (Simplicity)**: {complexity_metrics.get('semantic_metrics', {}).get('occams_razor_score', 'N/A')}/10",
        f"- **Readability Score**: {complexity_metrics.get('semantic_metrics', {}).get('readability_score', 'N/A')}/10",
        "",
        f"> **Summary**: {complexity_metrics.get('semantic_metrics', {}).get('complexity_analysis_summary', 'N/A')}"
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
    # Test stub
    print("Run via run_cellscientist.py")