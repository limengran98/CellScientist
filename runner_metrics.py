#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metrics extraction + log parsing + scoreboard printing.

Refactored out of run_cellscientist.py.
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from runner_config import get_nested
from runner_utils import safe_read_json, read_text


# =============================================================================
# Rich (optional)
# =============================================================================


def _maybe_console():
    try:
        from rich.console import Console

        return Console()
    except Exception:
        return None


def print_execution_plan(phase_map: Dict[str, Dict[str, Any]], dataset_name: str, console=None) -> None:
    """Pretty print phase configuration overview."""
    console = console if console is not None else _maybe_console()
    if not console:
        print(f"Plan for dataset: {dataset_name}")
        return

    from rich.table import Table

    table = Table(title=f"🧬 CellScientist Pipeline (Target: [bold green]{dataset_name}[/])")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Directory", style="blue")
    table.add_column("Model", style="magenta")
    table.add_column("Key Params", style="white")

    p1 = phase_map["Phase 1"]
    c1 = p1.get("_loaded_cfg", {})
    p1_model = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "llm", "model"])
    p1_runs = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "multi", "num_runs"])
    table.add_row("1. Design", p1["folder"], str(p1_model), f"Runs: {p1_runs}")

    p2 = phase_map["Phase 2"]
    c2 = p2.get("_loaded_cfg", {})
    p2_model = get_nested(c2, ["llm", "model"])
    p2_iters = get_nested(c2, ["experiment", "max_iterations"])
    table.add_row("2. Generate", p2["folder"], str(p2_model), f"Max Iters: {p2_iters}")

    p3 = phase_map["Phase 3"]
    c3 = p3.get("_loaded_cfg", {})
    p3_model = get_nested(c3, ["llm", "model"])
    p3_metric = get_nested(c3, ["review", "target_metric"])
    p3_dir = get_nested(c3, ["review", "direction"])
    table.add_row("3. Review", p3["folder"], str(p3_model), f"Target: {p3_metric} ({p3_dir})")

    console.print(table)
    console.print("")


# =============================================================================
# Metrics core
# =============================================================================


def planned_phase1_budget(phase1_cfg: Dict[str, Any]) -> int:
    nb_cfg = (((phase1_cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {})
    multi = nb_cfg.get("multi") or {}
    if not multi.get("enabled", False):
        return 0
    seeds = multi.get("seeds") or []
    variants = multi.get("prompt_variants") or []
    num_runs_cfg = int(multi.get("num_runs", 0) or 0)
    if not seeds or not variants or num_runs_cfg <= 0:
        return 0
    return min(num_runs_cfg, len(seeds), len(variants))


def extract_primary_score(metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
    if not metrics or not isinstance(metrics, dict):
        return None
    winner = metrics.get("winner")
    models = metrics.get("models") if isinstance(metrics.get("models"), dict) else None

    if not winner:
        if models:
            mk = [k for k in models.keys() if k != "config"]
            winner = mk[0] if mk else None
        else:
            mk = [k for k in metrics.keys() if k not in {"winner", "config", "models", "methods"}]
            winner = mk[0] if mk else None
    if not winner:
        return None

    m_data = metrics.get(winner)
    if not isinstance(m_data, dict) and models:
        m_data = models.get(winner)
    if not isinstance(m_data, dict):
        return None

    if isinstance(m_data.get("aggregate"), dict):
        val = m_data["aggregate"].get(metric_key)
        try:
            return float(val) if val is not None else None
        except Exception:
            return None

    pf = m_data.get("per_fold")
    if isinstance(pf, dict):
        vals = []
        for fold in pf.values():
            if isinstance(fold, dict):
                v = fold.get(metric_key)
                if v is None and isinstance(fold.get("metrics"), dict):
                    v = fold["metrics"].get(metric_key)
                try:
                    if v is not None:
                        vals.append(float(v))
                except Exception:
                    continue
        if vals:
            return sum(vals) / float(len(vals))
    return None


def mean_safe(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def find_scores_in_json(obj: Any) -> List[float]:
    out: List[float] = []
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(find_scores_in_json(x))
    elif isinstance(obj, dict):
        if isinstance(obj.get("scores"), list):
            out.extend(find_scores_in_json(obj["scores"]))
        else:
            for v in obj.values():
                out.extend(find_scores_in_json(v))
    return out


def pick_best(scores: List[float], direction: str = "maximize") -> Optional[float]:
    valid = [s for s in scores if isinstance(s, (int, float)) and s != -99 and s != float("inf")]
    if not valid:
        return None
    if direction.lower() == "minimize":
        return min(valid)
    return max(valid)


# =============================================================================
# Time Extraction Helpers (Log Parsing)
# =============================================================================

def _extract_timestamp(line: str) -> Optional[datetime]:
    """
    Attempts to extract a timestamp from the beginning of a log line.
    Supports formats:
    - [2025-01-01 12:00:00,123] ...
    - 2025-01-01 12:00:00,123 ...
    - [12:00:00] ...
    """
    # Try Regex for common patterns
    # 1. Standard full date time: YYYY-MM-DD HH:MM:SS
    m = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", line)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except:
            pass
            
    # 2. HH:MM:SS only (assume today/irrelevant date for diffs if within same day)
    m2 = re.search(r"(\d{2}:\d{2}:\d{2})", line)
    if m2:
        try:
            # Only time available, use dummy date
            t = datetime.strptime(m2.group(1), "%H:%M:%S")
            return t # Note: Date will be 1900-01-01, but delta works if runs don't cross midnight
        except:
            pass
            
    return None


def _sum_logs_duration(log_text: str, start_pat: str, end_pat: str, greedy_end: bool = False) -> float:
    """
    Calculates total duration by finding blocks between start_pat and end_pat.
    
    Args:
        greedy_end: If True, searches for the LAST end_pat before the NEXT start_pat.
                    (Useful for Phase 1/3 where multiple cells might run in one block).
    """
    total_seconds = 0.0
    lines = log_text.splitlines()
    
    current_start_time = None
    last_valid_end_time = None
    
    start_re = re.compile(start_pat)
    end_re = re.compile(end_pat)
    
    for line in lines:
        ts = _extract_timestamp(line)
        if not ts:
            continue
            
        # Check for Start
        if start_re.search(line):
            # If we were already in a block and greedy matching, commit the previous block
            if current_start_time and last_valid_end_time:
                 # Logic: New start means previous block is definitely over
                 delta = (last_valid_end_time - current_start_time).total_seconds()
                 if delta > 0: total_seconds += delta
                 last_valid_end_time = None
            
            # Start new block (or restart if we found a start before an end, handling retries)
            current_start_time = ts
            last_valid_end_time = None # Reset pending end
            continue
            
        # Check for End
        if current_start_time and end_re.search(line):
            if greedy_end:
                # Update candidate end time, but don't commit yet
                # We wait until the NEXT start or EOF to commit
                last_valid_end_time = ts
            else:
                # Immediate commit (Non-greedy)
                delta = (ts - current_start_time).total_seconds()
                if delta > 0: total_seconds += delta
                current_start_time = None # Close block
    
    # Commit pending greedy block at end of file
    if greedy_end and current_start_time and last_valid_end_time:
        delta = (last_valid_end_time - current_start_time).total_seconds()
        if delta > 0: total_seconds += delta
        
    return total_seconds


# =============================================================================
# Phase log parsers
# =============================================================================


def parse_phase1_log(log_text: str) -> Dict[str, Any]:
    run_ids = set(int(m.group(1)) for m in re.finditer(r"Executing design_analysis_\d{8}_\d{6}_Run(\d+)", log_text))
    finished_ids = set(int(m.group(1)) for m in re.finditer(r"Finished design_analysis_\d{8}_\d{6}_Run(\d+)", log_text))
    attempted = len(run_ids) if run_ids else len(finished_ids)
    succeeded = len(finished_ids)

    lines = log_text.splitlines()
    fail_positions = [i for i, ln in enumerate(lines) if "Failed (Cell" in ln or "Auto-Fix" in ln and "Failed" in ln]
    bug_runs: set[int] = set()
    for pos in fail_positions:
        for j in range(pos, min(pos + 50, len(lines))):
            m = re.search(r"Run(\d+)/", lines[j])
            if m:
                bug_runs.add(int(m.group(1)))
                break

    bug = len(bug_runs)
    clean_success = max(0, succeeded - len([r for r in bug_runs if r in finished_ids])) if attempted else 0
    
    # Phase 1: From "Starting Kernel" to LAST "Success (Cell xx)" in that run
    exec_time = _sum_logs_duration(
        log_text, 
        start_pat=r"Starting Kernel", 
        end_pat=r"Success \(Cell \d+\)", 
        greedy_end=True 
    )

    return {
        "attempted": attempted, 
        "succeeded": succeeded, 
        "bug": bug, 
        "clean_success": clean_success,
        "exec_time": exec_time
    }


def parse_phase2_log(log_text: str, metric: str) -> Dict[str, Any]:
    # Add success/early stop markers to prevent false negatives
    success_markers = [
        "Success threshold met",
        "Target metric reached",
        "Early stop triggered",
        "Optimization converged",
        "Stopping early",
        "Success!"
    ]
    bug_markers = [
        "Notebook Generation Failed",
        "LLM_GEN_FAILURE",
        "CRITICAL PARSE FAILURE",
        "[GRAPH] ❌",
        "Error in Node",
        "Initiating Adaptive Fix",
        "[FIX]",
        "Traceback",
        "Exception:",
    ]
    
    iters: Dict[int, Dict[str, Any]] = {}
    cur_iter: Optional[int] = None
    global_early_success = False

    for ln in log_text.splitlines():
        if any(x in ln for x in success_markers):
            global_early_success = True

        m = re.search(r"ITERATION\s+(\d+)/(\d+)", ln)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None, "explicit_success": False})
            continue

        if cur_iter is None:
            continue
            
        rec = iters.setdefault(cur_iter, {"bug": False, "score": None, "explicit_success": False})

        if any(x in ln for x in bug_markers):
            rec["bug"] = True

        mm = re.search(rf"\[CHECK\].*?\b{re.escape(metric)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if mm:
            try:
                rec["score"] = float(mm.group(1))
            except Exception:
                pass
        
        if any(x in ln for x in success_markers):
            rec["explicit_success"] = True

    attempted = len(iters)
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float))]
    
    bug = 0
    clean_success = 0
    succeeded = 0

    for idx, v in iters.items():
        is_bug = v["bug"]
        has_score = isinstance(v.get("score"), (int, float))
        is_success_stop = v["explicit_success"] or (global_early_success and idx == max(iters.keys()))
        
        if is_bug:
            bug += 1
        
        if has_score or is_success_stop:
            succeeded += 1
            if not is_bug:
                clean_success += 1

    succeeded = min(succeeded, attempted)
    clean_success = min(clean_success, attempted)
    
    # Phase 2: Start -> Shutdown is a clean closed block. No greedy needed.
    exec_time = _sum_logs_duration(
        log_text, 
        start_pat=r"Initializing Kernel in", 
        end_pat=r"Shutting down kernel\.", 
        greedy_end=False 
    )

    return {
        "attempted": attempted, 
        "succeeded": succeeded, 
        "bug": bug, 
        "clean_success": clean_success, 
        "scores": scores,
        "exec_time": exec_time
    }


def parse_phase3_log(log_text: str, metric: str) -> Dict[str, Any]:
    success_markers = [
        "Target metric reached",
        "Success threshold",
        "Optimization finished early",
        "Stopping early",
        "Success!"
    ]
    bug_markers = [
        "Invalid LLM response. Skipping",
        "LLM Generation Failed",
        "Report generation failed",
        "Static Fallback Report",
        "Request failed after",
        "Errors Found",
        "Final Execution Failed",
        "Logic Error",
        "Traceback",
        "Exception:",
    ]
    iters: Dict[int, Dict[str, Any]] = {}
    cur_iter: Optional[int] = None
    global_early_success = False

    for ln in log_text.splitlines():
        if any(x in ln for x in success_markers):
            global_early_success = True

        m = re.search(r"optimization \(Iter\s+(\d+)\)", ln, re.I)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            continue

        m2 = re.search(r"\[RESULT\]\s*Iteration\s+(\d+)\s+Summary", ln)
        if m2:
            cur_iter = int(m2.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            continue

        if cur_iter is None:
            continue
            
        if any(x in ln for x in bug_markers):
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            iters[cur_iter]["bug"] = True

        ms = re.search(r"Candidate Score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if ms:
            try:
                iters.setdefault(cur_iter, {"bug": False, "score": None})
                iters[cur_iter]["score"] = float(ms.group(1))
            except Exception:
                pass

    attempted = len(iters)
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float)) and v.get("score") != -999]
    
    bug = 0
    clean_success = 0
    succeeded = 0
    
    for idx, v in iters.items():
        is_bug = v["bug"]
        has_score = (isinstance(v.get("score"), (int, float)) and v.get("score") != -999)
        is_success_stop = (global_early_success and idx == max(iters.keys())) if iters else False

        if is_bug:
            bug += 1
        
        if has_score or is_success_stop:
            succeeded += 1
            if not is_bug:
                clean_success += 1
    
    # Phase 3: Start -> [EXEC] Cell xx Done. 
    # Use greedy_end=True to capture up to the LAST cell done in that iteration block.
    exec_time = _sum_logs_duration(
        log_text, 
        start_pat=r"\[EXEC\]\s*💉\s*Injecting Code Variables", 
        end_pat=r"\[EXEC\]\s*✅\s*Cell\s*\d+\s*Done\.", 
        greedy_end=True 
    )

    return {
        "attempted": attempted, 
        "succeeded": succeeded, 
        "bug": bug, 
        "clean_success": clean_success, 
        "scores": scores,
        "exec_time": exec_time
    }


# =============================================================================
# Artifact-based score extraction
# =============================================================================


def phase1_scores_from_artifacts(design_dir: str) -> Tuple[Optional[float], Optional[float]]:
    cand_files: List[str] = []
    for name in ["heuristic_scores.json", "scores.json", "heuristics.json", "run_scores.json"]:
        p = os.path.join(design_dir, name)
        if os.path.exists(p):
            cand_files.append(p)
    cand_files.extend(sorted(glob.glob(os.path.join(design_dir, "*score*.json"))))

    scores: List[float] = []
    for p in cand_files:
        obj = safe_read_json(p)
        if obj is None:
            continue
        vals = find_scores_in_json(obj)
        vals = [v for v in vals if isinstance(v, (int, float)) and not (v != v)]
        if vals:
            scores = vals
            break

    avg = mean_safe(scores) if scores else None
    best = max(scores) if scores else None

    if best is None:
        ref = safe_read_json(os.path.join(design_dir, "reference", "reference.json")) or {}
        if isinstance(ref.get("score"), (int, float)):
            best = float(ref["score"])
            if avg is None:
                avg = best
    return avg, best


def phase2_scores_from_artifacts(ge_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    scores: List[float] = []
    prompt_root = os.path.join(ge_dir, "prompt")
    if not os.path.exists(prompt_root):
        return scores

    run_dirs = []
    for d in glob.glob(os.path.join(prompt_root, "prompt_run_*")):
        try:
            mt = os.path.getmtime(d)
        except Exception:
            continue
        if t_start - 30 <= mt <= t_end + 30:
            run_dirs.append(d)
    run_dirs.sort(key=lambda p: os.path.getmtime(p))

    for d in run_dirs:
        met_path = os.path.join(d, "metrics.json")
        met = safe_read_json(met_path) or {}
        v = extract_primary_score(met, metric)
        if isinstance(v, (int, float)):
            scores.append(float(v))
    return scores


def phase3_scores_from_artifacts(rf_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    scores: List[float] = []
    best_path = None
    best_mtime = -1.0
    for name in os.listdir(rf_dir) if os.path.exists(rf_dir) else []:
        p = os.path.join(rf_dir, name)
        if not os.path.isdir(p) or not name.startswith("review_run_"):
            continue
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        if t_start - 30 <= mt <= t_end + 30 and mt > best_mtime:
            best_mtime = mt
            best_path = p
    if not best_path:
        return scores

    hist = safe_read_json(os.path.join(best_path, "history_state.json"))
    if isinstance(hist, list):
        for rec in hist:
            if not isinstance(rec, dict):
                continue
            sc = rec.get("score")
            if isinstance(sc, (int, float)) and sc != -999:
                scores.append(float(sc))
        return scores

    mb = safe_read_json(os.path.join(best_path, "metrics_best.json")) or {}
    v = extract_primary_score(mb, metric)
    if isinstance(v, (int, float)):
        scores.append(float(v))
    return scores


def rates(attempted: int, clean_success: int, succeeded: int, bug: int) -> Tuple[float, float, float]:
    """
    Revised logic:
    - Success Rate (total_success_rate): All successful runs / Attempted (Includes Auto-Fixed).
    - Clean Rate (zero_shot_sr): Runs that succeeded WITHOUT auto-fix / Attempted.
    - Bug Rate: Runs that triggered auto-fix / Attempted.
    """
    if attempted <= 0:
        return 0.0, 0.0, 0.0
    
    total_success_rate = succeeded / float(attempted)
    zero_shot_sr = clean_success / float(attempted)
    bug_rate = bug / float(attempted)
    
    return total_success_rate, zero_shot_sr, bug_rate


def print_final_scoreboard(summary: Dict[str, Any], console=None) -> None:
    stages = summary.get("stages", {})
    console = console if console is not None else _maybe_console()
    if console:
        from rich.table import Table

        table = Table(title=f"📊 Scoreboard (dataset={summary.get('dataset')})")
        table.add_column("Stage")
        table.add_column("Success Rate ↑", justify="right", style="green")
        table.add_column("Zero-Shot SR ↑", justify="right", style="cyan")
        table.add_column("Bug Rate ↓", justify="right", style="red")
        table.add_column("Avg@Budget", justify="right")
        table.add_column("Best@Budget", justify="right")
        table.add_column("Metric")
        table.add_column("Budget", justify="right")
        table.add_column("Attempted", justify="right")
        
        # [NEW] Add Non-Exec time column
        table.add_column("Non-Exec (s)", justify="right", style="yellow")
        table.add_column("Total Time (s)", justify="right")

        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})
            sr = row.get("success_rate")
            clean_sr = row.get("clean_rate")
            bugr = row.get("bug_rate")
            
            sr_s = f"{sr:.3f}" if isinstance(sr, (int, float)) else "-"
            clean_s = f"{clean_sr:.3f}" if isinstance(clean_sr, (int, float)) else "-"
            bug_s = f"{bugr:.3f}" if isinstance(bugr, (int, float)) else "-"
            
            avg = row.get("avg_at_budget")
            avg_s = f"{avg:.4f}" if isinstance(avg, (int, float)) else "-"
            best = row.get("best_at_budget")
            best_s = f"{best:.4f}" if isinstance(best, (int, float)) else "-"
            metric = str(row.get("best_metric", "-"))
            budget = row.get("budget")
            budget_s = str(budget) if budget is not None else "-"
            attempted = row.get("attempted")
            attempted_s = str(attempted) if attempted is not None else "-"
            
            tsec = row.get("time_sec")
            tsec_s = f"{tsec:.1f}" if isinstance(tsec, (int, float)) else "-"
            
            # [NEW] Calculate Non-Exec Time
            exec_time = row.get("exec_time", 0.0)
            non_exec = (tsec - exec_time) if isinstance(tsec, (int, float)) and isinstance(exec_time, (int, float)) else 0.0
            non_exec_s = f"{non_exec:.1f}"

            table.add_row(stage_name, sr_s, clean_s, bug_s, avg_s, best_s, metric, budget_s, attempted_s, non_exec_s, tsec_s)

        console.print(table)
    else:
        print("\n=== Scoreboard ===")
        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})
            tsec = row.get("time_sec", 0.0)
            exec_time = row.get("exec_time", 0.0)
            non_exec = tsec - exec_time if isinstance(tsec, (int, float)) else 0.0
            print(
                f"{stage_name}: Success={row.get('success_rate')}, ZeroShot={row.get('clean_rate')}, "
                f"BugRate={row.get('bug_rate')}, Avg={row.get('avg_at_budget')}, Best={row.get('best_at_budget')}, "
                f"Metric={row.get('best_metric')}, Non-Exec={non_exec:.1f}s, Time={row.get('time_sec')}"
            )