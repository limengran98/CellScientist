#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run_cellscientist.py
#
# Pipeline runner + robust metrics scoreboard:
# - Success Rate (clean): successful runs WITHOUT any bug/recovery signals
# - Robust SR: successful runs INCLUDING auto-fix/retries/fallbacks
# - Bug Rate: fraction of attempts that had any bug/recovery signals (even if eventually succeeded)
# - Avg@Budget: average primary metric over successful attempts only (crashes / no-metric are excluded)
# - Best@Budget: best primary metric over successful attempts only
#
# NOTE: Phase 1 metric is heuristic_score (not comparable to PCC). Total Avg/Best are computed on the
# pipeline final metric (usually Phase 3 target_metric) using Phase 2/3 scores only if they match.

import os
import sys
import json
import time
import datetime
import subprocess
import re
import glob
from typing import Dict, Any, List, Optional, Tuple

# =============================================================================
# ⚙️ Phase Map
# =============================================================================

PHASE_MAP = {
    "Phase 1": {
        "folder": "Design_Analysis",
        "script": "cellscientist_phase_1.py",
        "config": "design_analysis_config.json",
        "cmd_args": []
    },
    "Phase 2": {
        "folder": "Generate_Execution",
        "script": "cellscientist_phase_2.py",
        "config": "generate_execution_config.json",
        "cmd_args": ["run", "--use-idea"]
    },
    "Phase 3": {
        "folder": "Review_Feedback",
        "script": "cellscientist_phase_3.py",
        "config": "review_feedback_config.json",
        "cmd_args": []
    }
}

PHASE_LOG_NAME = {
    "Phase 1": "phase1.log",
    "Phase 2": "phase2.log",
    "Phase 3": "phase3.log",
}

# =============================================================================
# Rich (optional)
# =============================================================================

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except Exception:
    console = None

# =============================================================================
# IO: Tee Logger
# =============================================================================

class TeeStream:
    """Write-through stream to multiple underlying streams."""
    def __init__(self, *streams):
        self.streams = [s for s in streams if s is not None]
        self.encoding = getattr(self.streams[0], "encoding", "utf-8") if self.streams else "utf-8"

    def write(self, data: str):
        if data is None:
            return 0
        n = 0
        for s in self.streams:
            try:
                n = s.write(data)
            except Exception:
                pass
        return n

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self.streams:
            try:
                if hasattr(s, "isatty") and s.isatty():
                    return True
            except Exception:
                continue
        return False

    def fileno(self):
        for s in self.streams:
            if hasattr(s, "fileno"):
                try:
                    return s.fileno()
                except Exception:
                    continue
        raise OSError("No underlying fileno")

# =============================================================================
# Helpers
# =============================================================================

def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _ensure_project_cwd():
    """Allow running from anywhere; keep backwards compatible behavior."""
    root = _project_root()
    if os.path.exists(os.path.join(os.getcwd(), "Design_Analysis")):
        return
    if os.path.exists(os.path.join(root, "Design_Analysis")):
        os.chdir(root)
        return
    print("❌ Error: Run this script from the 'CellScientist' root directory (or keep the repo intact).")
    sys.exit(1)

def get_config_path(phase_info: Dict[str, Any]) -> str:
    """Return the config path for a phase.

    Backwards compatible:
    - If `phase_info["config"]` is an absolute path (or an existing path), use it directly.
    - Otherwise, treat it as a filename relative to the phase folder.
    """
    cfg = phase_info["config"]
    if not cfg:
        return os.path.join(phase_info["folder"], cfg)
    # absolute or already-resolved path
    if os.path.isabs(cfg) or os.path.exists(cfg):
        return cfg
    return os.path.join(phase_info["folder"], cfg)

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Pipeline-level config (optional)
# =============================================================================

def _deep_merge(dst: Any, src: Any) -> Any:
    """Recursively merge src into dst and return merged (does not mutate inputs)."""
    if isinstance(dst, dict) and isinstance(src, dict):
        out = dict(dst)
        for k, v in src.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    # for lists / scalars: src wins if not None
    return src if src is not None else dst

def _set_nested(d: Dict[str, Any], keys: List[str], value: Any):
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _load_pipeline_config() -> Optional[Dict[str, Any]]:
    """Load pipeline_config.json if present.

    Priority:
    1) env CELL_SCI_PIPELINE_CONFIG
    2) repo_root/pipeline_config.json
    """
    env_path = os.environ.get("CELL_SCI_PIPELINE_CONFIG")
    if env_path and os.path.exists(env_path):
        try:
            return load_json(env_path)
        except Exception:
            return None

    default_path = os.path.join(_project_root(), "pipeline_config.json")
    if os.path.exists(default_path):
        try:
            return load_json(default_path)
        except Exception:
            return None
    return None

def _apply_pipeline_overrides(phase_name: str, phase_cfg: Dict[str, Any], pipe_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply pipeline-level overrides to a single phase config."""
    cfg = dict(phase_cfg)

    # 1) dataset_name (common)
    if isinstance(pipe_cfg.get("dataset_name"), str) and pipe_cfg["dataset_name"].strip():
        cfg["dataset_name"] = pipe_cfg["dataset_name"].strip()

    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}
    # 2) CUDA / GPU selection (common)
    cuda_id = common.get("cuda_device_id", None)
    if cuda_id is not None:
        if phase_name == "Phase 1":
            _set_nested(cfg, ["phases", "task_analysis", "llm_notebook", "exec", "cuda_device_id"], cuda_id)
        elif phase_name == "Phase 2":
            _set_nested(cfg, ["exec", "cuda_device_id"], cuda_id)
        elif phase_name == "Phase 3":
            # Phase 3 mostly respects env CUDA_VISIBLE_DEVICES; keep here for future compatibility
            _set_nested(cfg, ["exec", "cuda_device_id"], cuda_id)

    # 3) LLM defaults (common)
    llm_common = pipe_cfg.get("llm") if isinstance(pipe_cfg.get("llm"), dict) else None
    if isinstance(llm_common, dict) and llm_common:
        if phase_name == "Phase 1":
            cur = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {}).get("llm") or {}
            merged = _deep_merge(cur, llm_common)
            _set_nested(cfg, ["phases", "task_analysis", "llm_notebook", "llm"], merged)
        else:
            cur = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
            cfg["llm"] = _deep_merge(cur, llm_common)

    # 4) Paths (common): merge into top-level `paths` if present
    paths_common = pipe_cfg.get("paths") if isinstance(pipe_cfg.get("paths"), dict) else None
    if isinstance(paths_common, dict) and paths_common:
        cur = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
        cfg["paths"] = _deep_merge(cur, paths_common)

    # 5) Phase-specific overrides (escape hatch)
    phase_overrides = pipe_cfg.get("phase_overrides") if isinstance(pipe_cfg.get("phase_overrides"), dict) else {}
    po = phase_overrides.get(phase_name) if isinstance(phase_overrides.get(phase_name), dict) else None
    if isinstance(po, dict) and po:
        cfg = _deep_merge(cfg, po)

    return cfg

def _materialize_merged_configs(pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Write merged per-phase configs under each phase folder and update PHASE_MAP in-place.

    Returns: {phase_name: merged_config_path}
    """
    merged_paths: Dict[str, str] = {}
    for phase_name, info in PHASE_MAP.items():
        base_cfg_path = get_config_path(info)
        base_cfg = load_json(base_cfg_path)

        merged_cfg = _apply_pipeline_overrides(phase_name, base_cfg, pipe_cfg)

        # write under phase folder to preserve relative-path semantics
        phase_folder = info["folder"]
        cache_dir = os.path.join(phase_folder, "_pipeline_cache")
        os.makedirs(cache_dir, exist_ok=True)

        base_name = os.path.basename(info["config"])
        if base_name.endswith(".json"):
            out_name = base_name[:-5] + ".merged.json"
        else:
            out_name = base_name + ".merged.json"

        out_path = os.path.abspath(os.path.join(cache_dir, out_name))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged_cfg, f, ensure_ascii=False, indent=2)

        info["config"] = out_path  # make phase run with merged config
        info["_loaded_cfg"] = merged_cfg
        merged_paths[phase_name] = out_path

    return merged_paths

def _pipeline_extra_env(pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Env vars to pass to all phase subprocesses."""
    env_out: Dict[str, str] = {}

    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}
    if common.get("cuda_visible_devices") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_visible_devices"])
    elif common.get("cuda_device_id") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_device_id"])

    env_cfg = pipe_cfg.get("env") if isinstance(pipe_cfg.get("env"), dict) else {}
    for k, v in env_cfg.items():
        if v is None:
            continue
        env_out[str(k)] = str(v)

    return env_out


def get_nested(data: Dict[str, Any], keys: List[str], default="N/A"):
    val: Any = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val

def validate_configs() -> str:
    dataset_names: Dict[str, str] = {}
    for name, info in PHASE_MAP.items():
        path = get_config_path(info)
        cfg = load_json(path)
        ds = cfg.get("dataset_name", "MISSING")
        dataset_names[name] = ds
        info["_loaded_cfg"] = cfg

    unique = set(dataset_names.values())
    if len(unique) > 1:
        if console:
            console.print(Panel("[bold red]CRITICAL ERROR: 'dataset_name' mismatch![/]", border_style="red"))
            for k, v in dataset_names.items():
                console.print(f"  - {k}: [red]{v}[/]")
        else:
            print("CRITICAL ERROR: 'dataset_name' mismatch!")
            print(dataset_names)
        sys.exit(1)

    return list(unique)[0]

def print_execution_plan(dataset_name: str):
    if not console:
        print(f"Plan for dataset: {dataset_name}")
        return

    table = Table(title=f"🧬 CellScientist Pipeline (Target: [bold green]{dataset_name}[/])")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Directory", style="blue")
    table.add_column("Model", style="magenta")
    table.add_column("Key Params", style="white")

    p1 = PHASE_MAP["Phase 1"]
    c1 = p1["_loaded_cfg"]
    p1_model = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "llm", "model"])
    p1_runs = get_nested(c1, ["phases", "task_analysis", "llm_notebook", "multi", "num_runs"])
    table.add_row("1. Design", p1["folder"], str(p1_model), f"Runs: {p1_runs}")

    p2 = PHASE_MAP["Phase 2"]
    c2 = p2["_loaded_cfg"]
    p2_model = get_nested(c2, ["llm", "model"])
    p2_iters = get_nested(c2, ["experiment", "max_iterations"])
    table.add_row("2. Generate", p2["folder"], str(p2_model), f"Max Iters: {p2_iters}")

    p3 = PHASE_MAP["Phase 3"]
    c3 = p3["_loaded_cfg"]
    p3_model = get_nested(c3, ["llm", "model"])
    p3_metric = get_nested(c3, ["review", "target_metric"])
    table.add_row("3. Review", p3["folder"], str(p3_model), f"Target: {p3_metric}")

    console.print(table)
    console.print("")

def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _atomic_write_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _results_root_for_dataset(dataset_name: str) -> str:
    return os.path.abspath(os.path.join(_project_root(), "results", dataset_name))

def _setup_logging(results_root: str) -> Tuple[str, str, Any]:
    # Timestamped logs dir (per pipeline run)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(results_root, f"logs_{ts}")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{ts}.log")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    log_fp = open(log_path, "a", encoding="utf-8")

    sys.stdout = TeeStream(sys.__stdout__, log_fp)  # type: ignore
    sys.stderr = TeeStream(sys.__stderr__, log_fp)  # type: ignore

    if console:
        console.print(f"📝 Logging console output to: [underline]{log_path}[/]")
    else:
        print(f"📝 Logging console output to: {log_path}")

    return logs_dir, log_path, log_fp

def _append_phase_header(phase_fp, dataset: str, phase_name: str, cmd: List[str], cwd: str):
    if not phase_fp:
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    phase_fp.write("\n" + "=" * 88 + "\n")
    phase_fp.write(f"[{ts}] dataset={dataset} | {phase_name}\n")
    phase_fp.write(f"cwd={cwd}\n")
    phase_fp.write(f"cmd={' '.join(cmd)}\n")
    phase_fp.write("=" * 88 + "\n")
    phase_fp.flush()

def _run_cmd_streamed(cmd: List[str], cwd: str, phase_fp=None, extra_env: Optional[Dict[str, str]] = None):
    env = os.environ.copy()
    if extra_env:
        for k, v in extra_env.items():
            if v is None:
                continue
            env[str(k)] = str(v)

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except Exception:
            pass
        if phase_fp:
            try:
                phase_fp.write(line)
                phase_fp.flush()
            except Exception:
                pass
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

# =============================================================================
# Metrics core
# =============================================================================

def _planned_phase1_budget(phase1_cfg: Dict[str, Any]) -> int:
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

def _extract_primary_score(metrics: Dict[str, Any], metric_key: str) -> Optional[float]:
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

def _mean_safe(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / float(len(vals))

def _read_text(path: str) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def _find_scores_in_json(obj: Any) -> List[float]:
    out: List[float] = []
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, list):
        for x in obj:
            out.extend(_find_scores_in_json(x))
    elif isinstance(obj, dict):
        # common patterns
        if isinstance(obj.get("scores"), list):
            out.extend(_find_scores_in_json(obj["scores"]))
        else:
            for v in obj.values():
                out.extend(_find_scores_in_json(v))
    return out

# =============================================================================
# Phase log parsers (for clean/robust/bug)
# =============================================================================

def _parse_phase1_log(log_text: str) -> Dict[str, Any]:
    """
    Phase 1 runs are parallel; we infer run ids and bug runs via nearest RunX path following a cell failure.
    """
    run_ids = set(int(m.group(1)) for m in re.finditer(r"Executing design_analysis_\d{8}_\d{6}_Run(\d+)", log_text))
    finished_ids = set(int(m.group(1)) for m in re.finditer(r"Finished design_analysis_\d{8}_\d{6}_Run(\d+)", log_text))
    attempted = len(run_ids) if run_ids else len(finished_ids)
    succeeded = len(finished_ids)

    lines = log_text.splitlines()
    fail_positions = [i for i, ln in enumerate(lines) if "Failed (Cell" in ln or "Auto-Fix" in ln and "Failed" in ln]
    bug_runs: set = set()
    for pos in fail_positions:
        for j in range(pos, min(pos + 50, len(lines))):
            m = re.search(r"Run(\d+)/", lines[j])
            if m:
                bug_runs.add(int(m.group(1)))
                break

    bug = len(bug_runs)
    clean_success = max(0, succeeded - len([r for r in bug_runs if r in finished_ids])) if attempted else 0

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "bug": bug,
        "clean_success": clean_success,
    }

def _parse_phase2_log(log_text: str, metric: str) -> Dict[str, Any]:
    """
    Iteration-level parsing.
    Success = iteration produced a numeric score for {metric}.
    Bug = iteration had any error/recovery signals (graph error, LLM parse failure, auto-fix, traceback).
    """
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
    for ln in log_text.splitlines():
        m = re.search(r"ITERATION\s+(\d+)/(\d+)", ln)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            continue

        if cur_iter is None:
            continue

        if any(x in ln for x in bug_markers):
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            iters[cur_iter]["bug"] = True

        # metric line
        mm = re.search(rf"\[CHECK\].*?\b{re.escape(metric)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if mm:
            try:
                iters.setdefault(cur_iter, {"bug": False, "score": None})
                iters[cur_iter]["score"] = float(mm.group(1))
            except Exception:
                pass

    attempted = len(iters)
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float))]
    succeeded = len(scores)
    bug = sum(1 for v in iters.values() if v.get("bug") is True)
    clean_success = sum(1 for v in iters.values() if isinstance(v.get("score"), (int, float)) and not v.get("bug"))

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "bug": bug,
        "clean_success": clean_success,
        "scores": scores,
    }

def _parse_phase3_log(log_text: str, metric: str) -> Dict[str, Any]:
    """
    Iteration-level parsing.
    Success = iteration produced Candidate Score.
    Bug = invalid LLM response / report fallback / execution recovery signals.
    """
    bug_markers = [
        "Invalid LLM response. Skipping",
        "LLM Generation Failed",
        "Report generation failed",
        "Static Fallback Report",
        "Request failed after",  # retry exhausted
        "Errors Found",
        "Final Execution Failed",
        "Logic Error",
        "Traceback",
        "Exception:",
    ]

    iters: Dict[int, Dict[str, Any]] = {}
    cur_iter: Optional[int] = None

    for ln in log_text.splitlines():
        m = re.search(r"optimization \(Iter\s+(\d+)\)", ln, re.I)
        if m:
            cur_iter = int(m.group(1))
            iters.setdefault(cur_iter, {"bug": False, "score": None})
            continue

        # result summary carries explicit iter id; prefer it
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

        # candidate score line (may appear without metric label)
        ms = re.search(r"Candidate Score:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", ln)
        if ms:
            try:
                iters.setdefault(cur_iter, {"bug": False, "score": None})
                iters[cur_iter]["score"] = float(ms.group(1))
            except Exception:
                pass

    attempted = len(iters)
    scores = [v["score"] for v in iters.values() if isinstance(v.get("score"), (int, float)) and v.get("score") != -999]
    succeeded = len(scores)
    bug = sum(1 for v in iters.values() if v.get("bug") is True)
    clean_success = sum(1 for v in iters.values() if isinstance(v.get("score"), (int, float)) and v.get("score") != -999 and not v.get("bug"))

    return {
        "attempted": attempted,
        "succeeded": succeeded,
        "bug": bug,
        "clean_success": clean_success,
        "scores": scores,
    }

# =============================================================================
# Phase stats from artifacts (+ log quality signals)
# =============================================================================

def _phase1_scores_from_artifacts(design_dir: str) -> Tuple[Optional[float], Optional[float]]:
    # Try to load a per-run score list if present
    cand_files: List[str] = []
    for name in ["heuristic_scores.json", "scores.json", "heuristics.json", "run_scores.json"]:
        p = os.path.join(design_dir, name)
        if os.path.exists(p):
            cand_files.append(p)

    # fallback: any json containing "score" in filename
    cand_files.extend(sorted(glob.glob(os.path.join(design_dir, "*score*.json"))))

    scores: List[float] = []
    for p in cand_files:
        obj = _safe_read_json(p)
        if obj is None:
            continue
        vals = _find_scores_in_json(obj)
        # Heuristic score is usually >1; filter obvious non-scores if mixed
        vals = [v for v in vals if isinstance(v, (int, float)) and not (v != v)]  # drop NaN
        if vals:
            scores = vals
            break

    avg = _mean_safe(scores) if scores else None
    best = max(scores) if scores else None

    # If not found, use reference.json best score
    if best is None:
        ref = _safe_read_json(os.path.join(design_dir, "reference", "reference.json")) or {}
        if isinstance(ref.get("score"), (int, float)):
            best = float(ref["score"])
            if avg is None:
                avg = best
    return avg, best

def _phase2_scores_from_artifacts(ge_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    """
    Prefer prompt_run_* metrics.json within the phase time window.
    """
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
        if t_start - 30 <= mt <= t_end + 30:  # buffer
            run_dirs.append(d)
    run_dirs.sort(key=lambda p: os.path.getmtime(p))

    for d in run_dirs:
        met_path = os.path.join(d, "metrics.json")
        met = _safe_read_json(met_path) or {}
        v = _extract_primary_score(met, metric)
        if isinstance(v, (int, float)):
            scores.append(float(v))
    return scores

def _phase3_scores_from_artifacts(rf_dir: str, metric: str, t_start: float, t_end: float) -> List[float]:
    scores: List[float] = []
    # find latest review_run_ within time window
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

    hist = _safe_read_json(os.path.join(best_path, "history_state.json"))
    if isinstance(hist, list):
        for rec in hist:
            if not isinstance(rec, dict):
                continue
            sc = rec.get("score")
            if isinstance(sc, (int, float)) and sc != -999:
                scores.append(float(sc))
        return scores

    # fallback: metrics_best.json
    mb = _safe_read_json(os.path.join(best_path, "metrics_best.json")) or {}
    v = _extract_primary_score(mb, metric)
    if isinstance(v, (int, float)):
        scores.append(float(v))
    return scores

def _rates(attempted: int, clean_success: int, succeeded: int, bug: int) -> Tuple[float, float, float]:
    if attempted <= 0:
        return 0.0, 0.0, 0.0
    success_rate = clean_success / float(attempted)
    robust_sr = succeeded / float(attempted)
    bug_rate = bug / float(attempted)
    return success_rate, robust_sr, bug_rate

def _print_final_scoreboard(summary: Dict[str, Any]):
    stages = summary.get("stages", {})

    if console:
        table = Table(title=f"📊 Scoreboard (dataset={summary.get('dataset')})")
        table.add_column("Stage")
        table.add_column("Success Rate ↑", justify="right")
        table.add_column("Robust SR ↑", justify="right")
        table.add_column("Bug Rate ↓", justify="right")
        table.add_column("Avg@Budget ↑", justify="right")
        table.add_column("Best@Budget ↑", justify="right")
        table.add_column("Metric")
        table.add_column("Budget", justify="right")
        table.add_column("Attempted", justify="right")
        table.add_column("Time ↓ (s)", justify="right")

        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})
            sr = row.get("success_rate")
            robust = row.get("robust_sr")
            bugr = row.get("bug_rate")
            sr_s = f"{sr:.3f}" if isinstance(sr, (int, float)) else "-"
            robust_s = f"{robust:.3f}" if isinstance(robust, (int, float)) else "-"
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
            table.add_row(stage_name, sr_s, robust_s, bug_s, avg_s, best_s, metric, budget_s, attempted_s, tsec_s)

        console.print(table)
    else:
        print("\n=== Scoreboard ===")
        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})
            print(
                f"{stage_name}: SR={row.get('success_rate')}, Robust={row.get('robust_sr')}, "
                f"BugRate={row.get('bug_rate')}, Avg={row.get('avg_at_budget')}, Best={row.get('best_at_budget')}, "
                f"Metric={row.get('best_metric')}, Budget={row.get('budget')}, Attempted={row.get('attempted')}, Time={row.get('time_sec')}"
            )

# =============================================================================
# Main
# =============================================================================

def main():
    _ensure_project_cwd()

    # Optional: pipeline-level config (single file to edit for common knobs)
    pipe_cfg = _load_pipeline_config()
    extra_env = None
    if isinstance(pipe_cfg, dict):
        try:
            _materialize_merged_configs(pipe_cfg)
            extra_env = _pipeline_extra_env(pipe_cfg)

            gpu_name = None
            if isinstance(pipe_cfg.get("common"), dict):
                gpu_name = pipe_cfg["common"].get("gpu_model") or pipe_cfg["common"].get("gpu_name")
            if console:
                console.print(Panel(
                    f"[bold]Pipeline config detected[/]\n"
                    f"- dataset_name: [green]{pipe_cfg.get('dataset_name','(from phase configs)')}[/]\n"
                    f"- CUDA_VISIBLE_DEVICES: [cyan]{(extra_env or {}).get('CUDA_VISIBLE_DEVICES','(default)')}[/]\n"
                    f"- GPU model: [magenta]{gpu_name or '(n/a)'}[/]",
                    border_style="blue"
                ))
            else:
                print("[INFO] Pipeline config detected:", pipe_cfg.get("dataset_name"))
        except Exception as e:
            if console:
                console.print(Panel(f"[bold yellow]WARN[/] Failed to apply pipeline_config.json: {e}", border_style="yellow"))
            else:
                print(f"[WARN] Failed to apply pipeline_config.json: {e}")

    ds_name = validate_configs()

    results_root = _results_root_for_dataset(ds_name)
    os.makedirs(results_root, exist_ok=True)

    logs_dir, pipeline_log_path, log_fp = _setup_logging(results_root)

    pipeline_start = time.time()
    print_execution_plan(ds_name)

    print(f"🚀 Pipeline starting for [{ds_name}] in 2 seconds...")
    time.sleep(2)

    stage_timings: Dict[str, Dict[str, float]] = {}
    phase_logs: Dict[str, str] = {}

    try:
        for name, info in PHASE_MAP.items():
            folder = info["folder"]
            script = info["script"]
            config = info["config"]
            extra_args = info["cmd_args"]

            cmd = ["python", script, "--config", config] if name != "Phase 1" else ["python", script, config]
            if extra_args:
                cmd.extend(extra_args)

            # Per-phase log (under timestamped logs_dir)
            phase_log_file = os.path.join(logs_dir, PHASE_LOG_NAME.get(name, f"{name}.log".replace(" ", "_").lower()))
            phase_logs[name] = phase_log_file

            phase_fp = None
            try:
                phase_fp = open(phase_log_file, "a", encoding="utf-8")
                _append_phase_header(phase_fp, ds_name, name, cmd, folder)
            except Exception as e:
                phase_fp = None
                print(f"[WARN] Failed to open phase log file {phase_log_file}: {e}")

            if console:
                console.rule(f"[bold blue]Running {name}[/]")
                console.print(f"📂 Context: [underline]{folder}[/]")
                console.print(f"💻 Command: [dim]{' '.join(cmd)}[/]\n")
            else:
                print(f"\n=== Running {name} (Dir: {folder}) ===")

            start_ts = time.time()
            stage_timings[name] = {"start": start_ts}

            try:
                _run_cmd_streamed(cmd, cwd=folder, phase_fp=phase_fp, extra_env=extra_env)
            except subprocess.CalledProcessError as e:
                if phase_fp:
                    try:
                        phase_fp.write(f"\n[ERROR] Phase failed. Exit Code={e.returncode}\n")
                        phase_fp.flush()
                    except Exception:
                        pass
                if console:
                    console.print(Panel(f"[bold red]❌ Pipeline Failed at {name}[/]\nExit Code: {e.returncode}", border_style="red"))
                else:
                    print(f"❌ Pipeline Failed at {name} with exit code {e.returncode}")
                raise
            finally:
                if phase_fp:
                    try:
                        phase_fp.write(f"\n[INFO] Phase finished at {datetime.datetime.now().isoformat()}\n")
                        phase_fp.flush()
                        phase_fp.close()
                    except Exception:
                        pass

            end_ts = time.time()
            stage_timings[name]["end"] = end_ts
            duration = end_ts - start_ts
            print(f"✅ {name} Completed ({duration:.1f}s)\n")

        pipeline_end = time.time()

        stage1_cfg = PHASE_MAP["Phase 1"].get("_loaded_cfg", {})
        stage2_cfg = PHASE_MAP["Phase 2"].get("_loaded_cfg", {})
        stage3_cfg = PHASE_MAP["Phase 3"].get("_loaded_cfg", {})

        base = _results_root_for_dataset(ds_name)
        design_dir = os.path.join(base, "design_analysis")
        ge_dir = os.path.join(base, "generate_execution")
        rf_dir = os.path.join(base, "review_feedback")

        # ---------------------
        # Phase 1
        # ---------------------
        p1_budget = _planned_phase1_budget(stage1_cfg)
        p1_metric = "heuristic_score"
        p1_log_text = _read_text(phase_logs.get("Phase 1", ""))
        p1_q = _parse_phase1_log(p1_log_text) if p1_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0}
        p1_avg, p1_best = _phase1_scores_from_artifacts(design_dir)

        p1_sr, p1_robust, p1_bug_rate = _rates(p1_q["attempted"], p1_q["clean_success"], p1_q["succeeded"], p1_q["bug"])

        p1 = {
            "budget": p1_budget,
            "attempted": p1_q["attempted"],
            "succeeded": p1_q["succeeded"],
            "clean_succeeded": p1_q["clean_success"],
            "bug_attempts": p1_q["bug"],
            "success_rate": p1_sr,
            "robust_sr": p1_robust,
            "bug_rate": p1_bug_rate,
            "avg_at_budget": p1_avg,
            "best_at_budget": p1_best,
            "best_metric": p1_metric,
        }

        # ---------------------
        # Phase 2
        # ---------------------
        p2_budget = int(get_nested(stage2_cfg, ["experiment", "max_iterations"], 0) or 0)
        p2_metric = str(get_nested(stage2_cfg, ["experiment", "primary_metric"], "PCC"))
        p2_t0 = stage_timings.get("Phase 2", {}).get("start", 0.0)
        p2_t1 = stage_timings.get("Phase 2", {}).get("end", time.time())

        p2_log_text = _read_text(phase_logs.get("Phase 2", ""))
        p2_q = _parse_phase2_log(p2_log_text, p2_metric) if p2_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0, "scores": []}

        # artifact scores as fallback if log missing
        if not p2_q.get("scores"):
            p2_q["scores"] = _phase2_scores_from_artifacts(ge_dir, p2_metric, p2_t0, p2_t1)
            p2_q["succeeded"] = len(p2_q["scores"])
            p2_q["attempted"] = p2_q["succeeded"]

        p2_avg = _mean_safe([float(x) for x in p2_q.get("scores", []) if isinstance(x, (int, float))])
        p2_best = max(p2_q["scores"]) if p2_q.get("scores") else None

        p2_sr, p2_robust, p2_bug_rate = _rates(p2_q["attempted"], p2_q["clean_success"], p2_q["succeeded"], p2_q["bug"])

        p2 = {
            "budget": p2_budget,
            "attempted": p2_q["attempted"],
            "succeeded": p2_q["succeeded"],
            "clean_succeeded": p2_q["clean_success"],
            "bug_attempts": p2_q["bug"],
            "success_rate": p2_sr,
            "robust_sr": p2_robust,
            "bug_rate": p2_bug_rate,
            "avg_at_budget": p2_avg,
            "best_at_budget": p2_best,
            "best_metric": p2_metric,
        }

        # ---------------------
        # Phase 3
        # ---------------------
        p3_budget = int(get_nested(stage3_cfg, ["review", "max_iterations"], 0) or 0)
        p3_metric = str(get_nested(stage3_cfg, ["review", "target_metric"], "PCC"))
        p3_t0 = stage_timings.get("Phase 3", {}).get("start", 0.0)
        p3_t1 = stage_timings.get("Phase 3", {}).get("end", time.time())

        p3_log_text = _read_text(phase_logs.get("Phase 3", ""))
        p3_q = _parse_phase3_log(p3_log_text, p3_metric) if p3_log_text else {"attempted": 0, "succeeded": 0, "bug": 0, "clean_success": 0, "scores": []}

        if not p3_q.get("scores"):
            p3_q["scores"] = _phase3_scores_from_artifacts(rf_dir, p3_metric, p3_t0, p3_t1)
            p3_q["succeeded"] = len(p3_q["scores"])
            p3_q["attempted"] = max(p3_q["succeeded"], 0)

        p3_avg = _mean_safe([float(x) for x in p3_q.get("scores", []) if isinstance(x, (int, float))])
        p3_best = max(p3_q["scores"]) if p3_q.get("scores") else None

        p3_sr, p3_robust, p3_bug_rate = _rates(p3_q["attempted"], p3_q["clean_success"], p3_q["succeeded"], p3_q["bug"])

        p3 = {
            "budget": p3_budget,
            "attempted": p3_q["attempted"],
            "succeeded": p3_q["succeeded"],
            "clean_succeeded": p3_q["clean_success"],
            "bug_attempts": p3_q["bug"],
            "success_rate": p3_sr,
            "robust_sr": p3_robust,
            "bug_rate": p3_bug_rate,
            "avg_at_budget": p3_avg,
            "best_at_budget": p3_best,
            "best_metric": p3_metric,
        }

        # ---------------------
        # Total
        # ---------------------
        total_attempted = (p1["attempted"] or 0) + (p2["attempted"] or 0) + (p3["attempted"] or 0)
        total_succeeded = (p1["succeeded"] or 0) + (p2["succeeded"] or 0) + (p3["succeeded"] or 0)
        total_clean = (p1["clean_succeeded"] or 0) + (p2["clean_succeeded"] or 0) + (p3["clean_succeeded"] or 0)
        total_bug = (p1["bug_attempts"] or 0) + (p2["bug_attempts"] or 0) + (p3["bug_attempts"] or 0)

        total_sr, total_robust, total_bug_rate = _rates(total_attempted, total_clean, total_succeeded, total_bug)

        # Total Avg/Best on FINAL metric only (Phase 3 target metric). If Phase 2 metric matches, include both.
        total_scores: List[float] = []
        if p2_metric == p3_metric and p2.get("avg_at_budget") is not None:
            total_scores.extend([float(x) for x in (p2_q.get("scores") or []) if isinstance(x, (int, float))])
        total_scores.extend([float(x) for x in (p3_q.get("scores") or []) if isinstance(x, (int, float))])

        total_avg = _mean_safe(total_scores)
        total_best = max(total_scores) if total_scores else None

        total_row = {
            "budget": (p1_budget or 0) + (p2_budget or 0) + (p3_budget or 0),
            "attempted": total_attempted,
            "succeeded": total_succeeded,
            "clean_succeeded": total_clean,
            "bug_attempts": total_bug,
            "success_rate": total_sr,
            "robust_sr": total_robust,
            "bug_rate": total_bug_rate,
            "avg_at_budget": total_avg,
            "best_at_budget": total_best,
            "best_metric": p3_metric,
            "time_sec": float(pipeline_end - pipeline_start),
        }

        # Attach times to per-phase rows
        p1["time_sec"] = stage_timings.get("Phase 1", {}).get("end", 0.0) - stage_timings.get("Phase 1", {}).get("start", 0.0)
        p2["time_sec"] = p2_t1 - p2_t0
        p3["time_sec"] = p3_t1 - p3_t0

        summary = {
            "dataset": ds_name,
            "generated_at": datetime.datetime.now().isoformat(),
            "logs_dir": logs_dir,
            "pipeline_log_path": pipeline_log_path,
            "phase_logs": phase_logs,
            "stages": {
                "Phase 1": p1,
                "Phase 2": p2,
                "Phase 3": p3,
                "Total": total_row,
            },
        }

        summary_path = os.path.join(results_root, "pipeline_summary.json")
        _atomic_write_json(summary_path, summary)

        _print_final_scoreboard(summary)

        if console:
            console.print(Panel("[bold green]🏆 CellScientist Workflow Completed Successfully![/]", border_style="green"))
            console.print(f"📌 Saved summary to: [underline]{summary_path}[/]")
        else:
            print("\n🏆 CellScientist Workflow Completed Successfully!")
            print(f"📌 Saved summary to: {summary_path}")

    finally:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
