#!/usr/bin/env python3
# run_cellscientist.py
#
# Adds:
# - Avg@Budget (AUC of best-so-far)
# - Robust SR (engineering stability, counts recovered/valid runs)
# Keeps:
# - Success Rate (original definition per phase, usually "goal success" for Phase 2)

import os
import sys
import json
import time
import datetime
import subprocess
from typing import Dict, Any, List, Optional, Tuple

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

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    console = None

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

def _project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def _ensure_project_cwd():
    root = _project_root()
    if os.path.exists(os.path.join(os.getcwd(), "Design_Analysis")):
        return
    if os.path.exists(os.path.join(root, "Design_Analysis")):
        os.chdir(root)
        return
    print("❌ Error: Run this script from the 'CellScientist' root directory (or keep the repo intact).")
    sys.exit(1)

def get_config_path(phase_info: Dict[str, Any]) -> str:
    return os.path.join(phase_info["folder"], phase_info["config"])

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

def _setup_logging(logs_dir: str, run_ts: str) -> Tuple[str, Any]:
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"pipeline_{run_ts}.log")

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

    return log_path, log_fp

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

def _run_cmd_streamed(cmd: List[str], cwd: str, phase_fp=None):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
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
# Metrics helpers
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

def _find_latest_dir(root: str, prefix: str, t_start: float, t_end: float) -> Optional[str]:
    if not root or not os.path.exists(root):
        return None
    best_path = None
    best_mtime = -1.0
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or not name.startswith(prefix):
            continue
        try:
            mt = os.path.getmtime(p)
        except Exception:
            continue
        if t_start - 5 <= mt <= t_end + 5 and mt > best_mtime:
            best_mtime = mt
            best_path = p
    return best_path

def _compute_auc_best_so_far(scores: List[Optional[float]], budget: int) -> Optional[float]:
    """
    Avg@Budget (AUC of best-so-far):
      mean(best_so_far_t) for t=1..budget
    """
    if budget <= 0:
        return None
    best = None
    series: List[float] = []
    for s in scores:
        if isinstance(s, (int, float)):
            best = float(s) if best is None else max(best, float(s))
        if best is not None:
            series.append(best)

    if best is None:
        return None

    if len(series) < budget:
        series.extend([best] * (budget - len(series)))
    if len(series) > budget:
        series = series[:budget]

    return sum(series) / float(len(series)) if series else None

def _extract_phase1_scores_from_hypergraph(hg: Dict[str, Any]) -> List[Optional[float]]:
    edges = hg.get("hyperedges") if isinstance(hg.get("hyperedges"), list) else []
    scores: List[Optional[float]] = []
    keys = ("heuristic_score", "score", "weight", "value", "reward")
    for e in edges:
        if not isinstance(e, dict):
            scores.append(None)
            continue
        v = None
        for k in keys:
            if k in e:
                try:
                    v = float(e.get(k))
                    break
                except Exception:
                    v = None
        scores.append(v)
    return scores

def _phase1_stats(dataset: str, phase1_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = _results_root_for_dataset(dataset)
    design_dir = os.path.join(base, "design_analysis")
    budget = _planned_phase1_budget(phase1_cfg)

    hg = _safe_read_json(os.path.join(design_dir, "hypergraph.json")) or {}
    edges = hg.get("hyperedges") if isinstance(hg.get("hyperedges"), list) else []
    succeeded = len(edges)

    ref_meta = _safe_read_json(os.path.join(design_dir, "reference", "reference.json")) or {}
    best_score = ref_meta.get("score")

    auc = None
    try:
        p1_scores = _extract_phase1_scores_from_hypergraph(hg)
        if any(isinstance(s, (int, float)) for s in p1_scores):
            auc = _compute_auc_best_so_far(p1_scores, budget if budget > 0 else max(1, len(p1_scores)))
        elif isinstance(best_score, (int, float)) and budget > 0:
            auc = float(best_score)
    except Exception:
        auc = None

    success_rate = (succeeded / float(budget)) if budget > 0 else (1.0 if succeeded > 0 else 0.0)

    # Robust SR: for Phase 1 we approximate as the same (a "successful" run == produced edge/artifact)
    robust_succeeded = succeeded
    robust_rate = success_rate

    return {
        "budget": budget,
        "attempted": budget,
        "succeeded": succeeded,
        "success_rate": success_rate,
        "robust_succeeded": robust_succeeded,
        "robust_success_rate": robust_rate,
        "avg_at_budget": auc,
        "best_at_budget": best_score,
        "best_metric": "heuristic_score",
    }

def _phase2_stats(dataset: str, phase2_cfg: Dict[str, Any], t_start: float, t_end: float) -> Dict[str, Any]:
    base = _results_root_for_dataset(dataset)
    ge_dir = os.path.join(base, "generate_execution")
    budget = int(get_nested(phase2_cfg, ["experiment", "max_iterations"], 0) or 0)
    pm = str(get_nested(phase2_cfg, ["experiment", "primary_metric"], "PCC"))

    summary_file = None
    try:
        cand = [
            os.path.join(ge_dir, f)
            for f in os.listdir(ge_dir)
            if f.startswith("phase2_loop_summary_") and f.endswith(".json")
        ]
        cand.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
        for p in cand:
            mt = os.path.getmtime(p)
            if t_start - 5 <= mt <= t_end + 5:
                summary_file = p
                break
    except Exception:
        summary_file = None

    attempted = 0

    # "Success Rate" keeps original meaning: goal/criteria met (if available), else fallback to valid
    goal_succeeded = 0

    # "Robust SR" counts engineering-stable iterations: produced metrics (valid), including those after auto-fix/retry
    robust_succeeded = 0

    best_score = None
    auc = None

    if summary_file:
        summ = _safe_read_json(summary_file) or {}

        attempted = int(summ.get("iterations_executed", 0) or 0)

        # Robust: valid iterations
        robust_succeeded = int(summ.get("valid_iterations", 0) or 0)

        # Goal: criteria met (if the summary tracks it), else fallback to robust
        if summ.get("criteria_met_iterations", None) is not None:
            goal_succeeded = int(summ.get("criteria_met_iterations", 0) or 0)
        else:
            goal_succeeded = robust_succeeded

        best_score = summ.get("best_score")

        iters = summ.get("iterations", [])
        scores: List[Optional[float]] = []
        if isinstance(iters, list):
            for it in iters:
                if isinstance(it, dict):
                    sc = it.get("score", None)
                    try:
                        scf = float(sc)
                        scores.append(None if scf <= -900 else scf)
                    except Exception:
                        scores.append(None)
        if scores:
            auc = _compute_auc_best_so_far(scores, budget if budget > 0 else max(1, len(scores)))
        elif isinstance(best_score, (int, float)) and budget > 0:
            auc = float(best_score)

    else:
        # Fallback: infer from most recent prompt_run
        prompt_root = os.path.join(ge_dir, "prompt")
        run_dir = _find_latest_dir(prompt_root, "prompt_run_", t_start, t_end)
        if run_dir:
            score = _extract_primary_score(_safe_read_json(os.path.join(run_dir, "metrics.json")) or {}, pm)
            attempted = 1
            if score is not None:
                robust_succeeded = 1
                goal_succeeded = 1
                best_score = score
                auc = float(score)

    success_rate = (goal_succeeded / float(attempted)) if attempted > 0 else 0.0
    robust_rate = (robust_succeeded / float(attempted)) if attempted > 0 else 0.0

    return {
        "budget": budget,
        "attempted": attempted,
        "succeeded": goal_succeeded,
        "success_rate": success_rate,
        "robust_succeeded": robust_succeeded,
        "robust_success_rate": robust_rate,
        "avg_at_budget": auc,
        "best_at_budget": best_score,
        "best_metric": pm,
    }

def _extract_phase3_candidate_score(metrics_path: str, metric: str) -> Optional[float]:
    data = _safe_read_json(metrics_path)
    if not isinstance(data, dict):
        return None
    models = data.get("models") if isinstance(data.get("models"), dict) else None
    if not models:
        return None
    non_baseline = [k for k in models.keys() if "baseline" not in k.lower() and "reference" not in k.lower()]
    target_key = non_baseline[-1] if non_baseline else (data.get("winner") or list(models.keys())[0])
    m_data = models.get(target_key, {})
    return _extract_primary_score({"models": {target_key: m_data}, "winner": target_key}, metric)

def _phase3_stats(dataset: str, phase3_cfg: Dict[str, Any], t_start: float, t_end: float) -> Dict[str, Any]:
    base = _results_root_for_dataset(dataset)
    rf_dir = os.path.join(base, "review_feedback")
    budget = int(get_nested(phase3_cfg, ["review", "max_iterations"], 0) or 0)
    metric = str(get_nested(phase3_cfg, ["review", "target_metric"], "PCC"))

    run_dir = _find_latest_dir(rf_dir, "review_run_", t_start, t_end)
    attempted = 0
    succeeded = 0
    best_score = None
    auc = None

    if run_dir:
        hist = _safe_read_json(os.path.join(run_dir, "history_state.json"))
        scores: List[Optional[float]] = []
        if isinstance(hist, list):
            attempted = len(hist)
            succeeded = sum(1 for x in hist if isinstance(x, dict) and str(x.get("status", "")).upper() != "CRASH")
            for x in hist:
                if not isinstance(x, dict):
                    scores.append(None)
                    continue
                sc = x.get("score", None)
                try:
                    scf = float(sc)
                    scores.append(None if scf <= -900 else scf)
                except Exception:
                    scores.append(None)

        best_score = _extract_phase3_candidate_score(os.path.join(run_dir, "metrics_best.json"), metric)

        if scores:
            auc = _compute_auc_best_so_far(scores, budget if budget > 0 else max(1, len(scores)))
        elif isinstance(best_score, (int, float)) and budget > 0:
            auc = float(best_score)

    success_rate = (succeeded / float(attempted)) if attempted > 0 else 0.0

    # Robust SR for Phase 3 == non-crash ratio (counts runs that produced a history record and didn't crash)
    robust_succeeded = succeeded
    robust_rate = success_rate

    return {
        "budget": budget,
        "attempted": attempted,
        "succeeded": succeeded,
        "success_rate": success_rate,
        "robust_succeeded": robust_succeeded,
        "robust_success_rate": robust_rate,
        "avg_at_budget": auc,
        "best_at_budget": best_score,
        "best_metric": metric,
    }

def _print_final_scoreboard(summary: Dict[str, Any]):
    stages = summary.get("stages", {})

    if console:
        table = Table(title=f"📊 Scoreboard (dataset={summary.get('dataset')})")
        table.add_column("Stage")
        table.add_column("Success Rate ↑", justify="right")
        table.add_column("Robust SR ↑", justify="right")
        table.add_column("Avg@Budget ↑", justify="right")
        table.add_column("Best@Budget ↑", justify="right")
        table.add_column("Metric")
        table.add_column("Budget", justify="right")
        table.add_column("Time ↓ (s)", justify="right")

        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})

            sr = row.get("success_rate")
            sr_s = f"{sr:.3f}" if isinstance(sr, (int, float)) else "-"

            rsr = row.get("robust_success_rate")
            rsr_s = f"{rsr:.3f}" if isinstance(rsr, (int, float)) else "-"

            avg = row.get("avg_at_budget")
            avg_s = f"{avg:.4f}" if isinstance(avg, (int, float)) else "-"

            best = row.get("best_at_budget")
            best_s = f"{best:.4f}" if isinstance(best, (int, float)) else "-"

            metric = str(row.get("best_metric", "-"))
            budget = row.get("budget")
            budget_s = str(budget) if budget is not None else "-"

            tsec = row.get("time_sec")
            tsec_s = f"{tsec:.1f}" if isinstance(tsec, (int, float)) else "-"

            table.add_row(stage_name, sr_s, rsr_s, avg_s, best_s, metric, budget_s, tsec_s)

        console.print(table)
    else:
        print("\n=== Scoreboard ===")
        for stage_name in ["Phase 1", "Phase 2", "Phase 3", "Total"]:
            row = stages.get(stage_name, {})
            print(
                f"{stage_name}: SR={row.get('success_rate')}, RobustSR={row.get('robust_success_rate')}, "
                f"Avg@Budget={row.get('avg_at_budget')}, Best@Budget={row.get('best_at_budget')}, "
                f"Metric={row.get('best_metric')}, Budget={row.get('budget')}, Time={row.get('time_sec')}"
            )

def main():
    _ensure_project_cwd()
    ds_name = validate_configs()

    results_root = _results_root_for_dataset(ds_name)
    os.makedirs(results_root, exist_ok=True)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = os.path.join(results_root, f"logs_{run_ts}")
    os.makedirs(logs_dir, exist_ok=True)

    log_path, log_fp = _setup_logging(logs_dir, run_ts)

    pipeline_start = time.time()
    print_execution_plan(ds_name)

    print(f"🚀 Pipeline starting for [{ds_name}] in 3 seconds...")
    time.sleep(3)

    stage_timings: Dict[str, Dict[str, float]] = {}

    try:
        for name, info in PHASE_MAP.items():
            folder = info["folder"]
            script = info["script"]
            config = info["config"]
            extra_args = info["cmd_args"]

            cmd = ["python", script, "--config", config] if name != "Phase 1" else ["python", script, config]
            if extra_args:
                cmd.extend(extra_args)

            phase_log_file = os.path.join(logs_dir, PHASE_LOG_NAME.get(name, f"{name}.log".replace(" ", "_").lower()))
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
                _run_cmd_streamed(cmd, cwd=folder, phase_fp=phase_fp)
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

        p1 = _phase1_stats(ds_name, stage1_cfg)
        p2 = _phase2_stats(ds_name, stage2_cfg, stage_timings.get("Phase 2", {}).get("start", 0.0), stage_timings.get("Phase 2", {}).get("end", time.time()))
        p3 = _phase3_stats(ds_name, stage3_cfg, stage_timings.get("Phase 3", {}).get("start", 0.0), stage_timings.get("Phase 3", {}).get("end", time.time()))

        p1["time_sec"] = stage_timings.get("Phase 1", {}).get("end", 0.0) - stage_timings.get("Phase 1", {}).get("start", 0.0)
        p2["time_sec"] = stage_timings.get("Phase 2", {}).get("end", 0.0) - stage_timings.get("Phase 2", {}).get("start", 0.0)
        p3["time_sec"] = stage_timings.get("Phase 3", {}).get("end", 0.0) - stage_timings.get("Phase 3", {}).get("start", 0.0)

        total_attempted = (p1.get("attempted", 0) or 0) + (p2.get("attempted", 0) or 0) + (p3.get("attempted", 0) or 0)
        total_succeeded = (p1.get("succeeded", 0) or 0) + (p2.get("succeeded", 0) or 0) + (p3.get("succeeded", 0) or 0)
        total_success_rate = (total_succeeded / float(total_attempted)) if total_attempted > 0 else 0.0

        total_robust_succeeded = (p1.get("robust_succeeded", 0) or 0) + (p2.get("robust_succeeded", 0) or 0) + (p3.get("robust_succeeded", 0) or 0)
        total_robust_rate = (total_robust_succeeded / float(total_attempted)) if total_attempted > 0 else 0.0

        auc_vals = [v for v in [p1.get("avg_at_budget"), p2.get("avg_at_budget"), p3.get("avg_at_budget")] if isinstance(v, (int, float))]
        total_auc = (sum(auc_vals) / float(len(auc_vals))) if auc_vals else None

        total_row = {
            "budget": (p1.get("budget", 0) or 0) + (p2.get("budget", 0) or 0) + (p3.get("budget", 0) or 0),
            "attempted": total_attempted,
            "succeeded": total_succeeded,
            "success_rate": total_success_rate,
            "robust_succeeded": total_robust_succeeded,
            "robust_success_rate": total_robust_rate,
            "avg_at_budget": total_auc,
            "best_at_budget": p3.get("best_at_budget") if p3.get("best_at_budget") is not None else p2.get("best_at_budget"),
            "best_metric": p3.get("best_metric") if p3.get("best_at_budget") is not None else p2.get("best_metric"),
            "time_sec": float(pipeline_end - pipeline_start),
        }

        summary = {
            "dataset": ds_name,
            "generated_at": datetime.datetime.now().isoformat(),
            "log_path": log_path,
            "logs_dir": logs_dir,
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
            console.print(f"🗂 Logs dir: [underline]{logs_dir}[/]")
        else:
            print("\n🏆 CellScientist Workflow Completed Successfully!")
            print(f"📌 Saved summary to: {summary_path}")
            print(f"🗂 Logs dir: {logs_dir}")

    finally:
        try:
            log_fp.flush()
            log_fp.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
