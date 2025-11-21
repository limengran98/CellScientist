# design_execution/report_builder.py
import os, json, glob, datetime
from typing import Optional, Any, Dict

def build_markdown_report(
    trial_dir: str,
    bdir: str = "",
    report_dir: Optional[str] = None,
    baseline_metrics_path: Optional[str] = None,
    primary_metric: Optional[str] = None,  # 现在还没用到，先预留
) -> str:
    trial_dir = os.path.abspath(trial_dir)
    trial_dir_name = os.path.basename(trial_dir.rstrip(os.sep))

    # report_dir: 如果没有显式传，就默认用 trials 上一层的 reports 目录
    if not report_dir:
        root = os.path.dirname(os.path.dirname(trial_dir))  # .../trials/<name>
        report_dir = os.path.join(root, "reports")
    os.makedirs(report_dir, exist_ok=True)

    # bdir: 如果没有传，就从 root/baselines 里找一个最近的
    if not bdir:
        root = os.path.dirname(os.path.dirname(trial_dir))
        bdirs = sorted(glob.glob(os.path.join(root, "baselines", "*")))
        bdir = bdirs[-1] if bdirs else "(none)"

    # 读 trial metrics
    trial_metrics = None
    for cand in ["metrics.json", "figs/metrics.json"]:
        p = os.path.join(trial_dir, cand)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    trial_metrics = json.load(f)
                break
            except Exception:
                pass

    # 读 baseline metrics（如果给了路径）
    baseline_metrics = None
    if baseline_metrics_path and os.path.exists(baseline_metrics_path):
        try:
            with open(baseline_metrics_path, "r", encoding="utf-8") as f:
                baseline_metrics = json.load(f)
        except Exception:
            baseline_metrics = None

    lines = []
    lines.append(f"# Trial Report: {trial_dir_name}")
    lines.append(f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_\n")

    lines.append("## Paths")
    lines.append(f"- Baselines dir: `{bdir}`")
    lines.append(f"- Trial dir: `{trial_dir}`\n")

    lines.append("## Baseline Metrics")
    if baseline_metrics:
        lines += ["```json", json.dumps(baseline_metrics, indent=2, ensure_ascii=False), "```"]
    else:
        lines.append("_No baseline metrics.json found._")

    lines.append("\n## Trial Metrics")
    if trial_metrics:
        lines += ["```json", json.dumps(trial_metrics, indent=2, ensure_ascii=False), "```"]
    else:
        lines.append("_No trial metrics.json found._")

    out = os.path.join(report_dir, f"Report_{trial_dir_name}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out
