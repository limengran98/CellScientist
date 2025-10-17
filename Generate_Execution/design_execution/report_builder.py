import os, json, glob, datetime
def build_markdown_report(root: str, trial_dir_name: str, baseline_metrics_path=None, primary_metric=None) -> str:
    report_dir = os.path.join(root, "reports"); os.makedirs(report_dir, exist_ok=True)
    tdir = os.path.join(root, "trials", trial_dir_name)
    bdirs = sorted(glob.glob(os.path.join(root, "baselines", "*"))); bdir = bdirs[-1] if bdirs else "(none)"
    trial_metrics = None
    for cand in ["metrics.json", "figs/metrics.json"]:
        p = os.path.join(tdir, cand)
        if os.path.exists(p):
            try:
                trial_metrics = json.loads(open(p, "r", encoding="utf-8").read()); break
            except Exception: pass
    baseline_metrics = None
    if baseline_metrics_path and os.path.exists(baseline_metrics_path):
        try: baseline_metrics = json.loads(open(baseline_metrics_path, "r", encoding="utf-8").read())
        except Exception: baseline_metrics = None
    lines = []
    lines.append(f"# Trial Report: {trial_dir_name}")
    lines.append(f"_Generated: {datetime.datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("## Paths")
    lines.append(f"- Baselines dir: `{bdir}`")
    lines.append(f"- Trial dir: `{tdir}`\n")
    lines.append("## Baseline Metrics")
    if baseline_metrics: lines += ["```json", json.dumps(baseline_metrics, indent=2, ensure_ascii=False), "```"]
    else: lines.append("_No baseline metrics.json found._")
    lines.append("\n## Trial Metrics")
    if trial_metrics: lines += ["```json", json.dumps(trial_metrics, indent=2, ensure_ascii=False), "```"]
    else: lines.append("_No trial metrics.json found._")
    out = os.path.join(report_dir, f"Report_{trial_dir_name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(lines)); return out
