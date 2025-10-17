import os, json

def pick_baseline_metrics(root: str, baseline_date: str, baseline_id: int) -> str:
    p = os.path.join(root, "baselines", baseline_date, f"metrics_baseline_{baseline_id:02d}.json")
    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            json.dump({}, f)
    return p

def record_metrics(tdir: str, metrics: dict) -> str:
    out = os.path.join(tdir, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2)
    return out
