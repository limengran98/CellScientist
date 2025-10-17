import os, glob, shutil, json, datetime
from typing import List
def migrate_reference_ipynb(source_dir: str, design_root: str, include_globs: List[str] = None) -> str:
    include_globs = include_globs or ["Baseline_*.ipynb"]
    date_dir = os.path.join(design_root, "baselines", datetime.date.today().isoformat())
    os.makedirs(date_dir, exist_ok=True)
    files = []
    for pattern in include_globs:
        for src in sorted(glob.glob(os.path.join(source_dir, pattern))):
            dst = os.path.join(date_dir, f"baseline_{len(files):02d}.ipynb")
            shutil.copy2(src, dst)
            files.append(os.path.basename(dst))
    with open(os.path.join(date_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"source_dir": source_dir, "include_globs": include_globs, "files": files}, f, indent=2)
    return date_dir
