import os, glob, re
import nbformat
def _safe_head(text: str, n=1500):
    return (text or "").strip()[:n]
def summarize_notebook(nb_path: str, max_chars: int = 1500) -> str:
    if not os.path.exists(nb_path): return ""
    nb = nbformat.read(nb_path, as_version=4)
    md_lines, code_hints = [], []
    for c in nb.cells:
        if c.cell_type == "markdown":
            src = c.source or ""
            for line in src.splitlines():
                if line.strip().startswith(("#", "-", "*")):
                    md_lines.append(line.strip())
        elif c.cell_type == "code":
            src = c.source or ""
            if re.search(r"\b(import|from)\b\s+(sklearn|torch|tensorflow|xgboost|lightgbm|keras)", src):
                code_hints.append(re.sub(r"\s+", " ", src.strip())[:220])
            if re.search(r"RandomForest|LogisticRegression|MLP|XGB|LightGBM|torch\.nn\.", src):
                code_hints.append(re.sub(r"\s+", " ", src.strip())[:220])
    text = "## Markdown cues\n" + "\n".join(md_lines[:40]) + "\n\n## Code hints\n" + "\n".join(code_hints[:20])
    return _safe_head(text, max_chars)
def summarize_folder_ipynb(folder: str, max_chars: int = 2000) -> str:
    if not folder or not os.path.isdir(folder): return ""
    parts = []
    for p in sorted(glob.glob(os.path.join(folder, "*.ipynb")))[:6]:
        s = summarize_notebook(p, max_chars=600)
        if s: parts.append(f"### {os.path.basename(p)}\n{s}")
    return _safe_head("\n".join(parts), max_chars)
