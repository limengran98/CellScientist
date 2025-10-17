import os, csv, json, requests
from typing import List, Dict
try:
    from .prompts import LITERATURE_SYSTEM
except Exception:
    LITERATURE_SYSTEM = """
You are a rigorous literature assistant. Return only verifiable items with DOI or persistent URLs.
Output MUST be a CSV with columns:
id,title,authors,year,venue,doi_or_url,keywords,summary,relation_to_project,code_or_data_url,notes
"""
from .llm_client import LLMClient, LLMUnavailable
import pandas as pd
def init_scaffold(root: str) -> str:
    lit_root = os.path.join(root, "literature")
    os.makedirs(lit_root, exist_ok=True)
    return lit_root
def openalex_search(query: str, per_page: int = 10) -> List[Dict]:
    url = "https://api.openalex.org/works"
    params = {"search": query, "per_page": per_page, "sort": "relevance_score:desc"}
    try:
        r = requests.get(url, params=params, timeout=20); r.raise_for_status()
        data = r.json(); items = []
        for w in data.get("results", []):
            items.append({
                "id": w.get("id"),
                "title": w.get("title"),
                "authors": "; ".join(a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])),
                "year": w.get("publication_year"),
                "venue": (w.get("host_venue") or {}).get("display_name"),
                "doi_or_url": w.get("doi") or (w.get("ids") or {}).get("openalex"),
                "keywords": "; ".join((kw.get("display_name") or "") for kw in w.get("keywords", [])[:8]),
                "summary": "", "relation_to_project": "", "code_or_data_url": "", "notes": "",
            })
        return items
    except Exception:
        return []
def save_rows_to_csv(rows: List[Dict], csv_path: str) -> None:
    if not rows:
        open(csv_path, "w", encoding="utf-8").write(""); return
    fieldnames = ["id","title","authors","year","venue","doi_or_url","keywords","summary","relation_to_project","code_or_data_url","notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in fieldnames})
def csv_to_md(csv_path: str, md_path: str) -> None:
    if (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0):
        open(md_path, "w", encoding="utf-8").write(""); return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        open(md_path, "w", encoding="utf-8").write(""); return
    if df.empty or df.columns.size == 0:
        open(md_path, "w", encoding="utf-8").write(""); return
    lines = []
    for _, row in df.iterrows():
        lines.append(f"### {str(row.get('title','')).strip()}")
        lines.append(f"- Authors: {row.get('authors','')}")
        lines.append(f"- Venue/Year: {row.get('venue','')} / {row.get('year','')}")
        lines.append(f"- DOI/URL: {row.get('doi_or_url','')}")
        if str(row.get('summary','')).strip(): lines.append(f"- Summary: {row.get('summary','')}")
        if str(row.get('relation_to_project','')).strip(): lines.append(f"- Relevance: {row.get('relation_to_project','')}")
        lines.append("")
    open(md_path, "w", encoding="utf-8").write("\n".join(lines))
def llm_summarize_literature(lit_root: str, query: str, llm_cfg: dict) -> str:
    md_path = os.path.join(lit_root, "auto_sections.md")
    text = open(md_path, "r", encoding="utf-8").read() if os.path.exists(md_path) else ""
    if not text.strip():
        syn = os.path.join(lit_root, "synthesis.md"); open(syn, "w", encoding="utf-8").write(""); return syn
    client = LLMClient(**(llm_cfg or {}))
    messages = [
        {"role": "system", "content": LITERATURE_SYSTEM},
        {"role": "user", "content": f"Topic: {query}\n\nSynthesize bullets (5-10 lines) focused on preprocessing/model design:\n{text[:6000]}"},
    ]
    out = client.chat(messages, temperature=0.2, max_tokens=800)
    syn = os.path.join(lit_root, "synthesis.md"); open(syn, "w", encoding="utf-8").write(out or ""); return syn
