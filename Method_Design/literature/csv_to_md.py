#!/usr/bin/env python3
import csv, sys, pathlib

def main(csv_path, md_path):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    out = ["# Related Work — Auto Generated Sections\n"]
    for r in rows:
        rid = r.get("id","").strip() or "R?"
        title = r.get("title","").strip() or "(no title)"
        year = r.get("year","").strip()
        venue = r.get("venue","").strip()
        authors = r.get("authors","").strip()
        link = r.get("doi_or_url","").strip()
        summary = r.get("summary","").strip()
        rel = r.get("relation_to_project","").strip()
        code = r.get("code_or_data_url","").strip()
        notes = r.get("notes","").strip()
        out.append(f"### [{rid}] {title} ({year}, {venue})\n- 作者：{authors}\n- 链接：{link}\n- 贡献摘要：{summary}\n- 与本项目关系：{rel}\n- 代码或数据链接：{code}\n- 备忘：{notes}\n")
    pathlib.Path(md_path).write_text("\n".join(out), encoding="utf-8")
    print(f"Written: {md_path}")

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Usage: csv_to_md.py papers.csv out.md")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
