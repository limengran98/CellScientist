# literature_agent.py
import os
import json
import arxiv
from pathlib import Path
from typing import List, Dict, Any

def _get_literature_dir(dataset_name: str) -> Path:
    """
    Returns the path: ../literature/${dataset_name}/
    Relative to the execution root (usually where the script is run).
    """
    # Assuming script runs from 'generate_execution' or similar folder
    # We want to go up one level, then into literature
    base = Path("..").resolve()
    lit_dir = base / "literature" / dataset_name
    lit_dir.mkdir(parents=True, exist_ok=True)
    return lit_dir

def search_arxiv(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Searches ArXiv and returns a list of paper metadata.
    """
    print(f"📚 [LIT] Searching ArXiv for: '{query}'...", flush=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    try:
        for r in client.results(search):
            results.append({
                "title": r.title,
                "authors": [a.name for a in r.authors],
                "published": r.published.strftime("%Y-%m-%d"),
                "summary": r.summary.replace("\n", " "),
                "url": r.entry_id
            })
    except Exception as e:
        print(f"⚠️ [LIT] ArXiv search failed: {e}", flush=True)
        
    return results

def get_literature_context(
    dataset_name: str, 
    task_keywords: str, 
    enabled: bool = True, 
    max_papers: int = 5
) -> str:
    """
    Main entry point.
    1. Checks cache at ../literature/{dataset_name}/arxiv_cache.json
    2. If missing and enabled, searches ArXiv and saves cache.
    3. Returns formatted string for LLM prompt.
    """
    if not enabled:
        return ""

    lit_dir = _get_literature_dir(dataset_name)
    cache_path = lit_dir / "arxiv_cache.json"
    
    papers = []

    # 1. Try Cache
    if cache_path.exists():
        try:
            print(f"📚 [LIT] Loading cached literature from {cache_path}", flush=True)
            with open(cache_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
        except Exception as e:
            print(f"⚠️ [LIT] Cache read failed: {e}", flush=True)

    # 2. Search if Cache Empty (and enabled)
    if not papers:
        # Construct a query relevant to the dataset/task
        # E.g., "Single cell perturbation {task_keywords}"
        query = f"single cell {task_keywords}"
        papers = search_arxiv(query, max_results=max_papers)
        
        # Save to Cache
        if papers:
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(papers, f, indent=2, ensure_ascii=False)
                print(f"📚 [LIT] Saved {len(papers)} papers to cache.", flush=True)
            except Exception as e:
                print(f"⚠️ [LIT] Cache write failed: {e}", flush=True)

    # 3. Format Output
    if not papers:
        return "No specific literature context available."

    context_str = "### Relevant Scientific Literature (ArXiv)\n"
    for i, p in enumerate(papers):
        context_str += f"**{i+1}. {p['title']}** ({p['published']})\n"
        context_str += f"   - *Authors*: {', '.join(p['authors'][:3])} et al.\n"
        context_str += f"   - *Abstract*: {p['summary']}\n\n"
    
    return context_str