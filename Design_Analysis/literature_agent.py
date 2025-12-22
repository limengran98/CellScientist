#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
literature_agent.py

Goal
- Provide a stable, cached literature retrieval + LLM synthesis pipeline.
- Output a reusable JSON knowledge pack for later phases (Phase 2/3).

Outputs (default)
- results/<dataset>/literature/papers.json            (raw normalized papers)
- results/<dataset>/literature/domain_knowledge.json  (LLM-synthesized knowledge JSON)
- results/<dataset>/literature/literature_agent.log   (debug log)

Design
- Multi-source retrieval (Semantic Scholar + PubMed) with timeouts, retries, and backoff.
- Atomic writes + a simple lock file to avoid multi-thread race conditions (Phase 1 parallel runs).
- Robust fallback: if network/LLM fails, still writes a minimal knowledge JSON.
"""
from __future__ import annotations

import os
import json
import time
import hashlib
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Utilities
# ----------------------------

def _project_root() -> Path:
    # .../CellScientist/Design_Analysis/literature_agent.py -> .../CellScientist
    return Path(__file__).resolve().parents[1]

def _results_root(dataset_name: str) -> Path:
    return _project_root() / "results" / (dataset_name or "default_dataset")

def _lit_dir(dataset_name: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    # Allow config override
    if cfg:
        p = ((cfg.get("paths") or {}).get("literature_dir")) or ((cfg.get("literature") or {}).get("output_dir"))
        if isinstance(p, str) and p.strip():
            # Make it absolute relative to project root if needed
            out = Path(p)
            if not out.is_absolute():
                out = _project_root() / out
            out.mkdir(parents=True, exist_ok=True)
            return out
    d = _results_root(dataset_name) / "literature"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def _truncate(s: str, n: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: max(0, n - 3)] + "..."

# ----------------------------
# Simple lock to avoid Phase1 parallel race
# ----------------------------

class _FileLock:
    def __init__(self, lock_path: Path, timeout_s: int = 60):
        self.lock_path = lock_path
        self.timeout_s = max(1, int(timeout_s))
        self._acquired = False

    def acquire(self) -> bool:
        start = time.time()
        while time.time() - start < self.timeout_s:
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self._acquired = True
                return True
            except FileExistsError:
                time.sleep(0.2 + random.random() * 0.3)
            except Exception:
                time.sleep(0.5)
        return False

    def release(self) -> None:
        if self._acquired:
            try:
                os.remove(str(self.lock_path))
            except Exception:
                pass
        self._acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

# ----------------------------
# Network fetch helpers (stdlib-only)
# ----------------------------

def _http_get_json(url: str, timeout_s: int = 15, headers: Optional[Dict[str, str]] = None, retries: int = 2) -> Optional[Dict[str, Any]]:
    headers = headers or {}
    last_err = None
    for i in range(max(1, retries) + 1):
        try:
            req = urllib.request.Request(url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                raw = r.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            last_err = e
            # backoff
            time.sleep(min(2.0, 0.5 * (i + 1)) + random.random() * 0.2)
    return None

def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: int = 30, headers: Optional[Dict[str, str]] = None, retries: int = 2) -> Optional[Dict[str, Any]]:
    headers = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for i in range(max(1, retries) + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                raw = r.read()
            return json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as e:
            last_err = e
            time.sleep(min(2.0, 0.5 * (i + 1)) + random.random() * 0.2)
    return None

# ----------------------------
# Retrieval: Semantic Scholar
# ----------------------------

def _search_semantic_scholar(query: str, limit: int, year_from: Optional[int], year_to: Optional[int], timeout_s: int = 15) -> List[Dict[str, Any]]:
    q = query.strip()
    if not q:
        return []
    fields = "title,authors,year,abstract,url,venue,citationCount,externalIds"
    params = {
        "query": q,
        "limit": str(max(1, min(int(limit or 5), 50))),
        "fields": fields,
        "offset": "0",
    }
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout_s=timeout_s, headers={"User-Agent": "CellScientist/1.0"})
    if not data or not isinstance(data.get("data"), list):
        return []
    out = []
    for p in data["data"]:
        try:
            year = p.get("year")
            if isinstance(year, int):
                if year_from and year < year_from:
                    continue
                if year_to and year > year_to:
                    continue
            authors = []
            if isinstance(p.get("authors"), list):
                authors = [a.get("name") for a in p["authors"] if isinstance(a, dict) and a.get("name")]
            ext = p.get("externalIds") or {}
            doi = ext.get("DOI") if isinstance(ext, dict) else None
            out.append({
                "source": "semantic_scholar",
                "title": p.get("title") or "",
                "authors": authors,
                "year": year,
                "abstract": p.get("abstract") or "",
                "url": p.get("url") or "",
                "venue": p.get("venue") or "",
                "citation_count": p.get("citationCount"),
                "doi": doi,
            })
        except Exception:
            continue
    return out

# ----------------------------
# Retrieval: PubMed (E-utilities)
# ----------------------------

def _pubmed_esearch_ids(query: str, retmax: int, timeout_s: int = 15) -> List[str]:
    q = query.strip()
    if not q:
        return []
    params = {
        "db": "pubmed",
        "term": q,
        "retmax": str(max(1, min(int(retmax or 5), 50))),
        "retmode": "json",
        "sort": "relevance",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout_s=timeout_s, headers={"User-Agent": "CellScientist/1.0"})
    try:
        ids = (data or {}).get("esearchresult", {}).get("idlist", [])
        return [str(x) for x in ids if str(x).strip()]
    except Exception:
        return []

def _pubmed_esummary(ids: List[str], timeout_s: int = 15) -> List[Dict[str, Any]]:
    if not ids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(ids[:200]),
        "retmode": "json",
    }
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
    data = _http_get_json(url, timeout_s=timeout_s, headers={"User-Agent": "CellScientist/1.0"})
    if not data:
        return []
    result = data.get("result", {})
    out = []
    for pid in ids:
        it = result.get(pid)
        if not isinstance(it, dict):
            continue
        title = it.get("title") or ""
        # 'pubdate' is like '2022 Jan 10'
        year = None
        pubdate = (it.get("pubdate") or "").strip()
        if pubdate[:4].isdigit():
            year = int(pubdate[:4])
        authors = []
        if isinstance(it.get("authors"), list):
            authors = [a.get("name") for a in it["authors"] if isinstance(a, dict) and a.get("name")]
        out.append({
            "source": "pubmed",
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": "",  # esummary doesn't include abstract reliably
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
            "venue": it.get("fulljournalname") or it.get("source") or "",
            "citation_count": None,
            "doi": it.get("elocationid") if isinstance(it.get("elocationid"), str) and "doi" in it.get("elocationid","").lower() else None,
            "pmid": pid,
        })
    return out

# ----------------------------
# Normalization / ranking / dedupe
# ----------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip().lower())

def _dedupe(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in papers:
        key = _norm_title(p.get("title","")) or (p.get("doi") or p.get("url") or "")
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def _rank(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Prefer more recent, higher citation_count if available
    def score(p: Dict[str, Any]) -> float:
        year = p.get("year") or 0
        cc = p.get("citation_count") or 0
        try:
            return float(year) * 10.0 + float(cc) * 0.01
        except Exception:
            return float(year) * 10.0
    return sorted(papers, key=score, reverse=True)

# ----------------------------
# LLM synthesis
# ----------------------------

def _resolve_llm_cfg_from_app_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer literature.llm, fallback to Phase1 llm_notebook llm, then env.
    lit_llm = (cfg.get("literature") or {}).get("llm") or {}
    if lit_llm:
        return lit_llm
    nb_llm = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {}).get("llm") or {}
    return nb_llm

def _openai_compat_chat_json(messages: List[Dict[str, str]], llm_cfg: Dict[str, Any], timeout_s: int = 60) -> Optional[Dict[str, Any]]:
    """
    OpenAI-compatible /v1/chat/completions call via urllib (no requests dependency here).
    Expects server supports response_format json_object; if not, we still try to parse JSON in content.
    """
    api_key = llm_cfg.get("api_key") or os.environ.get(llm_cfg.get("api_key_env","OPENAI_API_KEY"), os.environ.get("OPENAI_API_KEY"))
    base_url = llm_cfg.get("base_url") or os.environ.get(llm_cfg.get("base_url_env","OPENAI_BASE_URL"), os.environ.get("OPENAI_BASE_URL","https://api.openai.com/v1"))
    model = llm_cfg.get("model") or os.environ.get("OPENAI_MODEL","gpt-4o-mini")
    if not api_key:
        return None
    url = (base_url.rstrip("/") + "/chat/completions") if not base_url.rstrip("/").endswith("/v1") else (base_url.rstrip("/") + "/chat/completions")
    if base_url.rstrip("/").endswith("/v1"):
        url = base_url.rstrip("/") + "/chat/completions"
    else:
        url = base_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 4096)),
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "CellScientist/1.0",
    }
    data = _http_post_json(url, payload, timeout_s=timeout_s, headers=headers, retries=2)
    if not data:
        return None
    try:
        content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        content = content.strip()
        if not content:
            return None
        # strict parse first
        try:
            return json.loads(content)
        except Exception:
            pass
        # fenced JSON
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
        if m:
            return json.loads(m.group(1))
        m2 = re.search(r"(\{.*\})", content, flags=re.S)
        if m2:
            return json.loads(m2.group(1))
    except Exception:
        return None
    return None

def _fallback_knowledge(papers: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    # Minimal, always-valid JSON if LLM synthesis fails.
    titles = [p.get("title","") for p in papers if p.get("title")]
    return {
        "generated_at_utc": _now_iso(),
        "query": query,
        "summary": "LLM synthesis unavailable; fallback knowledge pack generated from metadata only.",
        "key_takeaways": [],
        "recommended_metrics": ["PCC", "MSE"],
        "recommended_baselines": [],
        "common_pitfalls": [],
        "papers": [{
            "title": p.get("title",""),
            "year": p.get("year"),
            "url": p.get("url",""),
            "doi": p.get("doi"),
            "source": p.get("source",""),
        } for p in papers[:20]],
        "notes": {
            "paper_count": len(papers),
            "titles_preview": titles[:10],
        }
    }

def synthesize_domain_knowledge_json(papers: List[Dict[str, Any]], cfg: Dict[str, Any], query: str, log_fp=None) -> Dict[str, Any]:
    lit_cfg = cfg.get("literature") or {}
    llm_cfg = _resolve_llm_cfg_from_app_cfg(cfg)
    max_in_prompt = int(lit_cfg.get("llm_max_papers_in_prompt", 8) or 8)
    used = papers[:max(1, min(max_in_prompt, len(papers)))]
    sys = (
        "You are a senior computational biology researcher. "
        "Given a small set of paper metadata (title/authors/year/abstract), "
        "output a compact JSON knowledge pack for downstream model design + debugging. "
        "Return ONLY valid JSON."
    )
    user = {
        "task": "Summarize key domain knowledge for single-cell perturbation / perturb-seq style modeling (as applicable). "
                "Extract methods, metrics, common failure modes, and actionable modeling advice.",
        "query": query,
        "papers": used,
        "output_schema": {
            "domain_overview": "string (<=1200 chars)",
            "key_concepts": [{"name": "string", "explanation": "string"}],
            "common_methods": [{"method": "string", "when_to_use": "string", "notes": "string"}],
            "recommended_metrics": ["string"],
            "recommended_baselines": ["string"],
            "common_pitfalls": [{"pitfall": "string", "symptom": "string", "fix": "string"}],
            "implementation_tips": [{"tip": "string", "why": "string"}],
            "paper_citations": [{"title": "string", "year": "int|null", "url": "string", "doi": "string|null", "key_takeaway": "string"}],
        }
    }
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    if not lit_cfg.get("llm_summarize", True):
        return _fallback_knowledge(papers, query)

    out = _openai_compat_chat_json(messages, llm_cfg=llm_cfg, timeout_s=int(lit_cfg.get("llm_timeout_seconds", 90) or 90))
    if isinstance(out, dict) and out:
        out.setdefault("generated_at_utc", _now_iso())
        out.setdefault("query", query)
        out.setdefault("paper_count", len(papers))
        return out

    return _fallback_knowledge(papers, query)

# ----------------------------
# Public API
# ----------------------------

def build_literature_query(cfg: Dict[str, Any]) -> str:
    lit_cfg = cfg.get("literature") or {}
    q = (lit_cfg.get("query") or "").strip()
    if q:
        return q
    # fallback: infer from dataset/task fields if provided
    ds = (cfg.get("dataset_name") or "").strip()
    extra = (lit_cfg.get("task_keywords") or lit_cfg.get("keywords") or "").strip()
    if extra:
        return f"{ds} {extra}".strip()
    # safe default
    return f"single cell perturbation modeling {ds}".strip()

def load_or_create_literature_knowledge(cfg: Dict[str, Any], dataset_name: str, *, force_refresh: bool = False) -> Tuple[Optional[Dict[str, Any]], Path]:
    """
    Returns: (domain_knowledge_json_or_None, path_to_domain_knowledge_json)
    Always attempts to ensure an on-disk JSON exists (fallback JSON if needed).
    """
    lit_cfg = cfg.get("literature") or {}
    enabled = bool(lit_cfg.get("enabled", False))
    lit_dir = _lit_dir(dataset_name, cfg=cfg)
    log_path = lit_dir / "literature_agent.log"
    papers_path = lit_dir / "papers.json"
    knowledge_path = lit_dir / (lit_cfg.get("output_filename") or "domain_knowledge.json")

    def _log(msg: str):
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fp:
                fp.write(f"[{_now_iso()}] {msg}\n")
        except Exception:
            pass

    # Fast path: load existing
    if knowledge_path.exists() and not force_refresh:
        try:
            return json.loads(knowledge_path.read_text(encoding="utf-8")), knowledge_path
        except Exception:
            pass

    # If disabled, still write minimal file so later phases can load reliably.
    query = build_literature_query(cfg)
    if not enabled:
        minimal = _fallback_knowledge([], query)
        _atomic_write_json(knowledge_path, minimal)
        return minimal, knowledge_path

    # Prevent parallel workers from clobbering
    lock = _FileLock(lit_dir / ".literature.lock", timeout_s=int(lit_cfg.get("lock_timeout_seconds", 90) or 90))
    with lock:
        # Re-check after acquiring lock
        if knowledge_path.exists() and not force_refresh:
            try:
                return json.loads(knowledge_path.read_text(encoding="utf-8")), knowledge_path
            except Exception:
                pass

        _log(f"START enabled={enabled} query={query}")

        # cache key
        year_from = lit_cfg.get("year_from")
        year_to = lit_cfg.get("year_to")
        try:
            year_from = int(year_from) if year_from is not None else None
        except Exception:
            year_from = None
        try:
            year_to = int(year_to) if year_to is not None else None
        except Exception:
            year_to = None

        max_papers = int(lit_cfg.get("max_papers", 8) or 8)
        timeout_s = int(lit_cfg.get("timeout_seconds", 15) or 15)
        sources = lit_cfg.get("sources") or ["semantic_scholar", "pubmed"]

        # TTL cache (optional)
        cache_ttl_days = int(lit_cfg.get("cache_ttl_days", 14) or 14)
        cache_path = lit_dir / "literature_cache.json"
        cache_key = _sha1(json.dumps({
            "query": query,
            "max_papers": max_papers,
            "sources": sources,
            "year_from": year_from,
            "year_to": year_to,
        }, sort_keys=True))

        # Try cache
        if cache_path.exists() and not force_refresh:
            try:
                c = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(c, dict) and c.get("cache_key") == cache_key:
                    ts = c.get("cached_at_utc") or ""
                    # TTL check
                    ok = True
                    try:
                        import datetime
                        dt = datetime.datetime.fromisoformat(ts.replace("Z",""))
                        age_days = (datetime.datetime.utcnow() - dt).total_seconds() / 86400.0
                        ok = age_days <= float(cache_ttl_days)
                    except Exception:
                        ok = True
                    if ok and isinstance(c.get("papers"), list):
                        papers = c["papers"]
                        _log(f"CACHE_HIT papers={len(papers)}")
                        _atomic_write_json(papers_path, {"generated_at_utc": _now_iso(), "query": query, "papers": papers})
                        know = synthesize_domain_knowledge_json(papers, cfg, query)
                        _atomic_write_json(knowledge_path, know)
                        _log("DONE(cache)")
                        return know, knowledge_path
            except Exception:
                pass

        papers: List[Dict[str, Any]] = []
        try:
            if "semantic_scholar" in [s.lower() for s in sources]:
                papers.extend(_search_semantic_scholar(query, limit=max_papers, year_from=year_from, year_to=year_to, timeout_s=timeout_s))
            if "pubmed" in [s.lower() for s in sources]:
                ids = _pubmed_esearch_ids(query, retmax=max_papers, timeout_s=timeout_s)
                papers.extend(_pubmed_esummary(ids, timeout_s=timeout_s))
        except Exception as e:
            _log(f"RETRIEVE_ERROR {type(e).__name__}: {e}")

        papers = _rank(_dedupe(papers))[:max_papers]
        _log(f"RETRIEVED papers={len(papers)}")

        # Save raw papers
        _atomic_write_json(papers_path, {"generated_at_utc": _now_iso(), "query": query, "papers": papers})

        # Update cache
        try:
            _atomic_write_json(cache_path, {
                "cached_at_utc": _now_iso(),
                "cache_key": cache_key,
                "papers": papers,
            })
        except Exception:
            pass

        # Synthesize knowledge
        know = synthesize_domain_knowledge_json(papers, cfg, query)
        _atomic_write_json(knowledge_path, know)
        _log("DONE")
        return know, knowledge_path


# =============================================================================
# Compatibility helpers (used by Phase 1/2/3 callers)
# =============================================================================

def get_literature_context(config: Dict[str, Any], dataset_name: str = "", force_refresh: bool = False) -> Dict[str, Any]:
    """Compatibility wrapper returning the knowledge JSON object (fail-open)."""
    try:
        ds = dataset_name or str(config.get("dataset_name") or "").strip()
        kp, _kp_path = load_or_create_literature_knowledge(config, ds, force_refresh=force_refresh)
        return kp or {}
    except Exception:
        return {}

def load_domain_knowledge_text(config: Dict[str, Any], dataset_name: str = "") -> str:
    """Load knowledge JSON and return a compact text (for LLM prompts)."""
    try:
        ds = dataset_name or str(config.get("dataset_name") or "").strip()
        kp, _kp_path = load_or_create_literature_knowledge(config, ds, force_refresh=False)
        max_chars = int(os.environ.get("DOMAIN_KNOWLEDGE_MAX_CHARS", "9000"))
        return json.dumps(kp or {}, ensure_ascii=False, indent=2)[:max_chars]
    except Exception:
        return ""
