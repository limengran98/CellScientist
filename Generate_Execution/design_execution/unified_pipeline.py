# unified_pipeline.py
from __future__ import annotations
import os, glob, shutil, datetime
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import nbformat

from .baseline_migrator import migrate_reference_ipynb
from .literature_manager import (
    openalex_search, save_rows_to_csv, csv_to_md, llm_summarize_literature
)
from .trial_manager import create_trial_from_baseline, propose_notebook_improvements
from .llm_client import LLMClient
from .patch_applier import apply_patch
from .report_builder import build_markdown_report
from .context_extractor import summarize_folder_ipynb, summarize_notebook
from .nb_autofix import execute_with_autofix

def _log(stage: str, msg: str) -> None:
    print(f"[{stage}] {msg}", flush=True)

def _latest_dir(path: str) -> Optional[str]:
    subs = [p for p in glob.glob(os.path.join(path, "*")) if os.path.isdir(p)]
    return sorted(subs)[-1] if subs else None

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _copy_stage1_to_reference_pool(cfg: Dict[str, Any]) -> Tuple[int, str]:
    src = (cfg.get("paths") or {}).get("stage1_analysis_dir")
    root = (cfg.get("paths") or {}).get("design_execution_root")
    dst = os.path.join(root, "reference_pool")
    if not src or not os.path.isdir(src):
        return 0, dst
    os.makedirs(dst, exist_ok=True)
    count = 0
    for p in sorted(glob.glob(os.path.join(src, "*.ipynb"))):
        shutil.copy2(p, os.path.join(dst, os.path.basename(p)))
        count += 1
    _log("INFO", f"Copied {count} Stage-1 notebooks to reference pool: {dst}")
    return count, dst

def _create_minimal_baseline(dst_dir: str, baseline_id: int = 0) -> str:
    """Create a minimal baseline notebook if user disables baseline or none exists."""
    os.makedirs(dst_dir, exist_ok=True)
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell(
        "# Minimal Baseline\nThis is a minimal baseline notebook created by the unified pipeline."
    ))
    path = os.path.join(dst_dir, f"baseline_{baseline_id:02d}.ipynb")
    nbformat.write(nb, path)
    _log("INFO", f"Minimal baseline created at: {path}")
    return path

def _pick_baseline_ipynb(baselines_root: str, baseline_id: int) -> str:
    bdir = _latest_dir(baselines_root)
    if not bdir:
        raise RuntimeError("No baselines found after migration/minimal creation.")
    nb = os.path.join(bdir, f"baseline_{baseline_id:02d}.ipynb")
    if not os.path.exists(nb):
        cands = sorted(glob.glob(os.path.join(bdir, "*.ipynb")))
        if not cands:
            raise FileNotFoundError(f"No ipynb found under {bdir}")
        nb = cands[0]
    return nb

def _run_literature(cfg: Dict[str, Any], enable: bool) -> Tuple[str, str]:
    root = (cfg.get("paths") or {}).get("design_execution_root")
    lit_root = os.path.join(root, "literature")
    os.makedirs(lit_root, exist_ok=True)
    if not enable:
        _log("INFO", "Literature search disabled.")
        Path(os.path.join(lit_root, "auto_sections.md")).write_text("", encoding="utf-8")
        return lit_root, ""
    lit = cfg.get("literature", {}) or {}
    query = (lit.get("query") or "").strip()
    n = int(lit.get("n", 10))
    llm_sum = bool(lit.get("llm_summarize", True))
    if not query:
        _log("INFO", "No literature query; skipping.")
        Path(os.path.join(lit_root, "auto_sections.md")).write_text("", encoding="utf-8")
        return lit_root, ""
    rows = openalex_search(query, per_page=n)
    if rows:
        csv_path = os.path.join(lit_root, "papers.csv")
        save_rows_to_csv(rows, csv_path)
        md_path = os.path.join(lit_root, "auto_sections.md")
        csv_to_md(csv_path, md_path)
        _log("INFO", f"Literature CSV: {csv_path}")
        _log("INFO", f"Markdown summary: {md_path}")
    else:
        Path(os.path.join(lit_root, "auto_sections.md")).write_text("", encoding="utf-8")
        _log("INFO", "No literature results; empty file created.")
    bullets = ""
    if llm_sum:
        try:
            syn = llm_summarize_literature(lit_root, query, cfg)
            bullets = Path(syn).read_text(encoding="utf-8")
            _log("INFO", f"LLM literature synthesis: {syn}")
        except Exception as e:
            _log("INFO", f"LLM synthesis failed: {e}")
    return lit_root, bullets

def _resolve_llm(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve LLM profile from cfg only (no llm_providers.json)."""
    llm = (cfg.get("llm") or {})
    providers = (cfg.get("providers") or {})
    provider = llm.get("provider") or cfg.get("default_provider")
    if not provider and providers:
        provider = next(iter(providers.keys()))
    prov = (providers.get(provider) or {}) if provider else {}
    model = llm.get("model") or prov.get("model") or (prov.get("models") or [None])[0] or "gpt-5"
    base_url = llm.get("base_url") or os.environ.get(f"{(provider or 'openai').upper()}_BASE_URL") or prov.get("base_url")
    api_key  = llm.get("api_key")  or os.environ.get(f"{(provider or 'openai').upper()}_API_KEY")  or prov.get("api_key")
    temperature = float(prov.get("temperature", llm.get("temperature", 0.0)))
    max_tokens  = int(prov.get("max_tokens",  llm.get("max_tokens", 20000)))
    print(f"[LLM] provider={provider or 'openai_compat'} model={model} base_url={base_url}")
    return {
        "provider": provider or "openai_compat",
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

class _VerboseLLM:
    """A thin proxy to print model/base_url on each call while delegating to the real LLMClient."""
    def __init__(self, inner: LLMClient, provider: str, model: str, base_url: Optional[str]):
        self._inner = inner
        self._provider = provider
        self._model = model
        self._base_url = base_url

    def __getattr__(self, name):
        # generic delegation (for attributes/methods not explicitly wrapped)
        return getattr(self._inner, name)

    # Common method names likely used across the codebase.
    def chat(self, *args, **kwargs):
        print(f"[LLM] provider={self._provider} model={self._model} base_url={self._base_url}")
        return self._inner.chat(*args, **kwargs)

    def chat_json(self, *args, **kwargs):
        print(f"[LLM] provider={self._provider} model={self._model} base_url={self._base_url}")
        return self._inner.chat_json(*args, **kwargs)

    def complete(self, *args, **kwargs):
        print(f"[LLM] provider={self._provider} model={self._model} base_url={self._base_url}")
        return self._inner.complete(*args, **kwargs)

    def complete_json(self, *args, **kwargs):
        print(f"[LLM] provider={self._provider} model={self._model} base_url={self._base_url}")
        return self._inner.complete_json(*args, **kwargs)

def _prepare_context(cfg: Dict[str, Any], baseline_nb: str) -> Dict[str, str]:
    root = (cfg.get("paths") or {}).get("design_execution_root")
    ref_pool = os.path.join(root, "reference_pool")
    reference_summary = summarize_folder_ipynb(ref_pool, max_chars=1200) if os.path.isdir(ref_pool) else ""
    baseline_summary = summarize_notebook(baseline_nb, max_chars=1200) if os.path.exists(baseline_nb) else ""
    _log("INFO", f"Prepared baseline summary from: {baseline_nb}")
    return {
        "reference_summary": reference_summary,
        "baseline_summary": baseline_summary,
        "baseline_nb": baseline_nb
    }

def _read_text(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""

def run_unified_pipeline(
    *,
    cfg: Dict[str, Any],
    baseline_id: int = 0,
    with_lit: bool = False,
    prompt_file: Optional[str] = None,
    prompt_text: Optional[str] = None,
    include_baseline: bool = True,
    which: str = "both",
) -> Dict[str, Any]:
    """
    Unified pipeline:
      - Optional text prompt (prompt_text or prompt_file)
      - Optional baseline code (baseline ipynb used as prior context) controlled by a single hyperparam.
    """
    paths = cfg.get("paths") or {}
    root = paths.get("design_execution_root")
    if not root:
        raise RuntimeError("paths.design_execution_root missing in config.")
    _ensure_dir(root)

    _log("INFO", "=== Unified pipeline: start ===")
    _copy_stage1_to_reference_pool(cfg)

    # Baseline migration or minimal creation based on the hyperparam
    if include_baseline:
        bsrc = paths.get("baseline_source_dir")
        if bsrc and os.path.isdir(bsrc) and glob.glob(os.path.join(bsrc, "*.ipynb")):
            bdir = migrate_reference_ipynb(bsrc, root, include_globs=["*.ipynb"])
            _log("INFO", f"Baselines migrated to: {bdir}")
        else:
            bdir = os.path.join(root, "baselines", datetime.datetime.now().strftime("minimal_%Y%m%d_%H%M%S"))
            _ensure_dir(bdir)
            _create_minimal_baseline(bdir, baseline_id=baseline_id)
        baseline_nb = _pick_baseline_ipynb(os.path.join(root, "baselines"), baseline_id)
    else:
        # When baseline is disabled, create a minimal one to keep pipeline structure consistent.
        bdir = os.path.join(root, "baselines", datetime.datetime.now().strftime("minimal_%Y%m%d_%H%M%S"))
        _ensure_dir(bdir)
        baseline_nb = _create_minimal_baseline(bdir, baseline_id=baseline_id)

    _lit_root, lit_bullets = _run_literature(cfg, enable=with_lit)
    ctx = _prepare_context(cfg, baseline_nb)

    tag = (cfg.get("trial") or {}).get("tag", "unified")
    seed = int((cfg.get("trial") or {}).get("seed", 22))
    tdir = create_trial_from_baseline(root, tag, baseline_id, seed)
    _log("INFO", f"Trial dir: {tdir}")

    llm_prof = _resolve_llm(cfg)
    raw_llm = LLMClient(
        provider=llm_prof["provider"],
        model=llm_prof["model"],
        api_key=llm_prof["api_key"],
        base_url=llm_prof["base_url"],
        temperature=llm_prof["temperature"],
        max_tokens=llm_prof["max_tokens"],
        timeout=(cfg.get("llm") or {}).get("timeout", 1800),
    )
    llm = _VerboseLLM(raw_llm, llm_prof["provider"], llm_prof["model"], llm_prof["base_url"])

    user_prompt = (prompt_text or "").strip()
    if not user_prompt:
        user_prompt = _read_text(prompt_file).strip()
    merged_bullets = lit_bullets.strip()
    if user_prompt:
        merged_bullets = (merged_bullets + "\n\n" if merged_bullets else "") + "## USER_PROMPT\n" + user_prompt

    patch_path = propose_notebook_improvements(
        cfg,
        tdir,
        related_work_bullets=merged_bullets,
        seed=seed,
        llm=llm,
        reference_summary=ctx["reference_summary"],
        baseline_summary=ctx["baseline_summary"],
        require_llm=True,
        llm_retries=int((cfg.get("llm") or {}).get("retries", 2)),
    )
    _log("INFO", f"LLM patch: {patch_path}")

    patched_nb = apply_patch(os.path.join(tdir, "notebook_unexec.ipynb"), patch_path)
    _log("INFO", f"Patched notebook: {patched_nb}")

    exec_cfg = (cfg.get("exec") or {})
    timeout_seconds = int(exec_cfg.get("timeout_seconds", 3600))
    preserve_source_in_exec = bool(exec_cfg.get("preserve_source_in_exec", True))
    save_intermediates = bool(exec_cfg.get("save_intermediates", False))

    executed_nb = None
    if which in ("baseline", "both"):
        executed_nb = execute_with_autofix(
            nb_path=baseline_nb,
            workdir=tdir,
            timeout_seconds=timeout_seconds,
            max_fix_rounds=int(exec_cfg.get("max_fix_rounds", 3)),
            max_cell_retries=int(exec_cfg.get("max_cell_retries", 2)),
            preserve_source_in_exec=preserve_source_in_exec,
            phase_cfg=cfg,
            verbose=True,
            save_intermediates=save_intermediates,
        )
        _log("INFO", f"Baseline executed -> {executed_nb}")

    if which in ("patched", "both"):
        executed_nb = execute_with_autofix(
            nb_path=patched_nb,
            workdir=tdir,
            timeout_seconds=timeout_seconds,
            max_fix_rounds=int(exec_cfg.get("max_fix_rounds", 3)),
            max_cell_retries=int(exec_cfg.get("max_cell_retries", 2)),
            preserve_source_in_exec=preserve_source_in_exec,
            phase_cfg=cfg,
            verbose=True,
            save_intermediates=save_intermediates,
        )
        _log("INFO", f"Patched executed -> {executed_nb}")

    report_path = None
    try:
        trial_name = os.path.basename(tdir)
        report_path = build_markdown_report(root, trial_name, None)
        _log("INFO", f"Report: {report_path}")
    except TypeError:
        try:
            report_path = build_markdown_report(root, trial_name, None, None)
            _log("INFO", f"Report: {report_path}")
        except Exception:
            _log("INFO", "Report generation skipped.")

    _log("INFO", "=== Unified pipeline: completed ===")
    return {
        "trial_dir": tdir,
        "patched_nb": patched_nb,
        "executed_nb": executed_nb,
        "report_path": report_path,
    }
