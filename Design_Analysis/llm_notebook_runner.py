# [NEW] Note: Auto-fix loop is implemented in run_llm_nb.py; this runner remains unchanged for stability.
"""
LLM-driven Jupyter Notebook generator & runner for Task_Analysis.

Workflow
--------
1) Summarize prior assets (CSV preview, optional paper PDF excerpt, user prompt)
2) Call an OpenAI-compatible LLM to RETURN a JSON spec for a notebook:
   {
     "title": "str",
     "cells": [{"type": "markdown"|"code", "source": "str"}, ...]
   }
   The LLM generates all code; nothing is hard-coded here.
3) Save .ipynb and execute it cell-by-cell (nbclient) to produce an executed notebook.

Configuration
-------------
All hyper-parameters can be provided via:
- A phase-scoped config (dict) under: phases.task_analysis.llm_notebook
- Or function args to `run_llm_notebook(...)`
- Environment variables for LLM credentials:
    OPENAI_API_KEY   (required)
    OPENAI_BASE_URL  (optional; for Azure or self-hosted/vLLM gateways)
    OPENAI_MODEL     (optional; used if model not provided in config/args)

This module does NOT require internet during notebook execution;
the generated code must rely on local data and standard python packages.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat as nbf

# ---------------- Utilities ----------------

def safe_read_csv_preview(csv_path: str, max_rows: int = 8, max_cols: int = 20) -> Dict[str, Any]:
    """Read a small preview of a CSV (if available) to provide schema/context to the LLM."""
    import pandas as pd
    p = Path(csv_path)
    if not p.exists():
        return {"exists": False, "error": f"Data not found: {csv_path}"}
    try:
        df = pd.read_csv(p, nrows=500)  # shallow preview
        cols = list(df.columns)[:max_cols]
        head = df[cols].head(max_rows).to_dict(orient="records")
        meta_cols = [c for c in df.columns if str(c).lower().startswith("metadata_")]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        return {
            "exists": True,
            "n_rows_preview": len(df),
            "n_cols": len(df.columns),
            "meta_cols": meta_cols[:30],
            "feature_cols_head": feat_cols[:30],
            "head_rows": head,
        }
    except Exception as e:
        return {"exists": True, "error": f"Preview failed: {e}"}

def read_pdf_excerpt(pdf_path: Optional[str], max_pages: int = 3, max_chars: int = 4000) -> str:
    """Extract a short excerpt from the first pages of a PDF if PyPDF2 is available."""
    if not pdf_path:
        return ""
    p = Path(pdf_path)
    if not p.exists():
        return ""
    try:
        import PyPDF2
        text = []
        with open(p, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i in range(min(len(reader.pages), max_pages)):
                try:
                    text.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pass
        return ("\n".join(text)).strip()[:max_chars]
    except Exception:
        return ""

def extract_json_block(text: str) -> Dict[str, Any]:
    """Extract a JSON object from raw LLM output, supporting fenced blocks."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    try:
        return json.loads(text)
    except Exception:
        pass
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        return json.loads(m2.group(1))
    raise ValueError("Could not parse JSON notebook spec from LLM response.")

def nb_from_spec(spec: Dict[str, Any]) -> nbf.NotebookNode:
    """Convert the LLM JSON spec to a Notebook object."""
    nb = nbf.v4.new_notebook()
    cells = []
    title = spec.get("title") or "LLM Notebook"
    cells.append(nbf.v4.new_markdown_cell(f"# {title}"))
    for cell in spec.get("cells", []):
        ctype = (cell.get("type") or "").lower()
        src = cell.get("source") or ""
        if ctype == "markdown":
            cells.append(nbf.v4.new_markdown_cell(src))
        elif ctype == "code":
            cells.append(nbf.v4.new_code_cell(src))
    nb["cells"] = cells
    return nb

# ---------------- Config helpers (phase-scoped) ----------------

def _phase_enabled(config: Dict[str, Any], phase_name: str = "task_analysis") -> bool:
    phases = config.get("workflow_phases") or []
    phase_cfg = (config.get("phases") or {}).get(phase_name) or {}
    return (phase_name in phases) and bool(phase_cfg.get("enabled", False))

def _get_phase_llm_nb_cfg(config: Dict[str, Any], phase_name: str = "task_analysis") -> Dict[str, Any]:
    return ((config.get("phases") or {}).get(phase_name) or {}).get("llm_notebook") or {}

def _resolve_paths(cfg: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    paths = cfg.get("paths") or {}
    data = paths.get("data") or "/mnt/data/CP_data.csv"
    paper = paths.get("paper") or "/mnt/data/BBBC036.pdf"
    preprocess = paths.get("preprocess") or "/mnt/data/BBBC036_data_process.ipynb"
    out = paths.get("out") or "/mnt/data/CP_llm.ipynb"
    out_exec = paths.get("out_exec") or paths.get("out-exec") or "/mnt/data/CP_llm_executed.ipynb"
    return data, paper, preprocess, out, out_exec

def _resolve_exec_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = cfg.get("exec") or {}
    return {
        "timeout_seconds": int(d.get("timeout_seconds", 1800)),
        "max_preview_rows": int(d.get("max_preview_rows", 8)),
        "max_preview_cols": int(d.get("max_preview_cols", 20)),
        "pdf_max_pages": int(d.get("pdf_max_pages", 3)),
        "force_json_mode": bool(d.get("force_json_mode", True)),
        "save_intermediate": bool(d.get("save_intermediate", True)),
        "allow_errors": bool(d.get("allow_errors", True)),
    }

def _resolve_llm_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    llm = cfg.get("llm") or {}
    # 1) Try provider-first via llm_providers.json (optional)
    base_dir = Path(__file__).resolve().parents[1]  # .../cellscientist
    prov_file = base_dir / "llm_providers.json"
    prov_name = (llm.get("provider") or "").strip() or None
    base_url = None
    model = llm.get("model")
    api_key = os.environ.get(llm.get("api_key_env") or "OPENAI_API_KEY")

    if prov_name and prov_file.exists():
        try:
            data = json.loads(prov_file.read_text(encoding="utf-8"))
            prov = data.get("providers", {}).get(prov_name)
            if prov:
                base_url = prov.get("base_url") or None
                if not model:
                    models = prov.get("models") or []
                    model = models[0] if models else None
        except Exception:
            pass

    # 2) Fallbacks (env & defaults)
    if not base_url:
        # Prefer explicit env var from config
        base_url_env = llm.get("base_url_env")
        if base_url_env and os.environ.get(base_url_env):
            base_url = os.environ.get(base_url_env)
        else:
            # Final fallback: your default gateway (OpenAI-compatible)
            base_url = os.environ.get("OPENAI_BASE_URL") or "https://vip.yi-zhan.top/v1"

    if not model:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    return {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
    }


def _resolve_language_and_headings(cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    language = (cfg.get("language") or "en").lower()
    headings_cfg = ((cfg.get("headings") or {}).get("sections")) or [
        "## Data Loading & Initial Exploration",
        "## Data Patterns",
        "## Hidden Information",
        "## Innovation Motivation",
        "## Experiment & Validation Suggestions",
    ]
    return language, headings_cfg

# ---------------- LLM client (OpenAI-compatible) ----------------

@dataclass
class LLMConfig:
    model: str
    api_key: Optional[str]
    base_url: Optional[str]

class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        if not cfg.api_key:
            raise RuntimeError("OPENAI_API_KEY (or configured api_key) is required.")
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url) if cfg.base_url else OpenAI(api_key=cfg.api_key)
        except Exception as e:
            raise RuntimeError("Install openai: `pip install openai`.") from e

    def chat_json(self, messages, force_json: bool = True):
        print(f"👉 Sending LLM request | model={self.cfg.model} | base_url={self.cfg.base_url or 'default OpenAI'}")

        if force_json:
            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
                print(f"✅ LLM responded | model={resp.model}")
                return json.loads(resp.choices[0].message.content or "{}")
            except Exception:
                pass

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=0.2,
        )
        print(f"✅ LLM responded | model={resp.model}")
        content = resp.choices[0].message.content or "{}"
        return extract_json_block(content)


# ---------------- High-level Runner ----------------

_BASE_SYSTEM_PROMPT = """You are an expert computational biologist and ML engineer.
Return ONLY a JSON object describing a runnable Jupyter Notebook with keys:
- "title": string
- "cells": array of {{"type": "markdown"|"code", "source": "string"}}

Constraints:
- Language: {language_label}.
- Alternate markdown and code logically; each code cell MUST be preceded by a markdown explanation.
- Code must be executable offline (no internet), using the given dataset path exactly as received.
- Prefer standard libs: pandas, numpy, matplotlib (seaborn optional), statsmodels, scikit-learn.
- The Notebook must explicitly connect data analysis with the scientific context of the provided paper.
- Include the following section headings (in order), using them exactly:
{headings_bulleted}

Section expectations:
1. **Data Loading & Initial Exploration**
   - Load the dataset, preview schema, show summary statistics and distributions.
   - Identify metadata vs. feature columns (treat dose/SMILES/plate IDs as covariates; keep features numeric).
   - Convert feature columns to numeric with coercion; replace +inf/-inf with NaN and report affected columns.
   - Diagnose missingness and distributional shape (skewness, kurtosis), and summarize per-column NaN ratios.
   - Adaptive handling (NO fixed hard thresholds):
     * Prefer imputation over dropping (median imputation by default).
     * If a column is near-constant or >95% missing **and** carries negligible variance after imputation, drop it with justification.
   - Do NOT force winsorization/clipping. If extreme outliers would numerically break scaling/PCA, apply the **minimum necessary** safeguard (e.g., quantile clipping or power transform) and document the decision.
   - Choose scaler based on diagnostics:
     * If features are heavy-tailed or contain outliers, use RobustScaler; otherwise use StandardScaler.
   - Ensure a fully finite numeric matrix before downstream steps; include assertions that fail early with clear messages if not satisfied.
   - Report final matrix shape, number of features retained, imputation summary, and any adaptive decisions. Relate these choices to the biological context of the paper (e.g., morphology features often heavy-tailed).
   - After completing preprocessing, insert a dedicated **Markdown cell** titled "## Biological Interpretation of the Dataset".  
   - In plain language, explain what the **rows** represent (e.g., individual cells, wells, compound–dose pairs, depending on dataset inspection).  
   - Explain what the **columns** represent (numeric features such as morphological descriptors, omics values, etc.).  
   - Clarify what the **metadata columns** represent (e.g., dose = treatment intensity, SMILES = compound identity, plate ID = batch effects).  
   - Relate the dataset schema directly to the biological experiment described in the paper (what kind of experiment generated it, what the features capture in terms of phenotype or cellular response).  
   - Provide a **concise, data-driven summary** in natural language, using real counts (e.g., number of samples, unique compounds, unique plates) derived from the dataset, for example:
    > “This dataset contains **{{n_rows}}** samples measured under **{{n_unique_compounds}}** compounds and **{{n_unique_batches}}** plates. Each row corresponds to a **{{unit_of_observation}}**, and each feature describes a **{{biological_property}}**. Collectively, these features capture biological responses such as {{contextual_examples_from_paper}}.”

2. **Data Patterns**
   - Build directly on the processed matrix from Step 1.
   - Perform bio-oriented EDA with **advanced visualizations** (prefer these, with graceful fallbacks):
     * **Heatmap + clustering dendrogram** of top-variable features across samples/conditions.
       - Preferred: seaborn.clustermap; Fallback: matplotlib + scipy (linkage + dendrogram).
     * **Distribution & correlation views**: KDE/violin or histogram for representative features; correlation matrix and top correlated pairs.
     * **Dimensionality reduction**: PCA (required); optionally UMAP/t-SNE if available. Color by condition/dose/plate; overlay point density if feasible.
     * **Batch & dose effects**: ANOVA or linear/mixed-effects models (statsmodels) with concise result tables.
   - Provide biological interpretations of observed trends (e.g., heterogeneity, dose-dependent morphological shifts, plate/batch structure).
   - Expected figures (names in captions): Fig-Heatmap-Dendro, Fig-PCA, Fig-CorrMatrix, Fig-DoseEffect.

3. **Hidden Information**
   - Leverage signals from Step 2 and integrate with the paper context to hypothesize hidden biology.
   - Use at least two of the following **advanced analyses** (choose what the data supports; keep offline):
     * **Marker identification** across conditions/groups with effect sizes and multiple-testing notes; visualize a **Volcano Plot** (log2FC vs -log10 p) for top features.
     * **Functional/Pathway enrichment (reasoned offline)**: if gene IDs/sets are available, perform simple over-representation using provided sets; otherwise provide **mechanistic reasoning** and a ranked feature-set summary. Visualize an **Enrichment Bubble Plot** (bubble size = set size, color = significance or effect).
     * **Network/module structure**: build a **feature correlation network** (NetworkX optional; fallback: adjacency threshold table + degree histogram) or at minimum a top-module correlation heatmap.
     * **Phenotype association**: regression/logistic models linking PCs or feature modules to dose/condition; report coefficients, CIs, and calibration notes.
     * **Model interpretability** (if a baseline is fit): show **feature importance**; if SHAP is not available, fall back to standardized coefficients or permutation importance.
   - Support hypotheses with appropriate tests (t/Wilcoxon/regression) and report effect sizes + 95% CIs where feasible.
   - Expected figures (names in captions): Fig-Volcano, Fig-Enrichment, Fig-NetworkOrModule, Fig-Association.

4. **Innovation Motivation**
   - Start with a concise Markdown summary of the key findings from Step 2 (Data Patterns) and Step 3 (Hidden Information).
     * Describe in natural language what the analyses revealed: e.g., distributional skew, hidden clusters, feature correlations, inferred biological mechanisms.
     * This summary acts as a bridge to motivate the next discussion.
   - Then, in Markdown, discuss:
     * Limitations of current methods (mapped to observed failure modes: instability due to outliers, plate leakage, poor dose-response fitting, etc.).
     * Unresolved questions revealed by the dataset (hidden subgroups, confounders, nonlinear responses).
     * Opportunities for innovation (methodological or biological).
   - Explicitly connect these insights back to the data-driven findings.
   - (Optional) Provide a baseline model in code (classification/regression, with leakage-safe splitting) to illustrate current performance and motivate improvements.

5. **Experiment & Validation Suggestions**
   - Extend logically from the innovation motivations in Step 4, with a tight integration of biological findings and computational model design.
   - In Markdown, propose next-step experiments that explicitly link:
     * **Biological discovery → Computational modeling**: e.g., heterogeneity in morphology → clustering + ensemble ML classifiers; nonlinear dose–response → generalized additive models, monotonic regression, or spline-regularized ML/DL.
     * **Mechanistic priors in modeling**: incorporate pathway/group priors into feature engineering, hierarchical models, or biologically-informed regularization.
     * **Cross-modal integration**: combine morphology with chemical descriptors/omics using multi-view learning (e.g., CCA, multimodal autoencoders, graph-based fusion).
     * **Generalization & robustness**: explicitly test whether models trained on one plate/dose/compound generalize to unseen conditions, with leakage-safe splits (plate-holdout, dose-stratified).
   - Define tasks & success criteria:
     * ML metrics (R²/MAE, AUROC/AUPRC, clustering indices).
     * Biological criteria (recovery of known markers, pathway-level consistency, phenotype relevance).
   - In code, provide a prototype pipeline skeleton (scikit-learn, XGBoost, or PyTorch) that illustrates how the processed data could be modeled:
     * Baseline: logistic regression or random forest.
     * Advanced: biologically-motivated model (e.g., monotonic regressor, graph-based model, multimodal fusion).
     * Training: leakage-safe split + evaluation with both ML metrics and biological interpretability checks.

General rules:
- Every code cell must be explained by a Markdown cell immediately before it.  
- Markdown must include clear subheadings (## Data Patterns, ## Hidden Information, etc.).  
- Keep all code concise and runnable, avoid heavy external dependencies.  
- The Notebook should demonstrate a flow from **data exploration → hidden insights → research motivation → proposed methods**, grounded in the paper’s context.
"""


def _system_prompt(language: str, headings: List[str]) -> str:
    lang_label = "English" if language.startswith("en") else language
    headings_bulleted = "\n".join([f"  {h}" for h in headings])
    return _BASE_SYSTEM_PROMPT.format(language_label=lang_label, headings_bulleted=headings_bulleted)

def make_messages(
    user_prompt: str,
    data_path: str,
    paper_excerpt: str,
    csv_preview: Dict[str, Any],
    language: str,
    headings: List[str],
) -> List[Dict[str, str]]:
    """Compose the system+user messages for the LLM."""
    context = {
        "user_prompt": user_prompt,
        "language": language,
        "data_path": data_path,
        "paper_excerpt": (paper_excerpt or "")[:2000],
        "csv_preview": csv_preview,
        "constraints": {
            "alternate_markdown_code": True,
            "no_internet": True,
            "allowed_libs": ["pandas","numpy","matplotlib","statsmodels","scikit-learn"],
            "required_headings": headings,
        }
    }
    return [
        {"role": "system", "content": _system_prompt(language, headings)},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False, indent=2)},
    ]

def run_llm_notebook(
    data_path: str,
    paper_pdf: Optional[str],
    preprocess_nb: Optional[str],
    user_prompt: str,
    out_path: str,
    executed_path: Optional[str] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 1800,
    force_json_mode: bool = True,
    max_preview_rows: int = 8,
    max_preview_cols: int = 20,
    pdf_max_pages: int = 3,
    language: str = "en",
    headings: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    save_intermediate: bool = True,
    allow_errors: bool = True,
) -> str:
    """
    Core runner with explicit arguments (no hardcoding).
    If you prefer phase-scoped config, use run_llm_notebook_with_config(...) instead.
    """
    if headings is None:
        headings = [
            "## Data Loading & Initial Exploration",
            "## Data Patterns",
            "## Hidden Information",
            "## Innovation Motivation",
            "## Experiment & Validation Suggestions",
        ]

    # Build context
    csv_preview = safe_read_csv_preview(data_path, max_rows=max_preview_rows, max_cols=max_preview_cols)
    paper_excerpt = read_pdf_excerpt(paper_pdf, max_pages=pdf_max_pages)

    # LLM messages
    messages = make_messages(user_prompt, data_path, paper_excerpt, csv_preview, language=language, headings=headings)

    # LLM config (explicit > env)
    model_final = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    api_key_final = api_key or os.environ.get("OPENAI_API_KEY")
    base_url_final = base_url or os.environ.get("OPENAI_BASE_URL")
    client = OpenAICompatClient(LLMConfig(model=model_final, api_key=api_key_final, base_url=base_url_final))

    # Generate JSON notebook spec
    spec = client.chat_json(messages, force_json=force_json_mode)

    # Build and write notebook
    nb = nb_from_spec(spec)
    out_p = Path(out_path); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(nbf.writes(nb), encoding="utf-8")

    # Save the "unexecuted" Notebook first
    if save_intermediate:
        out_p.write_text(nbf.writes(nb), encoding="utf-8")

    # Execute (optionally allow errors to not stop execution)
    from nbclient import NotebookClient
    # Key: allow_errors controls "continue on error"
    nb_client = NotebookClient(
        nb,
        timeout=timeout_seconds,
        kernel_name="python3",
        allow_errors=allow_errors
    )

    # Safety: prevent execution from stopping due to kernel-level exceptions
    errors_summary = []
    try:
        nb_client.execute()
    except Exception as e:
        # If it's not a cell-level error but an executor-level exception,
        # capture a short overview in metadata
        errors_summary.append(f"Notebook execution raised: {type(e).__name__}: {e}")

    # Write execution errors into notebook metadata (optional but useful)
    if errors_summary:
        nb.metadata["execution_errors"] = errors_summary

    # Save the "executed" Notebook
    exec_p = Path(executed_path or out_p.with_name(out_p.stem + "_executed.ipynb"))
    exec_p.write_text(nbf.writes(nb), encoding="utf-8")
    return str(exec_p)


# ---------------- Phase-scoped wrappers ----------------

def run_llm_notebook_with_config(config: Dict[str, Any], phase_name: str = "task_analysis") -> str:
    """Run only if the `phase_name` is active; read all params from phases.task_analysis.llm_notebook."""
    if not _phase_enabled(config, phase_name=phase_name):
        raise RuntimeError(f"Phase '{phase_name}' is not active (workflow_phases + phases.{phase_name}.enabled).")

    nb_cfg = _get_phase_llm_nb_cfg(config, phase_name=phase_name)

    # prompt & language/headings
    prompt = nb_cfg.get("prompt") or "Please generate an English analysis notebook following the required structure."
    language, headings = _resolve_language_and_headings(nb_cfg)

    # paths / exec / llm
    data, paper, preprocess, out, out_exec = _resolve_paths(nb_cfg)
    exec_cfg = _resolve_exec_cfg(nb_cfg)
    llm_basic = _resolve_llm_cfg(nb_cfg)

    # Run
    return run_llm_notebook(
        data_path=data,
        paper_pdf=paper,
        preprocess_nb=preprocess,
        user_prompt=prompt,
        out_path=out,
        executed_path=out_exec,
        model=llm_basic["model"],
        timeout_seconds=exec_cfg["timeout_seconds"],
        force_json_mode=exec_cfg["force_json_mode"],
        max_preview_rows=exec_cfg["max_preview_rows"],
        max_preview_cols=exec_cfg["max_preview_cols"],
        pdf_max_pages=exec_cfg["pdf_max_pages"],
        language=language,
        headings=headings,
        api_key=llm_basic["api_key"],
        base_url=llm_basic["base_url"],
        save_intermediate=bool(exec_cfg.get("save_intermediate", True)),
        allow_errors=bool(exec_cfg.get("allow_errors", True)),
    )

def run_llm_notebook_from_file(config_path: str, phase_name: str = "task_analysis") -> str:
    """Load a JSON config file and run the phase-scoped notebook generation/execution."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return run_llm_notebook_with_config(cfg, phase_name=phase_name)
