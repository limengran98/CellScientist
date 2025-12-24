#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pipeline-config loading + per-phase config materialization.

This module is a refactor of the logic previously embedded in run_cellscientist.py.
It keeps behavior compatible:

- Optional pipeline_config.json (or env CELL_SCI_PIPELINE_CONFIG) can override
  dataset_name / common env / llm fields / paths / per-phase overrides.
- A common "all-null" llm block behaves as a true no-op.
- If common GPU settings are provided, phase configs will have cuda_device_id
  wiped so children inherit CUDA_VISIBLE_DEVICES from the parent process.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from runner_utils import load_json, project_root


def get_config_path(phase_info: Dict[str, Any]) -> str:
    """Return the resolved config path for a phase."""
    cfg = phase_info.get("config")
    if not cfg:
        return os.path.join(phase_info["folder"], cfg)
    if os.path.isabs(cfg) or os.path.exists(cfg):
        return cfg
    return os.path.join(phase_info["folder"], cfg)


def get_nested(data: Dict[str, Any], keys: List[str], default="N/A"):
    val: Any = data
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


def deep_merge(dst: Any, src: Any) -> Any:
    """Recursively merge src into dst and return merged (does not mutate inputs)."""
    if isinstance(dst, dict) and isinstance(src, dict):
        out = dict(dst)
        for k, v in src.items():
            if k in out:
                out[k] = deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return src if src is not None else dst


def drop_none(obj: Any) -> Any:
    """Recursively drop None values from dict/list structures."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if v is None:
                continue
            vv = drop_none(v)
            if vv == {} or vv == []:
                continue
            out[k] = vv
        return out
    if isinstance(obj, list):
        out_list = []
        for v in obj:
            if v is None:
                continue
            vv = drop_none(v)
            if vv == {} or vv == []:
                continue
            out_list.append(vv)
        return out_list
    return obj


def set_nested(d: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_pipeline_config() -> Optional[Dict[str, Any]]:
    """Load pipeline_config.json if present (env overrides default path)."""
    env_path = os.environ.get("CELL_SCI_PIPELINE_CONFIG")
    if env_path and os.path.exists(env_path):
        try:
            return load_json(env_path)
        except Exception:
            return None

    default_path = os.path.join(project_root(), "pipeline_config.json")
    if os.path.exists(default_path):
        try:
            return load_json(default_path)
        except Exception:
            return None
    return None


def pipeline_extra_env(pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Env vars to pass to all phase subprocesses."""
    env_out: Dict[str, str] = {}
    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}
    if common.get("cuda_visible_devices") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_visible_devices"])
    elif common.get("cuda_device_id") is not None:
        env_out["CUDA_VISIBLE_DEVICES"] = str(common["cuda_device_id"])

    env_cfg = pipe_cfg.get("env") if isinstance(pipe_cfg.get("env"), dict) else {}
    for k, v in env_cfg.items():
        if v is None:
            continue
        env_out[str(k)] = str(v)
    return env_out


def apply_pipeline_overrides(
    phase_name: str,
    phase_cfg: Dict[str, Any],
    pipe_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply pipeline-level overrides to a single phase config."""
    cfg = dict(phase_cfg)

    # 1) dataset_name
    if isinstance(pipe_cfg.get("dataset_name"), str) and pipe_cfg["dataset_name"].strip():
        cfg["dataset_name"] = pipe_cfg["dataset_name"].strip()

    common = pipe_cfg.get("common") if isinstance(pipe_cfg.get("common"), dict) else {}

    # 2) GPU selection: wipe cuda_device_id in child configs if parent sets CUDA_VISIBLE_DEVICES
    cuda_visible = common.get("cuda_visible_devices")
    cuda_id = common.get("cuda_device_id")
    has_gpu_setting = (cuda_visible is not None) or (cuda_id is not None)
    if has_gpu_setting:
        config_val = None
        if phase_name == "Phase 1":
            set_nested(cfg, ["phases", "task_analysis", "llm_notebook", "exec", "cuda_device_id"], config_val)
        elif phase_name == "Phase 2":
            set_nested(cfg, ["exec", "cuda_device_id"], config_val)
        elif phase_name == "Phase 3":
            set_nested(cfg, ["exec", "cuda_device_id"], config_val)

    # 3) LLM defaults (legacy): pipeline_config.json -> llm can override Phase 1/2/3.
    llm_common_raw = pipe_cfg.get("llm") if isinstance(pipe_cfg.get("llm"), dict) else None
    llm_common = drop_none(llm_common_raw) if isinstance(llm_common_raw, dict) else None
    if isinstance(llm_common, dict) and llm_common:
        if phase_name == "Phase 1":
            cur = (((cfg.get("phases") or {}).get("task_analysis") or {}).get("llm_notebook") or {}).get("llm") or {}
            merged = deep_merge(cur, llm_common)
            set_nested(cfg, ["phases", "task_analysis", "llm_notebook", "llm"], merged)
        else:
            cur = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
            cfg["llm"] = deep_merge(cur, llm_common)

    # 4) Paths (common)
    paths_common = pipe_cfg.get("paths") if isinstance(pipe_cfg.get("paths"), dict) else None
    if isinstance(paths_common, dict) and paths_common:
        cur = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
        cfg["paths"] = deep_merge(cur, paths_common)

    # 5) Phase-specific overrides
    phase_overrides = pipe_cfg.get("phase_overrides") if isinstance(pipe_cfg.get("phase_overrides"), dict) else {}
    po = phase_overrides.get(phase_name) if isinstance(phase_overrides.get(phase_name), dict) else None
    if isinstance(po, dict) and po:
        cfg = deep_merge(cfg, po)

    return cfg


def materialize_merged_configs(phase_map: Dict[str, Dict[str, Any]], pipe_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Write merged per-phase configs under each phase folder and update phase_map in-place."""
    merged_paths: Dict[str, str] = {}
    dataset_tag = "default"
    if pipe_cfg.get("dataset_name"):
        dataset_tag = "".join(c for c in str(pipe_cfg["dataset_name"]) if c.isalnum() or c in ("-", "_"))

    for phase_name, info in phase_map.items():
        base_cfg_path = get_config_path(info)
        base_cfg = load_json(base_cfg_path)
        merged_cfg = apply_pipeline_overrides(phase_name, base_cfg, pipe_cfg)

        cache_dir = os.path.join(info["folder"], "_pipeline_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_name = os.path.basename(info["config"])
        if base_name.endswith(".json"):
            out_name = base_name[:-5] + f".{dataset_tag}.merged.json"
        else:
            out_name = base_name + f".{dataset_tag}.merged.json"

        out_path = os.path.abspath(os.path.join(cache_dir, out_name))
        with open(out_path, "w", encoding="utf-8") as f:
            import json

            json.dump(merged_cfg, f, ensure_ascii=False, indent=2)

        info["config"] = out_path
        info["_loaded_cfg"] = merged_cfg
        merged_paths[phase_name] = out_path

    return merged_paths


def validate_configs(phase_map: Dict[str, Dict[str, Any]]) -> str:
    """Ensure dataset_name matches across all phase configs and store loaded configs."""
    dataset_names: Dict[str, str] = {}
    for name, info in phase_map.items():
        path = get_config_path(info)
        cfg = load_json(path)
        ds = cfg.get("dataset_name", "MISSING")
        dataset_names[name] = ds
        info["_loaded_cfg"] = cfg

    unique = set(dataset_names.values())
    if len(unique) > 1:
        raise RuntimeError(f"CRITICAL ERROR: 'dataset_name' mismatch across phases: {dataset_names}")
    return list(unique)[0]
