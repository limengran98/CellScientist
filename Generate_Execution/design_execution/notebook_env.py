# -*- coding: utf-8 -*-
"""
Helpers for prompt-generated notebooks.

These functions provide a stable OUTPUT_DIR and a unified place to save
metrics.json 和中间产物（intermediate 目录下的各种 .npy/.json/.csv 等）。
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Any


def get_output_dir() -> str:
    """
    返回当前执行 notebook 的输出目录。

    我们约定：
      - prompt_pipeline 在执行阶段把 workdir 设置成 trial_dir
      - 所以这里直接用 os.getcwd() 就是当前 trial_dir
    """
    try:
        d = Path(os.getcwd()).resolve()
    except Exception:
        d = Path(".").resolve()
    return str(d)


def get_intermediate_dir(output_dir: str) -> str:
    """
    在 OUTPUT_DIR 下创建/返回 intermediate 子目录，用来丢所有中间产物。
    """
    inter = Path(output_dir) / "intermediate"
    inter.mkdir(parents=True, exist_ok=True)
    return str(inter)


def save_metrics(output_dir: str, metrics: dict) -> str:
    """
    把 metrics dict 写成 OUTPUT_DIR/metrics.json
    """
    path = Path(output_dir) / "metrics.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics or {}, f, indent=2)
    print(f"[INFO] metrics.json written to {path}")
    return str(path)


def save_intermediate(output_dir: str, name: str, obj: Any) -> str:
    """
    把中间结果统一写到 OUTPUT_DIR/intermediate/name.*

    规则：
      - numpy.ndarray -> .npy
      - pandas.DataFrame -> .csv
      - dict / list -> .json
      - 其他 -> 直接 str(obj) 写到 .txt
    """
    inter_dir = Path(output_dir) / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    base = inter_dir / name

    # numpy array
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, np.ndarray):
            np.save(str(base), obj)
            out = str(base) + ".npy"
            print(f"[INFO] Intermediate .npy saved: {out}")
            return out
    except Exception:
        pass

    # pandas DataFrame
    try:
        import pandas as pd  # type: ignore
        if isinstance(obj, pd.DataFrame):
            out = str(base) if str(base).endswith(".csv") else str(base) + ".csv"
            obj.to_csv(out, index=False)
            print(f"[INFO] Intermediate .csv saved: {out}")
            return out
    except Exception:
        pass

    # dict / list -> json
    if isinstance(obj, (dict, list)):
        out = str(base) if str(base).endswith(".json") else str(base) + ".json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        print(f"[INFO] Intermediate JSON saved: {out}")
        return out

    # fallback: plain text
    out = str(base) if str(base).endswith(".txt") else str(base) + ".txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(str(obj))
    print(f"[INFO] Intermediate text saved: {out}")
    return out
