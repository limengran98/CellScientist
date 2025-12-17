#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal explicit task-graph state for Phase 3 (reflection / decomposition / routing).

Design goals:
- Backward compatible: if LLM doesn't emit task fields, we still work.
- Low-intrusion: plain JSON file in workspace_dir.
- Graph (not tree): tasks can have dependencies; each iteration can touch multiple tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "task"


def _strip_leading_number(s: str) -> str:
    # "1. Model Architecture (Backbone/Encoder)" -> "Model Architecture (Backbone/Encoder)"
    return re.sub(r"^\s*\d+\s*[\.\)]\s*", "", s).strip()


def load_task_graph(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_task_graph(state: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def init_task_graph_from_config(optimization_hierarchy: List[str]) -> Dict[str, Any]:
    """
    Build an initial task graph from cfg['review']['optimization_hierarchy'].
    Creates:
      - root task T0
      - child subtasks from hierarchy list
    """
    tasks: Dict[str, Any] = {}
    edges: List[Dict[str, Any]] = []

    tasks["T0"] = {
        "name": "Root",
        "type": "root",
        "status": "active",
        "priority": 1.0,
        "expected_gain": 1.0,
        "uncertainty": 0.5,
        "dependencies": [],
        "children": [],
        "evidence": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    for idx, raw in enumerate(optimization_hierarchy or [], start=1):
        name = _strip_leading_number(raw)
        tid = f"T{idx}"
        tasks[tid] = {
            "name": name,
            "type": "subtask",
            "status": "active",
            "priority": 0.7,
            "expected_gain": 0.7,
            "uncertainty": 0.6,
            "dependencies": [],
            "children": [],
            "evidence": [],
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        tasks["T0"]["children"].append(tid)
        edges.append({"from": "T0", "to": tid, "type": "decomposes_to"})

    return {
        "version": 1,
        "tasks": tasks,
        "edges": edges,
        "iteration_map": {},
        "meta": {"created_at": _now_iso()},
    }


def find_task_ids_by_name(state: Dict[str, Any], names: List[str]) -> List[str]:
    """
    Resolve a list of task names (case-insensitive, substring match) to task IDs.
    """
    if not names:
        return []
    tasks = state.get("tasks", {})
    norm = [(n or "").strip().lower() for n in names if (n or "").strip()]
    found: List[str] = []
    for tid, t in tasks.items():
        tname = str(t.get("name", "")).strip().lower()
        for n in norm:
            if not n:
                continue
            if n == tname or n in tname:
                found.append(tid)
                break
    # stable unique
    seen=set()
    out=[]
    for tid in found:
        if tid not in seen:
            seen.add(tid); out.append(tid)
    return out


def ensure_task(state: Dict[str, Any], name: str, parent_name: Optional[str]=None) -> str:
    """
    Ensure a task exists. If not, create it (optionally under parent).
    Returns task id.
    """
    name = (name or "").strip()
    if not name:
        return "T0"

    existing = find_task_ids_by_name(state, [name])
    if existing:
        return existing[0]

    # new id
    tasks = state.setdefault("tasks", {})
    next_num = 0
    for tid in tasks.keys():
        m = re.match(r"^T(\d+)$", tid)
        if m:
            next_num = max(next_num, int(m.group(1)))
    new_id = f"T{next_num+1}"

    tasks[new_id] = {
        "name": name,
        "type": "subtask",
        "status": "active",
        "priority": 0.6,
        "expected_gain": 0.6,
        "uncertainty": 0.7,
        "dependencies": [],
        "children": [],
        "evidence": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    # attach to parent if provided
    if parent_name:
        parent_ids = find_task_ids_by_name(state, [parent_name])
        parent_id = parent_ids[0] if parent_ids else "T0"
    else:
        parent_id = "T0"

    # connect
    state.setdefault("edges", []).append({"from": parent_id, "to": new_id, "type": "decomposes_to"})
    tasks.setdefault(parent_id, {}).setdefault("children", []).append(new_id)
    tasks[parent_id]["updated_at"] = _now_iso()
    return new_id


def apply_decomposition_updates(
    state: Dict[str, Any],
    suggestion: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Apply (optional) decomposition updates from LLM suggestion:
      - new_subtasks: [{name,parent,rationale}]
      - dependency_updates: [{from,to,type}]
    Returns: (created_task_ids, created_edges)
    """
    created_tasks: List[str] = []
    created_edges: List[Dict[str, Any]] = []
    tasks = state.setdefault("tasks", {})

    new_subtasks = suggestion.get("new_subtasks", []) or []
    for st in new_subtasks:
        if not isinstance(st, dict):
            continue
        name = (st.get("name") or "").strip()
        parent = (st.get("parent") or "").strip() or None
        if not name:
            continue
        tid = ensure_task(state, name, parent_name=parent)
        if tid not in created_tasks:
            created_tasks.append(tid)
        # store rationale as a note evidence-less
        if st.get("rationale"):
            tasks[tid].setdefault("notes", []).append({
                "type": "rationale",
                "text": str(st.get("rationale")),
                "ts": _now_iso(),
            })
            tasks[tid]["updated_at"] = _now_iso()

    dep_updates = suggestion.get("dependency_updates", []) or []
    for ed in dep_updates:
        if not isinstance(ed, dict):
            continue
        frm = (ed.get("from") or "").strip()
        to = (ed.get("to") or "").strip()
        etype = (ed.get("type") or "depends_on").strip()
        if not frm or not to:
            continue
        frm_id = ensure_task(state, frm)
        to_id = ensure_task(state, to)
        edge = {"from": frm_id, "to": to_id, "type": etype}
        state.setdefault("edges", []).append(edge)
        created_edges.append(edge)
        # also list dependencies on to_id (simple)
        tasks[frm_id].setdefault("dependencies", [])
        if to_id not in tasks[frm_id]["dependencies"]:
            tasks[frm_id]["dependencies"].append(to_id)
            tasks[frm_id]["updated_at"] = _now_iso()

    return created_tasks, created_edges


def route_active_tasks(
    state: Dict[str, Any],
    suggestion: Dict[str, Any],
) -> List[str]:
    """
    Decide which tasks are "active" this iteration.
    Priority order:
      1) suggestion.subtasks_to_update (names)
      2) suggestion.focus_area (single name like 'Architecture'/'Fusion'/'Loss'/'All')
      3) fall back to root children
    """
    names = suggestion.get("subtasks_to_update")
    if isinstance(names, list) and names:
        ids = find_task_ids_by_name(state, [str(n) for n in names])
        if ids:
            return ids

    focus = str(suggestion.get("focus_area", "") or "").strip()
    if focus and focus.lower() != "all":
        ids = find_task_ids_by_name(state, [focus])
        if ids:
            return ids

    # fallback: all active children of root
    tasks = state.get("tasks", {})
    root = tasks.get("T0", {})
    children = root.get("children", []) or []
    # if empty, just root
    return children if children else ["T0"]


def update_after_iteration(
    state: Dict[str, Any],
    iteration: int,
    active_task_ids: List[str],
    improved: bool,
    score: float,
    target_metric: str,
    executed_notebook_path: Optional[str] = None,
    metrics_path: Optional[str] = None,
    reflection: Optional[str] = None,
) -> None:
    """
    Write evidence back to tasks and maintain iteration_map.
    Simple dynamics:
      - If improved: decrease uncertainty slightly; bump priority down a bit.
      - If not improved: bump uncertainty; keep priority or raise slightly.
    """
    tasks = state.setdefault("tasks", {})
    it_key = f"iter_{iteration}"
    state.setdefault("iteration_map", {})[it_key] = list(active_task_ids or [])

    ev = {
        "iteration": iteration,
        "improved": bool(improved),
        "score": float(score) if score is not None else None,
        "target_metric": target_metric,
        "executed_notebook_path": executed_notebook_path,
        "metrics_path": metrics_path,
        "reflection": reflection,
        "ts": _now_iso(),
    }

    for tid in active_task_ids or []:
        if tid not in tasks:
            continue
        t = tasks[tid]
        t.setdefault("evidence", []).append(ev)
        # simple attribute update
        unc = float(t.get("uncertainty", 0.5) or 0.5)
        pri = float(t.get("priority", 0.5) or 0.5)

        if improved:
            t["uncertainty"] = max(0.0, unc - 0.05)
            t["priority"] = max(0.0, pri - 0.02)
            # optionally mark done if very low uncertainty and multiple evidences
            if t["uncertainty"] < 0.15 and len(t.get("evidence", [])) >= 2:
                t["status"] = "stabilized"
        else:
            t["uncertainty"] = min(1.0, unc + 0.07)
            t["priority"] = min(1.0, pri + 0.01)

        t["updated_at"] = _now_iso()


def to_prompt_summary(state: Dict[str, Any], max_tasks: int = 12) -> str:
    """
    Compact summary injected into the LLM prompt.
    """
    tasks = state.get("tasks", {})
    # sort by priority desc then uncertainty desc
    def key(tid):
        t=tasks.get(tid,{})
        return (-(float(t.get("priority",0) or 0)), -(float(t.get("uncertainty",0) or 0)))
    tids = list(tasks.keys())
    # skip root in list
    tids = [t for t in tids if t != "T0"]
    tids.sort(key=key)
    tids = tids[:max_tasks]

    summary = {
        "tasks": [
            {
                "id": tid,
                "name": tasks[tid].get("name"),
                "status": tasks[tid].get("status"),
                "priority": round(float(tasks[tid].get("priority", 0) or 0), 3),
                "uncertainty": round(float(tasks[tid].get("uncertainty", 0) or 0), 3),
                "dependencies": tasks[tid].get("dependencies", []),
                "last_evidence_iter": (tasks[tid].get("evidence", [])[-1].get("iteration")
                                      if tasks[tid].get("evidence") else None),
            }
            for tid in tids
        ],
        "edges_count": len(state.get("edges", []) or []),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)
