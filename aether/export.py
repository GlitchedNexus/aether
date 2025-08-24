from __future__ import annotations

"""Aether export helpers.

This module provides lightweight exporters for scatterer selection results.
It focuses on tabular (CSV/JSON) outputs and leaves visualization/meshing to
higher layers.

Public API:
- ExportParams: configuration for export behavior.
- build_rows(...): construct a list[dict] rows from inputs.
- write_csv(path, rows, fields=None): write rows to CSV.
- write_json(path, rows, indent=2): write rows to JSON.
- export_selection(base_path, rows, formats=("csv","json")): convenience wrapper.

Design goals:
- Import-safe (no side effects), pure-Python with NumPy.
- Flexible schema: you can pass any per-scatterer arrays/columns; we'll merge by index.
- Robust type conversion so outputs are JSON/CSV friendly.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import csv
import json
import os

import numpy as np

__all__ = (
    "ExportParams",
    "build_rows",
    "write_csv",
    "write_json",
    "export_selection",
)


@dataclass(frozen=True)
class ExportParams:
    """Configuration for exporting selection results."""

    include_unselected: bool = False
    sort_desc_by: Optional[str] = "weight"
    fields: Optional[Sequence[str]] = None
    json_indent: int = 2


def _to_py_scalar(x):
    """Convert value to a JSON/CSV-friendly Python type."""
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (list, tuple)):
        return [_to_py_scalar(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_py_scalar(v) for k, v in x.items()}
    return x if isinstance(x, (str, bool, type(None), int, float)) else str(x)


def _ensure_len_equal(columns: Mapping[str, Sequence]) -> int:
    lengths = {k: len(v) for k, v in columns.items()}
    if not lengths:
        return 0
    n = next(iter(lengths.values()))
    if any(m != n for m in lengths.values()):
        raise ValueError(f"All column lengths must match, got: {lengths}")
    return n


def build_rows(
    *,
    indices: Sequence[int],
    weights: Sequence,
    selected_idx: Sequence[int],
    columns: Optional[Mapping[str, Sequence]] = None,
    params: Optional[ExportParams] = None,
) -> List[Dict[str, object]]:
    """Build a list of row dicts representing scatterers and selection metadata."""
    p = params or ExportParams()

    n = len(indices)
    if len(weights) != n:
        raise ValueError("weights must have the same length as indices")
    cols = dict(columns or {})
    if cols:
        _ensure_len_equal(cols | {"__indices__": indices})

    selected_set = set(int(i) for i in selected_idx)
    order = sorted((i for i in selected_set if 0 <= i < n), key=lambda i: (-float(weights[i]), int(i)))
    rank_of: Dict[int, int] = {i: r + 1 for r, i in enumerate(order)}

    rows: List[Dict[str, object]] = []
    include_all = p.include_unselected

    for i in range(n):
        sel = i in selected_set
        if not sel and not include_all:
            continue
        row: Dict[str, object] = {
            "id": int(indices[i]),
            "index": i,
            "weight": _to_py_scalar(weights[i]),
            "selected": bool(sel),
        }
        if sel:
            row["rank"] = rank_of[i]
        for k, seq in cols.items():
            row[k] = _to_py_scalar(seq[i])
        rows.append(row)

    if p.sort_desc_by is not None and len(rows) > 1:
        key = p.sort_desc_by
        try:
            def safe_float(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return float('-inf')
            rows.sort(key=lambda r: (r.get(key) is None, -safe_float(r.get(key, 0))))
        except Exception:
            pass

    return rows


def write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Optional[Sequence[str]] = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        header = list(fields or [])
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
        return os.path.abspath(path)

    if fields is None:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fields = sorted(all_keys)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    return os.path.abspath(path)


def write_json(path: str, rows: Sequence[Mapping[str, object]], indent: int = 2) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = [_to_py_scalar(r) for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    return os.path.abspath(path)


def export_selection(
    base_path: str,
    rows: Sequence[Mapping[str, object]],
    *,
    formats: Sequence[str] = ("csv", "json"),
    params: Optional[ExportParams] = None,
) -> Tuple[Optional[str], Optional[str]]:
    _ = params or ExportParams()

    csv_path = json_path = None
    fmts = [s.lower() for s in formats]

    if "csv" in fmts:
        csv_path = write_csv(f"{base_path}.csv", rows, fields=(_.fields if _ and _.fields else None))
    if "json" in fmts:
        json_path = write_json(f"{base_path}.json", rows, indent=(_.json_indent if _ else 2))

    return csv_path, json_path