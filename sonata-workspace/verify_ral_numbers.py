#!/usr/bin/env python3
"""Re-derive RAL headline metrics from eval JSON files."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


OK = "\u2713"
WARN = "\u26a0"
BAD = "\u2717"


TARGET_FILES = [
    "eval_finetuned_str80.json",
    "eval_scaffoldfree_str80_50.json",
    "eval_scaffoldfree_500.json",
    "eval_crossseq_seq00.json",
    "eval_crossseq_seq05.json",
    "eval_jitter_sig0.0.json",
    "eval_jitter_sig0.1.json",
    "eval_jitter_sig0.2.json",
    "eval_jitter_sig0.5.json",
    "eval_jitter_sig1.0.json",
    "eval_finetuned_fullval.json",
    "eval_lidiff_on_v2gt_50fr.json",
    "eval_scorelidar_on_v2gt_50fr.json",
]


def mean_std(values: Iterable[float]) -> tuple[float, float, int]:
    vals = [float(v) for v in values if isinstance(v, int | float) and math.isfinite(v)]
    if not vals:
        return math.nan, math.nan, 0
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var), len(vals)


def rel_dev(actual: float, claim: float) -> float:
    if not math.isfinite(actual) or not math.isfinite(claim):
        return math.inf
    denom = abs(claim) if claim != 0 else 1.0
    return abs(actual - claim) / denom


def status_for(pairs: Iterable[tuple[float, float]]) -> str:
    worst = max((rel_dev(actual, claim) for actual, claim in pairs), default=math.inf)
    if worst <= 0.01:
        return OK
    if worst <= 0.05:
        return WARN
    return BAD


def load_jsons() -> dict[str, Any]:
    entries = set(os.listdir("."))
    present = sum(name in entries for name in TARGET_FILES)
    print(f"LS target JSONs present: {present}/{len(TARGET_FILES)}")

    loaded: dict[str, Any] = {}
    seen_keysets: set[tuple[str, ...]] = set()
    for name in TARGET_FILES:
        path = Path(name)
        if not path.exists():
            print(f"MISSING: {name}")
            continue
        data = json.loads(path.read_text())
        loaded[name] = data
        keyset = top_keys(data)
        if keyset not in seen_keysets:
            seen_keysets.add(keyset)
            print(f"KEYS {name}: {','.join(keyset)}")
    return loaded


def top_keys(data: Any) -> tuple[str, ...]:
    if isinstance(data, dict):
        return tuple(str(k) for k in data.keys())
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return tuple(str(k) for k in data[0].keys())
    return (type(data).__name__,)


def per_frame_values(data: Any, field_names: Iterable[str]) -> list[float]:
    fields = set(field_names)
    candidates: list[list[float]] = []

    def add_numeric_list(items: Iterable[Any]) -> None:
        vals = [float(v) for v in items if isinstance(v, int | float) and math.isfinite(v)]
        if vals:
            candidates.append(vals)

    def metric_number(value: Any) -> float | None:
        if isinstance(value, int | float) and math.isfinite(value):
            return float(value)
        if isinstance(value, dict):
            for key in ("cd_mean", "mean", "cd_sq", "cd2", "cd", "value"):
                nested = value.get(key)
                if isinstance(nested, int | float) and math.isfinite(nested):
                    return float(nested)
        return None

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in fields:
                    if isinstance(value, list):
                        add_numeric_list(value)
                    elif isinstance(value, int | float):
                        candidates.append([float(value)])
                walk(value)
        elif isinstance(node, list):
            for field in fields:
                vals = []
                for item in node:
                    if isinstance(item, dict) and field in item:
                        value = metric_number(item[field])
                        if value is not None:
                            vals.append(value)
                if vals:
                    candidates.append(vals)
            for item in node:
                if isinstance(item, dict | list):
                    walk(item)

    walk(data)
    if not candidates:
        return []
    return max(candidates, key=len)


def variant_stats(data: dict[str, Any], variant: str) -> tuple[float, float, int]:
    stats = data.get("variants", {}).get(variant, {})
    return (
        float(stats.get("pred_cd_mean", math.nan)),
        float(stats.get("pred_cd_std", math.nan)),
        int(stats.get("n_frames", 0) or 0),
    )


def fmt(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "missing"
    return f"{mean:.4g}\u00b1{std:.4g}"


def add_row(
    rows: list[tuple[str, int, str, str, str, list[float]]],
    name: str,
    values: list[float],
    claim: tuple[float, float | None],
    paper: str,
) -> None:
    mean, std, n = mean_std(values)
    pairs = [(mean, claim[0])]
    if claim[1] is not None:
        pairs.append((std, claim[1]))
    rows.append((name, n, fmt(mean, std), paper, status_for(pairs), values[:5]))


def main() -> None:
    data = load_jsons()
    rows: list[tuple[str, int, str, str, str, list[float]]] = []

    if "eval_finetuned_str80.json" in data:
        mean, std, n = variant_stats(data["eval_finetuned_str80.json"], "A_fair_lidiff_match_kdtree")
        rows.append(("teacher-FT kdtree", n, "agg " + fmt(mean, std), "0.727\u00b10.378", status_for([(mean, 0.727), (std, 0.378)]), []))

    for name in ("eval_scaffoldfree_str80_50.json", "eval_scaffoldfree_500.json"):
        if name in data:
            mean, std, n = variant_stats(data[name], "A_fair_ego_bbox_lidiff_crop")
            rows.append((f"pre-FT scaffoldfree {n}", n, "agg " + fmt(mean, std), "~12.58\u00b18.14", status_for([(mean, 12.58), (std, 8.14)]), []))
            break

    cd2_fields = ("cd_sq", "cd2", "cd_squared", "chamfer_distance_sq", "chamfer_distance_squared", "per_frame_cd2")
    add_row(rows, "cross-seq 00", per_frame_values(data.get("eval_crossseq_seq00.json"), cd2_fields), (0.0239, 0.0041), "0.0239\u00b10.0041")
    add_row(rows, "cross-seq 05", per_frame_values(data.get("eval_crossseq_seq05.json"), cd2_fields), (0.0255, 0.0041), "0.0255\u00b10.0041")

    jitter_claims = {
        "0.0": 0.024,
        "0.1": 0.025,
        "0.2": 0.029,
        "0.5": 0.083,
        "1.0": 0.316,
    }
    for sigma, claim in jitter_claims.items():
        filename = f"eval_jitter_sig{sigma}.json"
        add_row(rows, f"jitter {sigma}", per_frame_values(data.get(filename), cd2_fields), (claim, None), f"~{claim:g}")

    full = data.get("eval_finetuned_fullval.json")
    add_row(rows, "full-val CD2", per_frame_values(full, cd2_fields), (0.030, None), "~0.030")
    add_row(rows, "full-val F@0.2", per_frame_values(full, ("f_score@0.2", "fscore@0.2", "f_at_0.2")), (0.826, None), "~0.826")

    for filename, label, diff_claim, refine_claim in [
        ("eval_lidiff_on_v2gt_50fr.json", "LiDiff 50fr", 3.41, 3.50),
        ("eval_scorelidar_on_v2gt_50fr.json", "ScoreLiDAR 50fr", 3.19, 3.15),
    ]:
        item = data.get(filename)
        diff_vals = per_frame_values(item, ("cd_diff",))
        refine_vals = per_frame_values(item, ("cd_refine",))
        d_mean, d_std, d_n = mean_std(diff_vals)
        r_mean, r_std, r_n = mean_std(refine_vals)
        n = min(d_n, r_n)
        computed = f"diff {fmt(d_mean, d_std)} / ref {fmt(r_mean, r_std)}"
        paper = f"~{diff_claim:g}/{refine_claim:g}"
        status = status_for([(d_mean, diff_claim), (r_mean, refine_claim)])
        rows.append((label, n, computed, paper, status, diff_vals[:5]))

    print("NOTE: Table III variant files have no per_frame; used stored aggregates.")
    print("claim | N | computed | paper | status")
    for name, n, computed, paper, status, first5 in rows:
        print(f"{name} | {n} | {computed} | {paper} | {status}")
        if status == BAD:
            print(" first5: " + ",".join(f"{v:.4g}" for v in first5))


if __name__ == "__main__":
    main()
