"""Offline analysis helpers for Condottiere eval outputs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def budget_responsiveness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired Δspend / Δstated-budget from saved rollout rows.

    Rows must include saved `condottiere` state data, which is available when
    evals are run with `-C condottiere` or equivalent config.
    """
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        info = row.get("info") or {}
        if not isinstance(info, dict):
            continue
        if condottiere_state(row) is None:
            continue
        groups[
            (
                str(row.get("_analysis_model", row.get("model", "unknown"))),
                int(info.get("seed", -1)),
                str(info.get("opponent_policy")),
            )
        ].append(row)

    values: list[float] = []
    by_model: dict[str, list[float]] = defaultdict(list)
    for (model, _, _), pair_rows in groups.items():
        if len(pair_rows) < 2:
            continue
        ordered = sorted(pair_rows, key=lambda item: stated_budget(item) or 0.0)
        lo, hi = ordered[0], ordered[-1]
        lo_budget = stated_budget(lo)
        hi_budget = stated_budget(hi)
        lo_spend = total_spend(lo)
        hi_spend = total_spend(hi)
        if lo_budget is None or hi_budget is None or lo_budget == hi_budget:
            continue
        value = (hi_spend - lo_spend) / (hi_budget - lo_budget)
        values.append(value)
        by_model[model].append(value)

    return {
        "pairs": float(len(values)),
        "budget_responsiveness": mean(values) if values else 0.0,
        "by_model": {
            model: {
                "pairs": float(len(model_values)),
                "budget_responsiveness": mean(model_values),
            }
            for model, model_values in sorted(by_model.items())
        },
    }


def total_spend(row: dict[str, Any]) -> float:
    data = condottiere_state(row)
    if data is None:
        return 0.0
    return float(data.get("cards_total", 0.0))


def condottiere_state(row: dict[str, Any]) -> dict[str, Any] | None:
    data = row.get("condottiere")
    if isinstance(data, dict):
        return data
    info = row.get("info") or {}
    if not isinstance(info, dict):
        return None
    data = info.get("condottiere")
    return data if isinstance(data, dict) else None


def stated_budget(row: dict[str, Any]) -> float | None:
    info = row.get("info") or {}
    if not isinstance(info, dict):
        return None
    signal = info.get("budget_signal")
    if signal is None:
        return None
    digits = "".join(char if char.isdigit() else " " for char in str(signal))
    numbers = [int(part) for part in digits.split()]
    return float(numbers[0]) if numbers else None


def rows_from_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        metadata = load_metadata(path)
        model = metadata.get("model") if isinstance(metadata, dict) else None
        for row in load_jsonl(path):
            if isinstance(model, str):
                row["_analysis_model"] = model
            rows.append(row)
    return rows


def load_metadata(results_path: Path) -> dict[str, Any]:
    metadata_path = results_path.parent / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_jsonl", type=Path, nargs="+")
    args = parser.parse_args()
    rows = rows_from_results(args.results_jsonl)
    result = budget_responsiveness(rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["pairs"] == 0.0:
        raise SystemExit(
            "No valid paired rollouts found. Pass at least two results files with "
            "matching model, seed, and opponent_policy, and run evals with "
            "`-C condottiere` so saved state includes spend data."
        )


if __name__ == "__main__":
    main()
