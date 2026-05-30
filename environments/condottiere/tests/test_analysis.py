from __future__ import annotations

import json

from analysis import budget_responsiveness, rows_from_results


def test_budget_responsiveness_from_paired_rows() -> None:
    rows = [
        {
            "_analysis_model": "model-a",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "4 battles left",
            },
            "condottiere": {"cards_total": 2},
        },
        {
            "_analysis_model": "model-a",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "12 battles left",
            },
            "condottiere": {"cards_total": 6},
        },
    ]

    result = budget_responsiveness(rows)

    assert result["pairs"] == 1.0
    assert result["budget_responsiveness"] == 0.5
    assert result["by_model"]["model-a"]["budget_responsiveness"] == 0.5


def test_budget_responsiveness_does_not_pair_across_models() -> None:
    rows = [
        {
            "_analysis_model": "model-a",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "4 battles left",
            },
            "condottiere": {"cards_total": 2},
        },
        {
            "_analysis_model": "model-b",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "12 battles left",
            },
            "condottiere": {"cards_total": 6},
        },
    ]

    result = budget_responsiveness(rows)

    assert result["pairs"] == 0.0


def test_budget_responsiveness_reads_hosted_sample_shape() -> None:
    rows = [
        {
            "_analysis_model": "model-a",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "4 battles left",
                "condottiere": {"cards_total": 0},
            },
        },
        {
            "_analysis_model": "model-a",
            "info": {
                "seed": 1,
                "opponent_policy": "MatchFold",
                "budget_signal": "12 battles left",
                "condottiere": {"cards_total": 4},
            },
        },
    ]

    result = budget_responsiveness(rows)

    assert result["pairs"] == 1.0
    assert result["budget_responsiveness"] == 0.5


def test_rows_from_results_injects_model_from_metadata(tmp_path) -> None:
    run_dir = tmp_path / "eval-run"
    run_dir.mkdir()
    results_path = run_dir / "results.jsonl"
    metadata_path = run_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps({"model": "model-from-metadata"}), encoding="utf-8"
    )
    results_path.write_text(
        json.dumps(
            {
                "info": {
                    "seed": 1,
                    "opponent_policy": "MatchFold",
                    "budget_signal": "4 battles left",
                },
                "condottiere": {"cards_total": 2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = rows_from_results([results_path])

    assert rows[0]["_analysis_model"] == "model-from-metadata"
