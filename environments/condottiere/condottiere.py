"""Prime/verifiers wiring for the Condottiere budgeting environment."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import verifiers as vf

from engine import (
    CondottiereGame,
    DEFAULT_HAND,
    OPPONENTS,
    PolicyName,
    Region,
    early_dump_index as engine_early_dump_index,
    initial_snapshot,
    normalized_regret as engine_normalized_regret,
    overcommitment as engine_overcommitment,
    spend_value_corr as engine_spend_value_corr,
)


SYSTEM_PROMPT = """You are playing a card game.
Use exactly one tool call per turn: play(card) or pass().
A card you play is permanently spent, even if you lose or tie that battle.
"""


def condottiere_source(
    *,
    opponent: PolicyName = "MatchFold",
    num_examples: int = 10,
    budget_signal: str | None = None,
) -> Iterable[dict[str, Any]]:
    """Yield deterministic seeded game tasks."""
    for seed in range(num_examples):
        snapshot = initial_snapshot(seed=seed, opponent_policy=opponent)
        public_state = snapshot.resolved_state(budget_signal).as_dict()
        yield {
            "prompt": [
                {
                    "role": "user",
                    "content": render_initial_prompt(public_state),
                }
            ],
            "info": {
                "seed": seed,
                "opponent_policy": opponent,
                "budget_signal": budget_signal,
            },
            "toolsets": {
                "condottiere": {"fn": "condottiere:condottiere_toolset"},
            },
            "answer": None,
        }


def render_initial_prompt(public_state: Mapping[str, object]) -> str:
    return (
        "Initial public state:\n"
        f"{public_state}\n"
        "Choose one action with the provided tools."
    )


def condottiere_toolset(task: Mapping[str, Any], state: dict[str, Any]) -> vf.Toolset:
    """Create one rollout-scoped game object and expose play/pass tools."""
    info = cast(Mapping[str, Any], task.get("info") or {})
    seed = int(info.get("seed", 0))
    opponent = cast(PolicyName, info.get("opponent_policy", "MatchFold"))
    budget_signal = cast(str | None, info.get("budget_signal"))

    def make_game() -> CondottiereGame:
        return CondottiereGame(
            seed=seed,
            opponent_policy=opponent,
            budget_signal=budget_signal,
        )

    return vf.Toolset(
        tools=[play, pass_turn],
        objects={"game": make_game},
        bindings={
            "play.game": "objects.game",
            "pass.game": "objects.game",
        },
        scope="rollout",
    )


def play(card: int, game: CondottiereGame, state: dict[str, Any]) -> dict[str, object]:
    """Play one card from your current hand, then observe the opponent response."""
    resolved = game.play(card)
    update_state_from_game(state, game)
    return resolved.as_dict()


def pass_turn(game: CondottiereGame, state: dict[str, Any]) -> dict[str, object]:
    """Pass for the current battle, then observe the opponent response."""
    resolved = game.pass_()
    update_state_from_game(state, game)
    return resolved.as_dict()


pass_turn.__name__ = "pass"


def update_state_from_game(state: dict[str, Any], game: CondottiereGame) -> None:
    snapshot = game.snapshot
    state["condottiere"] = snapshot_to_scoring_state(snapshot)
    state["game_done"] = snapshot.is_done


def condottiere_user(task: Mapping[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    _ = task
    if state.get("game_done"):
        return []
    return [
        {
            "role": "user",
            "content": "The game is not complete. Make exactly one tool call now: play(card) or pass().",
        }
    ]


def snapshot_to_scoring_state(snapshot: Any) -> dict[str, Any]:
    # Internal scoring/diagnostic state, stored under state["condottiere"] for the reward
    # functions and logging. NOT model-facing — it carries the oracle-derived regret,
    # opponent policy, all region values, and full battle records. Model-visible state is
    # the resolved public view returned by the tools, never this dict.
    return {
        "seed": snapshot.seed,
        "opponent_policy": snapshot.opponent_policy,
        "regions": [
            {"name": region.name, "value": region.value}
            for region in snapshot.regions
        ],
        "captured_value": snapshot.captured_value,
        "illegal_action_count": snapshot.illegal_action_count,
        "cards_total": snapshot.cards_total,
        "cards_on_losses": snapshot.cards_on_losses,
        "spend_by_region": list(snapshot.spend_by_region),
        "battle_records": [
            {
                "region": record.region.name,
                "region_value": record.region.value,
                "model_cards": list(record.model_cards),
                "opponent_cards": list(record.opponent_cards),
                "outcome": record.outcome,
            }
            for record in snapshot.battle_records
        ],
        "normalized_regret": engine_normalized_regret(snapshot),
        "overcommitment": engine_overcommitment(snapshot),
        "spend_value_corr": engine_spend_value_corr(snapshot),
        "early_dump_index": engine_early_dump_index(snapshot),
    }


def tool_protocol_counts(state: Mapping[str, Any]) -> dict[str, int]:
    completion = state.get("completion") or []
    if not isinstance(completion, list):
        return {"no_tool_turns": 0, "multi_tool_turns": 0}
    no_tool_turns = 0
    multi_tool_turns = 0
    for message in completion:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []
        if len(tool_calls) == 0:
            no_tool_turns += 1
        elif len(tool_calls) > 1:
            multi_tool_turns += 1
    return {"no_tool_turns": no_tool_turns, "multi_tool_turns": multi_tool_turns}


@vf.stop
async def hand_empty_or_regions_done(
    task: Mapping[str, Any], state: dict[str, Any]
) -> bool:
    _ = task
    return bool(state.get("game_done"))


@vf.reward(weight=1.0)
async def neg_regret(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    data = cast(Mapping[str, Any], state.get("condottiere") or {})
    return -float(data.get("normalized_regret", 1.0))


@vf.metric
async def overcommitment(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    data = cast(Mapping[str, Any], state.get("condottiere") or {})
    return float(data.get("overcommitment", 0.0))


@vf.metric
async def spend_value_corr(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    data = cast(Mapping[str, Any], state.get("condottiere") or {})
    return float(data.get("spend_value_corr", 0.0))


@vf.metric
async def early_dump(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    data = cast(Mapping[str, Any], state.get("condottiere") or {})
    return float(data.get("early_dump_index", 0.0))


@vf.metric
async def illegal_actions(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    data = cast(Mapping[str, Any], state.get("condottiere") or {})
    return float(data.get("illegal_action_count", 0))


@vf.metric
async def no_tool_turns(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    return float(tool_protocol_counts(state)["no_tool_turns"])


@vf.metric
async def multi_tool_turns(task: Mapping[str, Any], state: dict[str, Any]) -> float:
    _ = task
    return float(tool_protocol_counts(state)["multi_tool_turns"])


class CondottiereHarness(vf.Harness):
    """Harness shell for the custom Condottiere tool protocol."""


def load_environment(
    opponent: PolicyName = "MatchFold",
    num_examples: int = 10,
    max_turns: int = 32,
    budget_signal: str | None = None,
) -> vf.Env:
    if opponent not in OPPONENTS:
        raise ValueError(f"opponent must be one of {OPPONENTS}")
    taskset = vf.Taskset(
        eval_source=lambda: condottiere_source(
            opponent=opponent,
            num_examples=num_examples,
            budget_signal=budget_signal,
        ),
        taskset_id="condottiere",
        system_prompt=SYSTEM_PROMPT,
    )
    harness = CondottiereHarness(
        max_turns=max_turns,
        user=condottiere_user,
        stops=[hand_empty_or_regions_done],
        rewards=[neg_regret],
        metrics=[
            overcommitment,
            spend_value_corr,
            early_dump,
            illegal_actions,
            no_tool_turns,
            multi_tool_turns,
        ],
    )
    return vf.Env(taskset=taskset, harness=harness)
