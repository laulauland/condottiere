"""Standalone Condottiere v0 engine.

The engine is deliberately independent of verifiers so oracle, metrics, and
offline analysis can share exactly the same game semantics as the harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import random
from typing import Literal


PolicyName = Literal["Aggressive", "Conservative", "Hoarder", "MatchFold"]
BattleStatus = Literal["ongoing", "won", "lost", "tied"]
ActionKind = Literal["play", "pass"]
OPPONENTS: tuple[PolicyName, ...] = (
    "Aggressive",
    "Conservative",
    "Hoarder",
    "MatchFold",
)
REGION_NAMES = (
    "Florence",
    "Milan",
    "Venice",
    "Genoa",
    "Siena",
    "Naples",
    "Lucca",
    "Pisa",
)
DEFAULT_HAND = (1, 2, 3, 5, 6, 10)


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    card: int | None = None

    @staticmethod
    def play(card: int) -> Action:
        return Action("play", card)

    @staticmethod
    def pass_() -> Action:
        return Action("pass")


@dataclass(frozen=True)
class Region:
    name: str
    value: int


@dataclass(frozen=True)
class ResolvedState:
    region: str | None
    region_value: int | None
    your_line: tuple[int, ...]
    your_strength: int
    opp_line: tuple[int, ...]
    opp_strength: int
    your_hand: tuple[int, ...]
    battle_status: BattleStatus
    battles_remaining: int
    budget_signal: str
    last_battle: dict[str, object] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "region": self.region,
            "region_value": self.region_value,
            "your_line": list(self.your_line),
            "your_strength": self.your_strength,
            "opp_line": list(self.opp_line),
            "opp_strength": self.opp_strength,
            "your_hand": list(self.your_hand),
            "battle_status": self.battle_status,
            "battles_remaining": self.battles_remaining,
            "budget_signal": self.budget_signal,
            **({"last_battle": self.last_battle} if self.last_battle else {}),
            **({"error": self.error} if self.error else {}),
        }


@dataclass(frozen=True)
class BattleRecord:
    region: Region
    model_cards: tuple[int, ...]
    opponent_cards: tuple[int, ...]
    outcome: BattleStatus


@dataclass(frozen=True)
class GameSnapshot:
    seed: int
    opponent_policy: PolicyName
    regions: tuple[Region, ...]
    initial_model_hand: tuple[int, ...]
    region_index: int
    model_hand: tuple[int, ...]
    opponent_hand: tuple[int, ...]
    model_line: tuple[int, ...] = ()
    opponent_line: tuple[int, ...] = ()
    model_passed: bool = False
    opponent_passed: bool = False
    captured_value: int = 0
    battle_records: tuple[BattleRecord, ...] = ()
    illegal_action_count: int = 0
    last_status: BattleStatus = "ongoing"
    last_error: str | None = None

    @property
    def is_done(self) -> bool:
        return self.region_index >= len(self.regions)

    @property
    def current_region(self) -> Region | None:
        if self.region_index >= len(self.regions):
            return None
        return self.regions[self.region_index]

    @property
    def battles_remaining(self) -> int:
        return max(len(self.regions) - self.region_index - 1, 0)

    @property
    def model_strength(self) -> int:
        return sum(self.model_line)

    @property
    def opponent_strength(self) -> int:
        return sum(self.opponent_line)

    @property
    def cards_total(self) -> int:
        return sum(len(record.model_cards) for record in self.battle_records) + len(
            self.model_line
        )

    @property
    def cards_on_losses(self) -> int:
        total = sum(
            len(record.model_cards)
            for record in self.battle_records
            if record.outcome in {"lost", "tied"}
        )
        if self.last_status in {"lost", "tied"}:
            total += len(self.model_line)
        return total

    @property
    def spend_by_region(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (sum(record.model_cards), record.region.value)
            for record in self.battle_records
        )

    def resolved_state(self, budget_signal: str | None = None) -> ResolvedState:
        region = self.current_region
        last_battle = self.battle_records[-1] if self.battle_records else None
        battle_status = self.last_status if region is None else "ongoing"
        signal = budget_signal or (
            f"{len(self.model_hand)} cards remain across "
            f"{self.battles_remaining + (0 if self.is_done else 1)} visible/remaining battles."
        )
        return ResolvedState(
            region=region.name if region else None,
            region_value=region.value if region else None,
            your_line=self.model_line,
            your_strength=self.model_strength,
            opp_line=self.opponent_line,
            opp_strength=self.opponent_strength,
            your_hand=self.model_hand,
            battle_status=battle_status,
            battles_remaining=self.battles_remaining,
            budget_signal=signal,
            last_battle=record_as_dict(last_battle) if last_battle else None,
            error=self.last_error,
        )


class CondottiereGame:
    """Mutable facade around immutable snapshots."""

    def __init__(
        self,
        seed: int,
        opponent_policy: PolicyName,
        *,
        hand: tuple[int, ...] | None = None,
        regions: tuple[Region, ...] | None = None,
        budget_signal: str | None = None,
    ) -> None:
        self.budget_signal = budget_signal
        self.snapshot = initial_snapshot(
            seed=seed,
            opponent_policy=opponent_policy,
            hand=hand,
            regions=regions,
        )

    @property
    def captured_value(self) -> int:
        return self.snapshot.captured_value

    @property
    def illegal_action_count(self) -> int:
        return self.snapshot.illegal_action_count

    @property
    def is_done(self) -> bool:
        return self.snapshot.is_done

    def play(self, card: int) -> ResolvedState:
        self.snapshot = step_with_opponent(self.snapshot, Action.play(card))
        return self.snapshot.resolved_state(self.budget_signal)

    def pass_(self) -> ResolvedState:
        self.snapshot = step_with_opponent(self.snapshot, Action.pass_())
        return self.snapshot.resolved_state(self.budget_signal)

    def state(self) -> ResolvedState:
        return self.snapshot.resolved_state(self.budget_signal)


def initial_snapshot(
    seed: int,
    opponent_policy: PolicyName,
    *,
    hand: tuple[int, ...] | None = None,
    regions: tuple[Region, ...] | None = None,
) -> GameSnapshot:
    if opponent_policy not in OPPONENTS:
        raise ValueError(f"Unknown opponent policy: {opponent_policy}")
    rng = random.Random(seed)
    model_hand = tuple(sorted(hand or DEFAULT_HAND))
    opponent_hand = tuple(sorted(hand or DEFAULT_HAND))
    if regions is None:
        values = tuple(rng.sample(range(2, 9), 3))
        names = tuple(rng.sample(REGION_NAMES, 3))
        regions = tuple(Region(name, value) for name, value in zip(names, values))
    return GameSnapshot(
        seed=seed,
        opponent_policy=opponent_policy,
        regions=regions,
        initial_model_hand=model_hand,
        region_index=0,
        model_hand=model_hand,
        opponent_hand=opponent_hand,
    )


def legal_model_actions(snapshot: GameSnapshot) -> tuple[Action, ...]:
    if snapshot.is_done:
        return ()
    if not snapshot.model_hand:
        return (Action.pass_(),)
    if snapshot.model_passed:
        return (Action.pass_(),)
    return tuple(Action.play(card) for card in snapshot.model_hand) + (Action.pass_(),)


def step_with_opponent(snapshot: GameSnapshot, action: Action) -> GameSnapshot:
    after_model = apply_model_action(snapshot, action)
    if after_model.is_done or after_model.last_error:
        return after_model
    if after_model.model_passed:
        return auto_pass_remaining_if_empty(finish_battle(opponent_respond(after_model)))
    if after_model.opponent_passed:
        return auto_pass_remaining_if_empty(finish_battle(after_model))
    after_opp = opponent_respond(after_model)
    if after_opp.opponent_passed or after_opp.model_passed:
        return auto_pass_remaining_if_empty(finish_battle(after_opp))
    if not after_opp.model_hand:
        return auto_pass_remaining_if_empty(finish_battle(replace(after_opp, model_passed=True)))
    return after_opp


def apply_model_action(snapshot: GameSnapshot, action: Action) -> GameSnapshot:
    if snapshot.is_done:
        return replace(snapshot, last_error=None)
    if action.kind == "pass":
        return replace(snapshot, model_passed=True, last_error=None)
    card = action.card
    if card not in snapshot.model_hand:
        return replace(
            snapshot,
            illegal_action_count=snapshot.illegal_action_count + 1,
            last_error=f"Illegal action: card {card} is not in your hand.",
        )
    return replace(
        snapshot,
        model_hand=remove_one(snapshot.model_hand, card),
        model_line=(*snapshot.model_line, card),
        last_error=None,
        last_status="ongoing",
    )


def opponent_respond(snapshot: GameSnapshot) -> GameSnapshot:
    action = opponent_action(snapshot)
    if action.kind == "pass":
        return replace(snapshot, opponent_passed=True)
    card = action.card
    if card not in snapshot.opponent_hand:
        return replace(snapshot, opponent_passed=True)
    return replace(
        snapshot,
        opponent_hand=remove_one(snapshot.opponent_hand, card),
        opponent_line=(*snapshot.opponent_line, card),
    )


def opponent_action(snapshot: GameSnapshot) -> Action:
    hand = snapshot.opponent_hand
    if not hand:
        return Action.pass_()
    region = snapshot.current_region
    assert region is not None
    deficit = snapshot.model_strength - snapshot.opponent_strength
    if snapshot.opponent_policy == "Aggressive":
        if deficit >= 0:
            return Action.play(smallest_at_least(hand, deficit + 1) or max(hand))
        return Action.play(min(hand)) if region.value >= 5 else Action.pass_()
    if snapshot.opponent_policy == "Conservative":
        if region.value < 5:
            return Action.pass_()
        return Action.play(smallest_at_least(hand, max(deficit + 1, 1)) or max(hand))
    if snapshot.opponent_policy == "Hoarder":
        if snapshot.battles_remaining > 0 and max(hand) >= 6 and region.value < 7:
            affordable = tuple(card for card in hand if card < 6)
            if not affordable:
                return Action.pass_()
            return Action.play(smallest_at_least(affordable, deficit + 1) or min(affordable))
        return Action.play(smallest_at_least(hand, max(deficit + 1, 1)) or max(hand))
    if snapshot.opponent_policy == "MatchFold":
        if region.value < 4:
            return Action.pass_()
        needed = max(deficit, 1)
        card = smallest_at_least(hand, needed)
        return Action.play(card) if card is not None else Action.pass_()
    raise AssertionError(snapshot.opponent_policy)


def finish_battle(snapshot: GameSnapshot) -> GameSnapshot:
    region = snapshot.current_region
    if region is None:
        return snapshot
    if snapshot.model_strength > snapshot.opponent_strength:
        outcome: BattleStatus = "won"
        captured = snapshot.captured_value + region.value
    elif snapshot.model_strength < snapshot.opponent_strength:
        outcome = "lost"
        captured = snapshot.captured_value
    else:
        outcome = "tied"
        captured = snapshot.captured_value
    record = BattleRecord(
        region=region,
        model_cards=snapshot.model_line,
        opponent_cards=snapshot.opponent_line,
        outcome=outcome,
    )
    return replace(
        snapshot,
        region_index=snapshot.region_index + 1,
        model_line=(),
        opponent_line=(),
        model_passed=False,
        opponent_passed=False,
        captured_value=captured,
        battle_records=(*snapshot.battle_records, record),
        last_status=outcome,
        last_error=None,
    )


def auto_pass_remaining_if_empty(snapshot: GameSnapshot) -> GameSnapshot:
    if snapshot.model_hand or snapshot.is_done:
        return snapshot
    current = snapshot
    while current.current_region is not None:
        region = current.current_region
        assert region is not None
        record = BattleRecord(
            region=region,
            model_cards=(),
            opponent_cards=(),
            outcome="tied",
        )
        current = replace(
            current,
            region_index=current.region_index + 1,
            battle_records=(*current.battle_records, record),
            last_status="tied",
            last_error=None,
        )
    return current


def oracle_value(
    seed: int,
    opponent_policy: PolicyName,
    *,
    hand: tuple[int, ...] | None = None,
    regions: tuple[Region, ...] | None = None,
) -> int:
    snapshot = initial_snapshot(
        seed=seed, opponent_policy=opponent_policy, hand=hand, regions=regions
    )
    return _oracle(snapshot)


@lru_cache(maxsize=None)
def _oracle(snapshot: GameSnapshot) -> int:
    if snapshot.is_done:
        return snapshot.captured_value
    return max(_oracle(step_with_opponent(snapshot, action)) for action in legal_model_actions(snapshot))


def overcommitment(snapshot: GameSnapshot) -> float:
    return snapshot.cards_on_losses / max(snapshot.cards_total, 1)


def spend_value_corr(snapshot: GameSnapshot) -> float:
    pairs = snapshot.spend_by_region
    if len(pairs) < 2:
        return 0.0
    spend = [pair[0] for pair in pairs]
    values = [pair[1] for pair in pairs]
    mean_spend = sum(spend) / len(spend)
    mean_value = sum(values) / len(values)
    numerator = sum((x - mean_spend) * (y - mean_value) for x, y in zip(spend, values))
    denom_spend = math.sqrt(sum((x - mean_spend) ** 2 for x in spend))
    denom_value = math.sqrt(sum((y - mean_value) ** 2 for y in values))
    denom = denom_spend * denom_value
    return numerator / denom if denom else 0.0


def early_dump_index(snapshot: GameSnapshot) -> float:
    records = snapshot.battle_records
    if not records:
        return 0.0
    total = sum(sum(record.model_cards) for record in records)
    if total <= 0:
        return 0.0
    first_n = max(1, math.ceil(len(snapshot.regions) / 3))
    early = sum(sum(record.model_cards) for record in records[:first_n])
    return early / total


def normalized_regret(snapshot: GameSnapshot) -> float:
    opt = oracle_value(
        snapshot.seed,
        snapshot.opponent_policy,
        hand=snapshot.initial_model_hand,
        regions=snapshot.regions,
    )
    if opt <= 0:
        return 0.0
    return (opt - snapshot.captured_value) / opt


def remove_one(cards: tuple[int, ...], card: int | None) -> tuple[int, ...]:
    values = list(cards)
    values.remove(card)
    return tuple(values)


def smallest_at_least(cards: tuple[int, ...], target: int) -> int | None:
    eligible = [card for card in cards if card >= target]
    return min(eligible) if eligible else None


def record_as_dict(record: BattleRecord | None) -> dict[str, object] | None:
    if record is None:
        return None
    return {
        "region": record.region.name,
        "region_value": record.region.value,
        "your_cards": list(record.model_cards),
        "opp_cards": list(record.opponent_cards),
        "outcome": record.outcome,
    }
