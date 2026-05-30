from __future__ import annotations

from hypothesis import given, strategies as st

from engine import (
    OPPONENTS,
    Action,
    CondottiereGame,
    Region,
    initial_snapshot,
    legal_model_actions,
    oracle_value,
    overcommitment,
    spend_value_corr,
    step_with_opponent,
)


def test_seed_determinism() -> None:
    actions = [Action.play(1), Action.play(2), Action.pass_(), Action.play(3)]
    left = initial_snapshot(7, "MatchFold")
    right = initial_snapshot(7, "MatchFold")

    for action in actions:
        left = step_with_opponent(left, action)
        right = step_with_opponent(right, action)

    assert left == right


@given(
    seed=st.integers(min_value=0, max_value=100),
    policy=st.sampled_from(OPPONENTS),
    choices=st.lists(st.integers(min_value=0, max_value=8), min_size=1, max_size=20),
)
def test_card_conservation_for_random_legal_play(
    seed: int, policy: str, choices: list[int]
) -> None:
    snapshot = initial_snapshot(seed, policy)  # type: ignore[arg-type]
    initial_cards = len(snapshot.model_hand) + len(snapshot.opponent_hand)

    for choice in choices:
        actions = legal_model_actions(snapshot)
        if not actions:
            break
        snapshot = step_with_opponent(snapshot, actions[choice % len(actions)])
        spent = sum(
            len(record.model_cards) + len(record.opponent_cards)
            for record in snapshot.battle_records
        )
        in_play = len(snapshot.model_line) + len(snapshot.opponent_line)
        in_hand = len(snapshot.model_hand) + len(snapshot.opponent_hand)
        assert spent + in_play + in_hand == initial_cards


def test_illegal_action_returns_error_without_crashing() -> None:
    game = CondottiereGame(0, "MatchFold")
    resolved = game.play(99)

    assert game.illegal_action_count == 1
    assert resolved.error
    assert 99 not in resolved.your_hand


def test_oracle_dominates_greedy_legal_policy() -> None:
    for policy in OPPONENTS:
        snapshot = initial_snapshot(3, policy)
        while not snapshot.is_done:
            actions = legal_model_actions(snapshot)
            plays = [action for action in actions if action.kind == "play"]
            action = max(plays, key=lambda item: item.card or 0) if plays else Action.pass_()
            snapshot = step_with_opponent(snapshot, action)

        assert oracle_value(3, policy) >= snapshot.captured_value


def test_oracle_matches_independent_tiny_exhaustive_search() -> None:
    regions = (Region("A", 2), Region("B", 5))
    hand = (1, 3)

    for policy in OPPONENTS:
        assert oracle_value(0, policy, hand=hand, regions=regions) == brute_force_value(
            initial_snapshot(0, policy, hand=hand, regions=regions)
        )


def test_last_card_play_resolves_current_battle() -> None:
    regions = (Region("A", 2),)
    snapshot = initial_snapshot(
        0,
        "Conservative",
        hand=(10,),
        regions=regions,
    )

    snapshot = step_with_opponent(snapshot, Action.play(10))

    assert snapshot.is_done
    assert snapshot.captured_value == 2
    assert snapshot.battle_records[-1].outcome == "won"


def test_normalized_regret_uses_snapshot_initial_hand() -> None:
    regions = (Region("A", 2),)
    snapshot = initial_snapshot(
        0,
        "Conservative",
        hand=(10,),
        regions=regions,
    )

    snapshot = step_with_opponent(snapshot, Action.play(10))

    from engine import normalized_regret

    assert normalized_regret(snapshot) == 0.0


def test_empty_hand_auto_passes_remaining_regions() -> None:
    regions = (Region("A", 2), Region("B", 5))
    snapshot = initial_snapshot(
        0,
        "Conservative",
        hand=(10,),
        regions=regions,
    )

    snapshot = step_with_opponent(snapshot, Action.play(10))

    assert snapshot.is_done
    assert len(snapshot.battle_records) == 2
    assert snapshot.battle_records[1].model_cards == ()


def test_resolved_state_after_battle_reports_next_battle_as_ongoing() -> None:
    regions = (Region("A", 2), Region("B", 5))
    snapshot = initial_snapshot(
        0,
        "Conservative",
        hand=(1, 10),
        regions=regions,
    )

    snapshot = step_with_opponent(snapshot, Action.play(10))
    resolved = snapshot.resolved_state().as_dict()

    assert resolved["region"] == "B"
    assert resolved["battle_status"] == "ongoing"
    assert resolved["last_battle"]["region"] == "A"  # type: ignore[index]


def test_metrics_on_hand_built_loss() -> None:
    snapshot = initial_snapshot(0, "Aggressive")
    snapshot = step_with_opponent(snapshot, Action.play(max(snapshot.model_hand)))
    snapshot = step_with_opponent(snapshot, Action.pass_())

    assert 0.0 <= overcommitment(snapshot) <= 1.0
    assert -1.0 <= spend_value_corr(snapshot) <= 1.0


def brute_force_value(snapshot) -> int:
    frontier = [snapshot]
    best = 0
    while frontier:
        current = frontier.pop()
        if current.is_done:
            best = max(best, current.captured_value)
            continue
        for action in legal_model_actions(current):
            frontier.append(step_with_opponent(current, action))
    return best
