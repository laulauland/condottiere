from __future__ import annotations

import asyncio

from verifiers import load_environment
from condottiere import tool_protocol_counts


def test_environment_loads() -> None:
    env = load_environment("condottiere", num_examples=2)

    rows = env.taskset.eval_rows()
    assert len(rows) == 2
    assert type(env.harness).__name__ == "CondottiereHarness"


def test_tools_update_serializable_state() -> None:
    asyncio.run(_check_tools_update_serializable_state())


async def _check_tools_update_serializable_state() -> None:
    env = load_environment("condottiere", num_examples=1)
    task = env.taskset.to_task(env.taskset.eval_rows()[0])
    state = await env.harness.init_state(task)
    state = await env.harness.setup_state(task, state)

    tools = state.get_tools()
    assert sorted(tools) == ["pass", "play"]

    result = await tools["play"](card=1)
    assert result["your_hand"] == [2, 3, 5, 6, 10]
    assert "condottiere" in state
    state.strip_runtime_handles()
    state.assert_serializable()


def test_task_rows_do_not_override_harness_max_turns() -> None:
    env = load_environment("condottiere", num_examples=1, max_turns=17)
    row = env.taskset.eval_rows()[0]

    assert "max_turns" not in row
    assert env.harness.config.max_turns == 17


def test_tool_protocol_counts_assistant_tool_shape() -> None:
    counts = tool_protocol_counts(
        {
            "completion": [
                {"role": "assistant", "tool_calls": [{"name": "play"}]},
                {"role": "assistant", "tool_calls": []},
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "play"}, {"name": "pass"}],
                },
                {"role": "tool", "tool_calls": []},
            ]
        }
    )

    assert counts == {"no_tool_turns": 1, "multi_tool_turns": 1}
