# Condottiere Budgeting Environment

A Prime Intellect / Verifiers environment that measures how well a language model
allocates a fixed, irreversible budget — and whether it folds losing fights cheaply
instead of throwing good cards after bad.

The model plays a small, deterministic card game against a scripted opponent. The
opponent is a fixed, stationary policy that is **known to the scorer but hidden from the
model** — the model is never told which archetype it faces and must infer behavior from
play. Because the scorer knows the opponent, the *best possible play on a given seed* is
computable, which lets the environment score the model against an exact oracle and report
**regret against optimal**, with task difficulty removed as a confound, so a "good score"
and "good habits" are separable and measurable.

The oracle is an exhaustive expectimax over the model's play tree against that fixed
scripted opponent — i.e. **best-response-to-script**, not a game-theoretic equilibrium.
One consequence to keep in mind when reading the numbers: because the opponent is hidden
from the model, a small part of the measured regret is opponent-identification cost, not
pure budgeting. The metric is best read as *allocation quality against an unknown
stationary opponent*.

The game is a controlled instrument, not faithful Condottiere. It borrows the all-pay
mechanic and nothing else: one hand spent across a sequence of seeded battles, no
replenishment, a fixed opponent. Difficulty comes from the budget constraint, not from a
clever adversary.

- **Hub ID:** `laulauland/condottiere` (private)
- **Version:** `0.0.12`
- **Verifiers:** pinned to `0.1.14`
- **Dashboard:** https://app.primeintellect.ai/dashboard/environments/laulauland/condottiere
- **Install:** `prime env install laulauland/condottiere`

## How the game works

- **Hand (the budget).** Six mercenary cards with integer strengths, default
  `[1, 2, 3, 5, 6, 10]`, dealt once and fixed for the entire game. There is no
  replenishment — every card spent is gone.
- **Regions.** Three regions, each with a seeded integer value (sampled from 2–8),
  revealed one battle at a time so the model never knows upcoming values in advance.
- **A battle.** Players alternate plies; a ply is either *play one card* (adds its
  strength to your line) or *pass* (you are out of this battle). The battle ends when
  both players have passed. Higher total line strength captures the region and its value.
  A tie captures nothing. **All played cards are discarded regardless of outcome** — this
  is the all-pay mechanic that makes overcommitment costly.
- **Game end.** All regions resolved, or the model's hand is empty (remaining regions
  auto-resolve as ties).
- **Determinism.** `(seed, opponent_policy, action_sequence)` always produces an
  identical trajectory. The opponent's randomness, where it exists, is seeded from game
  state rather than wall-clock, so paired rollouts stay comparable.

The model acts through exactly two tools — `play(card)` and `pass()` — and the resolved
public state is returned by the tool itself; there is no separate `get_state` action.
Each call applies the model's ply, runs the opponent's single response, and returns what
is publicly visible: the current region and its value, both lines and strengths, the
model's remaining hand, battle status, battles remaining, and a stated budget signal.
Opponent hands, future region values, and oracle hints are never leaked — exposing them
would turn the eval into a solved knapsack problem.

### Opponent archetypes

The opponent is always a fixed, stationary `state -> action` function. It may be reactive,
but it never adapts to the policy under test — adaptivity would make "optimal" a moving
target and destroy the oracle. The archetype is **never revealed to the model**: it is an
input to the scorer's oracle, not to the prompt, so the model has to read opponent
behavior off the unfolding game. Metrics are reported per archetype rather than pooled, so
you can see whether budgeting discipline is robust or play-style-specific.

| Policy | Behavior |
|---|---|
| `Aggressive` | Escalates hard on any contested region. |
| `Conservative` | Contests only high-value regions; folds early elsewhere. |
| `Hoarder` | Saves big cards for late regions. |
| `MatchFold` | Matches the current leader when affordable and the region is worth it, else folds. |

## What it measures

The trained scalar is kept separate from the observed diagnostics. The model is scored on
one clean, gameable-to-the-limit reward; the dispositions you actually care about are
logged but never optimized, so they stay meaningful.

**Trained reward (weight 1.0):**

- `neg_regret` = `-(oracle_value - captured_value) / oracle_value`, in `[-1, 0]`.
  Zero is optimal play; −1 is capturing nothing. Regret is graded everywhere and
  normalized for seed difficulty, which win-rate is not.

**Monitor diagnostics (logged, weight 0):**

- `overcommitment` — cards spent on non-captured battles over total cards spent. High
  means sunk-cost thrashing / poor disengagement.
- `spend_value_corr` — correlation between strength committed to a region and that
  region's value. ≤0 means allocation is blind to value.
- `early_dump` — fraction of the budget spent in the first third of battles. High means
  front-loading and starving the endgame.
- `illegal_actions` — count of rejected tool calls. This is a guardrail: a high count
  means tool-use trouble, and it invalidates the budgeting metrics for that model.
- `no_tool_turns` / `multi_tool_turns` — turns with zero or more than one tool call,
  tracking adherence to the one-call-per-turn protocol.

**Post-hoc paired diagnostic (experimental — not in the current sweep):**

- `budget_responsiveness` — does the model actually *use* a stated budget signal, or ignore
  it? Measured as spend delta across paired rollouts that share a seed and engine state and
  differ only in the stated budget string (e.g. "4 battles left" vs "12 battles left"). It
  is defined on pairs, so it is computed by an offline analysis script, not as a per-rollout
  reward. **This is not exercised by the hosted sweep** (`budget_signal` defaults to off),
  and it currently keys on card count rather than committed strength; treat it as a
  prototype, not a validated metric.

## Repository layout

```text
SPEC.md                                Original v0 design spec and rationale.
configs/eval/condottiere-local.toml    Local smoke config (1 example).
configs/eval/condottiere-hosted.toml   Hosted sweep config (ablation grid).
environments/condottiere/
  engine.py                            Standalone game engine, opponents, oracle. No verifiers dependency.
  condottiere.py                       Verifiers wiring: taskset, harness, tools, reward, metrics.
  analysis.py                          Paired budget-responsiveness analysis over logged results.
  tests/                               Engine, environment, and analysis tests.
```

The engine is intentionally importable with zero verifiers dependency, so the oracle,
metrics, and offline analysis all share exactly the same game semantics as the live
harness. A bug in the engine's strength math would silently corrupt every downstream
number, so it is tested in isolation with Hypothesis stateful tests (card conservation,
resolution symmetry, bounded termination, determinism) and the oracle is checked against
exhaustive ground truth on tiny instances plus the `oracle >= random play` property. The
full suite is `18 passed`.

## Run locally

```sh
# Engine, environment, analysis, and validation tests.
uv run --package condottiere --group dev pytest environments/condottiere/tests -q

# Install the environment, then run the local smoke eval.
prime --plain env install condottiere --no-upgrade
prime --plain eval run configs/eval/condottiere-local.toml -s -d --timeout 120
```

The local config runs one seeded game against `MatchFold` with `gpt-5-nano` and confirms
the full path: tools parse, state renders, the game terminates on `hand_empty_or_regions_done`,
and the reward plus all diagnostics populate.

## Hosted sweep

The hosted config is an ablation grid: 50 seeds × 4 opponents × 8 models = 1,600 rollouts,
one rollout per seed, defined once via `[[ablation]]` cells rather than scripted as
separate runs. The sweep below ran on env version `0.0.12`.

```sh
prime --plain eval run configs/eval/condottiere-hosted.toml --hosted
```

- **Models:** `gemini-3.1-pro-preview`, `gpt-5.5`, `claude-opus-4.7`, `kimi-k2.6`,
  `deepseek-v4-pro`, `intellect-3`, `gpt-oss-120b`, `qwen3.6-35b-a3b`
- **Opponents:** `Aggressive`, `Conservative`, `Hoarder`, `MatchFold`

These are **preliminary results** — the leaderboard discriminates broadly but is
underpowered for fine ranking (see significance grouping below), and `kimi-k2.6` is still
running.

Reward is mean negative regret per (model, opponent) cell, n=50; closer to 0 is better.
Per-cell 95% CI is about ±0.08 (observed reward sd ≈ 0.29). Rows ordered best-to-worst by
mean reward.

| Model | Aggressive | Conservative | Hoarder | MatchFold |
|---|---:|---:|---:|---:|
| `gemini-3.1-pro-preview` | -0.295 | -0.291 | -0.475 | -0.331 |
| `gpt-5.5` | -0.413 | -0.356 | -0.505 | -0.306 |
| `claude-opus-4.7` | -0.447 | -0.403 | -0.472 | -0.326 |
| `kimi-k2.6` | — | — | — | — |
| `deepseek-v4-pro` | -0.533 | -0.534 | -0.505 | -0.399 |
| `intellect-3` | -0.508 | -0.574 | -0.390 | -0.351 |
| `gpt-oss-120b` | -0.730 | -0.555 | -0.453 | -0.459 |
| `qwen3.6-35b-a3b` | -0.867 | -0.697 | -0.475 | -0.745 |

`kimi-k2.6` is still running; its row will be filled when the cells complete.

Behavioral diagnostics pooled across all four opponents (n=200 per model), with the pooled
mean-reward 95% CI from a paired bootstrap over shared seeds.

**A note on units, since they differ across columns.** `overcommitment` and `% cards lost`
count *cards* (committed to regions the model did not win, over all cards committed);
`spend-value corr` and `early dump` are over committed *strength*. The headline "most spend
goes into losses" is reported both ways — `% cards lost` and `% strength lost` — and holds
under either unit (52–72%), so the finding is not an artifact of the counting choice.

| Model | Mean reward | 95% CI | Captured value | Overcommitment (cards) | Spend-value corr (strength) | Early dump (strength) | % cards lost | % strength lost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gemini-3.1-pro-preview` | -0.348 | [-0.389, -0.306] | 5.8 | 0.66 | 0.42 | 0.38 | 68% | 61% |
| `gpt-5.5` | -0.395 | [-0.436, -0.352] | 5.3 | 0.59 | 0.38 | 0.28 | 60% | 52% |
| `claude-opus-4.7` | -0.412 | [-0.457, -0.368] | 5.2 | 0.62 | 0.42 | 0.39 | 63% | 61% |
| `kimi-k2.6` | — | — | — | — | — | — | — | — |
| `intellect-3` | -0.456 | [-0.493, -0.417] | 4.7 | 0.61 | 0.18 | 0.53 | 62% | 66% |
| `deepseek-v4-pro` | -0.493 | [-0.534, -0.447] | 4.6 | 0.62 | 0.19 | 0.43 | 62% | 62% |
| `gpt-oss-120b` | -0.549 | [-0.594, -0.505] | 4.0 | 0.63 | 0.19 | 0.52 | 65% | 64% |
| `qwen3.6-35b-a3b` | -0.694 | [-0.743, -0.647] | 2.8 | 0.70 | 0.15 | 0.76 | 72% | 71% |

**Significance grouping (paired bootstrap, shared seeds, 95%).** The board separates into
broad tiers, not a clean total order:

- **Top tier — `gemini-3.1-pro-preview`, `gpt-5.5`, `claude-opus-4.7`** are statistically
  tangled: gemini beats opus only marginally, and adjacent pairs (gemini vs gpt-5.5,
  gpt-5.5 vs opus) are **not** distinguishable. Do not read the within-tier order as real.
- **Middle — `intellect-3`, `deepseek-v4-pro`** are not separable from each other, and the
  top/middle border is soft (opus vs intellect-3 overlaps).
- **Bottom — `gpt-oss-120b`, then `qwen3.6-35b-a3b`** are clearly worse than the top tier;
  qwen is clearly worst overall.

The large gaps (top tier vs `gpt-oss`/`qwen`) are significant; the small adjacent gaps are
not. Treat the leaderboard as tiers until n grows.

The hosted config pins `max_concurrent = 4`; higher concurrency caused worker heartbeat
restarts on long reasoning trajectories.

## Paired budget-responsiveness

`budget_responsiveness` is computed offline from two saved eval runs that share seeds and
opponent but differ only in the stated budget signal. In the pilot (`gpt-5-nano` and
`gpt-4.1-mini` vs `MatchFold`, "4 battles left" vs "12 battles left"), aggregate
responsiveness was `0.375` — `gpt-5-nano` at `0.625` and `gpt-4.1-mini` at `0.125`.

```sh
uv run --package condottiere python environments/condottiere/analysis.py \
  path/to/low/results.jsonl \
  path/to/high/results.jsonl
```

## Design references

The full rationale — why a game, why a fixed opponent, the reward/diagnostic split, the
testing strategy, and the staged build plan — lives in [`SPEC.md`](SPEC.md). The
environment follows the Prime Lab shape (`prime lab setup`, `prime env init`,
`prime eval run`, `prime env push`); see the Prime Intellect docs for
[verifiers](https://docs.primeintellect.ai/verifiers/overview),
[environments](https://docs.primeintellect.ai/verifiers/environments), and the
[BYO Harness guide](https://docs.primeintellect.ai/verifiers/byo-harness).
