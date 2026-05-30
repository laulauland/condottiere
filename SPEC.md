# Condottiere Budgeting Environment — Spec v0

A verifiers/Prime Intellect environment that measures an LLM's **resource-allocation discipline** and **disengagement** (cheap folding of losing battles) under a fixed, irreversible budget. Built as a controlled, oracle-backed proxy for agentic budgeting failures (token/tool/step exhaustion, sunk-cost thrashing) that real coding-agent traces exhibit but can't measure cleanly.

---

## 1. What this measures (and why a game)

Two model dispositions, isolated from task difficulty:

1. **Allocation** — does spend track value? (spend more on regions worth more)
2. **Disengagement** — does it fold losing battles cheaply, or thrash (sunk-cost)?

The game exists for one reason real environments can't give: **a computable oracle**. Because the opponent is a fixed known policy, "best possible play" is well-defined, so we can score *regret against optimal on the same seed* — removing task difficulty as a confound. This is the property SWE-bench-style traces lack (long trajectory = hard task OR bad model OR flaky harness, unattributable).

**This is an instrument, not a game.** Difficulty comes from the budget constraint, not from a clever adversary.

---

## 2. Scope & non-goals

**In scope (v0):**
- Single model seat vs. **scripted** opponent.
- Mercenaries-only deck (no special cards in v0 — see §11).
- **Fixed hand for the whole game** (no replenishment). One budget across all battles. This deliberately diverges from official Condottiere; we use the all-pay mechanic, not the full ruleset.
- Per-ply tool interface; minimal resolved-state returns.

**Explicit non-goals — do not build these into v0, they break the eval:**
- ❌ **Self-play / adaptive opponent.** Adaptivity destroys the oracle (optimal becomes a moving target). The opponent may be *reactive* and *varied* but must never *learn from the policy under test*. This is the load-bearing constraint.
- ❌ **Multiplayer (3–6 seat).** Kills oracle tractability; reintroduces noisy win-rate.
- ❌ **Negotiation / bluffing / table-talk.** Not verifiable; not the target skill.
- ❌ **`get_state` as a separate action.** Adds a second resource (info-gathering) that muddies card accounting. State is returned by actions instead (§5).
- ❌ Faithful Condottiere. We're not shipping the boardgame.

---

## 3. Game model (v0)

- **Hand (budget):** N mercenary cards with integer strengths, dealt once at game start, fixed for the whole game. Default N=6, e.g. `[1,2,3,5,6,10]` (seeded).
- **Regions:** ordered list, each with an integer value. Default 3 regions, values seeded. The order is revealed one battle at a time → preserves **future-demand uncertainty** (model doesn't know upcoming region values until it reaches them, unless we choose to reveal — see §11 open Q).
- **Battle:** players alternate plies. A ply = play one card (adds strength to your line) or pass. A player who passes is out of *this battle*. Battle ends when both have passed (or one passes after the other). Higher total line strength captures the region (+value). **All played cards are discarded regardless of outcome** (all-pay). Tie → nobody captures.
- **Game end:** all regions resolved OR model hand empty (remaining regions auto-pass).
- **Determinism:** `(seed, opponent_policy, action_sequence) → identical trajectory`. Required for paired rollouts (§8).

---

## 4. Architecture

Three layers, mapping to the verifiers spec (dataset + harness + rubric):

```text
engine.py        Pure Python. Game state machine + opponent policies + oracle.
                 No LLM. Fully unit/property tested in isolation.
harness          MultiTurnEnv subclass. Renders state→prompt, exposes tools,
                 steps engine, runs opponent inside the tool call.
rubric           neg_regret (trained, weight>0) + diagnostics (monitor, weight 0).
condottiere.py   load_environment(config) wiring it together.
```

Dependency/build order is bottom-up: correct engine → correct tools/parser → trustworthy metrics → meaningful reward. A bug in the engine's strength math silently corrupts every downstream number.

---

## 5. Tool interface

Two tools only. State is **returned by actions**, never fetched separately.

```python
play(card: int) -> ResolvedState
pass() -> ResolvedState
```

**Resolution granularity: one ply.** `play`/`pass` execute the model's action, then the opponent's single response, then return. The model sees each card exchange. (Battle-at-once resolution is rejected: we need per-card granularity to see *where* it overcommits — "played one card too many" is only visible if each card is an observable step.)

**`ResolvedState` (minimal — reveals only public info):**
```json
{
  "region": "Florence", "region_value": 4,
  "your_line": [6,5], "your_strength": 11,
  "opp_line": [5,6], "opp_strength": 11,
  "your_hand": [1,2,3,10],
  "battle_status": "ongoing|won|lost|tied",
  "battles_remaining": 1,
  "budget_signal": "<stated remaining budget — see §8>"
}
```

Do **not** leak: opponent hand, future region values (unless reveal-mode), oracle hints. Over-informative returns convert the eval back into a solved knapsack and destroy future-demand uncertainty.

**Determinism inside the tool:** the opponent's response (and any stochastic-opponent randomness) is realized *inside* the tool call. Seed it **from game state, not wall-clock** — otherwise paired rollouts diverge for the wrong reason.

**Illegal actions** (`play(7)` when 7 not in hand): return an error-state, never crash. Increment `illegal_action_count`. This is a **guardrail metric**: a model with high illegal-action count has tool-use trouble, and its budgeting metrics are suspect (tool competence confound).

---

## 6. Opponent policies

Scripted = a **fixed, known, stationary** function `state → action`. May be reactive and seeded-stochastic; never adaptive to the policy under test.

Ship a **suite of archetypes**, report metrics **per archetype** (richer than a pooled number; reveals whether budgeting is robust or play-style-specific):

| Policy | Behavior |
|---|---|
| `Aggressive` | Escalates hard on any contested region. |
| `Conservative` | Only contests high-value regions; folds early elsewhere. |
| `Hoarder` | Saves big cards for late regions. |
| `MatchFold` | Matches current leader if affordable & region worth it, else folds. |

**Seeded-stochastic allowed:** a policy may randomize (e.g. fold w.p. p). The oracle then maximizes *expected* value over the opponent's known decision tree. This closes the "model memorizes the one script" loophole — which is the legitimate worry behind any "shouldn't it adapt?" instinct — *without* introducing adaptivity.

---

## 7. Oracle

`oracle_value(seed, opponent_policy)` = total region-value captured by an optimal allocator against that fixed policy on that seed.

- **Tractable only because the opponent is fixed.** Restated because it's the whole design.
- **Computation:** v0 small instances (N≤6 cards, ≤3 regions) → exhaustive/expectimax search over the model's play tree against the (possibly stochastic) opponent tree. Memoize on game state.
- **Validation of the oracle itself** (do NOT property-test it against the opponent circularly):
  - Brute-force/exhaustive ground truth on tiny instances; assert fast oracle matches.
  - Legitimate property: `oracle_value ≥ value of any randomly sampled legal allocation` (optimality dominates random play; not circular because random ≠ the opponent).

---

## 8. Reward & metrics

**Separate the trained scalar from observed diagnostics.** Train on a clean gameable-to-the-limit signal; observe dispositions you never optimize (so they stay meaningful).

### Trained reward (weight > 0)

```python
normalized_regret = (oracle_value - captured_value) / oracle_value   # 0=optimal, 1=nothing
reward = -normalized_regret                                          # in [-1, 0]
```

Regret over win-rate: graded (gradient everywhere) and difficulty-normalized (removes seed hardness).

### Diagnostic metrics (monitor rubric, weight = 0, logged not trained)

| Metric | Definition | Reads |
|---|---|---|
| **Overcommitment** | cards spent on non-captured battles / total cards spent | sunk-cost / poor disengagement (high = bad) |
| **Spend–value corr** | corr(strength committed to region, region value) | rational allocation (≤0 = blind) |
| **Early-dump index** | fraction of budget spent in first third of battles (or 1 − temporal evenness) | front-loading / starve-the-endgame |
| **Budget responsiveness** | Δspend / Δstated-budget, **paired rollouts** | does it *use* budget info or ignore it (headline) |
| **Illegal-action count** | n rejected tool calls | guardrail — invalidates above if high |

**Budget responsiveness needs paired rollouts:** same seed, same engine state, prompt differs only in stated `battles_remaining` / budget signal (e.g. "4 left" vs "12 left"). Measure spend delta. Defined on *pairs* → compute in a **post-hoc analysis script** over logged trajectories, not as a per-rollout reward. This is the controlled version of the documented "budget-in-prompt but model self-regulates poorly" failure (BATS-style).

**Why kept separate:** a model can hit low regret while still front-loading (lucky seed ordering). Non-trained diagnostics catch "good score, bad habits" — the actual diagnostic value.

---

## 9. Verifiers integration

Pin the verifiers version — API moves fast (0.1.15.dev as of May 2026). Read `docs/environments.md` and `docs/byo-harness.md` before writing; method signatures shift across dev releases.

**Subclass `MultiTurnEnv` directly.** Everything in verifiers is built on it; for a custom game protocol (not plain tool-calling QA) you subclass it and override two methods:

- `env_response(messages, state) -> (new_messages, state)` — the engine step: apply the model's action, run the opponent inside it, return resolved state as the next message.
- `is_completed(messages, state) -> bool` — game over. **Always call `super().is_completed()` first** so `max_turns` and shared guards are respected, then return your own end condition.

**Tools are plain Python functions.** Verifiers extracts the tool schema from the function signature + docstring for OpenAI-compatible tool calling — so the docstring *is* part of your prompt surface; write `play`/`pass` carefully. State is **returned by the tool**, never fetched (no `get_state`).

**Lifecycle hooks** (the "good environment" details):
- `@vf.cleanup` — runs after each rollout → reset & re-seed the engine per game.
- `@vf.stop` — custom stop condition checked after each turn → "hand empty" early termination.
- `@vf.teardown` — once at shutdown.

**Free guards from MultiTurnEnv:** built-in stop conditions for errors, prompt-length, `max_turns`, `max_total_completion_tokens`; per-rollout wall-clock timeout via `--timeout` at eval time. Set `max_turns` well above your longest possible game (illegal-action retries extend it).

**Monitor rubrics stack automatically** (RubricGroup) — weight-0 diagnostics ride alongside the trained reward with no extra wiring.

**Dataset:** a HuggingFace Dataset with a `prompt` column; put seeds + opponent assignments in the `info` column (`answer` optional/omittable — we score on completion + state, not a target string).

```python
# environments/condottiere/condottiere.py
import verifiers as vf
from .engine import CondottiereGame, OPPONENTS, oracle_value

def condottiere_source():
    # yields tasks: prompt + info={seed, opponent_policy, budget_signal override}
    ...

@vf.reward(weight=1.0)
async def neg_regret(task, state) -> float:
    opt = oracle_value(task["seed"], task["opponent_policy"])
    return -(opt - state["game"].captured_value) / opt

@vf.reward(weight=0.0)  # monitor only — logged, never trained
async def overcommitment(task, state) -> float:
    g = state["game"]
    return g.cards_on_losses / max(g.cards_total, 1)

# ... spend_value_corr, early_dump, illegal_actions (all weight=0)

def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.Taskset(
            source=condottiere_source,
            rewards=[neg_regret, overcommitment, early_dump,
                     spend_value_corr, illegal_actions],
            config=config.taskset,
        ),
        harness=CondottiereHarness(),  # MultiTurnEnv subclass; play/pass tools
    )
```

Workflow:

```sh
prime env init condottiere --with-harness
prime env install condottiere
prime eval run condottiere -m openai/gpt-5-nano   # local; auto-uploads results
prime eval tui                                    # inspect trajectories
prime env push --path ./environments/condottiere
```

---

## 10. Testing

| Layer | Method |
|---|---|
| **Engine invariants** | Hypothesis stateful (`RuleBasedStateMachine`): generate random legal play sequences, assert invariants throughout |
| **Oracle** | Exhaustive ground-truth on tiny instances; `oracle ≥ random` property |
| **Metrics** | Unit tests on hand-built trajectories (e.g. "dump-everything-turn-1" → overcommitment high) |
| **Renderer/parser** | Round-trip property: `parse(render(state))` preserves legal-move set; parser rejects all malformed actions without crashing |

**Engine invariants to assert:**
- Card conservation: every card in exactly one place; total count constant.
- Resolution symmetry: outcome invariant to scoring order.
- Termination: every game ends in bounded plies.
- Determinism: `(seed, actions) → identical state`.
- (v1, special cards) interaction algebra: Winter idempotent; Drummer doubles *before* Spring's +3; Heroine/Courtesan immune to Winter/Spring/Drummer; Bishop removes exactly current-max mercenaries incl. own.

Parser/renderer bugs masquerade as model failures — a misread legal play inflates "overcommitment" for non-model reasons. Test this as hard as the engine.

---

## 11. Open questions / staged extensions

- **Region-value reveal:** reveal all upfront (pure allocation, weaker future-demand uncertainty) vs. one-at-a-time (preserves uncertainty, the game's edge over a knapsack). **Lean one-at-a-time**; consider a reveal-mode flag to A/B the effect.
- **Hand replenishment:** v0 fixed hand (strongest budget test). Variant: replenish between regions (models "budget refreshes per task") — different skill, separate config.
- **Special cards (v1):** add Winter/Spring/Drummer/Bishop for non-linear interaction richness. Cost: oracle combinatorics grow; parser/metric complexity up. Stage after v0 validated.
- **Info-gathering variant (separate env):** a `peek` action with real cost → tests *when to scout vs act*, arguably closer to the coding-agent file-read decision than card-spend. Do NOT fold into the budgeting eval; build as a sibling.

---

## 12. Validation plan (the result that justifies the whole thing)

Convergent validity, not just a leaderboard:

1. Define the diagnostics on the fixed-opponent game.
2. Run 3–4 models you *also* have SWE-bench/agent traces for.
3. Correlate game `overcommitment` against SWE-eval-style loop/thrash metrics on the same models.

If they correlate → the game predicts real coding behavior and the detour through a card game is earned. If null after a week → the game isn't measuring what you hoped; cut losses before investing a quarter. **De-risk in this order.**

Positioning: not "better than code telemetry" (loses that argument) but **the difficulty-controlled complement** — domain telemetry is the high-validity primary signal; the game answers what telemetry can't: *is poor budgeting a stable, difficulty-independent trait, and does it generalize across domains?*

---

## 13. Build / test / rollout plan

**Key economic fact:** this is an *eval*, not training. Cost = inference tokens, and the environment is text-light (~5–8K tokens/game, input-heavy from per-turn state re-render). No GPUs needed for evaluation. GPU pods ($1.50–3.50/hr) are for training only — deferred (see below). A full multi-model sweep is single-digit to low-tens of dollars at budget-API rates; tens at flagship rates.

### Local vs. hosted

Hosted Evaluations (part of Prime **Lab**, launched Feb 2026) run the env remotely and keep logs/trajectories on-platform — flip with `--hosted` on the same config. No billing setup required until you launch hosted jobs. Map this to phases:

- **Local** (`prime eval run`, no flag) for Phases 0–2 — fast iteration loop, results still auto-upload. First run ever = a 1-example eval to confirm account + endpoint + env + upload path.
- **Hosted** (`--hosted`) for Phase 3 sweep — fire and walk away; pull results with `prime eval samples <id>`. Same per-token cost; the win is not babysitting a long multi-cell run and having trajectories on-platform for validation.

The eval is fundamentally an **ablation study** (swap model / opponent archetype / stated-budget). Use the config's `[[ablation]]` model/endpoint overrides — define cells once instead of scripting N runs. Read the "Verifiers Evaluation" config schema for precedence rules before hand-rolling a sweep.

```toml
# configs/eval/condottiere-hosted.toml
model = "openai/gpt-4.1-mini"
num_examples = 50
rollouts_per_example = 1

[[eval]]
env_id = "yourname/condottiere"
env_args = { opponent = "MatchFold" }

# [[ablation]] cells: swap model / opponent / budget-signal here
```

```sh
prime eval run configs/eval/condottiere-hosted.toml --hosted
prime eval list ; prime eval get <id> ; prime eval samples <id>
prime eval models     # check live capacity + token pricing before a sweep
```

### Phases (de-risk cheapest-first)

| Phase | What | Where | Cost |
|---|---|---|---|
| 0 | Engine + oracle + parser, fully tested (§10) | local CPU | $0 |
| 1 | Plumbing eval, cheapest model, ~10 games — tools parse, state renders, game terminates, metrics populate | local | <$1 |
| 2 | **Pilot budget-responsiveness on 2 deliberately-different models** *before* full build-out | local | a few $ |
| 3 | Full sweep: seeds × opponents × 4–5 models | hosted | $10–50 |
| 4 | Convergent-validity correlation vs. SWE traces (§12) | local analysis | $0 API; your time |
| — | RL training against the env (prime-rl) | GPU cluster | **deferred — don't** |

**Phase 2 is placed first-among-risky deliberately.** Budget-responsiveness is the headline metric *and* the biggest risk: if models ignore a stated budget in the prompt (plausible), it reads ~0 for everyone and discriminates nothing. Pilot it for coffee money before investing in the full harness. Flat across obviously-different models ⇒ the metric (and part of the spec) needs rework — learn that early.

**Training is deferred on purpose.** The goal is an internal eval/diagnostic; the full diagnostic result comes from inference-only evaluation. `neg_regret` exists to keep the env *train-ready* for later, not to require training now. RL is a different budget universe (GPU-hours) — only enter it if a specific result demands it.

### Hosted + paired-rollout caution

Budget-responsiveness needs paired rollouts with a fixed seed (same engine state, only the stated-budget string differs). Hosted execution gives less control over pairing and seed determinism. **Before** committing this metric to a hosted sweep, confirm on a *local* run: (1) state-seeded RNG produces identical opponent play across a pair when only the budget string changes; (2) `prime eval samples` returns enough metadata to match the pair. If pairing is awkward through hosted, compute responsiveness from a local paired run and use hosted only for the unpaired metrics. Don't discover this after spending the sweep.

### Engineering-quality checklist (what separates a good env from a running one)

- `engine.py` is standalone and importable with zero verifiers dependency — testable in isolation, reusable for the oracle and offline analysis.
- Per-rollout reset via `@vf.cleanup`; never leak state between games.
- `max_turns` set above worst-case game length incl. illegal-action retries.
- `illegal_action_count` surfaced as a guardrail metric; document that high values invalidate the budgeting metrics for that model.
- Pin verifiers version in `pyproject.toml`; note it in the README.
- README documents: opponent archetypes, region/hand seeding, the budget-signal format, and which metrics are trained vs. monitor.
- Tool docstrings reviewed as prompt text, not just code.
