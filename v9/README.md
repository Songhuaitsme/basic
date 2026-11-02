# V9 candidate search

## E0 baseline audit

V9 did not exist at the start of this change. E0 is the untouched V8 source in
`v8/`, selected by `create_v9_runtime(search_stage=0)`. The work is on Git branch
`v9-candidate-search`. The V8 source files were originally untracked in the
working tree and were saved unchanged in E0 commit `d72f242`. Baseline SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `v8/scheduler/candidate_generator.py` | `461A4712E1C94C2EDB42FBEE58C33FC9CAA1E4CFD361467D138F39ACA7BBB74C` |
| `v8/v8_runtime.py` | `6637EAEFE3234EAE2A1D5A922FDC1BF83A9CA8F8DA8026AC0EF499556F9D20D1` |
| `v8/_entrypoint.py` | `7E1D28E09F02BDF98507122ACBFAA889E6A61B245BD352D648D89BF6BC3FC7C0` |

The runtime entry is `v8._entrypoint.use_v8_runtime` / `create_v8_runtime`.
The installed class is `V8CandidateGenerator`, inheriting V7, V5, V3 and V1.
V8 statically rejects nodes whose total CPU capacity is too small. V7 caches
static paths and durations; retry time bounds are recomputed. V5 caches records
and materialized metrics within one decision. V6 supplies exact batched energy
metrics. V1 handles the layered pool, full enumeration and hard feasibility.
K is configured as 1 in `shared/config.py` and has not changed.

The baseline Anchor samples at most four ticks per path and stops after the
first feasible tick. If all anchors fail it returns an empty pool. It then
round-robin ranks nodes by earliest completion, cost, green score, capacity
margin and hash, and samples more ticks only on chosen nodes. The original
`_adaptive_ticks` places early ticks before middle-window coverage. CPU, path
bandwidth and SLA checks occur in V1 `_feasible_item`; exact energy accounting
comes afterward. The `complete` mode enumerates the fixed path/grid space.

## Stages

| Stage | Change |
| --- | --- |
| E0 | Original V8 generator and runtime |
| E1 | Grid-aligned endpoints/middle/SLA anchors; resource-boundary rescue; multiple feasible times per node |
| E2 | Bounded exact-energy previews at forecast boundaries, including execution duration and existing green consumption |
| E3 | Equal total time-probe ceiling with deterministic priority for unresolved nodes, unexplored slots, feasible rate, energy and wait |

E1–E3 share the unchanged candidate features, DQN, reward, Wait Gate, resource
commit and accounting formulas. Energy only orders probes. Every emitted
candidate still passes the inherited CPU, bandwidth, forecast and SLA checks.
The earliest candidate offered to Wait Gate is the earliest **among explored
feasible slots**. It is not claimed to be globally earliest. A bounded search
with no result is reported as an unresolved empty search when untested slots
remain; it is not proof of physical infeasibility. The existing stream status
enum has no separate unresolved value, so `last_search_stats` carries this
distinction. `complete` still uses the inherited fixed-path enumeration.

`last_search_stats` and `path_search_statistics()["v9_search"]["totals"]`
record time probes, actual CPU and bandwidth check calls, energy preview calls,
exact energy evaluations, budgets, unresolved nodes and generation wall time.
Preview results are per-decision only. The inherited caches continue to handle
task retries and topology changes; per-decision previews cannot outlive a
resource snapshot, forecast change or decision boundary.

## Running evaluations later

Use the same frozen checkpoint and task trace in each command, changing only
`--search-stage 0`, `1`, `2`, `3`. For example:

```powershell
py -m v9.evaluate_v9 --search-stage 0 --policy candidate_dqn --model-path <checkpoint> --task-trace-input <trace.json> --seed 42
py -m v9.evaluate_v9 --search-stage 1 --policy candidate_dqn --model-path <checkpoint> --task-trace-input <trace.json> --seed 42
py -m v9.evaluate_v9 --search-stage 2 --policy candidate_dqn --model-path <checkpoint> --task-trace-input <trace.json> --seed 42
py -m v9.evaluate_v9 --search-stage 3 --policy candidate_dqn --model-path <checkpoint> --task-trace-input <trace.json> --seed 42
py -m v9.compare_reports artifacts/v9/e0/report.json artifacts/v9/e1/report.json artifacts/v9/e2/report.json artifacts/v9/e3/report.json
```

For small fixed snapshots, `v9.offline_ablation.compare_case` compares E0–E3
with a `complete` reference. Keep tasks, snapshots, topology, forecast versions
and random seeds fixed. Use equal `pool_time_samples_by_sla` for an equal
time-probe-ceiling comparison; E2/E3 also incur separate energy preview costs.
Do not attribute any later retraining gain to candidate search. Optional
retraining is available through `v9.train_v9` after frozen-model evaluation.

No offline candidate benchmark or full system evaluation has been run in this
change. Their quality, cost and scheduling results remain to be measured.
