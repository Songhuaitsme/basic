# V8: static hard-constraint admission

V8 builds on the unchanged V7 path cache and candidate policy. Before a path
query, it excludes a target only when configured CPU capacity is insufficient
by more than `max(time_tolerance, 1e-12)`. After the inherited transmission
calculation, it skips grids whose existing tick bounds have `first > last`.
Dynamic CPU and bandwidth feasibility, Anchor selection, node ranking, time
sampling, Pareto filtering, Candidate DQN, waiting gate, and reservation remain
on the inherited path.

`theoretical_slot_count` and evaluation `theoretical_time_slots` in V8 mean
**post-pruning search slots**. V7's values mean **pre-pruning theoretical
slots**. Both are reported separately by the V7/V8 comparison script.

## Run

```powershell
py -m v8.evaluate_v8 --policy candidate_dqn --model-path artifacts/v6/logs/candidate_dqn.pt --device cpu --seed 42 --warmup-days 0 --arrival-cutoff 24 --output artifacts/v8/evaluation/seed42.json
py -m v8.train_v8 --steps 600000 --seed 42
py -m unittest v8.tests.test_candidate_generator v8.tests.test_runtime v7.tests.test_path_cache -q
```

For paired timing, use the same model, seed, warm-up, and arrival cutoff for
`v7.evaluate_v7` and `v8.evaluate_v8`, with detailed profiling disabled.
Then run `py -m v8.compare_v7_v8 V7.json V8.json` (repeat path pairs as
needed). The comparison verifies identical input traces, exact metrics, and
non-timing decision fields, while reporting candidate generation times and
static-pruning counters. Do not infer a speedup from a single run.
