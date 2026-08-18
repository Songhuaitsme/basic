# Report notes

- Decision: determine whether the trained Candidate-DQN is ready to replace or advance beyond the equal-weight baseline.
- Scope: five frozen policies evaluated on seed 42 with identical task and exogenous trace hashes; 2,416 arrivals and 2,401 completions.
- Comparison baseline: equal-weight. Deltas are policy minus equal-weight.
- Confidence limit: one evaluation seed is diagnostic only. The repository quality gate requires at least 10 paired seeds.
- Chart map:
  - `cost_per_cpu`: horizontal bar of cost per completed CPU-hour by policy; category comparison; blue single-root palette.
  - `green_coverage`: horizontal bar of completed-task green coverage by policy; category comparison; olive single-root palette.
- Omitted richer uncertainty visual: seed 42 supplies no cross-seed distribution or confidence interval. Formal paired runs are required first.
- Validation: all five reports have `status=VALID`, equal seed, equal task trace hash, equal exogenous trace hash, 2,416 arrivals, 2,401 completions, and zero unsettled tasks.
