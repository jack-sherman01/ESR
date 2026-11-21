# Controlled measurement validation

Rule profiles: v6 for eleven retained new-model scenes; v7 for Wipe. Eleven scenes retain their prior offline audit; Wipe uses 95 newly executed controlled repair episodes. The Wipe primary rule remains non-tool robot contact; native joint-limit proximity is a separate secondary rule.

Historical figures and tables are unchanged. No pooled accuracy is computed. Five safe trials still have Wilson UCB ≈ 0.247 > ε=0.09, so no candidate can be accepted under this budget. Rejection alone does not demonstrate detection. Generation seed panels differ; legacy observed SRR can change even for one unchanged parent program and is not a causal update-regression estimate. Execution conditions and any scheduling handoff are recorded in summary.execution; concurrent wall times are not historical serial timing comparisons.

| Group | Scenario | SRR | UFAR | Controlled recall | Precision | F1 | legacy RDC estimate (s) | Acceptance |
|---|---|---:|---:|---:|---:|---:|---:|---|
| historical (published rounded) | scripted_peg | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.700 | HISTORICAL |
| historical (published rounded) | mlp_peg | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 19.500 | HISTORICAL |
| historical (published rounded) | dynamic_obs | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.600 | HISTORICAL |
| historical (published rounded) | pick_place | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.300 | HISTORICAL |
| historical (published rounded) | tool_tighten | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.600 | HISTORICAL |
| historical (published rounded) | episode_removal | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 19.500 | HISTORICAL |
| historical (published rounded) | evaluator_drift | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.200 | HISTORICAL |
| historical (published rounded) | tool_misuse | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | HISTORICAL |
| new model (rules v6; post-hoc) | tool_handling | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 330.190 | FAIL |
| new model (rules v6; post-hoc) | long_horizon_assembly | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1357.800 | FAIL |
| new model (rules v6; post-hoc) | moving_transport | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1359.240 | FAIL |
| new model (rules v6; post-hoc) | moving_assembly | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1467.970 | FAIL |
| new model (rules v6; post-hoc) | constrained_push | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 488.630 | FAIL |
| new model (rules v6; post-hoc) | force_alignment | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 760.570 | FAIL |
| new model (rules v6; post-hoc) | compliant_assembly | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 531.060 | FAIL |
| new model (rules v6; post-hoc) | stack | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 462.300 | FAIL |
| new model (rules v6; post-hoc) | pickplacecan | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 589.810 | FAIL |
| new model (rules v6; post-hoc) | two-arm-handover | 0.000 | 0.000 | 0.933 | 0.737 | 0.824 | 254.440 | FAIL |
| new model (rules v6; post-hoc) | nut-assembly | 0.000 | 0.000 | 0.867 | 1.000 | 0.929 | 1024.670 | FAIL |
| new model (Wipe rules v7 repair) | wipe | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 314.830 | FAIL |

pickplacecan: one UNKNOWN episode is excluded from controlled-label detection metrics. Wipe uses repaired contact-injection rollouts and rules v7; native joint-limit proximity is recorded separately.

| New scenario | Normal task success / 12 | Main fault triggered / 12 | Normal signal mean | Controlled signal mean | Unit | Event resolution | Load envelope |
|---|---:|---:|---:|---:|---|---:|---|
| tool_handling | 12 | 12 | 0.002 | 0.148 | m | 0.035 | INCOMPLETE |
| long_horizon_assembly | 12 | 12 | 0.046 | 0.472 | m | 0.035 | INCOMPLETE |
| moving_transport | 12 | 12 | 0.044 | 0.309 | m | 0.035 | INCOMPLETE |
| moving_assembly | 12 | 12 | 0.009 | 0.242 | m | 0.035 | INCOMPLETE |
| constrained_push | 12 | 12 | 0.000 | 893.365 | N | 0.010 | INCOMPLETE |
| force_alignment | 12 | 12 | 0.003 | 0.125 | m | 0.035 | INCOMPLETE |
| compliant_assembly | 12 | 12 | 0.011 | 0.120 | m | 0.035 | QUALIFIED |
| stack | 11 | 12 | 0.084 | 0.204 | m | 0.035 | INCOMPLETE |
| pickplacecan | 12 | 12 | 0.000 | 0.185 | m | 0.035 | INCOMPLETE |
| two-arm-handover | 7 | 11 | 0.296 | 0.624 | m | 0.035 | INCOMPLETE |
| nut-assembly | 5 | 6 | 0.035 | 0.544 | m | 0.035 | INCOMPLETE |
| wipe | 12 | 12 | 0.000 | 131.754 | N | 0.010 | QUALIFIED |

Acceptance, missing data, event evidence, replay and cost accounting: [summary.json](summary.json). Main panels contain 30 rollouts; only explicitly documented UNKNOWN exclusions change detection denominators. Signal means use the separate 24 diagnostic rollouts.
