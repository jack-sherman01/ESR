# Changelog

All notable changes to ESRB are documented here.

---

## [0.1.0] — 2026-11-09

Initial release of the **Evolutionary Safety Regression Benchmark**.

### Added

#### Framework
- `BenchmarkInstance` — composable ⟨Task, Operator, Hazard, Oracle, Protocol⟩ tuple
- `EvolutionLoop` — CE-EST agentic improvement loop with UCB safety gate
- `LineageTracker` — immutable SHA-256 generation records
- `RiskSensitiveSelector` — UCB-based rejection gate (`ε`, `α` configurable)
- `CounterexampleLibrary` — append-only, hashed counterexample store
- `ChangeImpactGraph` — NetworkX-based change-impact analysis

#### Nine hazard scenarios

| # | Name | Hazard type | MuJoCo |
|---|------|-------------|--------|
| 1 | `scripted_peg` | REWARD_HACKING | Yes |
| 2 | `mlp_peg` | MEMORY_CONTAMINATION | Yes |
| 3 | `dynamic_obs` | DISTRIBUTION_SHIFT | Yes |
| 4 | `pick_place` | DISTRIBUTION_SHIFT | Yes |
| 5 | `tool_tighten` | REWARD_HACKING | Yes |
| 6 | `episode_removal` | SAFETY_FORGETTING | Yes |
| 7 | `evaluator_drift` | EVALUATOR_DRIFT | Yes |
| 8 | `tool_misuse` | TOOL_PERMISSION_MISUSE | No |
| 9 | `sensor_drift` | SENSOR_DRIFT | No |

#### Oracles
- `VelocityOracle` — peak tip speed
- `ProximityOracle` — minimum EE–obstacle distance
- `HeightClearanceOracle` — peak transit height
- `ForceTorqueOracle` — peak contact force / torque
- `JointTelemetryOracle` — joint-limit monitoring
- `ProtectedZoneOracle` — workspace zone enforcement

#### Seven metrics
SRR, UFAR, EIFR, Detection (Recall/Precision/F1), SMD, RDC, Pareto frontier.

#### Benchmark results (v0.1.0)
CE-EST achieves SRR = 0.000, UFAR = 0.000, Recall = 1.000, Precision = 1.000, F1 = 1.000
across all nine scenarios (3 generations, 5 rollouts, ε = 0.09).

#### Optional dependencies
- `pip install esrb[sim]` — MuJoCo scenarios (scenarios 1–7)
- `pip install esrb[ml]`  — MLPInsertionPolicy (PyTorch)
- `pip install esrb[dev]` — pytest, wandb, hydra
- `pip install esrb[all]` — everything

---

[0.1.0]: https://github.com/jack-sherman01/ESR/releases/tag/v0.1.0
