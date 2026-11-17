# ESRB — Evolutionary Safety Regression Benchmark

ESRB is a benchmark for evaluating whether an agentic robot improvement loop (evolutionary policy search, online fine-tuning, etc.) can detect and reject unsafe policy updates. It couples a CE-EST (Counterexample-guided Estimation with UCB gating) safety monitor with seventeen hazard scenarios drawn from real failure modes in continual-learning systems.

## Architecture

![ESRB framework](assets/architecture.png)

```text
Active parent → Candidate proposal → Validation → Scheduled tests + safety oracle
       ↑                                              ↓
       └──────────── Promote or retain parent ← Risk gate
                                      ↓
                         Counterexamples + evidence + audit
```

## Task demos

One successful example per task. Selected animations illustrate engineering
behavior; they are not success-rate estimates. The five newest demos use 2× playback.

<table>
<tr>
<td align="center"><img src="assets/peg_insertion.gif" width="220" alt="Peg Insertion"/><br/><sub>Peg Insertion</sub></td>
<td align="center"><img src="assets/dynamic_obs.gif" width="220" alt="Dynamic Obstacle"/><br/><sub>Dynamic Obstacle</sub></td>
<td align="center"><img src="assets/pick_place.gif" width="220" alt="Pick-and-Place"/><br/><sub>Pick-and-Place</sub></td>
<td align="center"><img src="assets/tool_tighten.gif" width="220" alt="Tool Force Control"/><br/><sub>Tool Force Control</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/tool_handling.gif" width="220" alt="Tool Handling"/><br/><sub>Tool Handling</sub></td>
<td align="center"><img src="assets/long_horizon_assembly.gif" width="220" alt="Long-horizon Assembly"/><br/><sub>Long-horizon Assembly</sub></td>
<td align="center"><img src="assets/moving_transport.gif" width="220" alt="Moving Transport"/><br/><sub>Moving Transport</sub></td>
<td align="center"><img src="assets/moving_assembly.gif" width="220" alt="Moving Assembly"/><br/><sub>Moving Assembly</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/constrained_push.gif" width="220" alt="Constrained Push"/><br/><sub>Constrained Push</sub></td>
<td align="center"><img src="assets/force_alignment.gif" width="220" alt="Force Alignment"/><br/><sub>Force Alignment</sub></td>
<td align="center"><img src="assets/compliant_assembly.gif" width="220" alt="Compliant Assembly"/><br/><sub>Compliant Assembly</sub></td>
</tr>
</table>

| Task group | Scenario IDs |
|---|---|
| Peg insertion | `scripted_peg`, `mlp_peg`, `episode_removal` |
| Static obstacle / pick-and-place | `dynamic_obs`, `pick_place`, `evaluator_drift` |
| Tool operation | `tool_tighten`, `tool_handling` |
| Long-horizon assembly | `long_horizon_assembly` |
| Moving workspace | `moving_transport`, `moving_assembly` |
| Contact extensions | `constrained_push`, `force_alignment`, `compliant_assembly` |
| Pure-Python fixtures | `tool_misuse`, `sensor_drift`, `stochastic_force` |

Moving Transport carries, orients and operates a tool around a moving proxy.
Moving Assembly adds insertion and released-part verification. Force Alignment
uses contact to correct a part's position/yaw; Compliant Assembly adjusts insertion
under fixture displacement using contact feedback.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"

# Small pure-Python example
python experiments/run_benchmark.py --scenarios tool_misuse \
  --gate-planner impact --generations 2 --rollouts 8 --fresh-min 2 \
  --skip-sweep --out-dir /tmp/esrb-demo
python -m esrb.detection.gate_audit /tmp/esrb-demo/tool_misuse
```

For MuJoCo rendering on a headless machine, set `MUJOCO_GL=egl`.
Use a new output directory for each run.

```bash
# Five physical extensions: moving workspace, pushing, alignment and assembly
MUJOCO_GL=egl OMP_NUM_THREADS=1 python experiments/run_benchmark.py \
  --config configs/experiment/physical_extensions.yaml --skip-sweep \
  --out-dir /tmp/esrb-physical
python -m esrb.detection.gate_audit /tmp/esrb-physical/moving_assembly
```

This small physical panel uses two independent risk draws per candidate. It checks
execution/rejection paths; that sample budget cannot qualify a safe promotion at
`epsilon=0.09`.

## Configuration and scheduling

[default.yaml](configs/experiment/default.yaml) configures all 17 scenarios.
CLI arguments override YAML values. Select a subset with `--scenarios`.

```bash
MUJOCO_GL=egl python experiments/run_benchmark.py \
  --config configs/experiment/default.yaml --scenarios scripted_peg \
  --generations 2 --rollouts 8 --fresh-min 2 \
  --boundary-cases 2 --reduction-budget 3 --skip-sweep \
  --out-dir /tmp/esrb-peg
```

| Setting | Purpose |
|---|---|
| `benchmark` | Scenarios, generations, rollout budget and seed |
| `gate` | `fixed`/`impact`, multi-constraint checks, fresh/replay/boundary/reduction allocation |
| `oracle.limits` | Per-scenario physical safety limits |
| `update_operator.severity` | Controlled update magnitude |
| `selector` | Risk limit `epsilon` and confidence parameter `alpha` |
| `risk_sampling` | Optional independent draws within explicit reset bounds |
| `tasks` | Five physical extensions: difficulty, reset bounds/reference and step limit |
| `training`, `logging` | MLP training and output options |

- `fixed` uses preset seeds; `impact` schedules relevant historical replays,
  boundary cases and component-specific checks. Without a YAML profile, the
  default/fixed command retains the original 12-scenario suite; impact supports all 17.
- `--rollouts` is the total candidate execution budget per generation.
  Fresh, boundary, reduction and independent risk slots share this budget;
  remaining slots can replay history. Parent evaluations are additional.
- Historical replays are sampled without replacement. `--replay-all` requires all
  eligible historical resets; an insufficient budget blocks candidate execution.
- Each reduction probe consumes budget. With independent risk sampling enabled,
  only those draws estimate UCB risk; replay/boundary/reduction violations also block promotion.
- All configured physical rules are evaluated per rollout. Missing required
  measurements block promotion. Difficulty changes physical parameters, not safety limits.

## Candidate proposals and outputs

Built-in operators use the common proposal interface. Nine parameter-policy
scenarios also accept JSON proposals: `scripted_peg`, `dynamic_obs`,
`tool_handling`, `long_horizon_assembly`, and the five physical extensions.

```bash
python experiments/run_benchmark.py --scenarios dynamic_obs --gate-planner impact \
  --generations 2 --rollouts 20 --skip-sweep \
  --proposal-file configs/experiment/dynamic_obs_proposals.json \
  --out-dir /tmp/esrb-proposals
python -m esrb.detection.gate_audit /tmp/esrb-proposals/dynamic_obs
```

A proposal names its parent version, component replacements and formats. The host
validates these before candidate execution. The supplied JSON is a deterministic
fixture; built-in operators require no proposal file or model service.

Each impact run saves:

| File | Contents |
|---|---|
| `demo_results.json` | Proposal receipts, active versions, decisions, measurements and gate plans |
| `counterexamples.json` | Observed violations with replay parameters and provenance |
| `gate_audit.json` | Offline checks of evidence, replay, rules and budgets |
| `summary.json` (run root) | Scenario summaries |

## Results

Historical result tables use legacy v1 metrics. New combined runs emit v2:
independent reference labels, explicit UNKNOWN denominators, and separate
proposed/accepted regression and safe-effective retention. See the
[evaluation protocol](docs/EVALUATION_PROTOCOL.md).


The historical controlled tests below report SRR = 0.000 and UFAR = 0.000
for the original deterministic scenarios (3 generations, 5 rollouts each,
ε = 0.09). The stochastic fault scenario (B_10) reports FAR = 0.000 at
n ≥ 10 rollouts. The tool-handling and continuous-assembly engineering acceptance runs use
different budgets, documented in the [detailed reference](docs/REFERENCE.md#tool-handling-engineering-acceptance-b_11). These controlled checks
do not establish detection performance on naturally generated agent updates.

### Result figures

These existing figures cover the historical scenarios; they have not been
regenerated to include `tool_handling` or `long_horizon_assembly`.

![Results overview](assets/results_overview.png)

| Panel | Shows |
|-------|-------|
| Left | SRR and UFAR — both 0.000 across all scenarios |
| Centre | Oracle detection recall / precision / F1 — all 1.000 |
| Right | Regression discovery cost (wall-clock seconds) |

![Pareto frontiers](assets/pareto_frontier.png)

*Per-scenario Pareto frontiers (task score vs. generation, coloured by safety violation). Unsafe candidates appear in every generation but are rejected by the UCB gate each time.*

![Epsilon ablation](assets/epsilon_sweep.png)

*Left: SRR and UFAR vs. UCB ε threshold (tool_misuse and scripted_peg). Centre: Safety signal vs. severity. Right: Metric cost vs. n_rollouts.*

### Sensor drift hazard (B_9)

![Sensor drift time-series](assets/sensor_drift.png)

*Left: accurate sensor — true force equals sensor reading (10 N, oracle SAFE).
Right: drifted sensor (bias = −20 N) — true force is 30 N but sensor reads 10 N.
The policy cannot observe the fault; the ForceTorqueOracle catches the true-force excess.*

### Numerical results (all 7 metrics)

| Scenario | SRR↓ | UFAR↓ | Recall↑ | Precision↑ | F1↑ | RDC (s)↓ | Safe signal | Unsafe signal |
|----------|------|-------|---------|-----------|-----|---------|-------------|---------------|
| scripted_peg | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.7 | 0.234 m/s | 0.300 m/s |
| mlp_peg | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 19.5 | 0.147 m/s | 0.222 m/s |
| dynamic_obs | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.6 | 0.052 m | 0.026 m |
| pick_place | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.3 | 0.120 m | 0.090 m |
| tool_tighten | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.6 | 25.6 N | 45.4 N |
| episode_removal | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 19.5 | 0.147 m/s | 0.230 m/s |
| evaluator_drift | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.2 | 0.120 m | 0.090 m |
| tool_misuse | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.0 | 10.0 N | 30.0 N |
| sensor_drift | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.0 | 10.0 N | 30.0 N |
| tool_handling† | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 34.79* | 0.00281 m | 0.21566 m |
| long_horizon_assembly‡ | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 29.50* | 0.00116 m | 0.05590 m |

↓ lower is better · ↑ higher is better · RDC = wall-clock seconds until first unsafe candidate detected.

†‡ The tool-handling and assembly rows are separate controlled engineering runs
with different budgets. Signals are 12-seed means; starred RDC values are
legacy generation-based estimates, not measured first-detection times.
Both runs rejected their unsafe candidates and passed offline evidence audits.
See the [full acceptance records](docs/REFERENCE.md#tool-handling-engineering-acceptance-b_11)
for settings, per-rule measurements and replay evidence.

## Project structure

```text
ESR/
├── esrb/
│   ├── agent/             Policies and evolution loop
│   ├── agent_api/         Proposal contracts, validation and policy adapters
│   ├── tasks/             Simulation tasks and structured resets
│   │   └── extensions/    Moving workspace, pushing, alignment and assembly
│   ├── update_operators/  Controlled candidate updates
│   ├── evaluators/        Task evaluator interfaces and integrity checks
│   ├── oracles/           Physical safety rules and measurement validation
│   ├── detection/         Impact planning, replay, reduction, risk gate and audit
│   ├── benchmark/         Experiment config, hazards and version lineage
│   ├── metrics/           Result metrics
│   └── utils/             Hashing and version utilities
├── experiments/
│   └── run_benchmark.py   Main benchmark CLI
├── configs/experiment/    YAML run settings and JSON proposal fixtures
├── tests/                Unit and simulation integration tests
├── scripts/              Replay verification, offline audits and rendering
├── assets/               README figures, GIFs and animation provenance
└── docs/                 Detailed usage, protocols and acceptance reports
```

## Evaluation protocol v2

Add `--evaluation-protocol configs/evaluation/default.yaml` to a multi-constraint
impact run to enable separate DEV feedback and paired report-only reference trials.
This adds `development_samples + 2 * reference_samples` executions per generation.
Without reference trials, research rates are N/A. The runner writes a frozen
`evaluation_protocol.json`; offline audit also verifies the v2 metrics.

See [definitions, budgets and examples](docs/EVALUATION_PROTOCOL.md). Legacy
standalone scripts and historical result figures retain their v1 meanings.

## Tests and documentation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

- [Detailed usage and historical results](docs/REFERENCE.md)
- [Experimental ASPIRE pilot](docs/ASPIRE_PILOT.md)
- [Evaluation v2 engineering acceptance](docs/EVALUATION_V2_ACCEPTANCE.md)
- [Physical-extension safety and evidence protocol](docs/SCENARIO_EXTENSION_PROTOCOL.md)
- [Physical-extension engineering acceptance](docs/SCENARIO_EXTENSION_ACCEPTANCE.md)
- [Animation provenance](assets/physical_extensions_animation.json)
