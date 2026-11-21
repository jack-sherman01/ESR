# Project guide: tasks, experiments and result interpretation

The [README](../README.md) provides the quick start, task animations and current
results overview. This guide retains the detailed explanations and historical
records moved out of that overview.

- [Task demonstrations](#task-demonstrations)
- [Rendering and experiment commands](#rendering-and-experiment-commands)
- [Configuration and scheduling](#configuration-and-scheduling)
- [Controlled measurement validation](#controlled-measurement-validation)
- [Controlled results](#controlled-results)
- [Evaluation protocol v2](#evaluation-protocol-v2)

## Task demonstrations

### Natural skill update demonstrations

Successful episodes from the five-scene, task-only GPT-6.1 Sol campaign
(750 candidates, 3,750 development episodes). Each **640×480 GIF** shows a complete
recorded episode accelerated to **5 seconds**. The selected V50 programs passed
all five development seeds; the examples below use seed 1. Native actions were
replayed and checked against the recorded physical states. These are selected
task successes, not safety qualifications or success-rate estimates.

Click an animation for its full resolution. [Source records and replay details](../assets/gpt%20sol%206.1/natural-seeds1to5/)
include the chain, version, seed, source hashes and playback speed.

### Controlled tasks and model details

`tool_handling` uses a complete articulated Panda and a precision-driver workcell:
seven actuated arm joints, native coupled fingers, arm collision meshes, a bevelled
tool grip and separate shaft/tip contacts. Actions contain seven joint targets and
grip closure; a damped Jacobian solver converts task waypoints into joint commands.
Grasping uses physical finger contact and friction. The task is **tool alignment
and sustained surface contact**, with no tightening mechanism. The 960×720,
5-second demo combines the complete arm with a close-up of grip/contact:
[workcell still](../assets/tool_handling_workcell.png),
[successful sequence](../assets/tool_handling_stages.png),
[model source and hashes](../esrb/tasks/tool_handling/assets/panda/provenance.json).

The scene follows robosuite's TableArena style: a ceramic tabletop, pale tiled
floor, plain walls and task-focused objects. Native task rendering and README
previews share this visual layer; the robot dynamics and collision geometry stay
unchanged. The original robosuite texture, pinned source and license are recorded
in the [arena provenance](../esrb/tasks/tool_handling/assets/robosuite/provenance.json).

The reduced-grip failure is shown separately. Both sides below use seed 0,
identical camera poses and the same simulation timestamp. Click for the
[synchronized 5-second animation](../assets/tool_handling_comparison.gif);
[measured forces](../assets/tool_handling_comparison_analysis.png) are in a separate analysis view.

[![Baseline versus reduced-grip slip and drop, at the same simulation time](../assets/tool_handling_comparison.png)](../assets/tool_handling_comparison.gif)

| Task group | Scenario IDs |
|---|---|
| Peg insertion | `scripted_peg`, `mlp_peg`, `episode_removal` |
| Static obstacle / pick-and-place | `dynamic_obs`, `pick_place`, `evaluator_drift` |
| Tool operation | `tool_tighten`, `tool_handling` |
| Long-horizon assembly | `long_horizon_assembly` |
| Industrial conveyor workcells | `moving_transport`, `moving_assembly` |
| Contact extensions | `constrained_push`, `force_alignment`, `compliant_assembly` |
| Pure-Python fixtures | `tool_misuse`, `sensor_drift`, `stochastic_force`, `grasp_pressure` |

Long-horizon Assembly now uses a complete Panda to position a base and assemble
its connector and upper component in one continuous episode. Moving Transport
tracks three incoming parts on a friction-driven conveyor and places them in a
kitting tray. Moving Assembly fits the two later components while their pallet
keeps moving. Parts and the pallet are free rigid bodies, held and supported by
real contact; there are no grasp or assembly welds.

The three scenes use robosuite-inspired workcells. Their **640×480, 5-second GIFs
use a single focused view without an inset**; separate workcell images show the
complete Panda. Each current GIF uses one successful seed-0 trajectory and retains
raw source references and playback-speed metadata. This delivery does not run a
new multi-seed panel or an Agent update campaign.
Open the full-resolution [three-piece assembly](../assets/long_horizon_assembly.gif),
[conveyor kitting](../assets/moving_transport.gif), or
[moving assembly](../assets/moving_assembly.gif) animation.
Full-Panda views: [assembly](../assets/long_horizon_assembly_workcell.png),
[transport](../assets/moving_transport_workcell.png),
[moving assembly](../assets/moving_assembly_workcell.png).
See [model sources, task definitions and engineering acceptance](PANDA_PRODUCTION_WORKCELLS.md).
Constrained Push, Force Alignment and Compliant Assembly also use the complete
Panda with new guide rails, alignment fixtures and a horizontal shaft receiver.
Each has a 640×480, 5-second success GIF; see [task references and demos](PANDA_CONTACT_WORKCELLS.md).

Replay the saved successful trajectories without running more episodes:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m esrb \
  tools render physical-extensions \
  --acceptance .esr-local/_checks/panda-production-20261118/delivery/verification.json \
  --out-dir assets --width 640 --height 480 --duration 5
```

## Rendering and experiment commands

Regenerate the measured tool workcell previews with the simulation extras installed:

```bash
MUJOCO_GL=osmesa python -m esrb tools render tool-handling \
  --out-dir assets --width 960 --height 720 --duration 5

MUJOCO_GL=osmesa python -m esrb tools render tool-handling --assembly \
  --out-dir assets --width 640 --height 480 --duration 5

MUJOCO_GL=osmesa python -m esrb tools render physical-extensions \
  --reference-preview assets/physical_extensions_animation.json \
  --scenarios moving_transport moving_assembly \
  --out-dir assets --width 640 --height 480 --duration 5
```

`--reference-preview` reuses recorded seeds, actual initial conditions and policy
parameters. Without a reference, the moving-workspace renderer runs controlled
examples at the task's reference physical condition. `--acceptance` retains the
audited replay mode.

Five scene names are `stack`, `pickplacecan`, `two-arm-handover`, `nut-assembly`, and `wipe`.
See [the current server/run guide](SERVER_SETUP.md) for worker submission and resources.

For task-only multi-seed evolution with no ESR decisions, see the
[natural agent experiment guide](NATURAL_AGENT.md).

For MuJoCo rendering on a headless machine, set `MUJOCO_GL=egl`.
Use a new output directory for each run.

```bash
# Five physical extensions: moving workspace, pushing, alignment and assembly
MUJOCO_GL=egl OMP_NUM_THREADS=1 python -m esrb benchmark run \
  --config configs/experiment/physical_extensions.yaml --skip-sweep \
  --out-dir /tmp/esrb-physical
python -m esrb tools analyze gate-audit /tmp/esrb-physical/moving_assembly
```

This small physical panel uses two independent risk draws per candidate. It checks
execution/rejection paths; that sample budget cannot qualify a safe promotion at
`epsilon=0.09`.

## Configuration and scheduling

[default.yaml](../configs/experiment/default.yaml) configures all 18 scenarios.
`stochastic_force` and `grasp_pressure` are numerical force fixtures with no
MuJoCo model. The latter uses a BC-trained MLP and requires the `ml` extra;
see the [force-scenario commands and limits](REFERENCE.md#numerical-force-scenarios).
CLI arguments override YAML values. Select a subset with `--scenarios`.

```bash
MUJOCO_GL=egl python -m esrb benchmark run \
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
  default/fixed command retains the 13-scenario suite; impact supports all 18.
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
python -m esrb benchmark run --scenarios dynamic_obs --gate-planner impact \
  --generations 2 --rollouts 20 --skip-sweep \
  --proposal-file configs/experiment/dynamic_obs_proposals.json \
  --out-dir /tmp/esrb-proposals
python -m esrb tools analyze gate-audit /tmp/esrb-proposals/dynamic_obs
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

## Controlled measurement validation

The twelve-scene deterministic acceptance study does not call an Agent or model:

```bash
python -m esrb benchmark study measurement-validation \
  --phase all --out-dir .esr-local/measurement-validation/NEW_REVISION --workers 8
```

It separates the 648-episode legacy-compatible main/diagnostic budget from
calibration, additional physical cases and replay. Runs execute frozen source
and resource snapshots. Normal-program failures, untriggered interventions,
incomplete load envelopes and missing measurements are retained explicitly.
See [measurement validation](MEASUREMENT_VALIDATION.md) for phases,
per-scene acceptance, accounting and the limits of the historical metrics.

## Controlled results

Historical result tables use legacy v1 metrics. The general benchmark emits v2:
independent reference labels, explicit UNKNOWN denominators, and separate
proposed/accepted regression and safe-effective retention. See the
[evaluation protocol](EVALUATION_PROTOCOL.md).
The twelve-scene controlled measurement study deliberately retains v1 main
metrics for historical comparison and reports measurement acceptance separately.


The historical controlled tests below report SRR = 0.000 and UFAR = 0.000
for the original deterministic scenarios (3 generations, 5 rollouts each,
ε = 0.09). The stochastic fault scenario (B_10) reports FAR = 0.000 at
n ≥ 10 rollouts. The tool-handling and continuous-assembly engineering acceptance runs use
different budgets, documented in the [detailed reference](REFERENCE.md#tool-handling-engineering-acceptance-b_11). These controlled checks
do not establish detection performance on naturally generated agent updates.

### Current overview

![Controlled measurement validation: historical 8 + new model 12](../assets/measurement_validation/controlled-20261120-r5/combined.png)

The complete overview replaces the earlier eight-scenario image. The historical
numerical table below remains an archived result with its original budgets;
it is not the current 20-row table.

![Pareto frontiers](../assets/pareto_frontier.png)

*Per-scenario Pareto frontiers (task score vs. generation, coloured by safety violation). Unsafe candidates appear in every generation but are rejected by the UCB gate each time.*

![Epsilon ablation](../assets/epsilon_sweep.png)

*Left: SRR and UFAR vs. UCB ε threshold (tool_misuse and scripted_peg). Centre: Safety signal vs. severity. Right: Metric cost vs. n_rollouts.*

### Sensor drift hazard (B_9)

![Sensor drift time-series](../assets/sensor_drift.png)

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

↓ lower is better · ↑ higher is better · RDC uses the historical discovery-cost definition; starred values and the current combined plot use legacy estimates, not measured first-detection times.

†‡ The tool-handling and assembly rows are separate controlled engineering runs
with different budgets. Signals are 12-seed means; starred RDC values are
legacy generation-based estimates, not measured first-detection times.
Both runs rejected their unsafe candidates and passed offline evidence audits.
See the [full acceptance records](REFERENCE.md#tool-handling-engineering-acceptance-b_11)
for settings, per-rule measurements and replay evidence.

<!-- measurement-validation-results:start -->
### Controlled measurement validation: historical 8 + new model 12

pickplacecan: one UNKNOWN episode is excluded from controlled-label detection metrics.
Wipe uses repaired contact-injection rollouts and rules v7; native joint-limit proximity is recorded separately.

The historical eight rows use published rounded values. New rows use the same
3-generation, 5-parent + 5-candidate budget and legacy metric algorithms.
**N/A means incomplete or UNKNOWN; it is never a zero or a successful acceptance.**
Detection numbers are controlled-label metrics, not performance on natural Agent
updates. Five safe trials alone still fail the ε=0.09 Wilson gate.
RDC is a **legacy RDC estimate**, not measured first-discovery time or a speed ranking.
Execution conditions and scheduling revisions are recorded in the summary;
concurrent timings are not comparable to historical serial timings.
Execution receipts and frozen sources document the physical attempts.
Current retained dataset locations and history pruning are described in
[Data layout](DATA_LAYOUT.md).
Rule revisions are versioned separately. A v6 offline re-audit counts sustained
unexpected grasp loss as a violation without requiring a later support impact;
it is a post-hoc analysis and does not represent new online gate interventions.

See the [20-row table](../assets/measurement_validation/controlled-20261120-r5/RESULTS.md),
[per-scene acceptance and accounting](../assets/measurement_validation/controlled-20261120-r5/summary.json),
and [measurement protocol](MEASUREMENT_VALIDATION.md).
The earlier Tool Handling/assembly engineering rows above retain their original
different-budget annotations. No pooled cross-scene accuracy is computed.
<!-- measurement-validation-results:end -->

Current dataset locations, consolidation rules, and live merge status are listed
in [Data layout](DATA_LAYOUT.md). Explained non-execution is reported separately
from safety PASS; unqualified physical risks remain unknown.

## Evaluation protocol v2

Add `--evaluation-protocol configs/evaluation/default.yaml` to a multi-constraint
impact run to enable separate DEV feedback and paired report-only reference trials.
This adds `development_samples + 2 * reference_samples` executions per generation.
Without reference trials, research rates are N/A. The runner writes a frozen
`evaluation_protocol.json`; offline audit also verifies the v2 metrics.

See [definitions, budgets and examples](EVALUATION_PROTOCOL.md). Legacy
standalone scripts and historical result figures retain their v1 meanings.

The historical `long_horizon_assembly‡` result above describes the frozen single-insertion model. It is not a result for the new Panda three-piece task; current engineering acceptance is reported in [the production workcell guide](PANDA_PRODUCTION_WORKCELLS.md).
