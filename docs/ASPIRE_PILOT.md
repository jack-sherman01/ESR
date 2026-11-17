# ASPIRE + ESRB: two-round simulation pilot

For installing the public ASPIRE runtime, see [server setup](SERVER_SETUP.md).
This experimental pilot also needs a separate legacy ESRB checkout and its
research inputs; the public runtime installer alone does not supply them.

The pilot reuses the ASPIRE checkout and the v3 separate-measurement gate in
`ESR_benchmark` from the current `ESR` repository. The repositories both
package `esrb`; the new coordinator dispatches the old runner with its own
interpreter and working directory, rather than mixing the packages.

Task: LIBERO-Pro `libero_goal_swap/task_0` (open the middle drawer). Start from
the previously saved ASPIRE program. The Agent receives the active version,
that program, the public API and the official iterative-debugging worker
guide, and outcomes plus tool-span timings from development seeds 51 and 52. In
later rounds it receives the earlier program diff, development results, and
whether the earlier proposal became active. It proposes one
complete program per round. The scope is an adaptation of the official
single-GPU worker workflow: the coordinator executes all episodes. The
full ASPIRE curriculum, skill-library promotion, and hardware are out of scope.

The gate evaluates the parent and child on paired cases separate from
development. It uses the unchanged `safety_drawer_v3.json`; it may short-circuit
on invalid evidence or observed violations. Fresh risk draws are separate
from paired task cases and counterexample replays. The Agent receives no
hidden seeds or physical gate reports, only its public development feedback
and whether the previous candidate became active. The gate is a simulation
diagnostic with engineering limits, not a hardware safety certificate.

From the legacy repository, start the existing services in a separate terminal:

```bash
cd /path/to/ESR_benchmark
python3 -B -m aspire_native.services start --project "$PWD" \
  --output /tmp/esrb-aspire-services-new --without-curobo
```

When the supervisor reports ready, run from the current repository:

```bash
cd /path/to/ESR
LEGACY_ASPIRE=/path/to/ESR_benchmark
python3 -B scripts/run_aspire_pilot.py --legacy "$LEGACY_ASPIRE" \
  --baseline "$LEGACY_ASPIRE/runs/aspire_official_validation_20261103/campaign/libero-pro/libero_goal_swap/task_0/fix_code.py" \
  --model gpt-6.1-sol --out private/runs/aspire_pilot_new
```

Use a new output directory and a working NVIDIA environment with the original
ASPIRE dependencies. The Agent uses existing Codex authentication. No
additional service credentials are copied into the output.

Read `pilot_manifest.json` for settings and source fingerprints,
`round_*/agent/` for prompts, transcripts and submitted programs,
`round_*/round.json` for decisions and parent versions,
`gate/evaluations/` for per-version native captures, and `gate/store/state.json`
for the active version history. A run without `complete.json` is incomplete.
The pilot does not retry failed model calls automatically. If both slots
submitted the same executable AST and both were rejected on the same parent,
`--retry-duplicate` (in place of `--baseline`) performs one explicit refinement
in `round_2_refinement/` of that completed run. It retains all original records
and reuses the gate's existing active version and counterexample bank. The run files stay
under `private/`, outside Git.
