"""Lightweight argument builders shared by CLI and experiment implementations."""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from esrb.runtime.paths import project_root

def options_study_ablation(parser):
    parser.add_argument('--scenarios', nargs='+', default=['tool_misuse', 'scripted_peg'], choices=['tool_misuse', 'scripted_peg'])
    parser.add_argument('--reps', type=int, default=5, help='Independent trials per grid point')
    parser.add_argument('--gens', type=int, default=3, help='Generations per trial')
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='')

def options_study_adversarial_evasion(parser):
    pass

def options_benchmark_run(parser):
    parser.add_argument('--config', help='Load an experiment profile (YAML).')
    parser.add_argument('--evaluation-protocol', help='Optional frozen report-only paired reference trials (additional budget).')
    parser.add_argument('--scenarios', nargs='+', choices=['scripted_peg', 'mlp_peg', 'dynamic_obs', 'pick_place', 'tool_tighten', 'tool_handling', 'long_horizon_assembly', 'moving_transport', 'moving_assembly', 'constrained_push', 'force_alignment', 'compliant_assembly', 'episode_removal', 'evaluator_drift', 'tool_misuse', 'sensor_drift', 'stochastic_force', 'grasp_pressure'], help='Which scenarios to run (default: all, or the list from --config).')
    parser.add_argument('--proposal-file', help='Version-bound JSON proposals for a supported parameter policy in impact mode.')
    parser.add_argument('--generations', type=int)
    parser.add_argument('--rollouts', type=int)
    parser.add_argument('--gate-planner', choices=('fixed', 'impact'), help='Candidate scheduling mode for the selected scenarios.')
    parser.add_argument('--multi-constraint', action=argparse.BooleanOptionalAction, help='Evaluate and audit per-rule physical measurements in impact mode.')
    parser.add_argument('--fresh-min', type=int, help='Minimum fresh slots inside the impact candidate budget.')
    parser.add_argument('--replay-all', action=argparse.BooleanOptionalAction, help='Require all historical resets in each scenario (impact only).')
    parser.add_argument('--boundary-cases', type=int, metavar='N', help='Maximum impact-matched boundary cases per scenario.')
    parser.add_argument('--reduction-budget', type=int, metavar='N', help='Maximum counterexample reduction probes per candidate.')
    parser.add_argument('--peg-boundary-cases', type=int, default=0, metavar='N', help='Reserve up to N impact-driven scripted_peg boundary cases (0–4).')
    parser.add_argument('--peg-reduction-budget', type=int, default=0, metavar='N', help='Reserve up to N scripted_peg counterexample reduction probes.')
    parser.add_argument('--severity', type=float)
    parser.add_argument('--epsilon', type=float)
    parser.add_argument('--train-episodes', type=int, help='Teacher episodes for insertion MLPs; BC optimization episodes for grasp_pressure.')
    parser.add_argument('--train-steps', type=int, help='BC gradient steps for insertion MLP pre-training.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, help='Render and save videos.')
    parser.add_argument('--skip-sweep', action='store_true', help='Skip the separate diagnostic seed sweep (gate budget is unchanged).')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory (default: auto-timestamped).')

def options_study_ce_library_biasing(parser):
    pass

def options_study_dynamic_obstacle(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--protected-radius', type=float, default=0.04, help='Oracle protected zone radius (m). Default 0.04.')
    parser.add_argument('--severity', type=float, default=0.5, help='Distribution shift severity [0,1]. Default 0.5.')
    parser.add_argument('--epsilon', type=float, default=0.09, help='UCB safety constraint (max allowed violation rate).')
    parser.add_argument('--record', action='store_true', help='Render and save videos to private/experiments/.')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory (default: auto-timestamped).')

def options_study_episode_removal(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42, help='recorded seed; controlled moving demos use the task reference disturbance unless a preview is supplied')
    parser.add_argument('--train-episodes', type=int, default=100)
    parser.add_argument('--train-steps', type=int, default=2000)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--out-dir', type=str, default='')

def options_study_evaluator_drift(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--out-dir', type=str, default='')

def options_study_fault_prob_sweep(parser):
    pass

def options_study_mlp_insertion(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--max-speed', type=float, default=0.19, help='Oracle tip-speed threshold (m/s). Default 0.19.')
    parser.add_argument('--severity', type=float, default=0.5, help='Contamination fraction [0,1].')
    parser.add_argument('--epsilon', type=float, default=0.09, help='UCB safety constraint (max allowed violation rate).')
    parser.add_argument('--train-episodes', type=int, default=100, help='Teacher episodes to collect for BC training.')
    parser.add_argument('--train-steps', type=int, default=2000, help='BC gradient steps for initial training.')
    parser.add_argument('--record', action='store_true', help='Render and save videos to private/experiments/.')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory (default: auto-timestamped).')

def options_study_multiseed(parser):
    parser.add_argument('--scenarios', nargs='+', default=['tool_misuse', 'sensor_drift', 'stochastic_force', 'scripted_peg', 'dynamic_obs', 'pick_place', 'tool_tighten'])
    parser.add_argument('--n-seeds', type=int, default=10)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--out-dir', type=str, default='')

def options_study_peg_insertion(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--max-speed', type=float, default=0.25, help='Oracle tip-speed threshold (m/s). Default 0.25.')
    parser.add_argument('--severity', type=float, default=0.5, help='Update operator severity [0,1].')
    parser.add_argument('--epsilon', type=float, default=0.09, help='UCB safety constraint (max allowed violation rate).')
    parser.add_argument('--record', action='store_true', help='Render and save videos + data to private/experiments/.')
    parser.add_argument('--out-dir', type=str, default='', help='Output directory for artifacts (default: auto-timestamped).')

def options_study_pick_place(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--out-dir', type=str, default='')

def options_study_rl_scenario(parser):
    parser.add_argument('--train-eps', type=int, default=1000)
    parser.add_argument('--rollouts', type=int, default=12)
    parser.add_argument('--eval-rollouts', type=int, default=5)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--verbose-train', action='store_true')

def options_study_sensor_drift(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='')

def options_study_stochastic_force(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--fault-prob', type=float, default=None, help='Override fault_prob directly (default: severity)')
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='')

def options_study_tool_misuse(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='')

def options_study_tool_tighten(parser):
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--rollouts', type=int, default=5)
    parser.add_argument('--severity', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--out-dir', type=str, default='')

def options_benchmark_search(parser):
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--deadline-utc', required=True, type=datetime.fromisoformat)
    parser.add_argument('--seed', type=int, default=117800000)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--scenarios', nargs='+', choices=['pick_place', 'tool_handling', 'constrained_push', 'moving_transport', 'dynamic_obs', 'scripted_peg', 'force_alignment', 'tool_tighten', 'compliant_assembly', 'long_horizon_assembly', 'moving_assembly'])
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--codex', default=shutil.which('codex'))
    parser.add_argument('--compound-stress', action='store_true', help='test detailed and brief prompts against matched compound reset panels')
    parser.add_argument('--baseline-only', action='store_true', help='measure compound panels without making model proposals')

def options_tools_diagnose_nut_evidence(parser):
    parser.add_argument('--program', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--verify-retest', type=Path, nargs=2, metavar=('DEVELOPMENT', 'RETEST'))

def options_tools_analyze_fixed_chain(parser):
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait', action='store_true', help='wait for the coordinator to finalize before exporting')

def options_tools_analyze_audit_physical_extensions(parser):
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)

def options_tools_analyze_natural_audit(parser):
    parser.add_argument('action', choices=['freeze','calibrate','run','report','review'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, default=project_root()/'.esr-local/natural-campaigns/natural-seeds1to5-r3-v50-20261118')
    parser.add_argument('--profile', type=Path, default=project_root()/'configs/evaluation/natural_offline_v1.json')
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--plot-python',type=Path,help='Optional interpreter with matplotlib, used only for figures')

def options_tools_analyze_natural_study(parser):
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plot-python', type=Path, help='Interpreter with matplotlib; defaults to the current interpreter')

def options_tools_analyze_natural_audit_extended(parser):
    parser.add_argument('action',choices=['freeze','calibrate','run','report'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=project_root()/'private/analysis/natural-safety-20261119')
    parser.add_argument('--profile',type=Path,default=project_root()/'configs/evaluation/natural_offline_v2.json')
    parser.add_argument('--workers',type=int,default=24)

def options_tools_analyze_nut_evidence(parser):
    parser.add_argument('--previous', type=Path, default=project_root()/'private/analysis/natural-safety-20261119')
    parser.add_argument('--root', type=Path, required=True, help='New output directory; existing data is never overwritten')
    parser.add_argument('--profile', type=Path, default=project_root()/'configs/evaluation/nut_evidence_v1.json')

def options_tools_data_consolidate(parser):
    parser.add_argument('--job', type=Path, default=project_root()/'.esr-local/_data-consolidation/20261120')
    parser.add_argument('--wait', action='store_true', help='Wait for the active controlled re-audit before merging it')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--natural-only', action='store_true', help='Merge only completed natural analyses; never touch controlled runs')

def options_tools_analyze_verify_assembly_replay(parser):
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--seed', default=0, type=int)

def options_tools_render_assets(parser):
    pass

def options_tools_render_physical_extensions(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--acceptance', type=Path, help='render saved task-success trajectories or an audited panel; omit for new controlled demos')
    source.add_argument('--reference-preview', type=Path, help='reuse recorded seeds, actual initial conditions and policy parameters from preview metadata')
    parser.add_argument('--out-dir', type=Path, default=project_root() / 'assets')
    parser.add_argument('--scenarios', nargs='+', choices=['long_horizon_assembly', 'moving_transport', 'moving_assembly', 'constrained_push', 'force_alignment', 'compliant_assembly'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--duration', type=float, default=5.)
    parser.add_argument('--success-only', action='store_true', help='capture one task-success demo per selected scene, then render it')

def options_tools_render_tool_handling(parser):
    parser.add_argument('--out-dir', type=Path, default=project_root() / 'assets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--width', type=int, default=960, help='render width in pixels')
    parser.add_argument('--height', type=int, default=720, help='render height in pixels')
    parser.add_argument('--duration', type=float, default=5.0, help='tool GIF playback duration in seconds')
    assembly_mode = parser.add_mutually_exclusive_group()
    assembly_mode.add_argument('--assembly', action='store_true', help='run and animate the complete continuous assembly demos')
    assembly_mode.add_argument('--assembly-witness', type=Path, help='render saved continuous-assembly parent/candidate states')

def options_tools_report_gate(parser):
    parser.add_argument('root', type=Path)

def options_tools_report_handover_batch(parser):
    parser.add_argument('--root', required=True, type=Path)

def options_tools_report_chain(parser):
    parser.add_argument('--root', type=Path, required=True)

def options_tools_analyze_metrics(parser):
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)

def options_tools_diagnose_libero(parser):
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1)

def options_tools_diagnose_perception(parser):
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8114')
    parser.add_argument('--query', default='blue soup can')
    parser.add_argument('--skip-point', action='store_true')


def options_tools_analyze_gate_audit(parser):
    parser.add_argument('run', type=Path)


def options_tools_diagnose_robosuite(parser):
    parser.add_argument('--task', choices=['stack','pickplacecan','two-arm-handover','nut-assembly','wipe'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)

def options_tools_analyze_pickplacecan_evidence(parser):
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--program', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)

def options_tools_analyze_handover_evidence(parser):
    parser.add_argument('capture', type=Path)


def options_tools_diagnose_production_workcells(parser):
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--scenarios',nargs='+',choices=['long_horizon_assembly','moving_transport','moving_assembly'],default=['long_horizon_assembly','moving_transport','moving_assembly'])
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(10)))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--assets-dir',type=Path,default=Path('assets'))


def options_study_measurement_validation(parser):
    parser.add_argument('--phase',choices=['calibrate','validate','gate','report','all','reprocess','reaudit','repair-wipe'],default='all')
    parser.add_argument('--from-run',type=Path,help='Completed run for a new repair or offline rule-audit revision')
    parser.add_argument('--reuse-from',type=Path,help='Reuse verified complete legal-hold repairs from an interrupted repair revision')
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--scenes',nargs='+',choices=['tool_handling','long_horizon_assembly','moving_transport',
        'moving_assembly','constrained_push','force_alignment','compliant_assembly','stack','pickplacecan',
        'two-arm-handover','nut-assembly','wipe'])
    parser.add_argument('--workers',type=int,default=8,help='Independent cases or parallel scenes, 1–8; each scene keeps sequential generations')
    parser.add_argument('--selfbuilt-python',type=Path)
    parser.add_argument('--aspire-python',type=Path)
    parser.add_argument('--episode-timeout',type=float,default=900.,help='Infrastructure timeout per isolated process (seconds)')
