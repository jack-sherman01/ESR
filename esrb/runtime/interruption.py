"""Preserve incomplete artifacts; never archive a measured execution."""
from .io import save

def archive_incomplete_execution(root, version):
    candidate = root / f'round_01/version_{version:02d}'
    execution = candidate / 'execution'
    if (candidate / 'gate_decision.json').exists() or (execution / 'result.json').exists():
        raise ValueError('refuse to archive completed or measured execution')
    target = root / 'interrupted_attempts' / f'version_{version:02d}_launch01'
    target.mkdir(parents=True, exist_ok=False)
    if execution.exists():
        execution.rename(target / 'execution')
    if (candidate / 'execution.log').exists():
        (candidate / 'execution.log').rename(target / 'execution.log')
    save(target / 'interruption.json', {'version': version, 'status': 'INTERRUPTED_NO_RESULT',
         'reason': 'Interactive turn interruption terminated host background processes.',
         'result_present': False, 'gate_decision_present': False,
         'note': 'No recorded physics rows; do not infer that no unflushed computation occurred.'})
    return target
