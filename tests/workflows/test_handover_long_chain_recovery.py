"""Recovery must skip completed versions without changing episode logic."""
import ast
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from scripts.runners import run_handover_long_chain as core
from scripts.recovery.resume_handover_long_chain import continuation


def test_recovery_changes_only_loop_start_and_prior_identity_set():
    source = inspect.getsource(core.drive)
    original = ast.parse(source)
    resumed = continuation(source)
    function = resumed.body[0]
    seen = next(n for n in function.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'seen' for t in n.targets))
    loop = next(n for n in function.body if isinstance(n, ast.For))
    assert loop.iter.args[0].value == 48
    assert isinstance(seen.value, ast.DictComp)
    assert ast.unparse(seen.value.generators[0].iter) == 'range(1, 48)'
    seen.value = ast.Dict(keys=[], values=[])
    loop.iter.args[0] = ast.Constant(value=1)
    assert ast.dump(resumed, include_attributes=False) == ast.dump(original, include_attributes=False)
