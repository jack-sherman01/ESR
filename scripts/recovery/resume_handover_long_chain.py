"""Continue the frozen handover coordinator after a V48 submission timeout.

No measured version is replayed. Only the starting index and prior identities
of the original drive function change; its episode and feedback code is reused.
"""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import ast
import inspect
from pathlib import Path

from scripts.runners import run_handover_long_chain as core


def continuation(original):
    tree = ast.parse(original)
    function = tree.body[0]
    assert function.name == 'drive'
    seen = [n for n in function.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'seen' for t in n.targets)]
    loops = [n for n in function.body if isinstance(n, ast.For)]
    assert len(seen) == len(loops) == 1
    assert ast.dump(seen[0].value) == ast.dump(ast.Dict(keys=[], values=[]))
    assert ast.unparse(loops[0].iter) == 'range(1, VERSIONS + 1)'
    seen[0].value = ast.parse("{program_identity((folder / f'version_{v:02d}' / 'program.py').read_text()): v for v in range(1, 48)}", mode='eval').body
    loops[0].iter.args[0] = ast.Constant(value=48)
    return ast.fix_missing_locations(tree)


def main(root):
    manifest = core.read(root / 'resume_coordinator_manifest.json')
    assert core.sha(Path(__file__)) == manifest['source_sha256']
    assert core.sha(Path(core.__file__)) == manifest['original_coordinator_sha256']
    assert not (root / 'round_01/version_48').exists()
    assert not (root / 'round_01/generation_48').exists()
    assert not list((root / 'round_01/submissions').glob('v48_*'))
    core.state.load_rules(root)
    chain = core.state.load_chain(root, 1)
    assert len(chain['events']) == 47
    for v in range(1, 48):
        folder = root / f'round_01/version_{v:02d}'
        assert (folder / 'gate_decision.json').exists()
        assert (folder / 'execution/result.json').exists()
        assert core.sha(folder / 'program.py') == core.read(folder / 'proposal.json')['source_sha256']
    original = inspect.getsource(core.drive)
    tree = continuation(original)
    # Compile in the original module namespace: __file__ and episode commands
    # retain their original frozen paths, including all feedback logic.
    namespace = dict(core.__dict__)
    exec(compile(tree, core.__file__, 'exec'), namespace)
    namespace['drive'](root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['drive'])
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    main(args.root.resolve())
