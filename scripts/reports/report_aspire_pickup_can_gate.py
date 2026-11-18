"""Export already recorded PickPlaceCan admission decisions, without re-execution."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
from scripts.reports.report_aspire_stack_gate import export

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    root=parser.parse_args().root.resolve()
    import json
    if json.loads((root/'protocol.json').read_text()).get('task')!='pickup_can':
        raise ValueError('expected a PickPlaceCan batch')
    export(root)
