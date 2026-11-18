"""Audit a small saved engineering panel without launching simulation rollouts.

Repeated --run roots are ordered: a later root explicitly supersedes the same
scenario in an earlier root. Superseded files are retained and listed; they are
never overwritten or silently reinterpreted under a changed physical model.
"""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import sys


import mujoco
import numpy as np

from esrb.detection.gate_audit import audit_gate
from esrb.tasks.extensions.registry import EXTENSIONS
from esrb.utils.hashing import hash_artifact,hash_dict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,action='append',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error('out must be a new directory under an existing parent')
    selected,superseded={},[]
    for root in args.run:
        for name in EXTENSIONS:
            report=root/name/'demo_results.json'
            if report.is_file():
                if name in selected:
                    superseded.append({'scenario':name,'report':str(selected[name].resolve()),
                                       'digest':hash_artifact(selected[name])})
                selected[name]=report
    if set(selected)!=set(EXTENSIONS):
        parser.error('provide a saved report for every physical extension')
    records=[]
    for name,path in selected.items():
        report=json.loads(path.read_text())
        archive_path=path.with_name('counterexamples.json')
        archive=json.loads(archive_path.read_text())
        audit=audit_gate(report,archive)
        parents=[item['raw_result'] for row in report['generations'] for item in row['baseline_measurements']]
        candidates=[case['raw_result'] for row in report['generations'] for case in row['gate_plan']['executed']]
        records.append({'scenario':name,'source_report':str(path.resolve()),
            'report_file_digest':hash_artifact(path),'archive_file_digest':hash_artifact(archive_path),
            'audit':audit,'parent_task_success':sum(item['task_success'] for item in parents),
            'parent_count':len(parents),'candidate_task_success':sum(item['task_success'] for item in candidates),
            'candidate_count':len(candidates),
            'candidate_unsafe':sum(item['oracle']['status']=='VIOLATION' for item in candidates),
            'candidate_unknown':sum(any(r['status']=='UNKNOWN' for r in item['oracle']['rules']) for item in candidates),
            'accepted_generations':sum(row['accepted'] for row in report['generations']),
            'reduction_status':[row['gate_plan']['reduction']['status'] for row in report['generations']],
            'risk_samples':sum(row['gate_plan']['independent_risk']['sample_count'] for row in report['generations']),
        })
        print(name,audit['status'],flush=True)
        del report,archive,parents,candidates
    source_files=sorted(list(Path('esrb/tasks/extensions').rglob('*.py'))+
                        list(Path('esrb/tasks/extensions/assets').glob('*.xml'))+
                        [Path('esrb/oracles/extensions.py'),Path('esrb/oracles/assembly.py'),
                         Path('esrb/oracles/multiconstraint.py'),Path('esrb/detection/gate_audit.py'),
                         Path('experiments/run_benchmark.py')])
    sources={str(path):hash_artifact(path) for path in source_files}
    result={'schema':'esrb.physical-engineering-acceptance/1','status':'PASS',
            'scope':'controlled_engineering_only','model_calls':0,'hardware_actions':0,'research_samples':0,
            'versions':{'mujoco':mujoco.__version__,'numpy':np.__version__},'sources':sources,
            'source_manifest_scope':'physical_extension_modules_and_core_integration',
            'source_digest':hash_dict(sources),'superseded_reports':superseded,'scenarios':records}
    args.out.mkdir()
    (args.out/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(args.out/'acceptance.json')


if __name__=='__main__':
    main()
