"""Unified CLI for a frozen, retrospective natural-campaign audit."""
import json


def main(args):
    from .natural_audit import freeze,calibrate,execute,report
    if args.action=='freeze':
        items,_=freeze(args.root,args.campaign,args.profile)
        value={'frozen_episodes':len(items)}
    elif args.action=='calibrate':value=calibrate(args.root)
    elif args.action=='run':value=execute(args.root,args.workers)
    elif args.action=='report':value=report(args.root)
    else:
        from .natural_audit_review import review,figures
        value=review(args.root)
        if args.plot_python:
            import subprocess
            subprocess.run([str(args.plot_python),'-B','-c',
                'from pathlib import Path; import sys; from esrb.tools.analysis.natural_audit_review import figures; figures(Path(sys.argv[1]))',
                str(args.root.resolve())],check=True)
        else:
            import importlib.util
            if importlib.util.find_spec('matplotlib'):figures(args.root)
    print(json.dumps(value,ensure_ascii=False))
