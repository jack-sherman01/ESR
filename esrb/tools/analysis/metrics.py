"""Offline reproduction of version-2 evaluation, without new rollouts.

python -m esrb tools analyze metrics --report /tmp/esrb-eval/tool_misuse/demo_results.json \
  --out /tmp/esrb-evaluation-summary.json
"""
import argparse
import json
from pathlib import Path

from esrb.detection.gate_audit import audit_gate
from esrb.metrics.reporting import evaluate_report
from esrb.utils.hashing import hash_artifact


def main(args=None):
    from esrb.cli.options import options_tools_analyze_metrics
    parser = argparse.ArgumentParser()
    options_tools_analyze_metrics(parser)
    args = parser.parse_args() if args is None else args
    if args.out.exists() or args.out.is_symlink() or not args.out.parent.is_dir():
        parser.error('out must be a new file under an existing parent')
    report=json.loads(args.report.read_text())
    archive_path=args.archive or args.report.with_name('counterexamples.json')
    archive=json.loads(archive_path.read_text())
    audit=audit_gate(report,archive)
    result={'schema':'esrb.evaluation-summary/2',
            'report_digest':hash_artifact(args.report),'archive_digest':hash_artifact(archive_path),
            'audit':audit,'metrics':evaluate_report(report)}
    with args.out.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write('\n')
    print(args.out)
