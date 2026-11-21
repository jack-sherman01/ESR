"""Audit archived capture evidence without executing a policy."""
import argparse
import json
from pathlib import Path
from esrb.detection.aspire_handover import audit

def main(args=None):
    from esrb.cli.options import options_tools_analyze_handover_evidence
    parser=argparse.ArgumentParser()
    options_tools_analyze_handover_evidence(parser)
    args=parser.parse_args() if args is None else args
    print(json.dumps(audit(args.capture), indent=2))
