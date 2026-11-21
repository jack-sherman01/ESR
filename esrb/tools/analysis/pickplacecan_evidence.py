"""Audit archived capture evidence without executing a policy."""
import argparse
import json
from pathlib import Path
from esrb.detection.pick_place_can import audit_capture

def main(args=None):
    from esrb.cli.options import options_tools_analyze_pickplacecan_evidence
    parser=argparse.ArgumentParser()
    options_tools_analyze_pickplacecan_evidence(parser)
    args=parser.parse_args() if args is None else args
    print(json.dumps(audit_capture(args.capture, args.program, args.seed), indent=2, allow_nan=False))
