"""Audit a saved ESRB report and counterexample archive."""
import json
from esrb.detection.gate_audit import audit_gate

def main(args):
    root=args.run.resolve()
    result=audit_gate(json.loads((root/'demo_results.json').read_text()), json.loads((root/'counterexamples.json').read_text()))
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='PASS' else 1
