"""Load pure handover rules in the ASPIRE environment."""
import importlib.util
from pathlib import Path
from scripts.gates.stack_gate_rules import decide
PATH=Path(__file__).resolve().parents[2]/'esrb/detection/handover_gate.py'
SPEC=importlib.util.spec_from_file_location('esrb_handover_gate',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
HandoverSafety=MODULE.HandoverSafety
classify_geometry=MODULE.classify_geometry
validate_rules=MODULE.validate_rules
