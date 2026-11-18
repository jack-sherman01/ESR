"""Load pure task rules without optional detector dependencies."""
import importlib.util
from pathlib import Path
from scripts.gates.stack_gate_rules import decide
PATH=Path(__file__).resolve().parents[2]/'esrb/detection/pickup_can_gate.py'
SPEC=importlib.util.spec_from_file_location('esrb_pickup_can_gate',PATH)
MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)
PickupCanSafety=MODULE.PickupCanSafety
classify_geometry=MODULE.classify_geometry
validate_rules=MODULE.validate_rules
