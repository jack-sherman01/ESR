"""Load nut rules in ASPIRE's lightweight environment."""
import importlib.util
from pathlib import Path
from scripts.gates.stack_gate_rules import decide
from scripts.gates.simulation_capture import geom_group as classify_geometry

PATH = Path(__file__).resolve().parents[2] / 'esrb/detection/nut_assembly_gate.py'
SPEC = importlib.util.spec_from_file_location('esrb_nut_gate', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NutSafety = MODULE.NutSafety
validate_rules = MODULE.validate_rules
