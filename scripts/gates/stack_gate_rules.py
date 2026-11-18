"""Load the pure Stack gate without importing ESR's optional detector packages.

The ASPIRE runtime intentionally has its own environment; importing the complete
esrb.detection package would pull in unrelated NetworkX/ML dependencies.
"""
import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2]/'esrb/detection/stack_gate.py'
SPEC = importlib.util.spec_from_file_location('esrb_stack_gate',PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
StackSafety = MODULE.StackSafety
decide = MODULE.decide
classify_geometry = MODULE.classify_geometry
validate_rules = MODULE.validate_rules
