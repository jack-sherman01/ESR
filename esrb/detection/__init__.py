"""Detection APIs, loaded lazily so pure scene rules need no optional packages."""
from importlib import import_module

_EXPORTS = {'ChangeImpactGraph': ('change_impact', 'ChangeImpactGraph'), 'ComponentType': ('change_impact', 'ComponentType'), 'Counterexample': ('counterexample', 'Counterexample'), 'CounterexampleLibrary': ('counterexample', 'CounterexampleLibrary'), 'RiskSensitiveSelector': ('risk_selection', 'RiskSensitiveSelector'), 'NaiveSelector': ('risk_selection', 'NaiveSelector'), 'CandidateEvaluation': ('risk_selection', 'CandidateEvaluation'), 'ucb_safety_risk': ('risk_selection', 'ucb_safety_risk')}
__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value
