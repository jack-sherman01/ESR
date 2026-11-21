"""Closed-loop Panda controller for sequential three-piece assembly."""
from esrb.agent.production_policy import ProductionPolicy


class AssemblyPolicy(ProductionPolicy):
    name = "long_horizon_assembly"
