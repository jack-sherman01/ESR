from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from esrb.tasks.base import BaseTask
    from esrb.update_operators.base import BaseUpdateOperator
    from esrb.benchmark.hazard import HazardModel
    from esrb.oracles.base import BaseOracle
    from esrb.benchmark.generation_protocol import GenerationProtocol


@dataclass
class BenchmarkInstance:
    """
    B_i = (T_i, U_i, H_i, O_i, G_i) as defined in the ESRB proposal.

    T: task
    U: allowed update family
    H: hazard model
    O: independent safety oracle
    G: generation protocol (number of generations K, reset policy, etc.)
    """
    task: "BaseTask"
    update_operator: "BaseUpdateOperator"
    hazard_model: "HazardModel"
    oracle: "BaseOracle"
    generation_protocol: "GenerationProtocol"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.task.name}__{self.update_operator.name}"
