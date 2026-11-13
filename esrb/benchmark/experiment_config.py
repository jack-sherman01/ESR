"""Validated YAML profiles for the combined benchmark runner."""

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from esrb.oracles.multiconstraint import SCENARIO_LIMITS, scenario_oracle


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Run(_Section):
    num_generations: int = Field(default=3, ge=1)
    rollouts_per_generation: int = Field(default=5, ge=1)
    seed: int = Field(default=42, ge=0)


class _Benchmark(_Run):
    scenarios: list[str] = Field(default_factory=lambda: list(SCENARIO_LIMITS), min_length=1)

    @model_validator(mode="after")
    def check_scenarios(self):
        unknown = set(self.scenarios) - SCENARIO_LIMITS.keys()
        if unknown:
            raise ValueError(f"unknown scenarios: {', '.join(sorted(unknown))}")
        if len(self.scenarios) != len(set(self.scenarios)):
            raise ValueError("benchmark.scenarios cannot contain duplicates")
        return self


class _Oracle(_Section):
    limits: dict[str, dict[str, Annotated[float, Field(ge=0, allow_inf_nan=False)]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_limits(self):
        for scenario, limits in self.limits.items():
            if scenario not in SCENARIO_LIMITS:
                raise ValueError(f"unknown scenario in oracle.limits: {scenario}")
            scenario_oracle(scenario, limits=limits)
        return self


class _UpdateOperator(_Section):
    severity: float = Field(default=0.5, ge=0)


class _Selector(_Section):
    epsilon: float = Field(default=0.09, ge=0, le=1)
    alpha: Literal[0.1, 0.05, 0.01] = 0.1


class _Logging(_Section):
    wandb: Literal[False] = False
    log_dir: str = Field(default="private/runs", min_length=1)
    record: bool = False


class _Gate(_Section):
    planner: Literal["fixed", "impact"] = "impact"
    multi_constraint: bool | None = None
    fresh_min: int = Field(default=1, ge=1)
    replay_all: bool = False
    boundary_cases: int = Field(default=0, ge=0)
    reduction_budget: int = Field(default=0, ge=0)


class _Training(_Section):
    train_episodes: int = Field(default=100, ge=1)
    train_steps: int = Field(default=2000, ge=1)


class _Profile(_Section):
    update_operator: _UpdateOperator = Field(default_factory=_UpdateOperator)
    selector: _Selector = Field(default_factory=_Selector)
    logging: _Logging = Field(default_factory=_Logging)


class RiskSamplingConfig(_Section):
    distribution: Literal["uniform"] = "uniform"
    seed: int | None = Field(default=None, ge=0)
    samples: int = Field(ge=1)
    bounds: dict[str, dict[str, Annotated[list[float], Field(min_length=2, max_length=2)]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_scenarios(self):
        unknown = self.bounds.keys() - SCENARIO_LIMITS.keys()
        if unknown:
            raise ValueError(f"unknown scenarios in risk_sampling.bounds: {', '.join(sorted(unknown))}")
        return self


class ExperimentConfig(_Profile):
    benchmark: _Benchmark = Field(default_factory=_Benchmark)
    gate: _Gate = Field(default_factory=_Gate)
    training: _Training = Field(default_factory=_Training)
    oracle: _Oracle = Field(default_factory=_Oracle)
    risk_sampling: RiskSamplingConfig | None = None

    @property
    def scenarios(self):
        return self.benchmark.scenarios

    def limits_for(self, scenario):
        return self.oracle.limits.get(scenario, {})


class _LegacyBenchmark(_Run):
    name: Literal["peg_insertion_force_speed"]


class _LegacyOracle(_Section):
    type: Literal["force_torque"]
    max_force_n: float = Field(ge=0)
    max_torque_nm: float = Field(ge=0)
    max_tip_speed_ms: float | None = Field(default=None, ge=0)


class _LegacyUpdateOperator(_UpdateOperator):
    type: Literal["force_speed_injection"]


class LegacyPegConfig(_Profile):
    """Compatibility with the original scripted peg YAML and audit snapshots."""

    benchmark: _LegacyBenchmark
    oracle: _LegacyOracle
    update_operator: _LegacyUpdateOperator
    scenarios: ClassVar[tuple[str, ...]] = ("scripted_peg",)
    gate: ClassVar[_Gate] = _Gate(multi_constraint=True)
    training: ClassVar[_Training] = _Training()

    def limits_for(self, scenario):
        if scenario != "scripted_peg":
            return {}
        limits = {"peak_force_n": self.oracle.max_force_n, "peak_torque_nm": self.oracle.max_torque_nm}
        if self.oracle.max_tip_speed_ms is not None:
            limits["peak_tip_speed_ms"] = self.oracle.max_tip_speed_ms
        return limits


def validate_experiment_config(document):
    benchmark = document.get("benchmark") if isinstance(document, dict) else None
    model = LegacyPegConfig if isinstance(benchmark, dict) and "name" in benchmark else ExperimentConfig
    return model.model_validate(document)


def load_experiment_config(path: Path) -> ExperimentConfig | LegacyPegConfig:
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            mapping = super().construct_mapping(node, deep=deep)
            if len(mapping) != len(node.value):
                raise ValueError("duplicate YAML key")
            return mapping

    try:
        document = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    return validate_experiment_config(document)
