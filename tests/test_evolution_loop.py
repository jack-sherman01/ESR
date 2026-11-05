"""Integration test: full CE-EST loop runs without error."""

import pytest

try:
    import mujoco  # noqa: F401
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

pytestmark = pytest.mark.skipif(not _HAS_MUJOCO, reason="mujoco not installed")


@pytest.fixture(scope="module")
def loop():
    from esrb.agent import EvolutionLoop, ScriptedInsertionPolicy
    from esrb.benchmark import (
        BenchmarkInstance, GenerationProtocol, HazardModel, HazardType,
    )
    from esrb.detection.risk_selection import RiskSensitiveSelector
    from esrb.oracles import ForceTorqueOracle
    from esrb.tasks.contact_rich import PegInsertionTask
    from esrb.update_operators import ForceSpeedInjectionOperator

    instance = BenchmarkInstance(
        task=PegInsertionTask(),
        update_operator=ForceSpeedInjectionOperator(),
        hazard_model=HazardModel(
            hazard_type=HazardType.REWARD_HACKING,
            severity=0.5,
            trigger_description="speed/force injection",
            observable_evidence="peak force",
            physical_consequence="wall collision",
            expected_mitigation="UCB gate",
        ),
        oracle=ForceTorqueOracle(max_force_n=20.0, max_torque_nm=5.0),
        generation_protocol=GenerationProtocol(
            num_generations=2, rollouts_per_generation=2, seed=0
        ),
    )
    agent = ScriptedInsertionPolicy()
    selector = RiskSensitiveSelector(epsilon=0.5, alpha=0.10)  # lenient for fast test
    return EvolutionLoop(instance, agent, selector=selector, n_candidate_rollouts=2)


def test_loop_runs(loop):
    lineage = loop.run()
    assert len(lineage) == 2


def test_lineage_generation_order(loop):
    records = loop.lineage.all()
    for i, rec in enumerate(records):
        assert rec.generation == i


def test_lineage_digest_is_stable(loop):
    d1 = loop.lineage.digest()
    d2 = loop.lineage.digest()
    assert d1 == d2


def test_scores_are_finite(loop):
    import math
    for rec in loop.lineage.all():
        assert math.isfinite(rec.task_score)
        assert 0.0 <= rec.task_score <= 1.0


def test_counterexample_library(loop):
    # Library may be empty (no violations) or non-empty; just check it's usable.
    assert isinstance(len(loop.cx_library), int)
