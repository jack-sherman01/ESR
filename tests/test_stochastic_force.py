"""
Tests for StochasticForceOperator + StochasticForcePolicy + StochasticForceTask.

All tests are pure-logic (no MuJoCo required).  Coverage:
  StochasticForceOperator   — fault_prob metadata, severity threshold, deepcopy
  StochasticForcePolicy     — constant 10 N output, state_dict roundtrip
  StochasticForceTask       — per-seed fault sampling, force amplification
  Oracle integration        — no-fault SAFE, fault UNSAFE, stochastic sweep
  UCB detection reliability — demonstrates that large n catches low-p faults
"""

from __future__ import annotations

import numpy as np
import pytest

from esrb.agent.stochastic_force_policy import StochasticForcePolicy
from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import StochasticForceTask
from esrb.update_operators import StochasticForceOperator

_ORACLE_THRESHOLD = 20.0


def _run_ep(fault_prob: float = 0.0, seed: int = 0) -> dict:
    task = StochasticForceTask()
    task.configure(fault_prob=fault_prob)
    obs    = task.reset(seed=seed)
    policy = StochasticForcePolicy()
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


# ── StochasticForceOperator ───────────────────────────────────────────────────

class TestStochasticForceOperatorApply:
    def test_name(self):
        assert StochasticForceOperator.name == "stochastic_force"

    def test_candidate_is_deepcopy(self):
        p = StochasticForcePolicy()
        r = StochasticForceOperator().apply(p, seed=0, severity=0.5)
        assert r.candidate_policy is not p

    def test_fault_active_at_half_severity(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=0.5)
        assert r.metadata["has_fault"] is True
        assert r.metadata["fault_prob"] == pytest.approx(0.5)

    def test_fault_active_at_min_threshold(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=0.3)
        assert r.metadata["has_fault"] is True
        assert r.metadata["fault_prob"] == pytest.approx(0.3)

    def test_no_fault_below_threshold(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=0.2)
        assert r.metadata["has_fault"] is False
        assert r.metadata["fault_prob"] == pytest.approx(0.0)

    def test_fault_prob_at_full_severity(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=1.0)
        assert r.metadata["fault_prob"] == pytest.approx(1.0)

    def test_result_severity(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=0.7)
        assert r.severity == pytest.approx(0.7)

    def test_original_policy_unmodified(self):
        p  = StochasticForcePolicy()
        sd = p.state_dict()
        StochasticForceOperator().apply(p, seed=0, severity=0.5)
        assert p.state_dict() == sd

    def test_metadata_has_required_keys(self):
        r = StochasticForceOperator().apply(StochasticForcePolicy(), seed=0, severity=0.5)
        for key in ("fault_prob", "has_fault"):
            assert key in r.metadata


# ── StochasticForcePolicy ─────────────────────────────────────────────────────

class TestStochasticForcePolicy:
    def _obs(self, commanded: float = 0.0):
        from esrb.tasks.base import TaskObservation
        return TaskObservation(state=np.array([commanded], dtype=np.float32), info={})

    def test_act_output_shape(self):
        a = StochasticForcePolicy().act(self._obs())
        assert a.shape == (1,)

    def test_act_output_dtype(self):
        a = StochasticForcePolicy().act(self._obs())
        assert a.dtype == np.float32

    def test_outputs_constant_10n(self):
        p = StochasticForcePolicy()
        for _ in range(5):
            a = p.act(self._obs(float(np.random.rand())))
            assert float(a[0]) == pytest.approx(10.0)

    def test_state_dict_roundtrip(self):
        p  = StochasticForcePolicy(target_force_n=8.0)
        sd = p.state_dict()
        p2 = StochasticForcePolicy()
        p2.load_state_dict(sd)
        assert p2.target_force_n == pytest.approx(8.0)

    def test_state_dict_has_expected_key(self):
        sd = StochasticForcePolicy().state_dict()
        assert "target_force_n" in sd


# ── StochasticForceTask ───────────────────────────────────────────────────────

class TestStochasticForceTask:
    def test_no_fault_episode_has_contact_forces(self):
        ep = _run_ep(fault_prob=0.0)
        assert "contact_forces" in ep
        assert len(ep["contact_forces"]) == 10

    def test_no_fault_actual_force_equals_commanded(self):
        ep = _run_ep(fault_prob=0.0)
        for f_row, c in zip(ep["contact_forces"], ep["commanded_forces"]):
            assert f_row[0] == pytest.approx(c)

    def test_fault_episode_actual_force_amplified(self):
        # seed 0 with fault_prob=1.0 → fault is always active
        ep = _run_ep(fault_prob=1.0, seed=0)
        assert ep["has_fault"] is True
        for f_row, c in zip(ep["contact_forces"], ep["commanded_forces"]):
            assert f_row[0] == pytest.approx(c * 3.0)

    def test_no_fault_episode_actual_force_10n(self):
        ep = _run_ep(fault_prob=0.0, seed=0)
        for f_row in ep["contact_forces"]:
            assert f_row[0] == pytest.approx(10.0)

    def test_fault_episode_actual_force_30n(self):
        ep = _run_ep(fault_prob=1.0, seed=0)
        for f_row in ep["contact_forces"]:
            assert f_row[0] == pytest.approx(30.0)

    def test_fault_deterministic_given_seed(self):
        task = StochasticForceTask()
        task.configure(fault_prob=0.5)
        obs1 = task.reset(seed=7)
        ep1_fault = task._has_fault
        obs2 = task.reset(seed=7)
        ep2_fault = task._has_fault
        assert ep1_fault == ep2_fault

    def test_fault_stored_in_episode_data(self):
        ep = _run_ep(fault_prob=1.0, seed=0)
        assert ep["has_fault"] is True
        ep2 = _run_ep(fault_prob=0.0, seed=0)
        assert ep2["has_fault"] is False

    def test_fault_prob_stored_in_episode_data(self):
        ep = _run_ep(fault_prob=0.6, seed=0)
        assert ep["fault_prob"] == pytest.approx(0.6)

    def test_evaluate_success_true_when_forces_logged(self):
        task = StochasticForceTask()
        ep   = _run_ep()
        res  = task.evaluate(ep)
        assert res.success is True

    def test_configure_persists_across_reset(self):
        task = StochasticForceTask()
        task.configure(fault_prob=1.0)
        task.reset(seed=0)
        assert task._fault_prob == pytest.approx(1.0)
        task.reset(seed=1)
        assert task._fault_prob == pytest.approx(1.0)


# ── Oracle integration ────────────────────────────────────────────────────────

class TestOracleIntegration:
    _ORACLE = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    def test_no_fault_always_safe(self):
        v = self._ORACLE.evaluate(_run_ep(fault_prob=0.0), generation=0)
        assert v.safe
        assert v.signal_values["peak_force_n"] == pytest.approx(10.0)

    def test_full_fault_always_unsafe(self):
        v = self._ORACLE.evaluate(_run_ep(fault_prob=1.0), generation=0)
        assert not v.safe
        assert v.signal_values["peak_force_n"] == pytest.approx(30.0)

    def test_12_seed_no_fault_all_safe(self):
        oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
        for seed in range(12):
            v = oracle.evaluate(_run_ep(fault_prob=0.0, seed=seed), 0)
            assert v.safe, f"seed={seed}: no-fault should be SAFE"

    def test_12_seed_full_fault_all_unsafe(self):
        oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
        for seed in range(12):
            v = oracle.evaluate(_run_ep(fault_prob=1.0, seed=seed), 0)
            assert not v.safe, f"seed={seed}: full-fault should be UNSAFE"

    def test_stochastic_fault_mixed_verdicts(self):
        oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
        verdicts = []
        for seed in range(50):
            v = oracle.evaluate(_run_ep(fault_prob=0.5, seed=seed), 0)
            verdicts.append(v.safe)
        n_safe   = sum(verdicts)
        n_unsafe = 50 - n_safe
        # With p=0.5 and 50 seeds, we expect ~25 safe and ~25 unsafe
        assert n_safe > 10,   f"Expected some safe episodes, got {n_safe}/50"
        assert n_unsafe > 10, f"Expected some unsafe episodes, got {n_unsafe}/50"


# ── UCB detection reliability ─────────────────────────────────────────────────

class TestUCBDetectionReliability:
    """
    Key finding: stochastic hazards can evade detection with small n.

    With fault_prob=0.5 and n=5 rollouts, there's a ~3% chance k=0 (no
    unsafe rollouts observed), causing UCB < ε and false acceptance.
    With n=20 rollouts, the probability of k=0 drops below 0.001%.

    These tests verify the statistical behaviour, not individual outcomes.
    """
    _ORACLE = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    def _count_unsafe(self, fault_prob: float, seeds: list[int]) -> int:
        """Count how many episodes are oracle-unsafe across the given seeds."""
        return sum(
            1 for s in seeds
            if not self._ORACLE.evaluate(_run_ep(fault_prob, seed=s), 0).safe
        )

    def test_low_fault_prob_can_evade_small_n(self):
        # With fault_prob=0.1 and n=5 seeds, we may see 0 unsafe episodes.
        # This is expected — small n is insufficient for low-p detection.
        # We just verify the machinery runs without error.
        count = self._count_unsafe(0.1, list(range(5)))
        assert 0 <= count <= 5

    def test_large_n_reliable_for_half_fault_prob(self):
        # With fault_prob=0.5 and n=100 seeds, we should see ~50 unsafe.
        count = self._count_unsafe(0.5, list(range(100)))
        # Allow ±20% tolerance around expected 50
        assert 30 <= count <= 70, f"Expected ~50 unsafe/100, got {count}"

    def test_full_fault_prob_always_detected_at_n1(self):
        # Deterministic unsafe at fault_prob=1.0: even n=1 is sufficient.
        count = self._count_unsafe(1.0, [0])
        assert count == 1
