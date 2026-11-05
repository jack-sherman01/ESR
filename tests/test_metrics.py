from esrb.benchmark.lineage import GenerationRecord
from esrb.metrics import (
    safety_regression_rate,
    unsafe_false_acceptance_rate,
    detection_recall_precision,
)


def _rec(gen: int, task_score: float, safety_violated: bool) -> GenerationRecord:
    return GenerationRecord(
        generation=gen,
        policy_digest="abc",
        data_digest="def",
        evaluator_digest="ghi",
        update_operator="test_op",
        task_score=task_score,
        safety_violated=safety_violated,
        oracle_verdicts={},
    )


def test_srr_no_regression():
    records = [_rec(i, 0.8, False) for i in range(5)]
    assert safety_regression_rate(records) == 0.0


def test_srr_one_regression():
    records = [_rec(0, 0.8, False), _rec(1, 0.9, True), _rec(2, 0.9, True)]
    assert safety_regression_rate(records) == pytest.approx(1 / 2)


def test_ufar():
    oracle_unsafe = [True, False, True, True]
    gate_accepted = [True, False, False, True]
    # 2 unsafe accepted out of 3 unsafe total
    assert unsafe_false_acceptance_rate(oracle_unsafe, gate_accepted) == pytest.approx(2 / 3)


def test_detection_recall_precision():
    true_unsafe = [True, False, True, False, True]
    detected   = [True, False, False, False, True]
    stats = detection_recall_precision(true_unsafe, detected)
    assert stats.recall == pytest.approx(2 / 3)
    assert stats.precision == pytest.approx(1.0)


import pytest
