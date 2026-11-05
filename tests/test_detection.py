from esrb.detection import ChangeImpactGraph, ComponentType, RiskSensitiveSelector, CandidateEvaluation


def test_change_impact_policy_triggers_tests():
    graph = ChangeImpactGraph()
    tests = graph.required_tests({ComponentType.POLICY})
    assert "force_safety" in tests
    assert "regression_suite" in tests


def test_change_impact_evaluator_triggers_integrity():
    graph = ChangeImpactGraph()
    tests = graph.required_tests({ComponentType.EVALUATOR})
    assert "evaluator_integrity" in tests


def test_risk_selector_rejects_all_unsafe():
    selector = RiskSensitiveSelector(epsilon=0.05, alpha=0.1)
    candidates = [
        CandidateEvaluation(policy=None, task_scores=[0.9], safety_violations=[True], generation=1),
    ]
    assert selector.select(candidates) is None


def test_risk_selector_picks_safe_best():
    # Wilson UCB for 0 violations needs ~20 trials to clear epsilon=0.09.
    selector = RiskSensitiveSelector(epsilon=0.09, alpha=0.1)
    candidates = [
        CandidateEvaluation(policy="A", task_scores=[0.7] * 20, safety_violations=[False] * 20, generation=1),
        CandidateEvaluation(policy="B", task_scores=[0.9] * 20, safety_violations=[False] * 20, generation=1),
    ]
    result = selector.select(candidates)
    assert result is not None
    assert result.policy == "B"
