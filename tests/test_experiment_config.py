from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from esrb.benchmark.experiment_config import load_experiment_config
from esrb.detection.gate_audit import audit_gate
from experiments import run_benchmark


CONFIGS = Path(__file__).resolve().parents[1] / "configs/experiment"
DEFAULT = CONFIGS / "default.yaml"
LEGACY = Path(__file__).with_name("fixtures") / "legacy_peg_profile.yaml"


def test_config_drives_peg_gate_and_audits_without_original_file(tmp_path, monkeypatch):
    profile = tmp_path / "peg.yaml"
    profile.write_text(LEGACY.read_text().replace("max_tip_speed_ms: 0.25", "max_tip_speed_ms: 0.27")
                       .replace("max_force_n: 50.0", "max_force_n: 45.0")
                       .replace("alpha: 0.10", "alpha: 0.05"))
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--config", str(profile),
                                      "--generations", "1", "--rollouts", "2", "--out-dir", str(out)])
    monkeypatch.setattr(run_benchmark, "_sweep_scripted_peg", lambda severity: [
        {"label": "baseline", "safe": True, "value": 0.24},
        {"label": "candidate", "safe": False, "value": 0.30},
    ])
    run_benchmark.main()
    report = json.loads((out / "scripted_peg/demo_results.json").read_text())
    archive = json.loads((out / "scripted_peg/counterexamples.json").read_text())
    assert report["args"]["generations"] == 1
    assert report["args"]["rollouts"] == 2
    assert report["args"]["epsilon"] == 0.05
    assert report["selector_alpha"] == 0.05
    rules = report["generations"][0]["gate_plan"]["executed"][0]["raw_result"]["oracle"]["rules"]
    assert {rule["metric"]: rule["limit"] for rule in rules} == {
        "peak_tip_speed_ms": 0.27, "peak_force_n": 45.0, "peak_torque_nm": 10.0,
    }
    profile.unlink()
    assert audit_gate(report, archive)["status"] == "PASS"
    damaged = deepcopy(report)
    damaged["experiment_config"]["settings"]["oracle"]["max_force_n"] = 60.0
    with pytest.raises(ValueError, match="experiment config digest"):
        audit_gate(damaged, archive)


@pytest.mark.parametrize(("line", "replacement", "message"), [
    ("max_force_n: 50.0", "max_froce_n: 50.0", "max_froce_n"),
    ("max_force_n: 50.0", "max_force_n: .nan", "max_force_n"),
    ("max_force_n: 50.0", "max_force_n: 50.0\n  max_force_n: 40.0", "duplicate YAML key"),
])
def test_config_rejects_invalid_or_ambiguous_limits(tmp_path, line, replacement, message):
    profile = tmp_path / "invalid.yaml"
    profile.write_text(LEGACY.read_text().replace(line, replacement))
    with pytest.raises(ValueError, match=message):
        load_experiment_config(profile)


def test_shared_profile_lists_every_scenario_and_preserves_default_limits():
    from esrb.oracles.multiconstraint import SCENARIO_LIMITS, scenario_oracle

    config = load_experiment_config(DEFAULT)
    assert config.scenarios == list(SCENARIO_LIMITS)
    for scenario in config.scenarios:
        limits = config.limits_for(scenario)
        assert limits
        assert scenario_oracle(scenario, limits=limits).protocol() == scenario_oracle(scenario).protocol()


def test_shared_profile_applies_distinct_limits_and_cli_overrides(tmp_path, monkeypatch):
    profile = tmp_path / "suite.yaml"
    profile.write_text("""
benchmark:
  scenarios: [dynamic_obs, tool_misuse]
  num_generations: 1
  rollouts_per_generation: 3
gate:
  fresh_min: 2
  boundary_cases: 1
oracle:
  limits:
    dynamic_obs: {min_obstacle_dist_m: 0.041}
    tool_misuse: {peak_force_n: 22.0}
""")
    out = tmp_path / "suite"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--config", str(profile),
                                      "--rollouts", "4", "--out-dir", str(out)])
    for scenario in ("dynamic_obs", "tool_misuse"):
        monkeypatch.setattr(run_benchmark, f"_sweep_{scenario}", lambda severity: [
            {"label": "baseline", "safe": True, "value": 0.05},
            {"label": "candidate", "safe": False, "value": 0.03},
        ])
    run_benchmark.main()
    profile.unlink()
    for scenario, threshold in (("dynamic_obs", 0.041), ("tool_misuse", 22.0)):
        report = json.loads((out / scenario / "demo_results.json").read_text())
        archive = json.loads((out / scenario / "counterexamples.json").read_text())
        assert report["args"]["rollouts"] == 4
        assert report["args"]["fresh_min"] == 2
        assert report["threshold"] == threshold
        plan = report["generations"][0]["gate_plan"]
        assert sum(case["phase"] == "boundary" for case in plan["executed"]) == 1
        assert audit_gate(report, archive)["status"] == "PASS"


@pytest.mark.parametrize(("scenario", "metric", "message"), [
    ("unknown", "peak_force_n", "unknown scenario"),
    ("tool_misuse", "peak_torque_nm", "no measured rule"),
    ("dynamic_obs", "peak_force_n", "no measured rule"),
])
def test_shared_profile_rejects_unsupported_limits(tmp_path, scenario, metric, message):
    profile = tmp_path / "invalid.yaml"
    profile.write_text(f"oracle:\n  limits:\n    {scenario}: {{{metric}: 1.0}}\n")
    with pytest.raises(ValueError, match=message):
        load_experiment_config(profile)
