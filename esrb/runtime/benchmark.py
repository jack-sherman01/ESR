"""Shared controlled-benchmark execution, scheduling and evidence accounting."""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


from esrb.agent import (
    DynamicObstaclePolicy, EvolutionLoop, GraspForcePolicy, MLPInsertionPolicy,
    PickPlacePolicy, ScriptedInsertionPolicy, SensorDriftPolicy,
    StochasticForcePolicy, ToolTightenPolicy, ToolHandlingPolicy, AssemblyPolicy,
)
from esrb.agent_api.binding import apply_proposal
from esrb.agent_api.contracts import CandidateProposer, PublicFeedback, ProposalError, identity, require
from esrb.agent_api.controlled import ControlledProfile, TrustedOperatorProposer
from esrb.agent_api.policy_sources import OperatorProposer
from esrb.agent_api.profiles import BY_INSTANCE as PROPOSAL_INSTANCES, BY_SCENARIO as PROPOSAL_SCENARIOS
from esrb.benchmark import (
    BenchmarkInstance, GenerationProtocol, HazardModel, HazardType,
)
from esrb.benchmark.experiment_config import LegacyPegConfig, load_experiment_config
from esrb.benchmark.lineage import GenerationRecord
from esrb.detection.change_impact import ComponentType
from esrb.detection.counterexample import Counterexample, CounterexampleSource
from esrb.detection.gate_checks import available_checks, execute_gate_checks, has_oracle_evidence
from esrb.detection.gate_planner import plan_gate
from esrb.detection.minimize import minimize
from esrb.detection.gate_audit import TASKS_BY_NAME, audit_gate
from esrb.detection.gate_cases import boundary_catalog, control_steps, gate_case
from esrb.detection.risk_sampling import risk_cases, risk_summary, sampling_protocol
from esrb.detection.risk_selection import CandidateEvaluation, RiskSensitiveSelector
from esrb.metrics.protocol import load_protocol, freeze_evaluation, validate_frozen
from esrb.metrics.reference import collect_reference
from esrb.metrics.reporting import evaluate_report, print_metrics
from esrb.oracles import ForceTorqueOracle, HeightClearanceOracle, ProximityOracle, VelocityOracle
from esrb.oracles.multiconstraint import SCENARIO_LIMITS, MultiConstraintOracle, scenario_oracle, verdict_safety
from esrb.tasks.agentic import GraspForceTask, SensorDriftTask, StochasticForceTask
from esrb.tasks.contact_rich import PegInsertionTask
from esrb.tasks.dynamic_workspace import DynamicObstacleTask
from esrb.tasks.long_horizon import PickPlaceTask, LongHorizonAssemblyTask
from esrb.tasks.tool_handling import ToolTightenTask, ToolHandlingTask
from esrb.update_operators import (
    DistributionShiftOperator, EpisodeRemovalOperator, EvaluatorPerturbationOperator,
    ForceSpeedInjectionOperator, MemoryContaminationOperator,
    RewardPerturbationOperator, SensorDriftOperator, ShortcutBiasOperator,
    StochasticForceOperator, ToolMisuseOperator, ToolGripRelaxationOperator, AssemblyOrientationOperator,
)
from esrb.utils.hashing import hash_dict, hash_object

from esrb.tasks.extensions.registry import EXTENSIONS,ExtensionOperator
from esrb.tasks.extensions.settings import configured_task
ALL_SCENARIOS = list(SCENARIO_LIMITS)
_TEACHER_SEED_START = 2000

_ORACLE_TEST = {
    VelocityOracle: "dynamics_safety",
    ProximityOracle: "dynamics_safety",
    HeightClearanceOracle: "dynamics_safety",
    ForceTorqueOracle: "force_safety",
}


# ── Scenario builders ─────────────────────────────────────────────────────────

def _changed_components(parent, candidate, parent_state, candidate_state):
    """Classify actual changes to policy state and MLP training memory."""
    field_components = {
        "tool_permissions": ComponentType.TOOLS,
        "evaluator_threshold": ComponentType.EVALUATOR,
        "buffer_len": ComponentType.MEMORY,
        "summary_len": ComponentType.MEMORY,
    }
    changed = set()
    for field in parent_state.keys() | candidate_state.keys():
        if parent_state.get(field) != candidate_state.get(field):
            component = (ComponentType.REWARD if field.endswith("_penalty_weight")
                         else field_components.get(field, ComponentType.POLICY))
            changed.add(component)
    if isinstance(parent, MLPInsertionPolicy) and (
        parent.replay_buffer != candidate.replay_buffer
        or parent.episode_summary != candidate.episode_summary
    ):
        changed.add(ComponentType.MEMORY)
    return changed

def _record_gate_violation(loop, result, case, gen, policy_digest, task_config=None):
    """Keep each unsafe physical-oracle result with its reset and applied task config."""
    if verdict_safety(result.oracle_verdict) is not False:
        return None
    cx = Counterexample.from_episode(
        episode_data=result.episode_data,
        hazard_type=loop.instance.hazard_model.hazard_type,
        generation=gen,
        violated_constraints=result.oracle_verdict.violated_constraints,
        trigger_component=ComponentType.ROBOT_INTERFACE.value,
        source=CounterexampleSource(
            scenario=loop.instance.name,
            reset_key=case["reset_key"], seed=case["seed"], phase=case["phase"],
            policy_digest=policy_digest, task_config=task_config, disturbance=case["disturbance"],
        ),
    )
    loop.cx_library.add(cx)
    return cx.id

def _run_structured_case(loop, policy, case, generation):
    point = case["disturbance"]
    result = loop._run_rollout(
        policy, case["seed"], generation, reset_options={"disturbance": point},
    )
    if result.episode_data["reset_readback"] != point:
        raise ValueError("task did not restore the requested reset disturbance")
    return result

class _ReductionStopped(Exception):
    """The probe series cannot retain a newly verified violation."""

def _reduce_panel(task, plan, executed, results, execute_case):
    """Spend reserved slots on measured reduction probes, then fill with fresh cases."""
    initial = next((
        row for row, result in zip(executed, results, strict=True)
        if row["phase"] != "risk" and not result.oracle_verdict.safe and has_oracle_evidence(result)
        and any(abs(value - task.reset_reference[field]) > 1e-6
                for field, value in row["disturbance"].items())
    ), None)
    next_seed = 1 + max(case["seed"] for case in plan["cases"] if case["phase"] not in {"replay", "risk"})

    def run_extra(case):
        plan["cases"].append(case)
        return execute_case(case)

    reduction = {"status": "no_off_reference_unsafe_case", "evaluations": 0}
    if initial is not None:
        history = []

        def probe(point):
            result = run_extra(gate_case(
                task, initial["seed"], "reduction", point, origin_evidence_id=initial["evidence_id"],
            ))
            unsafe = not result.oracle_verdict.safe if has_oracle_evidence(result) else None
            evidence = {"unsafe": unsafe, "evidence_id": executed[-1]["evidence_id"]}
            history.append({"environment": point, "result": evidence})
            if unsafe is None:
                raise _ReductionStopped("missing_probe_evidence")
            if len(history) == 1 and not unsafe:
                raise _ReductionStopped("witness_not_reproduced")
            return evidence

        try:
            reduction = minimize(
                initial["disturbance"], task.reset_reference, probe,
                budget=plan["reduction_reserved"],
            )
            reduction["status"] = "verified"
        except _ReductionStopped as exc:
            reduction = {
                "status": str(exc), "evaluations": len(history), "history": history,
                "original": initial["disturbance"],
            }
        reduction["origin_evidence_id"] = initial["evidence_id"]
    for index in range(plan["reduction_reserved"] - reduction["evaluations"]):
        run_extra(gate_case(task, next_seed + index, "fresh"))
    return reduction

def _raw_gate_result(result, case, policy_digest):
    return {
        "policy_digest": policy_digest, "case": case,
        "episode_data": result.episode_data, "oracle": asdict(result.oracle_verdict),
        "task_score": result.task_score, "task_success": result.task_success,
    }

def _panel_safety(labels):
    if False in labels:
        return False
    return True if labels and None not in labels else None

def run_loop(instance, agent, selector, severity, n_rollouts, *,
             gate_planner="fixed", fresh_min=1, training_seed_stop=0, replay_all=False,
             peg_boundary_cases=0, peg_reduction_budget=0, boundary_cases=0, reduction_budget=0,
             proposer: CandidateProposer | None = None, risk_protocol=None, evaluation_protocol=None):
    """Run the CE-EST loop, returning (EvolutionLoop, list[gen_dict])."""
    use_impact = gate_planner == "impact"
    task = instance.task
    if evaluation_protocol is not None:
        validate_frozen(evaluation_protocol)
        frozen_settings=evaluation_protocol['settings']
        require(use_impact and isinstance(instance.oracle,MultiConstraintOracle),
                'reference evaluation requires multi-constraint impact execution')
        require(frozen_settings['oracle']==instance.oracle.protocol()
                and frozen_settings['generations']==instance.generation_protocol.num_generations
                and frozen_settings['budgets']['candidate_per_generation']==n_rollouts
                and frozen_settings['gate_sampling']==risk_protocol
                and frozen_settings['gate_selector']['epsilon']==selector.epsilon
                and frozen_settings['gate_selector']['alpha']==selector.alpha,
                'runtime differs from frozen evaluation protocol')
    multi = instance.oracle if isinstance(instance.oracle, MultiConstraintOracle) else None
    if risk_protocol is not None and multi is None:
        raise ValueError("independent risk sampling requires the multi-constraint impact gate")
    physical_tests = multi.physical_tests if multi is not None else frozenset()
    if instance.name == "scripted_peg_speed_injection":
        boundary_cases = boundary_cases or peg_boundary_cases
        reduction_budget = reduction_budget or peg_reduction_budget
    policy_profile = PROPOSAL_INSTANCES.get(instance.name)
    if policy_profile is None:
        raise ProposalError("this scenario has no registered proposal consumer")
    controlled = isinstance(policy_profile, ControlledProfile)
    if controlled and proposer is not None:
        raise ProposalError("this scenario only accepts the built-in operator")
    if not controlled and proposer is None:
        proposer = OperatorProposer(
            instance.update_operator, instance.generation_protocol.seed, severity, policy_profile,
        )
    manifest_id = proposer.manifest.id if proposer is not None else None
    public_history = []
    parent_config = policy_profile.nominal_task_config if controlled else None
    if parent_config is not None:
        task.configure(**parent_config)
    loop  = EvolutionLoop(instance, agent, selector=selector, n_candidate_rollouts=n_rollouts)
    proto = instance.generation_protocol
    gen_rows: list[dict[str, Any]] = []
    fresh_start = max(
        training_seed_stop,
        proto.seed + (proto.num_generations - 1) * 1000 + proto.rollouts_per_generation,
    )

    clock_start = time.perf_counter()
    execution_events = []

    def record_event(result, role, case, *, task=task):
        steps = control_steps(result.episode_data)
        physics = None
        if hasattr(task, '_model') and hasattr(task, '_data'):
            physics = int(round(float(task._data.time) / float(task._model.opt.timestep)))
        execution_events.append({
            'generation': gen, 'role': role, 'phase': case.get('phase',role),
            'seed': case['seed'], 'control_steps': steps, 'physics_steps': physics,
            'safe': verdict_safety(result.oracle_verdict) if has_oracle_evidence(result) else None,
            'elapsed_s': time.perf_counter()-clock_start,
        })

    for gen in range(proto.num_generations):
        event_start = len(execution_events)
        source = proposer
        if controlled:
            source = TrustedOperatorProposer(policy_profile, instance.update_operator, loop.agent, proto.seed, severity)
        if manifest_id is None:
            manifest_id = source.manifest.id
        feedback = PublicFeedback(instance.name, gen + 1, tuple(public_history))
        try:
            child, receipt = apply_proposal(
                source, manifest_id, policy_profile.snapshot(loop.agent), feedback, policy_profile.validators,
            )
            if controlled:
                candidate, candidate_config = source.consume(child)
            else:
                candidate, candidate_config = policy_profile.build(child["policy"]), None
        except ProposalError as exc:
            raise ProposalError(f"generation {gen + 1}: {exc}") from exc

        parent_seed = proto.seed + gen * 1000
        parent_seeds = range(parent_seed, parent_seed + proto.rollouts_per_generation)
        parent_cases = [gate_case(task, seed, "parent") if use_impact else
                        {'seed':seed,'phase':'parent'} for seed in parent_seeds]
        results = []
        for case in parent_cases:
            measured = (_run_structured_case(loop, loop.agent, case, gen) if use_impact else
                        loop._run_rollout(loop.agent, case['seed'], gen))
            results.append(measured)
            record_event(measured,'parent',case)
        mean_score = sum(r.task_score for r in results) / len(results)
        baseline_labels = [verdict_safety(r.oracle_verdict) for r in results]
        any_unsafe = False in baseline_labels
        last_sigs  = results[-1].oracle_verdict.signal_values

        parent_state = loop.agent.state_dict()
        parent_digest = hash_object(parent_state)
        for offset, r in enumerate(results):
            if use_impact:
                _record_gate_violation(loop, r, parent_cases[offset], gen, parent_digest, parent_config)
            elif verdict_safety(r.oracle_verdict) is False:
                cx = Counterexample.from_episode(
                    episode_data=r.episode_data,
                    hazard_type=instance.hazard_model.hazard_type,
                    generation=gen,
                    violated_constraints=r.oracle_verdict.violated_constraints,
                    trigger_component=ComponentType.POLICY.value,
                )
                loop.cx_library.add(cx)

        plan = None
        executed = []
        check_results, check_blockers = {}, []
        reduction = None
        if use_impact:
            candidate_state = candidate.state_dict()
            candidate_digest = hash_object(candidate_state)
            changed = _changed_components(loop.agent, candidate, parent_state, candidate_state)
            if candidate_config != parent_config:
                changed.add(ComponentType.DATA)
            primary_test = multi.primary_test if multi is not None else _ORACLE_TEST[type(instance.oracle)]
            checks = available_checks(
                instance.task, primary_test, loop.agent, candidate, parent_state, candidate_state,
                physical_tests=physical_tests,
            )
            plan = plan_gate(
                loop.impact_graph, loop.cx_library, changed=changed, scenario=instance.name,
                generation=gen, budget=n_rollouts, fresh_min=fresh_min,
                fresh_seed=fresh_start + gen * n_rollouts,
                supported_tests=set(checks),
                replay_all=replay_all,
                replay_seed=proto.seed,
                boundary_catalog=boundary_catalog(task, primary_test) if boundary_cases else (),
                boundary_limit=boundary_cases, reduction_budget=reduction_budget,
                risk_cases=risk_cases(task, risk_protocol, instance.name, gen) if risk_protocol is not None else (),
            )
            cand_results = []

            def execute_case(case):
                case.update(gate_case(task, case["seed"], case["phase"], case.get("disturbance")))
                result = _run_structured_case(loop, candidate, case, gen)
                record_event(result,'candidate',case)
                cx_id = _record_gate_violation(
                    loop, result, case, gen, candidate_digest, candidate_config,
                )
                cand_results.append(result)
                raw = _raw_gate_result(result, case, candidate_digest)
                executed.append({
                    **case, "safe": verdict_safety(result.oracle_verdict),
                    "signal_values": result.oracle_verdict.signal_values,
                    "new_counterexample_id": cx_id, "raw_result": raw, "evidence_id": hash_dict(raw),
                    "control_steps": control_steps(result.episode_data),
                })
                return result

            try:
                if candidate_config is not None:
                    instance.task.configure(**candidate_config)
                for case in plan["cases"]:
                    execute_case(case)
                if plan["reduction_reserved"]:
                    reduction = _reduce_panel(task, plan, executed, cand_results, execute_case)
            finally:
                if parent_config is not None:
                    instance.task.configure(**parent_config)
            check_results, check_blockers = execute_gate_checks(
                plan, checks, cand_results, primary_test, physical_tests=physical_tests,
            )
        else:
            try:
                if candidate_config is not None:
                    task.configure(**candidate_config)
                cand_results = []
                for r in range(loop.n_candidate_rollouts):
                    case={'seed':parent_seed+r,'phase':'fixed'}
                    measured=loop._run_rollout(candidate,case['seed'],gen)
                    cand_results.append(measured)
                    record_event(measured,'candidate',case)
            finally:
                if parent_config is not None:
                    task.configure(**parent_config)

        feedback_results = [
            result for case, result in zip(plan["cases"], cand_results, strict=True)
            if case["phase"] not in {"reduction", "risk"}
        ] if use_impact else cand_results
        risk_results = feedback_results
        independent_risk = None
        if risk_protocol is not None:
            independent_risk = risk_summary(executed, gen, selector.alpha)
            risk_results = [
                result for case, result in zip(plan["cases"], cand_results, strict=True)
                if case["phase"] == "risk"
            ]
            if independent_risk["regression_violations"]:
                check_blockers.append("regression_violation")
        if multi is not None and any(
            r.oracle_verdict.missing_required for r in results + cand_results
        ):
            check_blockers.append("measurement_unknown")
        selected = None
        if risk_results and not check_blockers:
            cand_eval = CandidateEvaluation(
                policy=candidate,
                task_scores=[r.task_score for r in risk_results],
                safety_violations=[not r.oracle_verdict.safe for r in risk_results],
                generation=gen,
            )
            selected = selector.select([cand_eval])

        # Gate selection is already fixed. These trials never enter its inputs.
        reference_panel = None
        if evaluation_protocol is not None:
            reference_panel = collect_reference(
                instance,loop.agent,candidate,gen,evaluation_protocol,profile=policy_profile,
                receipt=receipt,parent_config=parent_config,candidate_config=candidate_config,
                encode=_raw_gate_result,record_event=record_event,
            )

        loop.lineage.record(GenerationRecord(
            generation=gen,
            policy_digest=parent_digest,
            data_digest="",
            evaluator_digest="",
            update_operator=source.manifest.adapter_id,
            task_score=mean_score,
            safety_violated=any_unsafe,
            oracle_verdicts=last_sigs,
        ))

        promoted = selected is not None and parent_config is None
        active_version = receipt["base_version"]
        if promoted:
            if isinstance(loop.agent, MLPInsertionPolicy):
                # Carry the validated weights, memory, optimizer and RNG together.
                loop.agent = candidate
                active_version = receipt["candidate_version"]
            else:
                loop.agent.load_state_dict(candidate.state_dict())
                active_version = identity(policy_profile.snapshot(loop.agent))
                require(active_version == receipt["candidate_version"], "promoted state differs from proposal")

        candidate_labels = [verdict_safety(r.oracle_verdict) for r in cand_results]
        row = {
            "generation":     gen,
            "baseline_score": mean_score,
            "baseline_safe":  _panel_safety(baseline_labels),
            "baseline_sigs":  last_sigs,
            "candidate_safe": _panel_safety(candidate_labels),
            "candidate_sigs": cand_results[-1].oracle_verdict.signal_values if cand_results else {},
            "accepted":       selected is not None,
            # per-rollout data for multi-metric computation
            "baseline_rollout_safe":    baseline_labels,
            "baseline_rollout_success": [
                bool(r.episode_data.get("success", r.task_score > 0.5))
                for r in results
            ],
            "candidate_rollout_safe":    candidate_labels,
            "candidate_rollout_success": [
                bool(r.episode_data.get("success", r.task_score > 0.5))
                for r in cand_results
            ],
        }
        if use_impact:
            baseline_raw = [
                _raw_gate_result(result, case, parent_digest)
                for result, case in zip(results, parent_cases, strict=True)
            ]
            row["baseline_measurements"] = [
                {"raw_result": raw, "evidence_id": hash_dict(raw)} for raw in baseline_raw
            ]
        if plan is not None:
            row["gate_plan"] = {
                **plan, "candidate_policy_digest": candidate_digest,
                "executed": executed, "used_rollouts": len(executed),
                "test_results": check_results, "blocking_tests": check_blockers,
                "counterexamples_after_generation": len(loop.cx_library),
            }
            if independent_risk is not None:
                row["gate_plan"]["independent_risk"] = independent_risk
            if candidate_config is not None:
                row["gate_plan"]["candidate_task_config"] = candidate_config
            row["gate_plan"]["reduction"] = reduction
            row["gate_plan"]["used_control_steps"] = sum(case["control_steps"] for case in executed)
        row["proposal_receipt"] = {
            **receipt,
            "accepted": row["accepted"], "promoted": promoted,
            "active_version_after": active_version,
        }
        row['execution_events'] = execution_events[event_start:]
        if reference_panel is not None:
            row['reference_evaluation'] = reference_panel
        public_scores = ([item['raw_result']['task_score'] for item in reference_panel['development']]
                         if reference_panel is not None else [r.task_score for r in feedback_results])
        public_history.append({
            "generation": gen + 1, "slot": 0, "task_scores": public_scores,
        })
        gen_rows.append(row)

    return loop, gen_rows

def _run_stochastic_force_loop(instance, agent, selector, severity, n_rollouts):
    """Custom CE-EST loop for stochastic_force: baseline uses fault_prob=0,
    candidate uses fault_prob=severity (stochastic: some rollouts may be safe)."""
    oracle    = instance.oracle
    operator  = instance.update_operator
    proto     = instance.generation_protocol
    gen_rows: list[dict[str, Any]] = []

    for gen in range(proto.num_generations):
        # Baseline: no fault
        instance.task.configure(fault_prob=0.0)
        results = []
        for r in range(proto.rollouts_per_generation):
            obs  = instance.task.reset(seed=proto.seed + gen * 1000 + r)
            ep: dict = {}
            done = False
            while not done:
                obs, _, done, ep = instance.task.step(agent.act(obs))
            task_res = instance.task.evaluate(ep)
            o_v      = oracle.evaluate(ep, gen)
            results.append((task_res.task_score, o_v, ep))

        mean_score  = sum(s for s, _, _ in results) / len(results)
        any_unsafe  = any(not v.safe for _, v, _ in results)
        last_b_sigs = results[-1][1].signal_values

        # Operator → stochastic fault config
        update_result = operator.apply(agent, seed=proto.seed + gen, severity=severity)
        candidate     = update_result.candidate_policy
        c_fault_prob  = update_result.metadata["fault_prob"]

        # Candidate rollouts: stochastic fault
        cand_task = StochasticForceTask()
        cand_task.configure(fault_prob=c_fault_prob)
        cand_results = []
        for r in range(n_rollouts):
            obs  = cand_task.reset(seed=proto.seed + gen * 1000 + r)
            ep: dict = {}
            done = False
            while not done:
                obs, _, done, ep = cand_task.step(candidate.act(obs))
            task_res = cand_task.evaluate(ep)
            o_v      = oracle.evaluate(ep, gen)
            cand_results.append((task_res.task_score, o_v, ep))

        last_c_sigs = cand_results[-1][1].signal_values
        cand_eval   = CandidateEvaluation(
            policy=candidate,
            task_scores=[s for s, _, _ in cand_results],
            safety_violations=[not v.safe for _, v, _ in cand_results],
            generation=gen,
        )
        selected = selector.select([cand_eval])

        gen_rows.append({
            "generation":     gen,
            "baseline_score": mean_score,
            "baseline_safe":  not any_unsafe,
            "baseline_sigs":  last_b_sigs,
            "candidate_safe": all(v.safe for _, v, _ in cand_results),
            "candidate_sigs": last_c_sigs,
            "accepted":       selected is not None,
            "baseline_rollout_safe":    [v.safe for _, v, _ in results],
            "baseline_rollout_success": [
                bool(ep.get("success", s > 0.5)) for s, _, ep in results
            ],
            "candidate_rollout_safe":    [v.safe for _, v, _ in cand_results],
            "candidate_rollout_success": [
                bool(ep.get("success", s > 0.5)) for s, _, ep in cand_results
            ],
        })

    class _FakeLoop:
        def __init__(self):
            from esrb.benchmark.lineage import LineageTracker
            from esrb.detection.counterexample import CounterexampleLibrary
            self.cx_library = CounterexampleLibrary()
            self.lineage    = LineageTracker()
        def digest(self): return "stochastic_force"

    fake_loop = _FakeLoop()
    for row in gen_rows:
        fake_loop.lineage.record(GenerationRecord(
            generation=row["generation"],
            policy_digest="",
            data_digest="",
            evaluator_digest="",
            update_operator=operator.name,
            task_score=row["baseline_score"],
            safety_violated=not row["baseline_safe"],
            oracle_verdicts=row["baseline_sigs"],
        ))

    return fake_loop, gen_rows

def _run_sensor_drift_loop(instance, agent, selector, severity, n_rollouts):
    """Custom CE-EST loop for sensor_drift: baseline uses accurate sensor,
    candidate uses drifted sensor (true_force=30, sensor_bias=-20)."""
    oracle    = instance.oracle
    operator  = instance.update_operator
    proto     = instance.generation_protocol
    gen_rows: list[dict[str, Any]] = []

    for gen in range(proto.num_generations):
        # ── Baseline rollouts: accurate sensor ────────────────────────────────
        instance.task.configure(true_force=10.0, sensor_bias=0.0)
        results = []
        for r in range(proto.rollouts_per_generation):
            obs  = instance.task.reset(seed=proto.seed + gen * 1000 + r)
            ep: dict = {}
            done = False
            while not done:
                obs, _, done, ep = instance.task.step(agent.act(obs))
            task_res = instance.task.evaluate(ep)
            o_v      = oracle.evaluate(ep, gen)
            results.append((task_res.task_score, o_v, ep))

        mean_score = sum(s for s, _, _ in results) / len(results)
        any_unsafe = any(not v.safe for _, v, _ in results)
        last_b_sigs = results[-1][1].signal_values

        # ── Apply operator → get drifted config ───────────────────────────────
        update_result = operator.apply(agent, seed=proto.seed + gen, severity=severity)
        candidate     = update_result.candidate_policy
        c_true_force  = update_result.metadata["true_force_n"]
        c_sensor_bias = update_result.metadata["sensor_bias_n"]

        # ── Candidate rollouts: drifted sensor ────────────────────────────────
        cand_task = SensorDriftTask()
        cand_task.configure(true_force=c_true_force, sensor_bias=c_sensor_bias)
        cand_results = []
        for r in range(n_rollouts):
            obs  = cand_task.reset(seed=proto.seed + gen * 1000 + r)
            ep: dict = {}
            done = False
            while not done:
                obs, _, done, ep = cand_task.step(candidate.act(obs))
            task_res = cand_task.evaluate(ep)
            o_v      = oracle.evaluate(ep, gen)
            cand_results.append((task_res.task_score, o_v, ep))

        last_c_sigs = cand_results[-1][1].signal_values
        cand_eval   = CandidateEvaluation(
            policy=candidate,
            task_scores=[s for s, _, _ in cand_results],
            safety_violations=[not v.safe for _, v, _ in cand_results],
            generation=gen,
        )
        selected = selector.select([cand_eval])

        gen_rows.append({
            "generation":     gen,
            "baseline_score": mean_score,
            "baseline_safe":  not any_unsafe,
            "baseline_sigs":  last_b_sigs,
            "candidate_safe": all(v.safe for _, v, _ in cand_results),
            "candidate_sigs": last_c_sigs,
            "accepted":       selected is not None,
            "baseline_rollout_safe":    [v.safe for _, v, _ in results],
            "baseline_rollout_success": [
                bool(ep.get("success", s > 0.5)) for s, _, ep in results
            ],
            "candidate_rollout_safe":    [v.safe for _, v, _ in cand_results],
            "candidate_rollout_success": [
                bool(ep.get("success", s > 0.5)) for s, _, ep in cand_results
            ],
        })

    class _FakeLoop:
        def __init__(self):
            from esrb.benchmark.lineage import LineageTracker
            from esrb.detection.counterexample import CounterexampleLibrary
            self.cx_library = CounterexampleLibrary()
            self.lineage    = LineageTracker()
        def digest(self): return "sensor_drift"

    fake_loop = _FakeLoop()
    for row in gen_rows:
        fake_loop.lineage.record(GenerationRecord(
            generation=row["generation"],
            policy_digest="",
            data_digest="",
            evaluator_digest="",
            update_operator=operator.name,
            task_score=row["baseline_score"],
            safety_violated=not row["baseline_safe"],
            oracle_verdicts=row["baseline_sigs"],
        ))

    return fake_loop, gen_rows
