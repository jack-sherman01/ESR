from __future__ import annotations

from typing import Any

from esrb.benchmark.hazard import HazardType
from esrb.benchmark.instance import BenchmarkInstance
from esrb.benchmark.lineage import GenerationRecord, LineageTracker
from esrb.detection.change_impact import ChangeImpactGraph, ComponentType
from esrb.detection.counterexample import Counterexample, CounterexampleLibrary
from esrb.detection.risk_selection import CandidateEvaluation, RiskSensitiveSelector
from esrb.oracles.base import OracleVerdict
from esrb.utils.hashing import hash_object
from .base import BaseAgent


# Maps each HazardType to the components it touches (for change-impact analysis).
_HAZARD_COMPONENTS: dict[HazardType, set[ComponentType]] = {
    HazardType.REWARD_HACKING:         {ComponentType.POLICY, ComponentType.REWARD},
    HazardType.EVALUATOR_DRIFT:        {ComponentType.EVALUATOR},
    HazardType.MEMORY_CONTAMINATION:   {ComponentType.MEMORY},
    HazardType.POLICY_REGRESSION:      {ComponentType.POLICY},
    HazardType.DISTRIBUTION_SHIFT:     {ComponentType.DATA},
    HazardType.TOOL_PERMISSION_MISUSE: {ComponentType.TOOLS},
    HazardType.SAFETY_FORGETTING:      {ComponentType.MEMORY, ComponentType.DATA},
}


class RolloutResult:
    __slots__ = ("task_score", "oracle_verdict", "episode_data")

    def __init__(
        self,
        task_score: float,
        oracle_verdict: OracleVerdict,
        episode_data: dict[str, Any],
    ) -> None:
        self.task_score = task_score
        self.oracle_verdict = oracle_verdict
        self.episode_data = episode_data


class EvolutionLoop:
    """
    CE-EST agentic improvement loop.

    Per generation:
      1. Collect rollouts with the current policy; query the independent oracle.
      2. Add any new safety violations to the counterexample library.
      3. Apply the benchmark's update operator to produce a candidate policy.
      4. Run change-impact analysis to determine which safety tests to rerun.
      5. Evaluate the candidate with the independent oracle (N rollouts).
      6. Risk-sensitive update selection (UCB safety constraint).
      7. Record an immutable GenerationRecord in the lineage tracker.
      8. Advance to the selected candidate if it passes; otherwise retain current.
    """

    def __init__(
        self,
        instance: BenchmarkInstance,
        agent: BaseAgent,
        selector: RiskSensitiveSelector | None = None,
        n_candidate_rollouts: int = 5,
    ) -> None:
        self.instance = instance
        self.agent = agent
        self.lineage = LineageTracker()
        self.cx_library = CounterexampleLibrary()
        self.impact_graph = ChangeImpactGraph()
        self.selector = selector or RiskSensitiveSelector()
        self.n_candidate_rollouts = n_candidate_rollouts

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self) -> LineageTracker:
        proto = self.instance.generation_protocol

        for gen in range(proto.num_generations):
            # ── 1. Evaluate current policy ────────────────────────────────────
            results = [
                self._run_rollout(self.agent, proto.seed + gen * 1000 + r, gen)
                for r in range(proto.rollouts_per_generation)
            ]
            mean_score = sum(r.task_score for r in results) / len(results)
            any_unsafe = any(not r.oracle_verdict.safe for r in results)
            last_signals = results[-1].oracle_verdict.signal_values

            # ── 2. Counterexample library update ──────────────────────────────
            for r in results:
                if not r.oracle_verdict.safe:
                    cx = Counterexample.from_episode(
                        episode_data=r.episode_data,
                        hazard_type=self.instance.hazard_model.hazard_type,
                        generation=gen,
                        violated_constraints=r.oracle_verdict.violated_constraints,
                        trigger_component=ComponentType.POLICY.value,
                    )
                    self.cx_library.add(cx)

            # ── 3–4. Update operator + change-impact analysis ─────────────────
            update_result = self.instance.update_operator.apply(
                self.agent, seed=proto.seed + gen
            )
            candidate = update_result.candidate_policy
            changed = _HAZARD_COMPONENTS.get(update_result.hazard_type, {ComponentType.POLICY})
            required_tests = self.impact_graph.required_tests(changed)

            # ── 5. Evaluate candidate with oracle ─────────────────────────────
            cand_results = [
                self._run_rollout(candidate, proto.seed + gen * 1000 + r, gen)
                for r in range(self.n_candidate_rollouts)
            ]
            cand_eval = CandidateEvaluation(
                policy=candidate,
                task_scores=[r.task_score for r in cand_results],
                safety_violations=[not r.oracle_verdict.safe for r in cand_results],
                generation=gen,
            )

            # ── 6. Risk-sensitive update selection ────────────────────────────
            selected = self.selector.select([cand_eval])

            # ── 7. Record immutable lineage entry ─────────────────────────────
            rec = GenerationRecord(
                generation=gen,
                policy_digest=hash_object(self.agent.state_dict()),
                data_digest="",
                evaluator_digest="",
                update_operator=self.instance.update_operator.name,
                task_score=mean_score,
                safety_violated=any_unsafe,
                oracle_verdicts=last_signals,
            )
            self.lineage.record(rec)

            # ── 8. Advance to selected candidate (or stay put) ────────────────
            if selected is not None:
                self.agent.load_state_dict(candidate.state_dict())

        return self.lineage

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_rollout(
        self,
        agent: BaseAgent,
        seed: int,
        generation: int,
    ) -> RolloutResult:
        obs = self.instance.task.reset(seed=seed)
        episode_data: dict[str, Any] = {}
        done = False
        while not done:
            action = agent.act(obs)
            obs, _reward, done, episode_data = self.instance.task.step(action)
        # episode_data is the full accumulated log from the final step
        result = self.instance.task.evaluate(episode_data)
        oracle_v = self.instance.oracle.evaluate(episode_data, generation)
        return RolloutResult(
            task_score=result.task_score,
            oracle_verdict=oracle_v,
            episode_data=episode_data,
        )
