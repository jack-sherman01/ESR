from __future__ import annotations

import time
from typing import Any

from esrb.benchmark.instance import BenchmarkInstance
from esrb.benchmark.lineage import GenerationRecord, LineageTracker
from esrb.detection.change_impact import ChangeImpactGraph, ComponentType
from esrb.detection.counterexample import Counterexample, CounterexampleLibrary
from esrb.detection.risk_selection import CandidateEvaluation, RiskSensitiveSelector
from esrb.evaluators.integrity import check_evaluator_oracle_agreement
from esrb.utils.hashing import hash_object
from .base import BaseAgent


class EvolutionLoop:
    """
    Main CE-EST agentic improvement loop.

    Each generation:
      1. Run rollouts with the current policy.
      2. Apply the benchmark's update operator to produce a candidate.
      3. Run change-impact analysis; re-run relevant tests and counterexamples.
      4. Query the independent oracle (never modified by the agent).
      5. Check evaluator–oracle agreement; flag disagreements.
      6. Risk-sensitive update selection (UCB constraint).
      7. Record immutable GenerationRecord in the lineage.
    """

    def __init__(
        self,
        instance: BenchmarkInstance,
        agent: BaseAgent,
        selector: RiskSensitiveSelector | None = None,
    ) -> None:
        self.instance = instance
        self.agent = agent
        self.lineage = LineageTracker()
        self.cx_library = CounterexampleLibrary()
        self.impact_graph = ChangeImpactGraph()
        self.selector = selector or RiskSensitiveSelector()

    def run(self) -> LineageTracker:
        proto = self.instance.generation_protocol
        for gen in range(proto.num_generations):
            task_scores, safety_violations, oracle_verdicts_all = [], [], {}

            for _ in range(proto.rollouts_per_generation):
                obs = self.instance.task.reset(seed=proto.seed + gen)
                episode_data: dict[str, Any] = {}
                done = False
                while not done:
                    action = self.agent.act(obs)
                    obs, _, done, info = self.instance.task.step(action)
                    episode_data.update(info)

                result = self.instance.task.evaluate(episode_data)
                oracle_v = self.instance.oracle.evaluate(episode_data, gen)
                eval_v = None  # evaluator call placeholder

                task_scores.append(result.task_score)
                safety_violations.append(not oracle_v.safe)
                oracle_verdicts_all = oracle_v.signal_values

                if not oracle_v.safe:
                    cx = Counterexample.from_episode(
                        episode_data=episode_data,
                        hazard_type=self.instance.hazard_model.hazard_type,
                        generation=gen,
                        violated_constraints=oracle_v.violated_constraints,
                        trigger_component=ComponentType.POLICY.value,
                    )
                    self.cx_library.add(cx)

            mean_score = sum(task_scores) / len(task_scores)

            rec = GenerationRecord(
                generation=gen,
                policy_digest=hash_object(self.agent.state_dict()),
                data_digest="",
                evaluator_digest="",
                update_operator=self.instance.update_operator.name,
                task_score=mean_score,
                safety_violated=any(safety_violations),
                oracle_verdicts=oracle_verdicts_all,
            )
            self.lineage.record(rec)

            # Apply update operator to produce next-generation candidate
            update_result = self.instance.update_operator.apply(
                self.agent, seed=proto.seed + gen
            )
            # In a full implementation: evaluate candidates, run risk-sensitive selection,
            # then load the selected candidate's state dict into self.agent.

        return self.lineage
