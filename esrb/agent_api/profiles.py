"""Trusted policy consumers selected by the benchmark scenario."""

from esrb.agent.dynamic_obstacle_policy import DynamicObstaclePolicy
from esrb.agent.grasp_force_policy import GraspForcePolicy
from esrb.agent.mlp_policy import MLPInsertionPolicy
from esrb.agent.pick_place_policy import PickPlacePolicy
from esrb.agent.scripted_policy import ScriptedInsertionPolicy
from esrb.agent.sensor_drift_policy import SensorDriftPolicy
from esrb.agent.stochastic_force_policy import StochasticForcePolicy
from esrb.agent.rl_grasp_policy import RLGraspPolicy
from esrb.agent.tool_tighten_policy import ToolTightenPolicy
from esrb.agent.tool_handling_policy import ToolHandlingPolicy
from esrb.agent.assembly_policy import AssemblyPolicy
from .contracts import Consumer
from .controlled import ControlledProfile
from .policy_sources import ParameterPolicy

SCRIPTED_PEG = ParameterPolicy(
    "scripted peg", ScriptedInsertionPolicy,
    Consumer("policy", "esrb.scripted-peg.policy/1", "scripted-insertion-parameters/1"),
    "scripted_peg_json",
)
DYNAMIC_OBS = ParameterPolicy(
    "dynamic obstacle", DynamicObstaclePolicy,
    Consumer("policy", "esrb.dynamic-obstacle.policy/1", "dynamic-obstacle-parameters/1"),
    "dynamic_obs_json",
)
TOOL_HANDLING = ParameterPolicy(
    "tool handling", ToolHandlingPolicy,
    Consumer("policy", "esrb.tool-handling.policy/2", "panda-tool-handling-parameters/1"),
    "tool_handling_json",
)
ASSEMBLY = ParameterPolicy(
    "long horizon assembly", AssemblyPolicy,
    Consumer("policy", "esrb.assembly.policy/2", "panda-three-piece-parameters/1"),
    "long_horizon_assembly_json",
)
_MLP = ControlledProfile(MLPInsertionPolicy, ("policy", "memory"))
_EVALUATOR_FIELDS = {"evaluator": ("evaluator_threshold",)}
BY_INSTANCE = {
    "scripted_peg_speed_injection": SCRIPTED_PEG,
    "dynamic_obstacle_distribution_shift": DYNAMIC_OBS,
    "mlp_peg_memory_contamination": _MLP,
    "mlp_peg_episode_removal": _MLP,
    "pick_place_shortcut_bias": ControlledProfile(PickPlacePolicy, ("policy",), _EVALUATOR_FIELDS),
    "pick_place_evaluator_drift": ControlledProfile(PickPlacePolicy, ("evaluator",), _EVALUATOR_FIELDS),
    "tool_handling_grip_relaxation": TOOL_HANDLING,
    "long_horizon_assembly_orientation": ASSEMBLY,
    "tool_tighten_reward_perturbation": ControlledProfile(
        ToolTightenPolicy, ("reward",), {"reward": ("force_penalty_weight",)},
    ),
    "grasp_force_tool_misuse": ControlledProfile(
        GraspForcePolicy, ("tools",), {"tools": ("tool_permissions",)},
    ),
    "sensor_drift_calibration_fault": ControlledProfile(
        SensorDriftPolicy, ("data",), nominal_task_config={"true_force": 10.0, "sensor_bias": 0.0},
        task_config_metadata={"true_force": "true_force_n", "sensor_bias": "sensor_bias_n"},
    ),
    "stochastic_force_intermittent_fault": ControlledProfile(
        StochasticForcePolicy, ("data",), nominal_task_config={"fault_prob": 0.0},
        task_config_metadata={"fault_prob": "fault_prob"},
    ),
    "grasp_pressure_target_shift": ControlledProfile(
        RLGraspPolicy, ("data",), nominal_task_config={"target_pressure_n": 15.0},
        task_config_metadata={"target_pressure_n": "candidate_target_n"},
    ),
}
BY_SCENARIO = {"scripted_peg": SCRIPTED_PEG, "dynamic_obs": DYNAMIC_OBS, "tool_handling": TOOL_HANDLING, "long_horizon_assembly": ASSEMBLY}

from esrb.tasks.extensions.registry import EXTENSIONS
for _name, (_task, _policy) in EXTENSIONS.items():
    _revision=2
    _profile=ParameterPolicy(_name,_policy,Consumer("policy",f"esrb.{_name}.policy/{_revision}",f"{_name}-parameters/{_revision}"),_name+"_json")
    BY_SCENARIO[_name]=_profile
    BY_INSTANCE[_name+"_controlled_update"]=_profile
