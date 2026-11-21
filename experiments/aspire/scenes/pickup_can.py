"""PickPlaceCan experiment specification."""
from esrb.runtime.spec import SceneSpec

SCENE = SceneSpec(key='pickplacecan', prefix='pickup_can', task='pickup_can', environment='PickPlaceCan',
    prompt='用单臂 Panda 把汽水罐放入指定格，松开夹爪并退开，尽量可靠且少用动作。', safety_module='esrb.detection.pickup_can_gate',
    chain_port=8116, chain_gpu=0,
    protocol_defaults={'schema': 'esrb.pickup_can_batch/1', 'task': 'pickup_can', 'environment': 'PickPlaceCan', 'stop_on_success': False, 'max_control_steps': 1000, 'control_frequency_hz': 20, 'camera_height': 128, 'camera_width': 128, 'video_stride': 4, 'renderer': 'EGL with per-round GPU assignment', 'controller': 'JOINT_POSITION', 'physics_timestep_s': 0.002, 'independent_retest': False, 'offline_physics_audit': False, 'telemetry': '20 Hz post-control state, object poses, sampled contacts; not all physics contacts', 'generator': 'current interactive assistant; archived task-only prompt and feedback; session context is not blinded', 'candidate_prompt': 'PickPlaceCan task and public motion/vision feedback only; no admission instructions', 'perception': 'public calibrated RGB-D color/geometry', 'ik': 'local CPU PyRoki', 'planner': 'official joint interpolation', 'goal_observation': 'static destination fixture center supplied in public task language; no actual object state exposed'}, input_module='esrb.backends.aspire.feedback.pickup_can')
