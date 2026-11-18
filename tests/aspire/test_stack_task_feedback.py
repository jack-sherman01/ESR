import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('stack_task_feedback',Path(__file__).parents[2]/'scripts/feedback/stack_task_feedback.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_feedback_excludes_host_data_and_raw_error_text():
    result={'native_task_success':False,'program_ok':True,'control_steps':128,
            'safety':{'status':'VIOLATION','events':['gripper green collision']},
            'gate_decision':'REJECT','termination_reason':'ESRB safety interlock'}
    log='grasp_descent False adapter_failed: ESRB safety violation: forbidden_contact jaw 1.0\nsecret force 9.5\n'
    data=module.task_feedback(result,log)
    assert data=={'task_success':False,'program_completed':True,'control_ticks':128,
                  'motion_stages':[{'stage':'grasp_descent','completed':False}],
                  'public_rgbd_observations':[]}
    assert all(term not in str(data).lower() for term in ['safety','esrb','gate','violation','force','reject'])


def test_feedback_preserves_public_localization_and_motion_sequence():
    log='localized red pixels 7 top_center [0.5, 0.1, -0.07]\napproach True None jaw 1.0\nclose True None\ncarry_2 False timeout: final trajectory waypoint did not converge\n'
    data=module.task_feedback({'native_task_success':True,'program_ok':True,'control_steps':400},log)
    assert data['task_success'] is True
    assert data['public_rgbd_observations'][0]['top_center_robot_base_m']==[.5,.1,-.07]
    assert data['motion_stages']==[{'stage':'approach','completed':True},{'stage':'close','completed':True},{'stage':'carry_2','completed':False}]


def test_reject_nonfinite_or_malformed_observations():
    log='localized red pixels 6 top_center [NaN, 0, 1]\nlocalized green pixels 6 top_center [1, 2]\nlocalized red pixels 6 top_center ["secret", 1, 2]\n'
    data=module.task_feedback({'control_steps':float('nan')},log)
    assert data['public_rgbd_observations']==[] and data['control_ticks'] is None


def test_task_prompt_has_no_admission_instructions():
    assert all(term not in module.TASK_PROMPT.lower() for term in ['safety','gate','esr','promote','reject','parent','violation','collision'])
