"""Engineering cases distinguish seating, ordinary failure, drop and contact."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
from esrb.tools.diagnostics.nut_evidence import Capture


def evaluator(nut=(0., 0., .81), open_fraction=1.):
    instance = Capture.__new__(Capture)
    data = SimpleNamespace(body_xpos=np.array([nut, (0., 0., .85)]), qpos=np.array([open_fraction*.04]),
                           body_xmat=np.array([np.eye(3).reshape(9)] * 2), time=0.,
                           get_site_xpos=lambda name: np.array([0., 0., 1.1]))
    instance.sim = SimpleNamespace(data=data)
    instance.nut, instance.peg, instance.table_z = 0, 1, .8
    instance.finger_qpos = 0
    instance.adapter = SimpleNamespace(get_robot_state=lambda: SimpleNamespace(gripper_positions={"primary": open_fraction}))
    instance.env = SimpleNamespace(robots=[SimpleNamespace(gripper=SimpleNamespace(important_sites={"grip_site": "eef"}))])
    instance.protocol = dict(peg_halfwidth_m=.016, nut_hole_halfwidth_m=.02275, nut_halfthickness_m=.01,
                             task_max_tilt_rad=.0873, task_seat_tolerance_m=.005)
    instance.grasped = instance.lifted = instance.drop = False
    return instance


def contact(first, second):
    return dict(group1=first, group2=second, pair="--".join(sorted([first, second])), wrench_contact_raw=[1.,0.,0.,0.,0.,0.])


def test_seated_released_nut_and_ordinary_incomplete_insertion_are_separate():
    complete = evaluator().task_frame([])
    incomplete = evaluator(nut=(0.,0.,.87)).task_frame([])
    assert complete["independent_task_frame"]
    assert not incomplete["independent_task_frame"]
    assert not incomplete["drop_observed"] and incomplete["unexpected_contacts"] == []


def test_closed_gripper_does_not_qualify_a_seated_nut():
    assert not evaluator(open_fraction=0.).task_frame([])["independent_task_frame"]


def test_nut_peg_contact_is_expected_but_gripper_peg_contact_is_flagged():
    assert evaluator().task_frame([contact("square_nut", "square_peg")])["unexpected_contacts"] == []
    assert evaluator().task_frame([contact("gripper", "square_peg")])["unexpected_contacts"] == ["gripper--square_peg"]


def test_lifted_nut_dropped_off_peg_is_flagged():
    instance = evaluator(nut=(.1,0.,.88))
    instance.task_frame([contact("gripper", "square_nut")])
    assert instance.lifted
    instance.sim.data.body_xpos[0] = (.1, 0., .81)
    assert instance.task_frame([contact("square_nut", "table")])["drop_observed"]
