import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from scripts.gates.nut_gate_rules import NutSafety, classify_geometry


def rules():
    return json.loads((ROOT / 'configs/experiment/nut_safety_gate.json').read_text())


def contact(*groups, force=1.):
    return {'groups': list(groups), 'normal_force_n': force, 'geom1': 1, 'geom2': 2}


def test_insertion_contact_is_allowed_but_gripper_peg_is_forbidden():
    monitor = NutSafety(rules(), .82)
    assert monitor.observe(0., .002, [0., 0., .9], [contact('square_nut', 'square_peg')], seated=False) is None
    event = monitor.observe(.002, .004, [0., 0., .9], [contact('gripper', 'square_peg')], seated=False)
    assert event['kind'] == 'forbidden_contact'
    assert event['physics_index'] == 1


@pytest.mark.parametrize('seated,expected', [(False, 'lifted_nut_returned_to_table_without_grasp'), (True, None)])
def test_seated_release_is_distinguished_from_drop(seated, expected):
    monitor = NutSafety(rules(), .82)
    monitor.observe(0., .002, [0., 0., .9], [contact('square_nut', 'gripper')], seated=False)
    event = monitor.observe(.002, .004, [0., 0., .83], [contact('square_nut', 'table')], seated=seated)
    assert (event['kind'] if event else None) == expected


def test_floor_contact_is_forbidden_before_lift():
    monitor = NutSafety(rules(), .82)
    assert monitor.observe(0., .002, [0., 0., .01], [contact('square_nut', 'floor')], seated=False)['kind'] == 'nut_on_floor'


def test_reset_airborne_settling_is_not_a_robot_lift_or_drop():
    monitor = NutSafety(rules(), .82)
    monitor.observe(0., .002, [0., 0., .9], [], seated=False)
    assert not monitor.lifted
    assert monitor.observe(.002, .004, [0., 0., .83], [contact('square_nut', 'table')], seated=False) is None


def test_missing_sampling_and_unclassified_robot_contact_cannot_pass():
    monitor = NutSafety(rules(), .82)
    with pytest.raises(ValueError, match='missing'):
        monitor.observe(.002, .004, [0., 0., .83], [], seated=False)
    monitor = NutSafety(rules(), .82)
    with pytest.raises(ValueError, match='unclassified'):
        monitor.observe(0., .002, [0., 0., .83], [contact('gripper', 'other')], seated=False)
    assert monitor.report(elapsed_physics_steps=1, records_written=0)['status'] == 'UNKNOWN'


def test_nut_and_gripper_peg_geometries_remain_distinct():
    assert classify_geometry('SquareNut_g0', ['SquareNut']) == 'square_nut'
    assert classify_geometry(None, ['peg1']) == 'square_peg'
    assert classify_geometry('gripper0_finger1_collision', ['gripper0_leftfinger', 'robot0_base']) == 'gripper'


def test_nut_long_chain_has_fifty_versions_without_smoke_or_retest_requirement(tmp_path):
    from scripts.runners import run_nut_long_chain as module
    root, capsule = tmp_path / 'trial', tmp_path / 'capsule'
    module.init(root, capsule, seed=6719229)
    protocol = module.read(root / 'protocol.json')
    module.state.load_rules(root)
    assert protocol['versions_per_round'] == 50
    assert protocol['task'] == 'nut_assembly'
    assert protocol['environment'] == 'NutAssemblySquare'
    assert protocol['seeds'] == [6719229]
    assert protocol['independent_retest'] is False
    assert 'smoke_reference' not in protocol
    assert 'scripts/runners/run_aspire_nut_batch.py' in protocol['safety_gate']['implementation_sha256']
    assert module.read(root / 'worker_contract.json')['initial_program'] is None
