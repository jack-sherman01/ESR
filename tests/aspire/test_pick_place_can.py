import pytest

from esrb.detection.pick_place_can import StablePlacementWindow


def stable_sample(window, **overrides):
    sample = {
        "dt": 0.002, "in_goal": True, "gripper_contact": False,
        "upright": True, "still": True, "target_support": True,
    }
    sample.update(overrides)
    window.observe(**sample)


def test_terminal_placement_requires_full_window_and_final_support():
    window = StablePlacementWindow(required_s=0.20)
    for index in range(99):
        stable_sample(window, target_support=index != 70)
    assert not window.passed

    stable_sample(window, target_support=False)
    assert not window.passed

    stable_sample(window)
    assert window.passed


@pytest.mark.parametrize(
    "unstable", [{"in_goal": False}, {"gripper_contact": True}, {"upright": False}, {"still": False}],
)
def test_unstable_sample_restarts_terminal_window(unstable):
    window = StablePlacementWindow(required_s=0.20)
    stable_sample(window, dt=0.20)
    assert window.passed

    stable_sample(window, **unstable)
    stable_sample(window)
    assert not window.passed
    assert window.elapsed_s == pytest.approx(0.002)
