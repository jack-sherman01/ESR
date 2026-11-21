"""Keep target, gripper and inactive-object contact evidence distinct."""
from esrb.backends.aspire import evidence as capture


def test_peg_contact_identity_keeps_nut_and_finger_distinct():
    nut = capture.geom_group("SquareNut_g0", ["SquareNut_main"])
    gripper = capture.geom_group("gripper0_finger1_collision", ["gripper0_leftfinger", "robot0_link7"])
    peg = capture.geom_group(None, ["peg1"])
    assert nut == "square_nut" and gripper == "gripper" and peg == "square_peg"
    assert capture.pair_category(nut, peg) != capture.pair_category(gripper, peg)
    assert capture.pair_category(peg, nut) == capture.pair_category(nut, peg)


def test_unused_round_nut_is_separate_from_target_drop_evidence():
    assert capture.geom_group("RoundNut_g1", ["RoundNut_main"]) == "inactive_round_nut"
