"""A small left/right gap must not hide falls or degradation of the right hand."""

import copy

from supermimic.eval.hand_tracking import comparable_hands


def report(left, right, completed=True):
    return {
        "all_windows_completed": completed,
        "aggregate": {
            "left": {"palm_orientation_mean_deg": left, "wrist_joint_mae_deg": [0, 0, left]},
            "right": {"palm_orientation_mean_deg": right, "wrist_joint_mae_deg": [0, 0, right]},
        },
    }


def test_paired_improvement_passes():
    assert comparable_hands(report(22, 20), report(80, 20))["passed"]


def test_parity_from_right_degradation_fails():
    result = comparable_hands(report(40, 40), report(80, 20))
    assert not result["passed"]
    assert not result["checks"]["right_palm_preserved"]


def test_early_failure_or_persistent_roll_error_fails():
    baseline = report(80, 20)
    assert not comparable_hands(report(22, 20, completed=False), baseline)["passed"]
    candidate = copy.deepcopy(report(22, 20))
    candidate["aggregate"]["left"]["wrist_joint_mae_deg"][2] = 70
    assert not comparable_hands(candidate, baseline)["passed"]
