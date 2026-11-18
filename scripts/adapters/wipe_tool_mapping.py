"""Wipe-only conversion at the adapter/PyRoki boundary."""
import numpy as np


def provider_input_transform(target_grip, hand_from_grip):
    """Compensate provider's -0.1m hand-z translation after tool calibration."""
    hand = np.asarray(target_grip) @ np.linalg.inv(np.asarray(hand_from_grip))
    provider_input = hand.copy()
    provider_input[:3,3] += hand[:3,:3] @ np.array([0.,0.,.1])
    return provider_input
