"""Finite engineering comparisons; equality is permitted by this protocol."""
import math


def exceeds(value,limit):
    if not math.isfinite(value) or not math.isfinite(limit):raise ValueError('nonfinite load or threshold')
    return value>limit
