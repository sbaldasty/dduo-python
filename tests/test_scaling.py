"""Scaling by a constant must stay sound under DDuo's row-counting LInf mode.

For a column read from a CSV, LInf sensitivity 1 means "one row differs",
not "each value moves by at most 1"; multiplying every element by 0.5 does
not change how many rows differ. Before the fix, `(v * 0.5).clip(0, 10).sum()`
got sensitivity 5 although one changed row can move it by 10. Division by a
constant scaled by c instead of 1/|c|, and negative constants produced
negative sensitivities.
"""

import math

import pytest

import duet
from duet.duet_vals import L1, LInf

from conftest import sensitivity


def clipped_sum(x):
    return sensitivity(x.clip(0, 10).sum())


@pytest.mark.parametrize("scaled", [
    lambda v: v * 0.5,
    lambda v: 0.5 * v,
    lambda v: v / 2,
    lambda v: v // 2,
])
def test_shrinking_a_row_distance_value_does_not_shrink_sensitivity(people, scaled):
    assert clipped_sum(scaled(people["age"])) == 10


def test_growing_a_row_distance_value_is_still_conservative(people):
    assert clipped_sum(people["age"] * 2) == 20


def test_dividing_by_a_small_constant_grows_sensitivity(people):
    # v / 0.5 == 2 * v; this used to scale by 0.5
    assert sensitivity(people["age"] / 0.5) == 2


def test_negative_constants_scale_by_magnitude(people):
    assert sensitivity(people["age"] * -3) == 3
    assert sensitivity((people["age"] * -1).clip(-10, 0).sum()) == 10


def test_multiplying_by_zero_is_constant(people):
    assert sensitivity(people["age"] * 0) == 0


def test_dividing_by_zero_is_unbounded(people):
    assert math.isinf(sensitivity(people["age"] / 0))


def test_value_distance_modes_still_scale_down(people):
    clipped = people["age"].clip(0, 10)
    assert clipped.mode == L1()
    assert sensitivity((clipped * 0.5).sum()) == 5


def test_map_does_not_shrink_row_distance_sensitivity():
    ls = duet.DuetWrapper([1, 2, 3], duet.SensEnv({duet.DataSource("ls"): 1.0}), LInf())
    assert sensitivity(duet.map(lambda x: x * 0.5, ls)) == 1


def test_mechanism_accepts_negated_value(people):
    with duet.EDOdometer():
        duet.laplace((people["age"] * -1).clip(-100, 0).sum(), epsilon=1.0)


def test_floor_division_of_value_distance_is_unbounded(people):
    # values clipped to [2, 4] differ by at most 2, yet // 3 can jump by 1
    assert math.isinf(sensitivity((people["age"].clip(2, 4) // 3).sum()))


def test_binning_before_clipping_keeps_clip_width(people):
    # binning first (row-level), clipping after: one changed row, width 10
    assert clipped_sum(people["age"] // 10) == 10
