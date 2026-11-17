"""Scaling by a constant, under each meaning of a sensitivity.

A row distance (Rows, e.g. a column read from a CSV) counts changed rows:
multiplying every element by a nonzero constant does not change how many
rows differ. A value distance (L1/L2/LInf, e.g. after clipping) bounds how
far the value moves, so it scales by |c|.

History: before Rows existed both meanings shared LInf, and
`(v * 0.5).clip(0, 10).sum()` got sensitivity 5 although one changed row can
move it by 10; division scaled by c instead of 1/|c|; negative constants
produced negative sensitivities.
"""

import math

import pytest

import duet
from duet.duet_vals import L1, LInf, Rows

from conftest import sensitivity


def clipped_sum(x):
    return sensitivity(x.clip(0, 10).sum())


@pytest.mark.parametrize("scaled", [
    lambda v: v * 0.5,
    lambda v: 0.5 * v,
    lambda v: v / 2,
    lambda v: v // 2,
    lambda v: v * 2,
    lambda v: v * -3,
    lambda v: v / 0.5,
])
def test_scaling_rows_keeps_the_row_count(people, scaled):
    assert people["age"].mode == Rows()
    assert clipped_sum(scaled(people["age"])) == 10


def test_multiplying_rows_by_zero_is_constant(people):
    assert sensitivity(people["age"] * 0) == 0


def test_dividing_rows_by_zero_is_still_row_local(people):
    assert sensitivity(people["age"] / 0) == 1


@pytest.fixture
def clipped(people):
    x = people["age"].clip(0, 10)
    assert x.mode == L1()
    return x


@pytest.mark.parametrize("scaled, expected", [
    (lambda x: x * 0.5, 5),
    (lambda x: x / 2, 5),
    (lambda x: x / 0.5, 20),
    (lambda x: x * -3, 30),
    (lambda x: -3 * x, 30),
    (lambda x: x * 0, 0),
])
def test_value_distances_scale_by_magnitude(clipped, scaled, expected):
    assert sensitivity(scaled(clipped).sum()) == expected


def test_dividing_a_value_distance_by_zero_is_unbounded(clipped):
    assert math.isinf(sensitivity(clipped / 0))


def test_floor_division_of_value_distance_is_unbounded(people):
    # values clipped to [2, 4] differ by at most 2, yet // 3 can jump by 1
    assert math.isinf(sensitivity((people["age"].clip(2, 4) // 3).sum()))


def test_binning_rows_before_clipping_keeps_clip_width(people):
    assert clipped_sum(people["age"] // 10) == 10


def test_map_over_rows_keeps_the_row_count():
    ls = duet.DuetWrapper([1, 2, 3], duet.SensEnv({duet.DataSource("ls"): 1.0}), Rows())
    assert sensitivity(duet.map(lambda x: x * 0.5, ls)) == 1


def test_map_over_a_value_distance_scales():
    ls = duet.DuetWrapper([1, 2, 3], duet.SensEnv({duet.DataSource("ls"): 1.0}), LInf())
    assert sensitivity(duet.map(lambda x: x * 0.5, ls)) == 0.5


def test_mechanism_accepts_negated_value(people):
    with duet.EDOdometer():
        duet.laplace((people["age"] * -1).clip(-100, 0).sum(), epsilon=1.0)
