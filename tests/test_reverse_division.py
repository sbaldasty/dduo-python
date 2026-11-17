"""Division with the sensitive value on the right, and in-place division.

Before the fix, `c / x` kept x's sensitivity although 1/x is unbounded near
0, and `x /= c` computed `c / x` rather than `x / c`.
"""

import math

import pandas
import pytest

import duet

from conftest import sensitivity


@pytest.mark.parametrize("divide", [
    lambda c, x: c / x,
    lambda c, x: c // x,
])
def test_dividing_by_a_value_distance_is_unbounded(people, divide):
    x = duet.DuetWrapper(people["age"].val, duet.SensEnv({duet.DataSource("x"): 1.0}), duet.L1())
    assert math.isinf(sensitivity(divide(100, x)))


@pytest.mark.parametrize("divide", [
    lambda c, x: c / x,
    lambda c, x: c // x,
])
def test_dividing_by_rows_is_row_local(people, divide):
    result = divide(100, people["age"])
    assert sensitivity(result) == 1
    assert math.isinf(sensitivity(result.sum()))           # nothing bounds 100 / age
    assert sensitivity(result.clip(0, 10).sum()) == 10


def test_reverse_division_computes_c_over_x(people):
    pandas.testing.assert_series_equal((100 / people["age"]).val, 100 / people["age"].val)


def test_in_place_division_divides_the_value(people):
    x = people["age"]
    x /= 2
    pandas.testing.assert_series_equal(x.val, people["age"].val / 2)
    assert sensitivity(x) == sensitivity(people["age"] / 2)


def test_in_place_floor_division_divides_the_value(people):
    x = people["age"]
    x //= 10
    pandas.testing.assert_series_equal(x.val, people["age"].val // 10)
    assert sensitivity(x) == sensitivity(people["age"] // 10)


def test_in_place_division_of_a_constant_by_a_sensitive_value(people):
    c = 100
    c /= people["age"]
    assert sensitivity(c) == 1
    assert math.isinf(sensitivity(c.sum()))


def test_reverse_multiplication_still_scales(people):
    assert sensitivity((3 * people["age"].clip(0, 10)).sum()) == 30
