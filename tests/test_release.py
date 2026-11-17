"""Turning row distances into value distances, at sums and at releases.

`clip` no longer converts anything: it sets public bounds and keeps the row
distance. A sum across rows, or a mechanism, converts a row count s into a
value distance s * per_row_change(bounds) * elements_per_row. Rows with no
bounds -- nothing clipped -- are unbounded.
"""

import math

import numpy
import pytest

import duet
import duet.numpy as dnp
from duet.duet_vals import L1, LInf, Rows, per_row_change, release_sensitivity

from conftest import sensitivity


def rows(value, bounds=None, sens=1.0):
    return duet.DuetWrapper(numpy.asarray(value, dtype=float),
                            duet.SensEnv({duet.DataSource("x"): sens}), Rows(), bounds)


# ── per_row_change: both neighbour relations ───────────────────────────────

@pytest.mark.parametrize("bounds, expected", [
    ((0, 100), 100),    # 0 inside: replace-one and add/remove-one agree
    ((-5, 5), 10),
    ((10, 20), 20),     # 0 outside: adding/removing a row moves a sum by up to 20
    ((-20, -10), 20),
    ((3, 3), 3),
    ((0, math.inf), math.inf),
])
def test_per_row_change(bounds, expected):
    assert per_row_change(bounds) == expected


def test_sum_clipped_away_from_zero_covers_add_remove(people):
    assert sensitivity(people["age"].clip(10, 20).sum()) == 20


# ── release_sensitivity and the mechanisms ─────────────────────────────────

def test_release_of_unbounded_rows_is_unbounded(people):
    assert math.isinf(release_sensitivity(people["age"]))


def test_release_of_bounded_rows_uses_bounds(people):
    assert release_sensitivity(people["age"].clip(0, 100)) == 100


def test_release_of_value_distances_is_unchanged(people):
    assert release_sensitivity(people.shape[0]) == 1
    assert release_sensitivity(people["age"].clip(0, 100).sum()) == 100


def test_release_of_rows_with_several_elements_per_row():
    assert release_sensitivity(rows([[0, 1], [1, 0], [1, 1]], bounds=(0, 1), sens=2)) == 4


@pytest.mark.parametrize("mechanism, params, scale", [
    (duet.laplace, {"epsilon": 0.5}, lambda s: s / 0.5),
    (duet.gauss, {"epsilon": 0.5, "delta": 1e-5}, lambda s: math.sqrt(2 * math.log(1.25 / 1e-5)) * s / 0.5),
])
def test_mechanisms_calibrate_to_bounded_rows(people, mechanism, params, scale):
    one_row = people["age"].clip(0, 50)[3]
    assert one_row.mode == Rows() and one_row.bounds == (0, 50)
    numpy.random.seed(1)
    released = mechanism(one_row, **params)
    numpy.random.seed(1)
    draw = (numpy.random.laplace if mechanism is duet.laplace else numpy.random.normal)(0, scale(50))
    assert released == pytest.approx(one_row.val + draw)


# ── sums ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("total", [
    lambda x: dnp.sum(x),
    lambda x: dnp.sum(x, axis=0),
    lambda x: x.sum(),
    lambda x: x.sum(axis=0),
])
def test_sum_across_rows_counts_every_element_of_a_row(total):
    result = total(rows([[0, 1], [1, 0], [1, 1]], bounds=(0, 1)))
    assert result.mode == L1()
    assert sensitivity(result) == 2


@pytest.mark.parametrize("total", [
    lambda x: dnp.sum(x, axis=1),
    lambda x: x.sum(axis=1),
    lambda x: x.sum(axis=-1),
])
def test_sum_within_rows_is_row_local(total):
    result = total(rows([[0, 1], [1, 0], [1, 1]], bounds=(0, 1)))
    assert result.mode == Rows()
    assert sensitivity(result) == 1


def test_non_lipschitz_function_of_clipped_rows_is_bounded(people):
    # exp has no Lipschitz constant, but each row stays within [1, e]
    result = dnp.exp(people["age"].clip(0, 1).values).sum()
    assert sensitivity(result) == pytest.approx(math.e)


# ── bounds through map / list / zip ────────────────────────────────────────

def test_map_result_bounds_are_the_hull_of_the_elements():
    result = duet.map(lambda x: dnp.sign(x), rows([-2.0, 0.0, 3.0]))
    assert result.mode == Rows()
    assert result.bounds == (-1, 1)
    assert sensitivity(dnp.sum(result)) == 2


def test_list_and_zip_bounds_are_hulls(people):
    a, b = people["age"].clip(0, 10), people["hours"].clip(-5, 5)
    assert duet.zip(a, b).bounds == (-5, 10)
    assert duet.list([a, b]).bounds == (-5, 10)


# ── supporting fixes ───────────────────────────────────────────────────────

def test_scaling_a_zero_sensitivity_by_infinity_stays_zero():
    env = duet.SensEnv({duet.DataSource("a"): 0, duet.DataSource("b"): 2}).scale(math.inf)
    assert sorted(env.sens.values()) == [0, math.inf]


def test_public_wrappers_have_the_bounds_of_their_value(people):
    public = duet.DuetWrapper(numpy.array([2.0, 7.0]), duet.SensEnv({}), LInf())
    assert (people["age"].clip(0, 10) + public[0]).bounds == (2, 12)
