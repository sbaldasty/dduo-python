"""Public bounds: every element of a wrapped value lies in `bounds` for every
possible dataset. They come only from public constants and propagate by
interval arithmetic; (-inf, inf) means nothing is known.
"""

import math

import numpy
import pandas
import pytest

import duet
import duet.numpy as dnp
import duet.pandas as dpd
from duet.duet_vals import UNBOUNDED

INF = math.inf


def private(value, source="other"):
    return duet.DuetWrapper(value, duet.SensEnv({duet.DataSource(source): 1.0}), duet.LInf())


# ── where bounds come from ─────────────────────────────────────────────────

def test_raw_data_has_no_bounds(people):
    assert people.bounds == UNBOUNDED
    assert people["age"].bounds == UNBOUNDED


@pytest.mark.parametrize("clip, expected", [
    (lambda s: s.clip(0, 100), (0, 100)),
    (lambda s: s.clip(lower=0, upper=100), (0, 100)),
    (lambda s: s.clip(upper=100), (-INF, 100)),
    (lambda s: s.clip(lower=0), (0, INF)),
    (lambda s: s.clip(numpy.zeros(10), numpy.arange(10.0)), (0, 9)),
])
def test_clip_sets_bounds(people, clip, expected):
    assert clip(people["age"]).bounds == expected


def test_data_dependent_clip_limit_bounds_nothing_on_its_side(people):
    assert people["age"].clip(0, private(50.0)).bounds == (0, INF)


def test_clipping_an_already_bounded_value_tightens(people):
    assert people["age"].clip(0, 100).clip(-5, 50).bounds == (0, 50)


@pytest.mark.parametrize("count", [
    lambda df: df.shape[0],
    lambda df: df.count(),
    lambda df: df["age"].value_counts(),
    lambda df: df.groupby("age").size(),
])
def test_counts_are_nonnegative(people, count):
    assert count(people).bounds == (0, INF)


@pytest.mark.parametrize("compare", [
    lambda s: s > 30, lambda s: s < 30, lambda s: s >= 30, lambda s: s <= 30, lambda s: s == 30,
])
def test_comparisons_are_boolean(people, compare):
    assert compare(people["age"]).bounds == (0, 1)


# ── propagation ────────────────────────────────────────────────────────────

@pytest.fixture
def x(people):
    return people["age"].clip(0, 100)


@pytest.mark.parametrize("expr, expected", [
    (lambda x: x + 5, (5, 105)),
    (lambda x: 5 + x, (5, 105)),
    (lambda x: x - 5, (-5, 95)),
    (lambda x: 5 - x, (-95, 5)),
    (lambda x: -x, (-100, 0)),
    (lambda x: x * 2, (0, 200)),
    (lambda x: -2 * x, (-200, 0)),
    (lambda x: x / 4, (0, 25)),
    (lambda x: x // 30, (0, 3)),
    (lambda x: abs(x - 30), (0, 70)),
    (lambda x: x + numpy.full(10, 2.0), (2, 102)),
])
def test_interval_arithmetic(x, expr, expected):
    assert expr(x).bounds == pytest.approx(expected)


def test_product_of_two_clipped_columns(people):
    w, x = people["hours"].clip(0, 2), people["age"].clip(0, 100)
    product = w * x
    assert product.bounds == (0, 200)
    # finite sensitivity for such products is the next step; for now it stays
    assert math.isinf(product.senv.get_max())


def test_division_by_an_interval_containing_zero_is_unbounded(x):
    assert (1 / x).bounds == UNBOUNDED
    assert (1 / (x + 1)).bounds == pytest.approx((1 / 101, 1))


@pytest.mark.parametrize("clip, expected", [
    ((0, 100), (0, INF)),
    ((-10, -1), (-INF, 0)),
    ((-5, 5), UNBOUNDED),
])
def test_sum_keeps_only_the_sign(people, clip, expected):
    assert people["age"].clip(*clip).sum().bounds == expected


def test_selection_and_conversion_keep_bounds(people, x):
    assert x[people["hours"] > 20].bounds == (0, 100)
    assert x.values.bounds == (0, 100)
    assert x.to_numpy().bounds == (0, 100)


def test_methods_without_a_bounds_rule_are_unbounded(x):
    assert x.cumsum().bounds == UNBOUNDED


def test_rewrapping_intersects_bounds(x):
    assert duet.DuetWrapper(x, duet.SensEnv({}), duet.L1(), (50, 200)).bounds == (50, 100)


def test_str_shows_known_bounds(x, people):
    assert "bounds=(0.0, 100.0)" in str(x)
    assert "bounds=" not in str(people)


# ── duet.numpy ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr, expected", [
    (lambda x: dnp.exp(x / 100), (1, math.e)),
    (lambda x: dnp.abs(x - 30), (0, 70)),
    (lambda x: dnp.sqrt(x), (0, 10)),
    (lambda x: dnp.sqrt(x - 1), UNBOUNDED),
    (lambda x: dnp.sign(x - 30), (-1, 1)),
    (lambda x: dnp.subtract(x, 10), (-10, 90)),
    (lambda x: dnp.sum(x.values), (0, INF)),
    (lambda x: dnp.where(numpy.ones(10, dtype=bool), x, x - 200), (-200, 100)),
    (lambda x: dnp.linalg.norm(x.values), (0, INF)),
])
def test_numpy_functions_propagate_bounds(x, expr, expected):
    assert expr(x).bounds == pytest.approx(expected)


def test_numpy_zeros_and_array_split(x):
    assert dnp.zeros(private(3)).bounds == (0, 0)
    assert all(c.bounds == (0, 100) for c in dnp.array_split(x.values, 2))


# ── bounds never depend on the data ────────────────────────────────────────

def test_bounds_are_the_same_on_any_dataset(tmp_path):
    def program(df):
        w, x = df["hours"].clip(0, 2), df["age"].clip(0, 100)
        return [v.bounds for v in (w * x, (w * x).sum(), df.shape[0], x / 4 - 3)]

    results = []
    for i, rows in enumerate([[(30, 1)], [(90, 5), (10, 0), (55, 2)] * 40]):
        path = tmp_path / f"d{i}.csv"
        pandas.DataFrame(rows, columns=["age", "hours"]).to_csv(path, index=False)
        results.append(program(dpd.read_csv(str(path))))
    assert results[0] == results[1]
