"""Elementwise binary operations: row-local on rows, Lipschitz on values.

If every sensitive operand is a row distance, element i of the result
depends only on row i of each operand, so the result is row-local whatever
the operation -- products and comparisons included -- and bounds turn it
into a value distance when it is summed or released. pandas operands pair
records by index label, so one changed record changes one label. Before
this, every product or comparison of sensitive values was unbounded.
"""

import math

import numpy
import pytest

import duet
import duet.numpy as dnp
from duet.duet_vals import L1, LInf, Rows

from conftest import sensitivity


def value(x, mode=L1(), sens=1.0, source="v"):
    return duet.DuetWrapper(x, duet.SensEnv({duet.DataSource(source): sens}), mode)


def public(x):
    return duet.DuetWrapper(x, duet.SensEnv({}), LInf())


# ── the motivating case ────────────────────────────────────────────────────

def test_product_of_clipped_columns_has_exact_sensitivity(people):
    w, x = people["hours"].clip(0, 2), people["age"].clip(0, 100)
    product = w * x
    assert product.mode == Rows()
    assert product.bounds == (0, 200)
    assert sensitivity(product.sum()) == 200


def test_clipping_a_product_of_raw_columns(people):
    assert sensitivity((people["hours"] * people["age"]).clip(0, 500).sum()) == 500


def test_product_of_two_counts_is_unbounded(people):
    # counts are value distances: n * n moves by 2n + 1, so no row-local rule
    n = people.shape[0]
    assert math.isinf(sensitivity(n * n))


# ── row-local operations ───────────────────────────────────────────────────

def test_comparison_sums_to_a_count(people):
    over_30 = people["age"] > 30
    assert over_30.mode == Rows()
    assert sensitivity(over_30.sum()) == 1


def test_conjunction_of_comparisons(people):
    both = (people["age"] > 30) & (people["hours"] > 20)
    assert sensitivity(both.sum()) == 1


def test_quotient_of_rows_uses_its_bounds(people):
    q = people["age"].clip(1, 2) / people["hours"].clip(1, 4)
    assert q.bounds == (0.25, 2)
    assert sensitivity(q.sum()) == 2


def test_columns_from_different_sources_add_their_counts(people):
    other = duet.DuetWrapper(people["age"].val, duet.SensEnv({duet.DataSource("other"): 1.0}), Rows())
    combined = people["age"] + other
    assert {str(k): v for k, v in combined.senv.sens.items()}.values() and sorted(combined.senv.sens.values()) == [1, 1]


def test_filtered_and_full_columns_align_by_label(people):
    over_30 = people["age"][people["age"] > 30]
    assert sensitivity(over_30 + people["hours"]) == 1


def test_positional_arrays_add_counts_from_one_source(people):
    a, b = people["age"].values, people["hours"].values
    assert sensitivity(a * b) == 2


def test_map_of_a_non_lipschitz_product_is_row_local():
    rows = duet.DuetWrapper([1.0, 2.0, 3.0], duet.SensEnv({duet.DataSource("r"): 1.0}), Rows())
    assert sensitivity(duet.map(lambda x: x * x, rows)) == 1


def test_selection_with_row_local_condition(people):
    picked = dnp.where((people["age"] > 30).values, people["age"].values, people["hours"].values)
    assert picked.mode == Rows()
    assert sensitivity(picked) == 3   # positional: three operands from one source


def test_column_condition_broadcast_within_rows(people):
    grid = numpy.ones((10, 3))
    cond = (people["age"] > 30).values[:, None]
    assert sensitivity(dnp.where(cond, grid, 0)) == 1


# ── broadcasting across rows ───────────────────────────────────────────────

def test_one_row_broadcast_to_a_column_is_unbounded(people):
    assert math.isinf(sensitivity(people["age"] + people["age"][3]))


def test_series_broadcast_against_a_dataframe_is_unbounded(people):
    assert math.isinf(sensitivity(people - people["age"]))


def test_value_distance_broadcast_to_a_column_is_unbounded(people):
    assert math.isinf(sensitivity(people["age"] - people.shape[0]))


def test_value_distance_combined_with_a_single_row(people):
    assert sensitivity(people["age"][3] - people.shape[0]) == 1


def test_numpy_subtract_broadcast_across_rows_is_unbounded(people):
    ages = people["age"].values
    assert math.isinf(sensitivity(dnp.subtract(ages, ages[0])))


# ── value distances ────────────────────────────────────────────────────────

def test_adding_value_distances_in_one_metric():
    total = value(numpy.ones(3), sens=2) + value(numpy.ones(3), sens=3, source="w")
    assert total.mode == L1() and sensitivity(total) == 3


def test_adding_value_distances_in_different_metrics_is_unbounded():
    assert math.isinf(sensitivity(value(numpy.ones(3)) + value(numpy.ones(3), mode=LInf(), source="w")))


@pytest.mark.parametrize("expr, expected", [
    (lambda x: x * numpy.array([1.0, -4.0, 2.0]), 8),
    (lambda x: numpy.array([1.0, -4.0, 2.0]) * x, 8),
    (lambda x: x * public(3.0), 6),
    (lambda x: x / numpy.array([2.0, 4.0, 8.0]), 1),
])
def test_scaling_a_value_distance_by_public_data(expr, expected):
    assert sensitivity(expr(value(numpy.ones(3), sens=2))) == expected


@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
@pytest.mark.parametrize("expr", [
    lambda x, y: x * y,
    lambda x, y: x / y,
    lambda x, y: x > y,
    lambda x, y: x // y,
    lambda x, y: x / numpy.array([1.0, 0.0, 1.0]),
    lambda x, y: 1 / x,
])
def test_value_distances_without_a_lipschitz_bound_are_unbounded(expr):
    x = value(numpy.ones(3), sens=2)
    y = value(numpy.ones(3), sens=1, source="w")
    assert math.isinf(sensitivity(expr(x, y)))


def test_broadcast_l1_value_distance_is_unbounded():
    assert math.isinf(sensitivity(value(numpy.ones(3)) + value(1.0, source="w")))


def test_broadcast_linf_value_distance_is_kept():
    total = value(numpy.ones(3), mode=LInf(), sens=2) + value(1.0, mode=LInf(), sens=3, source="w")
    assert sensitivity(total) == 3


# ── norms within rows, and mixed selections (the vectorised gradient path) ──

def test_norm_within_rows_is_row_local(people):
    norms = dnp.linalg.norm(people.values, ord=2, axis=1)
    assert norms.mode == Rows()
    assert sensitivity(norms) == 1
    assert norms.bounds == (0, math.inf)


@pytest.mark.parametrize("axis", [None, 0])
def test_norm_across_rows_is_unbounded(people, axis):
    assert math.isinf(sensitivity(dnp.linalg.norm(people.values, ord=2, axis=axis)))


def test_selection_with_value_distance_condition_stays_rows(people):
    cond = people.shape[0] > 3           # a comparison of a count: value distance
    picked = dnp.where(cond, people["age"].values, people["hours"].values)
    assert picked.mode == Rows()
    assert math.isinf(sensitivity(picked))


def test_vectorised_l2_clipping(people):
    vs = people.values
    norms = dnp.linalg.norm(vs, ord=2, axis=1)
    clipped = dnp.where((norms > 1.0)[:, None], vs / norms[:, None], vs)
    assert clipped.mode == Rows()
    assert duet.mode_switch(Rows, duet.L2)(lambda v, b: v)(clipped, 1.0).mode == duet.L2(1.0)
