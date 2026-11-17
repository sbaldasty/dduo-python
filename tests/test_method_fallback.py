"""Methods without a sensitivity rule must not keep their input's sensitivity.

Before the fix, any method missing from `sens_effects` was treated as
sensitivity-preserving, so `cumsum()` -- which spreads a change in one row
across every later row -- kept sensitivity 1, and a clipped cumulative sum
came out at 10 instead of 10 * n.
"""

import math

import pytest

from conftest import sensitivity


def test_unknown_method_is_unbounded(people):
    assert math.isinf(sensitivity(people["age"].cumsum()))
    assert math.isinf(sensitivity(people["age"].cumsum().clip(0, 10).sum()))


def test_unknown_attribute_is_unbounded(people):
    assert math.isinf(sensitivity(people["age"].index))


def test_numpy_sum_needs_clipping_like_series_sum(people):
    assert math.isinf(sensitivity(people.values.sum()))


@pytest.mark.parametrize("convert", [
    lambda s: s.values,
    lambda s: s.to_numpy(),
])
def test_series_to_numpy_keeps_sensitivity(people, convert):
    clipped = people["age"].clip(0, 10)
    assert sensitivity(convert(clipped).sum()) == 10


def test_dataframe_to_numpy_keeps_sensitivity(people):
    assert sensitivity(people.to_numpy()) == 1


def test_groupby_size_histogram_has_sensitivity_one(people):
    # the histogram in notebooks/Simple Examples.ipynb
    assert sensitivity(people.groupby("age").size()) == 1


def test_groupby_aggregates_without_rules_are_unbounded(people):
    assert math.isinf(sensitivity(people.groupby("age").sum()))


def test_sklearn_dataset_fields_keep_sensitivity():
    sk = pytest.importorskip("duet.sklearn")
    iris = sk.load_iris()
    assert sensitivity(iris.data) == 1
    assert sensitivity(iris.target) == 1
