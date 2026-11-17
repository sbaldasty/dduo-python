"""DistanceMetric equality must distinguish the kind of metric, not just its bound.

Before the fix, `L1() == LInf()` was True (both have bound inf), so the mode
checks in `sens_sum` and `mode_switch` never fired and an unclipped column
summed with sensitivity 1.
"""

import math

import pytest

import duet
from duet.duet_vals import L1, L2, LInf

from conftest import sensitivity


def test_metrics_of_different_kinds_are_unequal():
    assert L1() != LInf()
    assert L2() != LInf()
    assert L1(5) != L2(5)


def test_metrics_of_same_kind_compare_by_bound():
    assert L1() == L1()
    assert L2(5) == L2(5)
    assert L2(5) != L2(6)


def test_hash_agrees_with_equality():
    assert len({L1(), LInf(), L2(), L1()}) == 3


def test_comparison_with_non_metric_is_false():
    assert L1() != 1
    assert L1() != None  # noqa: E711


def test_unclipped_sum_has_unbounded_sensitivity(people):
    assert math.isinf(sensitivity(people["age"].sum()))


def test_clipped_sum_is_unchanged(people):
    assert sensitivity(people["age"].clip(0, 100).sum()) == 100


def test_clipping_twice_does_not_grow_sensitivity(people):
    twice = people["age"].clip(0, 100).clip(0, 50)
    assert twice.mode == L1()
    assert sensitivity(twice.sum()) == 100


def test_mode_switch_rejects_wrong_input_mode():
    @duet.mode_switch(LInf, L2)
    def passthrough(v, b):
        return v

    source = duet.DataSource("x")
    unbounded = duet.DuetWrapper([3.0, 4.0], duet.SensEnv({source: 1.0}), LInf())
    assert passthrough(unbounded, 2.0).mode == L2(2.0)

    already_l1 = duet.DuetWrapper([3.0, 4.0], duet.SensEnv({source: 1.0}), L1())
    with pytest.raises(ValueError, match="unexpected mode"):
        passthrough(already_l1, 2.0)
