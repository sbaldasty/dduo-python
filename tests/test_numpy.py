"""Sensitivity rules in duet.numpy, DDuo's wrappers around numpy functions.

Before the fixes: `sum` gave non-L2 inputs an *empty* sensitivity
environment; elementwise functions, `subtract`, `where`, `dot` and
`array_split` relabelled clipped (value-distance) inputs as LInf while
keeping the number; `dot` added sensitivities even for a product of two
sensitive values; `sqrt` did not run; and importing duet.numpy replaced
`numpy.linalg.norm` for the whole process.
"""

import math

import numpy
import pytest

import duet
import duet.numpy as dnp
from duet.duet_vals import L1, L2, LInf

from conftest import sensitivity

SRC = duet.DataSource("x")


def wrap(value, mode, sens=1.0):
    return duet.DuetWrapper(numpy.asarray(value, dtype=float), duet.SensEnv({SRC: sens}), mode)


def public(value):
    return duet.DuetWrapper(numpy.asarray(value, dtype=float), duet.SensEnv({}), LInf())


# ── sum ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sum_", [dnp.sum, dnp.sum2])
def test_sum_of_unclipped_data_is_unbounded(sum_):
    assert math.isinf(sensitivity(sum_(wrap([1, 2, 3], LInf()))))


def test_sum_of_clipped_data_keeps_sensitivity():
    assert sensitivity(dnp.sum(wrap([1, 2, 3], L1(), sens=10))) == 10


def test_sum_of_row_clipped_gradients_over_rows_is_unchanged():
    grads = wrap([[1, 0], [0, 1], [1, 1]], L2(4.0), sens=4.0)
    assert sensitivity(dnp.sum(grads, axis=0)) == 4.0
    assert sensitivity(dnp.sum(grads, 0)) == 4.0


def test_sum_of_l2_vector_scales_by_sqrt_of_count():
    v = wrap([1, 2, 3, 4], L2(20.0), sens=20.0)
    assert sensitivity(dnp.sum(v)) == pytest.approx(40.0)


# ── elementwise functions ──────────────────────────────────────────────────

@pytest.mark.parametrize("fn", [dnp.exp, dnp.sign, dnp.sqrt])
def test_non_lipschitz_function_of_clipped_data_is_unbounded(fn):
    result = fn(wrap([1, 4, 9], L1(), sens=0.5))
    assert result.mode == LInf()
    assert math.isinf(sensitivity(result))


@pytest.mark.parametrize("fn", [dnp.exp, dnp.sign, dnp.sqrt, dnp.abs])
def test_elementwise_function_of_rows_keeps_row_count(fn):
    assert sensitivity(fn(wrap([1, 4, 9], LInf()))) == 1


def test_abs_is_1_lipschitz():
    result = dnp.abs(wrap([-1, 2], L1(), sens=3))
    assert result.mode == L1()
    assert sensitivity(result) == 3


def test_sqrt_computes_square_roots():
    numpy.testing.assert_allclose(dnp.sqrt(wrap([1, 4, 9], LInf())).val, [1, 2, 3])


# ── subtract / where ───────────────────────────────────────────────────────

def test_subtract_in_one_metric_adds_sensitivities():
    result = dnp.subtract(wrap([1, 2], L1(), sens=2), wrap([3, 4], L1(), sens=3))
    assert result.mode == L1()
    assert sensitivity(result) == 5


def test_subtract_of_public_value_keeps_metric():
    result = dnp.subtract(wrap([1, 2], L2(5.0), sens=5.0), numpy.array([1.0, 1.0]))
    assert result.mode == L2(5.0)
    assert sensitivity(result) == 5.0


def test_subtract_across_metrics_is_unbounded():
    assert math.isinf(sensitivity(dnp.subtract(wrap([1], L1()), wrap([1], LInf()))))


def test_where_with_public_condition_keeps_branch_metric():
    result = dnp.where(numpy.array([True, False]), wrap([1, 2], L1(), sens=2), wrap([3, 4], L1(), sens=3))
    assert result.mode == L1()
    assert sensitivity(result) == 5


def test_where_with_sensitive_condition_on_clipped_branches_is_unbounded():
    cond = wrap([1, 0], LInf()) > 0  # comparisons are already unbounded
    cond = duet.DuetWrapper(cond.val, duet.SensEnv({SRC: 1.0}), LInf())
    assert math.isinf(sensitivity(dnp.where(cond, wrap([1, 2], L1()), wrap([3, 4], L1()))))


# ── dot ────────────────────────────────────────────────────────────────────

def test_dot_of_two_sensitive_values_is_unbounded():
    assert math.isinf(sensitivity(dnp.dot(wrap([[1, 2]], LInf()), wrap([3, 4], LInf()))))


def test_dot_of_record_matrix_with_public_weights_is_row_local():
    theta = public([0.5, 0.5])
    assert sensitivity(dnp.dot(wrap([[1, 2], [3, 4]], LInf()), theta)) == 1


@pytest.mark.parametrize("dot", [
    lambda s, p: dnp.dot(s, p),   # 1-D: sums over the sensitive records
    lambda s, p: dnp.dot(p, s),   # sensitive on the right: mixes its entries
])
def test_dot_summing_over_sensitive_entries_is_unbounded(dot):
    assert math.isinf(sensitivity(dot(wrap([1, 2], LInf()), public([1, 1]))))


# ── array_split / linalg ───────────────────────────────────────────────────

def test_array_split_keeps_metric():
    chunks = dnp.array_split(wrap([1, 2, 3, 4], L1(), sens=2), 2)
    assert all(c.mode == L1() and sensitivity(c) == 2 for c in chunks)


def test_importing_duet_numpy_leaves_numpy_alone():
    assert type(numpy.linalg.norm([3.0, 4.0])) is numpy.float64
    assert dnp.linalg.svd is numpy.linalg.svd


def test_duet_norm_is_unbounded():
    assert math.isinf(sensitivity(dnp.linalg.norm(wrap([3, 4], LInf()))))


def test_utilities_l2_clip_works_without_the_numpy_patch():
    clipped = duet.L2_clip(wrap([3, 4], LInf()), 1.0)
    assert clipped.mode == L2(1.0)
    numpy.testing.assert_allclose(clipped.val, [0.6, 0.8])
