"""`duet.list` and `duet.map` must never drop a data source from the result.

Before the fix, both fell back to an empty sensitivity environment (i.e.
"depends on no data") when their elements disagreed on mode or data source,
and `map` reached that fallback even for `map(lambda x: x + 1, ls)`.
"""

import itertools
import math

import pytest

import duet
from duet.duet_vals import L1, L2, LInf


@pytest.fixture
def ls():
    return duet.DuetWrapper([1, 2, 3], duet.SensEnv({duet.DataSource("ls"): 1.0}), LInf())


@pytest.fixture
def other():
    return duet.DuetWrapper(10, duet.SensEnv({duet.DataSource("other"): 1.0}), LInf())


def sens(x):
    return {str(k): v for k, v in x.senv.sens.items()}


# ── map ────────────────────────────────────────────────────────────────────

def test_map_of_elementwise_function_keeps_list_sensitivity(ls):
    assert sens(duet.map(lambda x: x + 1, ls)) == {"DataSource(ls)": 1.0}


def test_map_scales_by_the_functions_sensitivity(ls):
    assert sens(duet.map(lambda x: x * 2, ls)) == {"DataSource(ls)": 2.0}


def test_map_composes_other_sources_sequentially(ls, other):
    # each of the 3 applications reads `other` once
    assert sens(duet.map(lambda x: x + other, ls)) == {
        "DataSource(ls)": 1.0, "DataSource(other)": 3.0}


def test_map_with_mixed_result_modes_is_unbounded(ls, other):
    calls = itertools.count()

    def f(x):
        y = x + other
        return y if next(calls) == 0 else duet.DuetWrapper(y.val, y.senv, L1())

    result = duet.map(f, ls)
    assert result.mode == LInf()
    assert set(sens(result)) == {"DataSource(ls)", "DataSource(other)"}
    assert all(math.isinf(v) for v in result.senv.sens.values())


# ── list ───────────────────────────────────────────────────────────────────

def _scalar(source, mode=None):
    return duet.DuetWrapper(1.0, duet.SensEnv({source: 1.0}), mode or LInf())


def test_list_with_mixed_modes_is_unbounded():
    src = duet.DataSource("ls")
    result = duet.list([_scalar(src, LInf()), _scalar(src, L1())])
    assert sens(result) == {"DataSource(ls)": math.inf}


def test_list_with_several_sources_is_unbounded_in_each():
    result = duet.list([_scalar(duet.DataSource("a")), _scalar(duet.DataSource("b"))])
    assert sens(result) == {"DataSource(a)": math.inf, "DataSource(b)": math.inf}


def test_list_single_mode_and_source_is_unchanged():
    src = duet.DataSource("ls")
    result = duet.list([_scalar(src, L2(5)), _scalar(src, L2(5))])
    assert result.mode == L2(5)
    assert sens(result) == {"DataSource(ls)": 5}


def test_list_of_public_values_stays_public():
    public = duet.DuetWrapper(1.0, duet.SensEnv({}), LInf())
    assert sens(duet.list([public, public])) == {}


def test_map_identity_does_not_zero_later_elements(ls):
    # The first call returns its argument unchanged; later calls double it.
    # Each element must still be tracked against the element variable, so the
    # map's sensitivity is the largest per-element one: 2.
    calls = itertools.count()
    result = duet.map(lambda x: x if next(calls) == 0 else x * 2, ls)
    assert sens(result) == {"DataSource(ls)": 2.0}


def test_map_leaves_values_returned_by_f_untouched(ls):
    kept = []

    def f(x):
        kept.append(x)
        return x

    duet.map(f, ls)
    assert all(len(x.senv.sens) == 1 for x in kept)
