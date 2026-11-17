"""Row distance (Rows) versus value distance (L1, L2, LInf).

Data sources produce Rows: sensitivity s means one change to the source
changes at most s rows, each arbitrarily. Value distances bound how far the
value itself moves. Before Rows existed both meanings shared LInf, which
forced conservative workarounds and ruled out row-local rules (products of
two LInf values could not be row-local, because an LInf scalar such as a
count is a genuine value distance: n * n moves by 2n + 1).
"""

import math

import numpy
import pytest

import duet
import duet.numpy as dnp
from duet.duet_vals import L1, L2, LInf, Rows

from conftest import sensitivity


def rows(value, sens=1.0, source="ls"):
    return duet.DuetWrapper(value, duet.SensEnv({duet.DataSource(source): sens}), Rows())


# ── where Rows come from ───────────────────────────────────────────────────

def test_read_csv_produces_rows(people):
    assert people.mode == Rows()
    assert people["age"].mode == Rows()
    assert people[people["hours"] > 20].mode == Rows()


def test_numpy_load_produces_rows(tmp_path):
    path = tmp_path / "x.npy"
    numpy.save(path, numpy.arange(6.0).reshape(3, 2))
    assert dnp.load(str(path)).mode == Rows()


def test_sklearn_datasets_produce_rows():
    sk = pytest.importorskip("duet.sklearn")
    iris = sk.load_iris()
    assert iris.mode == Rows()
    assert all(part.mode == Rows() for part in sk.train_test_split(iris.data, iris.target))


def test_to_dict_keeps_the_mode(people):
    import duet.pandas as dpd
    assert dpd.to_dict(people["age"]).mode == Rows()


# ── leaving Rows ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("count", [
    lambda df: df.shape[0],
    lambda df: df.count(),
    lambda df: df["age"].value_counts(),
    lambda df: df.groupby("age").size(),
])
def test_counts_are_value_distances(people, count):
    result = count(people)
    assert result.mode == LInf()
    assert sensitivity(result) == 1


def test_clip_sets_bounds_and_keeps_rows(people):
    clipped = people["age"].clip(0, 100)
    assert clipped.mode == Rows()
    assert clipped.bounds == (0, 100)
    assert sensitivity(clipped) == 1


def test_sum_turns_bounded_rows_into_a_value_distance(people):
    total = people["age"].clip(0, 100).sum()
    assert total.mode == L1()
    assert sensitivity(total) == 100


def test_clipping_a_value_distance_keeps_it(people):
    counts = people["age"].value_counts()
    assert counts.mode == LInf()
    clipped = counts.clip(0, 5)
    assert clipped.mode == LInf()
    assert sensitivity(clipped) == 1


def test_mode_switch_takes_rows(people):
    grads = people.values
    assert duet.L2_clip(grads[0], 1.0).mode == L2(1.0)
    with pytest.raises(ValueError, match="unexpected mode"):
        duet.L2_clip(duet.DuetWrapper(numpy.ones(2), duet.SensEnv({duet.DataSource("v"): 1.0}), LInf()), 1.0)


# ── map ────────────────────────────────────────────────────────────────────

def test_map_over_rows_is_row_local_even_for_non_lipschitz_f():
    result = duet.map(lambda x: dnp.exp(x), rows([1.0, 2.0, 3.0]))
    assert result.mode == Rows()
    assert sensitivity(result) == 1


def test_map_over_rows_ignoring_the_element_has_no_own_sensitivity():
    other = duet.DuetWrapper(2.0, duet.SensEnv({duet.DataSource("other"): 0.5}), LInf())
    result = duet.map(lambda x: other, rows([1.0, 2.0, 3.0]))
    assert {str(k): v for k, v in result.senv.sens.items()}["DataSource(ls)"] == 0


def test_map_over_rows_reading_another_source_is_unbounded_in_it():
    # a change to `other` changes every element, and the number of elements
    # is private: counting them would leak it through the noise scale
    other = duet.DuetWrapper(2.0, duet.SensEnv({duet.DataSource("other"): 0.5}), LInf())
    result = duet.map(lambda x: x + other, rows([1.0, 2.0, 3.0]))
    assert result.mode == Rows()
    sens = {str(k): v for k, v in result.senv.sens.items()}
    assert sens["DataSource(ls)"] == 1.0
    assert math.isinf(sens["DataSource(other)"])


def test_map_clipping_each_row_gives_count_times_bound():
    # the per-example gradient clipping pattern in examples/grad-renyi.py
    data = rows([numpy.array([3.0, 4.0]), numpy.array([0.3, 0.4])], sens=2.0)
    result = duet.map(lambda r: duet.L2_clip(r, 1.0), data)
    assert result.mode == L2(1.0)
    assert sensitivity(result) == 2.0
    numpy.testing.assert_allclose(result.val[0], [0.6, 0.8])


def test_map_over_a_value_distance_measures_lipschitz_constant():
    ls = duet.DuetWrapper([1.0, 2.0], duet.SensEnv({duet.DataSource("ls"): 1.0}), LInf())
    assert sensitivity(duet.map(lambda x: x * 3, ls)) == 3
