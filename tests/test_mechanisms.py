"""Mechanism fixes: independent noise per element, and a checked SVT.

Before: gauss and renyi_gauss added one noise draw to every element of a
vector (releasing the exact differences between elements); laplace and
renyi_gauss_vec sized vector noise with len(), wrong for 2-D values.
above_threshold assumed every query had sensitivity 1 without checking,
charged nothing when no query passed, and charged only the passing query's
sources.
"""

import numpy
import pytest

import duet

from conftest import sensitivity


def bounded_rows(value, bounds=(0, 1)):
    return duet.DuetWrapper(numpy.asarray(value, dtype=float),
                            duet.SensEnv({duet.DataSource("x"): 1.0}), duet.Rows(), bounds)


# ── per-element noise ──────────────────────────────────────────────────────

@pytest.mark.parametrize("mechanism, params", [
    (duet.gauss, {"epsilon": 0.5, "delta": 1e-5}),
    (duet.renyi_gauss, {"alpha": 10, "epsilon": 0.5}),
    (duet.renyi_gauss_vec, {"alpha": 10, "epsilon": 0.5}),
    (duet.laplace, {"epsilon": 0.5}),
])
@pytest.mark.parametrize("shape", [(5,), (4, 3)])
def test_vector_noise_is_independent_per_element(mechanism, params, shape):
    x = bounded_rows(numpy.zeros(shape))
    released = numpy.asarray(mechanism(x, **params))
    assert released.shape == shape
    assert len(numpy.unique(released)) == released.size


@pytest.mark.parametrize("mechanism, params, sampler", [
    (duet.gauss, {"epsilon": 0.5, "delta": 1e-5}, numpy.random.normal),
    (duet.laplace, {"epsilon": 0.5}, numpy.random.laplace),
])
def test_vector_noise_matches_one_draw_of_the_values_shape(mechanism, params, sampler):
    x = bounded_rows(numpy.arange(6.0).reshape(3, 2), bounds=(0, 6))
    release = mechanism(x, details=True, **params)
    numpy.random.seed(9)
    released = mechanism(x, **params)
    numpy.random.seed(9)
    expected = x.val + sampler(0, release.scale, size=(3, 2))
    numpy.testing.assert_allclose(released, expected)


def test_scalar_noise_is_one_draw(people):
    numpy.random.seed(9)
    released = duet.laplace(people.shape[0], epsilon=0.5)
    numpy.random.seed(9)
    assert released == 10 + numpy.random.laplace(0, 2.0)


def test_series_release_keeps_its_index(people):
    released = duet.gauss(people["age"].clip(0, 100), epsilon=1.0, delta=1e-5)
    assert list(released.index) == list(people["age"].val.index)


# ── above_threshold ────────────────────────────────────────────────────────

def counts_over(*cutoffs):
    return [lambda df, c=c: (df["age"] > c).sum() for c in cutoffs]


def test_picks_a_query_with_count_queries(people):
    chosen = duet.above_threshold(counts_over(200, 150, 0), people, T=5, epsilon=100)
    assert chosen == 2


def test_query_above_the_declared_sensitivity_raises_before_releasing(people):
    queries = counts_over(30) + [lambda df: df["age"].clip(0, 10).sum()]
    with duet.EDOdometer() as odo:
        numpy.random.seed(3)
        state = numpy.random.get_state()[1].copy()
        with pytest.raises(ValueError, match="query 1 has sensitivity 10"):
            duet.above_threshold(queries, people, T=5, epsilon=1.0)
        assert (numpy.random.get_state()[1] == state).all()   # no noise drawn
    assert odo.privacy_env._sum_costs() == [0, 0]


def test_declared_sensitivity_scales_the_noise(people):
    queries = [lambda df: df["age"].clip(0, 10).sum()]
    numpy.random.seed(4)
    chosen = duet.above_threshold(queries, people, T=60, epsilon=1.0, sensitivity=10)
    numpy.random.seed(4)
    t_hat = 60 + numpy.random.laplace(0, 20)
    nu = numpy.random.laplace(0, 40)
    total = people["age"].clip(0, 10).sum().val
    assert chosen == (0 if total + nu >= t_hat else -1)


def test_charges_even_when_no_query_passes(people):
    with duet.EDOdometer() as odo:
        chosen = duet.above_threshold(counts_over(200), people, T=1000, epsilon=0.5)
    assert chosen == -1
    assert odo.privacy_env._sum_costs() == [0.5, 0]


def test_charges_every_source_the_queries_read(people, tmp_path):
    import duet.pandas as dpd
    import pandas
    path = tmp_path / "other.csv"
    pandas.DataFrame({"age": [1, 2]}).to_csv(path, index=False)
    other = dpd.read_csv(str(path))
    queries = [lambda df: (df["age"] > 0).sum(), lambda df: (other["age"] > 0).sum()]
    with duet.EDOdometer() as odo:
        duet.above_threshold(queries, people, T=-1000, epsilon=0.5)
    assert len(odo.privacy_env.fp) == 2
    assert all(v == 0.5 for v in odo.privacy_env.fp.values())
