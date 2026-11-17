"""`details=True`: a mechanism's output together with its public facts.

A Release records the released value, the mechanism, the noise scale, the
sensitivity it was calibrated to, and public bounds on each element of the
true value -- what a caller needs to turn the release into a posterior
(e.g. truncated at zero for a count).
"""

import dataclasses
import math

import numpy
import pandas
import pytest

import duet
import duet.pandas as dpd
from duet.duet_vals import Release

GAUSS_SIGMA = lambda sens, eps, delta: math.sqrt(2 * math.log(1.25 / delta)) * sens / eps


def run(mechanism, val, seed=5, **params):
    """The default output and the details=True output, from the same draw."""
    numpy.random.seed(seed)
    plain = mechanism(val, **params)
    numpy.random.seed(seed)
    release = mechanism(val, details=True, **params)
    return plain, release


@pytest.mark.parametrize("mechanism, params, kind, scale", [
    (duet.laplace, {"epsilon": 0.5}, "laplace", lambda s: s / 0.5),
    (duet.laplace, {"ε": 0.5}, "laplace", lambda s: s / 0.5),
    (duet.gauss, {"epsilon": 0.5, "delta": 1e-5}, "gaussian", lambda s: GAUSS_SIGMA(s, 0.5, 1e-5)),
    (duet.renyi_gauss, {"alpha": 10, "epsilon": 0.5}, "gaussian", lambda s: math.sqrt(s ** 2 * 10 / (2 * 0.5))),
])
def test_scalar_release_details(people, mechanism, params, kind, scale):
    total = people["age"].clip(0, 100).sum()
    plain, release = run(mechanism, total, **params)
    assert isinstance(release, Release)
    assert release.value == plain
    assert release.mechanism == kind
    assert release.sensitivity == 100
    assert release.scale == pytest.approx(scale(100))
    assert release.bounds == (0, math.inf)


def test_default_return_is_unchanged(people):
    assert not isinstance(duet.laplace(people.shape[0], epsilon=1.0), Release)


def test_count_bounds_are_nonnegative(people):
    _, release = run(duet.laplace, people.shape[0], epsilon=1.0)
    assert release.sensitivity == 1
    assert release.bounds == (0, math.inf)


def test_vector_release_bounds_are_elementwise(people):
    clipped = people["age"].clip(10, 50)
    plain, release = run(duet.laplace, clipped, epsilon=1.0)
    pandas.testing.assert_series_equal(release.value, plain)
    assert release.bounds == (10, 50)
    assert release.sensitivity == 50          # 0 lies outside [10, 50]


def test_renyi_gauss_vec_details(people):
    v = people["age"].clip(0, 1).values
    plain, release = run(duet.renyi_gauss_vec, v, alpha=10, epsilon=1.0)
    numpy.testing.assert_allclose(release.value, plain)
    assert release.bounds == (0, 1)
    assert release.scale == pytest.approx(math.sqrt(10 / 2))


def test_untracked_value_released_exactly():
    release = duet.renyi_gauss(3.0, alpha=10, epsilon=1.0, details=True)
    assert (release.value, release.scale, release.sensitivity, release.bounds) == (3.0, 0, 0, (3, 3))


def test_details_charge_the_accountant_once(people):
    with duet.EDOdometer() as odo:
        duet.laplace(people.shape[0], epsilon=0.25, details=True)
    assert odo.privacy_env._sum_costs() == [0.25, 0]


def test_release_is_immutable(people):
    release = duet.laplace(people.shape[0], epsilon=1.0, details=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        release.scale = 0


def test_public_facts_do_not_depend_on_the_data(tmp_path):
    def program(df):
        w, x = df["hours"].clip(0, 2), df["age"].clip(0, 100)
        r = duet.laplace((w * x).sum(), epsilon=1.0, details=True)
        return r.mechanism, r.scale, r.sensitivity, r.bounds

    facts = []
    for i, rows in enumerate([[(30, 1)], [(90, 5), (10, 0), (55, 2)] * 40]):
        path = tmp_path / f"d{i}.csv"
        pandas.DataFrame(rows, columns=["age", "hours"]).to_csv(path, index=False)
        facts.append(program(dpd.read_csv(str(path))))
    assert facts[0] == facts[1] == ("laplace", 200.0, 200.0, (0.0, math.inf))
