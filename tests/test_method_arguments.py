"""Arguments to tracked method calls: keyword arguments and clip's bounds.

Before the fix, a DuetWrapper passed as a keyword argument contributed no
sensitivity, wrappers were handed to pandas as-is rather than unwrapped, and
`sens_clip` read its bounds only positionally -- so `clip(lower=0,
upper=10)` and one-sided clips crashed, and data-dependent bounds were
trusted as public.
"""

import math

import numpy
import pandas
import pytest

import duet

from conftest import sensitivity


def private(value, source="other"):
    return duet.DuetWrapper(value, duet.SensEnv({duet.DataSource(source): 1.0}), duet.LInf())


def public(value):
    return duet.DuetWrapper(value, duet.SensEnv({}), duet.LInf())


@pytest.mark.parametrize("clip", [
    lambda s: s.clip(0, 10),
    lambda s: s.clip(lower=0, upper=10),
    lambda s: s.clip(0, upper=10),
    lambda s: s.clip(numpy.zeros(10), numpy.full(10, 10.0)),
])
def test_clip_bounds_by_position_or_keyword(people, clip):
    assert sensitivity(clip(people["age"]).sum()) == 10


@pytest.mark.parametrize("clip", [
    lambda s: s.clip(upper=10),
    lambda s: s.clip(lower=0),
    lambda s: s.clip(None, 10),
])
def test_one_sided_clip_is_unbounded(people, clip):
    assert math.isinf(sensitivity(clip(people["age"]).sum()))


@pytest.mark.parametrize("clip", [
    lambda s, b: s.clip(0, b),
    lambda s, b: s.clip(lower=0, upper=b),
])
def test_data_dependent_clip_bound_is_unbounded(people, clip):
    bound = private(10.0)
    assert math.isinf(sensitivity(clip(people["age"], bound).sum()))


def test_public_wrapped_bounds_are_unwrapped(people):
    clipped = people["age"].clip(public(0), upper=public(50))
    pandas.testing.assert_series_equal(clipped.val, people["age"].val.clip(0, 50))
    assert sensitivity(clipped.sum()) == 50


def test_keyword_argument_sensitivity_is_counted():
    d = public({})
    result = d.update(secret=private(1.0, "kw"))
    assert "DataSource(kw)" in {str(k) for k in result.senv.sens}


def test_keyword_arguments_reach_the_method_unwrapped():
    d = public({})
    d.update(secret=private(1.0))
    assert d.val == {"secret": 1.0}
