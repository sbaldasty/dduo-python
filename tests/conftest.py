import numpy as np
import pandas
import pytest

import duet
import duet.pandas as dpd


@pytest.fixture(autouse=True)
def isolated_accountant():
    """Undo any odometers/filters a test registers on DDuo's global accountant.

    `duet.accountant` is module-level state, so a test that fails inside a
    `with duet.EDOdometer():` block (or registers one directly) would
    otherwise keep charging it in every later test.
    """
    acct = duet.accountant
    saved = {name: list(getattr(acct, name)) for name in ("ods", "fs", "advods", "advfs")}
    yield acct
    for name, items in saved.items():
        getattr(acct, name)[:] = items


@pytest.fixture(autouse=True)
def seeded_numpy():
    """DDuo's mechanisms draw from NumPy's global RNG; make tests repeatable."""
    np.random.seed(0)


@pytest.fixture
def people(tmp_path):
    """A small sensitive DataFrame read through DDuo's own entry point."""
    path = tmp_path / "people.csv"
    pandas.DataFrame({
        "age": [23, 35, 47, 51, 62, 18, 29, 90, 41, 38],
        "hours": [40, 50, 20, 45, 10, 30, 60, 5, 40, 35],
    }).to_csv(path, index=False)
    return dpd.read_csv(str(path))


def sensitivity(x):
    """The largest sensitivity DDuo tracks for `x` across its data sources."""
    return x.senv.get_max()
