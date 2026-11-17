"""Baseline behaviour the fixes in this branch must not disturb."""

import duet

from conftest import sensitivity


def test_read_csv_tracks_one_row_sensitivity(people):
    assert sensitivity(people) == 1


def test_count_has_sensitivity_one(people):
    assert sensitivity(people.shape[0]) == 1


def test_clipped_sum_has_clip_width_sensitivity(people):
    assert sensitivity(people["age"].clip(0, 100).sum()) == 100


def test_laplace_charges_active_odometer(people):
    with duet.EDOdometer() as odo:
        duet.laplace(people.shape[0], epsilon=0.5)
        duet.laplace(people.shape[0], epsilon=0.25)
    eps, delta = odo.privacy_env._sum_costs()
    assert eps == 0.75
    assert delta == 0


def test_odometer_unregisters_on_exit(people):
    with duet.EDOdometer():
        pass
    assert duet.accountant.ods == []
