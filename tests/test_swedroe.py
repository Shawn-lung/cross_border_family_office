"""Tests for Swedroe 5/25 rebalancing rule boundary conditions."""

from app.utils.portfolio_math import swedroe_5_25_rebalance_required


def test_absolute_drift_4_point_9_percent_does_not_trigger_rebalance():
    """A 15% target drifting to 19.9% stays below the 5% absolute threshold."""
    target = {"SXRV": 15.0}
    current = {"SXRV": 19.9}

    assert swedroe_5_25_rebalance_required(target, current) is False


def test_absolute_drift_5_point_1_percent_triggers_rebalance():
    """A 15% target drifting to 20.1% breaches the 5% absolute threshold."""
    target = {"SXRV": 15.0}
    current = {"SXRV": 20.1}

    assert swedroe_5_25_rebalance_required(target, current) is True


def test_relative_drift_24_point_9_percent_of_target_does_not_trigger_rebalance():
    """A 5% target at 6.245% remains under the 25% relative threshold."""
    target = {"ZPRX": 5.0}
    current = {"ZPRX": 6.245}

    assert swedroe_5_25_rebalance_required(target, current) is False


def test_relative_drift_25_point_1_percent_of_target_triggers_rebalance():
    """A 5% target at 6.255% exceeds the 25% relative threshold."""
    target = {"ZPRX": 5.0}
    current = {"ZPRX": 6.255}

    assert swedroe_5_25_rebalance_required(target, current) is True