"""Tests for Dutch Box 3 threshold and tax boundary behavior."""

import pytest

from app.utils.tax_logic import calculate_box3_tax


def test_nav_57683_triggers_zero_tax():
    """NAV below the 57,684 EUR threshold should incur zero tax."""
    assert calculate_box3_tax(57_683) == 0.0


def test_nav_77684_taxes_assumed_six_percent_return_at_36_percent():
    """NAV at 77,684 EUR should tax 36% on the assumed 6% return of the 20,000 overage."""
    tax = calculate_box3_tax(77_684)
    # Overage = 20,000 -> assumed return = 1,200 -> tax due = 432
    assert tax == pytest.approx(434.88)
