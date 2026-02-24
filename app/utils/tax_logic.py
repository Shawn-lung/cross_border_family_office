"""Dutch Box 3 threshold and wealth-tax calculation helpers."""

from __future__ import annotations

BOX3_EXEMPTION_LIMIT_EUR = 57_684.0
BOX3_TAX_RATE = 0.36
BOX3_ASSUMED_YIELD = 0.0604

def calculate_box3_tax(nav_eur: float) -> float:
    """Calculate Dutch Box 3 tax for a given NAV in EUR based on assumed yields."""
    if nav_eur <= BOX3_EXEMPTION_LIMIT_EUR:
        return 0.0

    overage = nav_eur - BOX3_EXEMPTION_LIMIT_EUR
    assumed_return = overage * BOX3_ASSUMED_YIELD
    return assumed_return * BOX3_TAX_RATE
