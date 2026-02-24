"""Portfolio mathematics utilities for allocation drift and rebalancing decisions."""

from __future__ import annotations


def swedroe_5_25_rebalance_required(
    target_weights: dict[str, float],
    current_weights: dict[str, float],
) -> bool:
    """Return True when any asset breaches the Swedroe 5/25 drift threshold.

    Inputs are expected as percentage weights (for example, 40.0 for 40%).
    Rebalance is required when:
    - Absolute drift is strictly greater than 5.0 percentage points, or
    - Relative drift is strictly greater than 25% of target for small sleeves.

    In this project, "small sleeves" are target allocations at or below 10%.
    """
    target_keys = set(target_weights)
    current_keys = set(current_weights)
    if target_keys != current_keys:
        raise ValueError("target_weights and current_weights must contain identical asset keys.")

    for asset, target in target_weights.items():
        if target <= 0:
            raise ValueError(f"Target weight for {asset} must be positive.")

        current = current_weights[asset]
        absolute_drift = abs(current - target)
        if absolute_drift > 5.0:
            return True

        if target <= 10.0 and (absolute_drift / target) > 0.25:
            return True

    return False
