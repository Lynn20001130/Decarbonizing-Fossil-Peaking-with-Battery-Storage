"""
3R Metrics (CSR / PSR / OSR) calculation module for natural gas peak-shaving.

CSR (Cost Storage Ratio):   the storage penetration at which average unit
                            storage cost exceeds the gas-fired generation cost.
PSR (Profit Storage Ratio): the storage penetration at which unit profit
                            becomes negative.
OSR (Optimal Storage Ratio): the storage penetration at which marginal storage
                             cost exceeds the gas-fired generation cost.
"""

import numpy as np
import pandas as pd
from scipy import interpolate


# ---------------------------------------------------------------------------
# Helper: linear extrapolation to the y-axis (x = 0)
# ---------------------------------------------------------------------------

def extrapolate_to_y_axis(x_data, y_data):
    """Linearly extrapolate to x=0 using the first two data points.

    If the extrapolated y value is negative it is clamped to 0.

    Parameters
    ----------
    x_data : array-like
        X values (must have at least two elements).
    y_data : array-like
        Corresponding Y values.

    Returns
    -------
    x_with_zero : np.ndarray
        ``x_data`` with 0 prepended.
    y_with_zero : np.ndarray
        ``y_data`` with the extrapolated (or clamped) value prepended.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    # Linear extrapolation from the first two points
    slope = (y_data[1] - y_data[0]) / (x_data[1] - x_data[0])
    y_at_zero = y_data[0] - slope * x_data[0]

    # Clamp to 0 if negative
    if y_at_zero < 0:
        y_at_zero = 0.0

    x_with_zero = np.concatenate([[0.0], x_data])
    y_with_zero = np.concatenate([[y_at_zero], y_data])
    return x_with_zero, y_with_zero


# ---------------------------------------------------------------------------
# Intersection finders
# ---------------------------------------------------------------------------

def find_second_intersection(x_storage, y_storage, x_gas, y_gas):
    """Find the intersection between the storage cost curve and the gas cost line.

    Uses ``scipy.interpolate.interp1d`` to build a continuous gas-cost
    interpolant, then detects a sign change in
    ``diff = y_storage - gas_interp(x_storage)``.

    Parameters
    ----------
    x_storage, y_storage : array-like
        Storage cost curve.
    x_gas, y_gas : array-like
        Gas cost line (two or more points).

    Returns
    -------
    x_intersect : float or None
        The x coordinate of the intersection (linearly interpolated between
        the two bracketing points).  Returns ``None`` if the intersection
        is at x <= 0.05 or no sign change is found.
    """
    x_storage = np.asarray(x_storage, dtype=float)
    y_storage = np.asarray(y_storage, dtype=float)
    x_gas = np.asarray(x_gas, dtype=float)
    y_gas = np.asarray(y_gas, dtype=float)

    gas_interp = interpolate.interp1d(
        x_gas, y_gas, bounds_error=False, fill_value="extrapolate"
    )

    diff = y_storage - gas_interp(x_storage)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    for idx in sign_changes:
        # Linear interpolation between bracketing points
        x0, x1 = x_storage[idx], x_storage[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
        if x_cross > 0.05:
            return float(x_cross)

    return None


def find_marginal_intersection(x_marginal, y_marginal, x_gas, y_gas):
    """Find the intersection for the marginal cost curve and the gas cost line.

    Similar to :func:`find_second_intersection` but with a threshold set at
    5 % of the x-data range instead of a fixed 0.05.

    Parameters
    ----------
    x_marginal, y_marginal : array-like
        Marginal cost curve.
    x_gas, y_gas : array-like
        Gas cost line.

    Returns
    -------
    x_intersect : float or None
    """
    x_marginal = np.asarray(x_marginal, dtype=float)
    y_marginal = np.asarray(y_marginal, dtype=float)
    x_gas = np.asarray(x_gas, dtype=float)
    y_gas = np.asarray(y_gas, dtype=float)

    gas_interp = interpolate.interp1d(
        x_gas, y_gas, bounds_error=False, fill_value="extrapolate"
    )

    diff = y_marginal - gas_interp(x_marginal)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    threshold = 0.05 * (x_marginal.max() - x_marginal.min())

    for idx in sign_changes:
        x0, x1 = x_marginal[idx], x_marginal[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
        if x_cross > threshold:
            return float(x_cross)

    return None


# ---------------------------------------------------------------------------
# CSR
# ---------------------------------------------------------------------------

def compute_csr(results_df, gas_price, gas_power_per_m3, generator_cost):
    """Compute the Cost Storage Ratio (CSR).

    CSR is the storage penetration (%) at which the average unit storage cost
    equals the unit cost of gas-fired generation.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``peak_supply_kwh``, ``annual_cost``,
        ``total_demand_kwh``.
    gas_price : float
        Gas price (currency per m^3).
    gas_power_per_m3 : float
        Electricity generated per cubic metre of gas (kWh / m^3).
    generator_cost : float
        Additional generator capital / O&M cost per kWh.

    Returns
    -------
    csr : float
        CSR value in percent (0 -- 100).
    """
    df = results_df.copy()
    gas_unit_cost = gas_price / gas_power_per_m3 + generator_cost

    peak_supply = df["peak_supply_kwh"].values.astype(float)
    annual_cost = df["annual_cost"].values.astype(float)
    total_demand = df["total_demand_kwh"].values[0]

    # Only keep rows with positive supply
    mask = peak_supply > 0
    peak_supply = peak_supply[mask]
    annual_cost = annual_cost[mask]

    if len(peak_supply) == 0:
        return 0.0

    y_unit = annual_cost / peak_supply

    # Normalize
    x_norm = peak_supply / total_demand
    y_max = max(np.max(y_unit), gas_unit_cost)
    y_norm = y_unit / y_max
    gas_norm = gas_unit_cost / y_max

    # Sort by x
    order = np.argsort(x_norm)
    x_norm = x_norm[order]
    y_norm = y_norm[order]

    # Extrapolate to y-axis
    if len(x_norm) >= 2:
        x_norm, y_norm = extrapolate_to_y_axis(x_norm, y_norm)

    # Check trivial cases
    if np.all(y_norm <= gas_norm):
        return 100.0
    if np.all(y_norm >= gas_norm):
        return 0.0

    # Gas cost as a horizontal line spanning the x range
    x_gas = np.array([x_norm.min(), x_norm.max()])
    y_gas = np.array([gas_norm, gas_norm])

    x_inter = find_second_intersection(x_norm, y_norm, x_gas, y_gas)
    if x_inter is not None:
        return float(x_inter * 100.0)

    # Fallback: if no intersection found but not trivially above/below
    return 0.0


# ---------------------------------------------------------------------------
# PSR
# ---------------------------------------------------------------------------

def compute_psr(results_df, gas_price, electricity_price,
                gas_power_per_m3, generator_cost):
    """Compute the Profit Storage Ratio (PSR).

    PSR is the storage penetration (%) at which unit profit becomes negative.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``peak_supply_kwh``, ``unit_profit``,
        ``total_demand_kwh``.
    gas_price : float
        Gas price (currency per m^3).
    electricity_price : float
        Electricity selling price (currency per kWh).
    gas_power_per_m3 : float
        Electricity generated per cubic metre of gas (kWh / m^3).
    generator_cost : float
        Additional generator cost per kWh.

    Returns
    -------
    psr : float
        PSR value in percent (0 -- 100).
    """
    df = results_df.copy()
    gas_unit_cost = gas_price / gas_power_per_m3 + generator_cost
    gas_unit_profit = electricity_price - gas_unit_cost

    peak_supply = df["peak_supply_kwh"].values.astype(float)
    unit_profit = df["unit_profit"].values.astype(float)
    total_demand = df["total_demand_kwh"].values[0]

    mask = peak_supply > 0
    peak_supply = peak_supply[mask]
    unit_profit = unit_profit[mask]

    if len(peak_supply) == 0:
        return 0.0

    # Normalize x
    x_norm = peak_supply / total_demand

    # Normalize y: shift and scale to [0, 1] range
    y_min = np.min(unit_profit)
    y_max = np.max(unit_profit)
    y_range = y_max - y_min if y_max != y_min else 1.0
    y_norm = (unit_profit - y_min) / y_range

    # Normalize the zero-profit line the same way
    zero_norm = (0.0 - y_min) / y_range

    # Sort by x
    order = np.argsort(x_norm)
    x_norm = x_norm[order]
    y_norm = y_norm[order]

    # Find where normalised profit crosses the zero line
    # (from positive to negative, i.e. profit drops below zero)
    diff = y_norm - zero_norm
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    for idx in sign_changes:
        x0, x1 = x_norm[idx], x_norm[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
        return float(x_cross * 100.0)

    # No crossing found
    if np.all(unit_profit >= 0):
        return 100.0
    return 0.0


# ---------------------------------------------------------------------------
# OSR
# ---------------------------------------------------------------------------

def compute_osr(results_df, gas_price, gas_power_per_m3, generator_cost):
    """Compute the Optimal Storage Ratio (OSR).

    OSR is the storage penetration (%) at which the marginal storage cost
    equals the unit cost of gas-fired generation.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``peak_supply_kwh``, ``annual_cost``,
        ``satisfy_rate_pct``.
    gas_price : float
        Gas price (currency per m^3).
    gas_power_per_m3 : float
        Electricity generated per cubic metre of gas (kWh / m^3).
    generator_cost : float
        Additional generator cost per kWh.

    Returns
    -------
    osr : float
        OSR value in percent (0 -- 100).
    """
    df = results_df.copy()
    gas_unit_cost = gas_price / gas_power_per_m3 + generator_cost

    # Sort by peak_supply and deduplicate
    df = df.sort_values("peak_supply_kwh").drop_duplicates(
        subset=["peak_supply_kwh"], keep="first"
    ).reset_index(drop=True)

    peak_supply = df["peak_supply_kwh"].values.astype(float)
    annual_cost = df["annual_cost"].values.astype(float)
    satisfy_rate = df["satisfy_rate_pct"].values.astype(float)

    if len(peak_supply) < 2:
        return 0.0

    # Marginal cost = d(annual_cost) / d(peak_supply)
    d_cost = np.diff(annual_cost)
    d_supply = np.diff(peak_supply)

    # Avoid division by zero
    valid = d_supply != 0
    if not np.any(valid):
        return 0.0

    marginal_cost = np.full_like(d_cost, np.nan)
    marginal_cost[valid] = d_cost[valid] / d_supply[valid]

    # X-axis: midpoint satisfy rates
    x_mid = (satisfy_rate[:-1] + satisfy_rate[1:]) / 2.0

    # Remove NaN entries
    finite_mask = np.isfinite(marginal_cost)
    marginal_cost = marginal_cost[finite_mask]
    x_mid = x_mid[finite_mask]

    if len(x_mid) < 2:
        return 0.0

    # Normalize marginal cost
    y_max = max(np.max(marginal_cost), gas_unit_cost)
    y_norm = marginal_cost / y_max
    gas_norm = gas_unit_cost / y_max

    # Extrapolate to y-axis
    x_mid, y_norm = extrapolate_to_y_axis(x_mid, y_norm)

    # Gas cost horizontal line
    x_gas = np.array([x_mid.min(), x_mid.max()])
    y_gas = np.array([gas_norm, gas_norm])

    x_inter = find_marginal_intersection(x_mid, y_norm, x_gas, y_gas)

    max_satisfy_rate = 100.0

    if x_inter is not None:
        return float((x_inter / max_satisfy_rate) * 100.0)

    # Fallback
    if np.all(y_norm <= gas_norm):
        return 100.0
    return 0.0


# ---------------------------------------------------------------------------
# Province-level 3R wrapper
# ---------------------------------------------------------------------------

def compute_province_3r(results, gas_price, elec_price,
                        gas_power_per_m3, generator_cost):
    """Compute all three ratios (CSR, PSR, OSR) for a single province.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain at least ``peak_supply_kwh``,
        ``annual_cost``, ``unit_profit``, ``total_demand_kwh``,
        ``satisfy_rate_pct``.
    gas_price : float
        Gas price (currency per m^3).
    elec_price : float
        Electricity selling price (currency per kWh).
    gas_power_per_m3 : float
        Electricity generated per cubic metre of gas (kWh / m^3).
    generator_cost : float
        Additional generator cost per kWh.

    Returns
    -------
    tuple or None
        ``(csr, psr, osr, total_demand)`` if computation succeeds,
        otherwise ``None``.
    """
    if not results:
        return None

    df = pd.DataFrame(results)

    required_cols = {
        "peak_supply_kwh", "annual_cost", "unit_profit",
        "total_demand_kwh", "satisfy_rate_pct",
    }
    if not required_cols.issubset(df.columns):
        return None

    try:
        csr = compute_csr(df, gas_price, gas_power_per_m3, generator_cost)
        psr = compute_psr(df, gas_price, elec_price,
                          gas_power_per_m3, generator_cost)
        osr = compute_osr(df, gas_price, gas_power_per_m3, generator_cost)
        total_demand = df["total_demand_kwh"].values[0]
        return (csr, psr, osr, total_demand)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# National-level aggregation
# ---------------------------------------------------------------------------

def aggregate_province_3r(province_3r_list):
    """Compute weighted national 3R from province-level results.

    Parameters
    ----------
    province_3r_list : list[tuple]
        Each element is ``(csr, psr, osr, weight)`` where *weight* is
        typically the province total demand.

    Returns
    -------
    dict
        ``{'CSR': float, 'PSR': float, 'OSR': float}`` as weighted averages.
    """
    if not province_3r_list:
        return {"CSR": 0.0, "PSR": 0.0, "OSR": 0.0}

    csr_vals = np.array([t[0] for t in province_3r_list])
    psr_vals = np.array([t[1] for t in province_3r_list])
    osr_vals = np.array([t[2] for t in province_3r_list])
    weights = np.array([t[3] for t in province_3r_list], dtype=float)

    total_weight = weights.sum()
    if total_weight == 0:
        return {"CSR": 0.0, "PSR": 0.0, "OSR": 0.0}

    return {
        "CSR": float(np.sum(csr_vals * weights) / total_weight),
        "PSR": float(np.sum(psr_vals * weights) / total_weight),
        "OSR": float(np.sum(osr_vals * weights) / total_weight),
    }
