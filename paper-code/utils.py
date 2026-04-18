"""
Shared utility functions for the natural gas peak-shaving analysis pipeline.
Provides I/O helpers, price lookups, and financial calculations.
"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------

def calculate_crf(discount_rate: float, years: int) -> float:
    """
    Calculate the Capital Recovery Factor (CRF).

    Parameters
    ----------
    discount_rate : float
        Annual discount rate (e.g. 0.036 for 3.6%).
    years : int
        Project lifetime in years.

    Returns
    -------
    float
        The CRF value.
    """
    return (discount_rate * (1 + discount_rate) ** years) / \
           ((1 + discount_rate) ** years - 1)


def calculate_gas_unit_cost(gas_price: float,
                            gas_power_per_m3: float,
                            generator_cost: float) -> float:
    """
    Calculate the unit cost of gas-fired electricity generation.

    Parameters
    ----------
    gas_price : float
        Natural gas price per cubic metre (Yuan/m^3).
    gas_power_per_m3 : float
        Electricity output per cubic metre of gas (kWh/m^3).
    generator_cost : float
        Generator operation & maintenance cost (Yuan/kWh).

    Returns
    -------
    float
        Total unit cost of gas-generated electricity (Yuan/kWh).
    """
    fuel_cost_per_kwh = gas_price / gas_power_per_m3
    return fuel_cost_per_kwh + generator_cost


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_curtailment_data(filepath: str) -> pd.DataFrame:
    """
    Load provincial daily curtailment data from a CSV file.

    The CSV is expected to contain daily curtailed energy values
    (one row per day, columns per province or a single-province file).

    Parameters
    ----------
    filepath : str
        Path to the curtailment CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the curtailment data.
    """
    df = pd.read_csv(filepath)
    print(f"[load_curtailment_data] Loaded {len(df)} rows from {filepath}")
    return df


def load_electricity_prices(filepath: str) -> dict:
    """
    Load electricity prices from an Excel file.

    Uses fuzzy column-name matching to locate the province name column
    and the price column, tolerating minor header variations.

    Parameters
    ----------
    filepath : str
        Path to the electricity price Excel file.

    Returns
    -------
    dict
        Mapping of province name (str) -> electricity price (float, Yuan/kWh).
    """
    df = pd.read_excel(filepath)

    # Fuzzy match for province column
    province_col = None
    price_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in ['province', 'region', 'area',
                                           'name', 'sheng', 'diqu']):
            province_col = col
        if any(kw in col_lower for kw in ['price', 'cost', 'rate',
                                           'jiage', 'danjia', 'dianjia']):
            price_col = col

    if province_col is None:
        # Fallback: assume first column is province
        province_col = df.columns[0]
        print(f"[load_electricity_prices] Province column not identified; "
              f"falling back to first column: '{province_col}'")

    if price_col is None:
        # Fallback: assume second column is price
        price_col = df.columns[1]
        print(f"[load_electricity_prices] Price column not identified; "
              f"falling back to second column: '{price_col}'")

    prices = dict(zip(df[province_col].astype(str), df[price_col].astype(float)))
    print(f"[load_electricity_prices] Loaded prices for {len(prices)} provinces")
    return prices


def load_gas_prices(filepath: str) -> dict:
    """
    Load natural gas prices from an Excel file.

    Parameters
    ----------
    filepath : str
        Path to the gas price Excel file.

    Returns
    -------
    dict
        Mapping of province name (str) -> gas price (float, Yuan/m^3).
    """
    df = pd.read_excel(filepath)

    # Use first two columns as province name and gas price
    province_col = df.columns[0]
    price_col = df.columns[1]

    prices = dict(zip(df[province_col].astype(str), df[price_col].astype(float)))
    print(f"[load_gas_prices] Loaded gas prices for {len(prices)} provinces")
    return prices


# ---------------------------------------------------------------------------
# Province name matching
# ---------------------------------------------------------------------------

# Suffixes to strip for fuzzy matching
_PROVINCE_SUFFIXES = re.compile(
    r'(省|市|自治区|壮族|回族|维吾尔)'
)


def _normalize_province(name: str) -> str:
    """Strip common administrative suffixes for matching."""
    return _PROVINCE_SUFFIXES.sub('', name).strip()


def get_price(prices_dict: dict, province_name: str) -> float:
    """
    Look up a price by province name with fuzzy matching.

    Strips administrative suffixes (e.g. Province, City, Autonomous Region
    identifiers) from both the query and the dictionary keys so that
    '内蒙古自治区' matches '内蒙古', etc.

    Parameters
    ----------
    prices_dict : dict
        Mapping of province name -> price value.
    province_name : str
        The province name to look up.

    Returns
    -------
    float
        The matched price value.

    Raises
    ------
    KeyError
        If no matching province is found.
    """
    # Try exact match first
    if province_name in prices_dict:
        return prices_dict[province_name]

    # Fuzzy match by stripping suffixes
    query_norm = _normalize_province(province_name)
    for key, value in prices_dict.items():
        if _normalize_province(key) == query_norm:
            return value

    raise KeyError(
        f"Province '{province_name}' not found in price dictionary. "
        f"Available keys: {list(prices_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Peak demand loading
# ---------------------------------------------------------------------------

def load_peak_demand(filepath: str) -> pd.DataFrame:
    """
    Load peak demand data from an Excel file.

    The raw file has provinces as rows and time periods as columns.
    This function transposes the data so that rows represent days,
    adds a 'month_day' column, and removes February 29 entries
    to standardise on a 365-day year.

    Parameters
    ----------
    filepath : str
        Path to the peak demand Excel file.

    Returns
    -------
    pd.DataFrame
        Transposed DataFrame with a 'month_day' column and no Feb-29 rows.
    """
    df = pd.read_excel(filepath, index_col=0)

    # Transpose: provinces become columns, time periods become rows
    df = df.T.reset_index(drop=True)

    # Build a month_day index for a non-leap year (365 days)
    dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    month_day = dates.strftime('%m-%d').tolist()

    # If the data has 366 rows, remove the Feb-29 entry
    feb29_idx = None
    if len(df) == 366:
        try:
            feb29_idx = month_day.index('02-29') if '02-29' in month_day else None
        except ValueError:
            pass

        # Use a leap-year range to find Feb 29 position
        leap_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        leap_md = leap_dates.strftime('%m-%d').tolist()
        if '02-29' in leap_md:
            idx_to_drop = leap_md.index('02-29')
            df = df.drop(index=idx_to_drop).reset_index(drop=True)
            print("[load_peak_demand] Removed February 29 row (leap year data)")

    # Ensure we have exactly 365 rows
    if len(df) != 365:
        print(f"[load_peak_demand] Warning: expected 365 rows, got {len(df)}")

    # Re-generate month_day for a standard 365-day year
    standard_dates = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    df.insert(0, 'month_day', standard_dates.strftime('%m-%d').tolist()[:len(df)])

    print(f"[load_peak_demand] Loaded peak demand with shape {df.shape}")
    return df
