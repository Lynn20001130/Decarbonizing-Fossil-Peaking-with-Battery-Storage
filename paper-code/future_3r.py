"""
Future CSR/PSR/OSR Calculation (2025-2035)

Computes 3R replacement metrics for each province across years 2025-2035
using storage optimization results from the yearly analysis.
Outputs provincial and national weighted/median summaries.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import interpolate
import warnings

warnings.filterwarnings('ignore')

# Configuration
YEARS = list(range(2025, 2036))

# Path template - user should adjust
STORAGE_PATH_TEMPLATE = "path/to/yearly_analysis/{year}/storage_detail.xlsx"
ENERGY_PRICE_PATH = Path("path/to/energy_prices.xlsx")
ELECTRICITY_PRICE_PATH = Path("path/to/electricity_prices.xlsx")
OUTPUT_DIR = Path("path/to/output")

# Gas parameters
GAS_HEAT_VALUE = 35544
GAS_EFFICIENCY = 0.536
GENERATOR_COST = 0.43
GAS_POWER_PER_M3 = GAS_HEAT_VALUE * GAS_EFFICIENCY / 3600


def calculate_gas_unit_cost(gas_price):
    """Calculate gas generation unit cost (yuan/kWh)."""
    return gas_price / GAS_POWER_PER_M3 + GENERATOR_COST


def extrapolate_to_y_axis(x_data, y_data):
    """Linearly extrapolate curve to x=0."""
    if len(x_data) < 2:
        return x_data, y_data
    x1, x2 = x_data[0], x_data[1]
    y1, y2 = y_data[0], y_data[1]
    slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
    y0 = max(y1 - slope * x1, 0)
    return np.insert(x_data, 0, 0), np.insert(y_data, 0, y0)


def find_second_intersection(x_storage, y_storage, x_gas, y_gas):
    """Find intersection between storage cost curve and gas cost line."""
    if len(x_storage) < 2 or len(x_gas) < 2:
        return None
    gas_interp = interpolate.interp1d(x_gas, y_gas, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
    diff = y_storage - gas_interp(x_storage)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    for idx in sign_changes:
        x1, x2 = x_storage[idx], x_storage[idx + 1]
        d1, d2 = diff[idx], diff[idx + 1]
        if d2 - d1 != 0:
            x_intersect = x1 - d1 * (x2 - x1) / (d2 - d1)
            if x_intersect > 0.05:
                return x_intersect
    return None


def find_marginal_intersection(x_marginal, y_marginal, x_gas, y_gas):
    """Find intersection for marginal cost curve (excludes near-origin)."""
    if len(x_marginal) < 2 or len(x_gas) < 2:
        return None
    gas_interp = interpolate.interp1d(x_gas, y_gas, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
    diff = y_marginal - gas_interp(x_marginal)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None
    x_range = x_marginal[-1] - x_marginal[0]
    threshold = x_marginal[0] + x_range * 0.05 if x_range > 0 else 0.05
    for idx in sign_changes:
        x1, x2 = x_marginal[idx], x_marginal[idx + 1]
        d1, d2 = diff[idx], diff[idx + 1]
        if d2 - d1 != 0:
            x_intersect = x1 - d1 * (x2 - x1) / (d2 - d1)
            if x_intersect > threshold:
                return x_intersect
    return None


def compute_csr(storage_df, gas_price):
    """Compute Cost Storage Ratio.

    This version reads from the yearly analysis Excel output.
    The Excel has Chinese column names from the original yearly analysis:
    '调峰供应(kWh)' -> peak supply
    '年化成本(元)' -> annual cost
    '总调峰需求(kWh)' -> total demand

    Note: Column names here match the yearly_analysis.py output format.
    """
    # Support both Chinese (from original output) and English column names
    peak_col = 'peak_supply_kwh' if 'peak_supply_kwh' in storage_df.columns else '调峰供应(kWh)'
    cost_col = 'annual_cost' if 'annual_cost' in storage_df.columns else '年化成本(元)'
    demand_col = 'total_demand_kwh' if 'total_demand_kwh' in storage_df.columns else '总调峰需求(kWh)'

    peak_supply = storage_df[peak_col].values
    storage_cost = storage_df[cost_col].values
    supply_at_100 = storage_df[demand_col].values[0]
    if supply_at_100 <= 0:
        return 0.0

    valid_mask = peak_supply > 0
    x_valid = peak_supply[valid_mask]
    y_unit = storage_cost[valid_mask] / x_valid
    if len(x_valid) == 0:
        return 0.0

    gas_unit_cost = calculate_gas_unit_cost(gas_price)
    y_max = max(y_unit.max(), gas_unit_cost)
    if y_max <= 0:
        y_max = 1

    x_norm = x_valid / supply_at_100
    y_norm = y_unit / y_max
    x_norm, y_norm = extrapolate_to_y_axis(x_norm, y_norm)

    x_gas = np.array([0, 1])
    y_gas = np.array([gas_unit_cost / y_max, gas_unit_cost / y_max])

    ix = find_second_intersection(x_norm, y_norm, x_gas, y_gas)
    if ix is not None:
        return ix * 100

    gas_interp = interpolate.interp1d(x_gas, y_gas, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')
    gas_at_x = gas_interp(x_norm)
    if np.all(y_norm <= gas_at_x + 1e-6):
        return 100.0
    elif np.all(y_norm >= gas_at_x - 1e-6):
        return 0.0
    return 0.0


def compute_psr(storage_df, gas_price, electricity_price):
    """Compute Profit Storage Ratio."""
    peak_col = 'peak_supply_kwh' if 'peak_supply_kwh' in storage_df.columns else '调峰供应(kWh)'
    profit_col = 'unit_profit' if 'unit_profit' in storage_df.columns else '单位供电利润(元/kWh)'
    demand_col = 'total_demand_kwh' if 'total_demand_kwh' in storage_df.columns else '总调峰需求(kWh)'

    peak_supply = storage_df[peak_col].values
    storage_unit_profit = storage_df[profit_col].values
    total_demand = storage_df[demand_col].values[0]

    valid_mask = ~pd.isna(storage_unit_profit)
    peak_supply = peak_supply[valid_mask]
    storage_unit_profit = storage_unit_profit[valid_mask]

    if len(peak_supply) == 0:
        return 0.0

    gas_unit_profit = electricity_price - calculate_gas_unit_cost(gas_price)
    supply_at_100 = total_demand if total_demand > 0 else 1000

    y_min = min(storage_unit_profit.min(), gas_unit_profit, 0)
    y_max = max(storage_unit_profit.max(), gas_unit_profit, 0)
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = 1

    x_norm = peak_supply / supply_at_100 if supply_at_100 > 0 else peak_supply
    y_norm = (storage_unit_profit - y_min) / y_range
    zero_norm = (0 - y_min) / y_range

    diff = y_norm - zero_norm
    for idx in range(len(diff) - 1):
        if diff[idx] >= 0 and diff[idx + 1] < 0:
            x1, x2 = x_norm[idx], x_norm[idx + 1]
            d1, d2 = diff[idx], diff[idx + 1]
            if d1 - d2 != 0:
                return (x1 + d1 * (x2 - x1) / (d1 - d2)) * 100

    if np.all(y_norm >= zero_norm - 1e-6):
        return x_norm[-1] * 100
    elif np.all(y_norm <= zero_norm + 1e-6):
        return 0.0
    return 0.0


def compute_osr(storage_df, gas_price):
    """Compute Optimal Storage Ratio (marginal cost intersection)."""
    peak_col = 'peak_supply_kwh' if 'peak_supply_kwh' in storage_df.columns else '调峰供应(kWh)'
    cost_col = 'annual_cost' if 'annual_cost' in storage_df.columns else '年化成本(元)'
    demand_col = 'total_demand_kwh' if 'total_demand_kwh' in storage_df.columns else '总调峰需求(kWh)'

    # Also need satisfy_rate for OSR x-axis
    rate_col = 'satisfy_rate_pct' if 'satisfy_rate_pct' in storage_df.columns else '调峰满足率(%)'

    peak_supply = storage_df[peak_col].values
    storage_cost = storage_df[cost_col].values
    supply_at_100 = storage_df[demand_col].values[0]
    if supply_at_100 <= 0:
        return 0.0

    sort_indices = np.argsort(peak_supply)
    peak_supply = peak_supply[sort_indices]
    storage_cost = storage_cost[sort_indices]

    # Get satisfy_rate if available
    if rate_col in storage_df.columns:
        satisfy_rate = storage_df[rate_col].values[sort_indices]
    else:
        satisfy_rate = peak_supply / supply_at_100 * 100

    _, unique_indices = np.unique(peak_supply, return_index=True)
    unique_indices = np.sort(unique_indices)
    x_unique = peak_supply[unique_indices]
    y_unique = storage_cost[unique_indices]
    satisfy_unique = satisfy_rate[unique_indices]

    max_satisfy_rate = 100.0

    if len(x_unique) < 2:
        return 0.0

    delta_cost = np.diff(y_unique)
    delta_supply = np.diff(x_unique)
    valid_mask = delta_supply > 0
    delta_cost = delta_cost[valid_mask]
    delta_supply = delta_supply[valid_mask]

    if len(delta_cost) == 0:
        return 0.0

    marginal_cost = delta_cost / delta_supply

    satisfy_left = satisfy_unique[:-1][valid_mask]
    satisfy_right = satisfy_unique[1:][valid_mask]
    x_marginal_pct = (satisfy_left + satisfy_right) / 2

    gas_unit_cost = calculate_gas_unit_cost(gas_price)
    y_max = max(marginal_cost.max(), gas_unit_cost)
    if y_max <= 0:
        y_max = 1

    y_marginal_norm = marginal_cost / y_max
    x_marginal_pct, y_marginal_norm = extrapolate_to_y_axis(x_marginal_pct, y_marginal_norm)

    x_gas = np.array([0, max_satisfy_rate])
    y_gas_norm = np.array([gas_unit_cost / y_max, gas_unit_cost / y_max])

    intersection_x = find_marginal_intersection(x_marginal_pct, y_marginal_norm, x_gas, y_gas_norm)
    if intersection_x is not None:
        return (intersection_x / max_satisfy_rate) * 100 if max_satisfy_rate > 0 else 0.0

    if len(x_marginal_pct) > 1:
        gas_interp = interpolate.interp1d(x_gas, y_gas_norm, kind='linear',
                                           bounds_error=False, fill_value='extrapolate')
        gas_at_x = gas_interp(x_marginal_pct)
        if np.all(y_marginal_norm <= gas_at_x + 1e-6):
            return 100.0
        elif np.all(y_marginal_norm >= gas_at_x - 1e-6):
            return 0.0
    return 0.0


def load_gas_prices():
    """Load gas prices from Excel."""
    df = pd.read_excel(ENERGY_PRICE_PATH)
    # Support both Chinese and English column names
    province_col = '省份' if '省份' in df.columns else 'province'
    price_col = '天然气价格（元/立方米）' if '天然气价格（元/立方米）' in df.columns else 'gas_price'
    return dict(zip(df[province_col], df[price_col]))


def load_electricity_prices():
    """Load electricity prices from Excel with fuzzy column matching."""
    df = pd.read_excel(ELECTRICITY_PRICE_PATH)
    province_col = None
    price_col = None
    for col in df.columns:
        if any(k in col for k in ['省', '自治区', '直辖市', 'province']):
            province_col = col
        if any(k in col for k in ['基准价', '电价', 'price']):
            price_col = col
    if province_col is None or price_col is None:
        print(f"  Error: cannot identify price file columns: {list(df.columns)}")
        return {}
    prices = {}
    for _, row in df.iterrows():
        province = str(row[province_col]).strip()
        price = float(row[price_col]) if pd.notna(row[price_col]) else 0.0
        prices[province] = price
    return prices


def get_electricity_price(electricity_prices, province_name):
    """Fuzzy match electricity price for a province."""
    if province_name in electricity_prices:
        return electricity_prices[province_name]
    short = province_name.replace('省', '').replace('市', '').replace('自治区', '').replace('壮族', '').replace('回族', '').replace('维吾尔', '')
    for key, value in electricity_prices.items():
        key_short = key.replace('省', '').replace('市', '').replace('自治区', '').replace('壮族', '').replace('回族', '').replace('维吾尔', '')
        if short in key_short or key_short in short:
            return value
    return 0.0


def compute_all(gas_prices, electricity_prices):
    """Compute 3R for all provinces across all years.
    Returns {province: {'csr': {year: val}, 'psr': {year: val}, 'osr': {year: val}, 'peak_demand': {year: val}}}
    """
    results = {}
    for year in YEARS:
        path = Path(STORAGE_PATH_TEMPLATE.format(year=year))
        print(f"Processing year {year}...")
        if not path.exists():
            print(f"  File not found, skipping: {path}")
            continue

        excel_file = pd.ExcelFile(path)
        for province in excel_file.sheet_names:
            if province not in gas_prices:
                continue
            df = pd.read_excel(excel_file, sheet_name=province)
            gas_price = gas_prices[province]
            elec_price = get_electricity_price(electricity_prices, province)

            if province not in results:
                results[province] = {'csr': {}, 'psr': {}, 'osr': {}, 'peak_demand': {}}

            results[province]['csr'][year] = round(compute_csr(df, gas_price), 2)
            results[province]['psr'][year] = round(compute_psr(df, gas_price, elec_price), 2)
            results[province]['osr'][year] = round(compute_osr(df, gas_price), 2)

            demand_col = 'total_demand_kwh' if 'total_demand_kwh' in df.columns else '总调峰需求(kWh)'
            results[province]['peak_demand'][year] = df[demand_col].values[0]
        print(f"  Done")
    return results


def compute_national(results):
    """Compute national weighted average and median for each year.
    Returns {year: {'CSR_weighted_avg': val, 'CSR_median': val, 'PSR_weighted_avg': val, ...}}
    """
    national = {}
    for year in YEARS:
        csr_vals, psr_vals, osr_vals, weights = [], [], [], []
        for province, data in results.items():
            csr = data['csr'].get(year)
            psr = data['psr'].get(year)
            osr = data['osr'].get(year)
            w = data['peak_demand'].get(year)
            if all(v is not None for v in [csr, psr, osr, w]) and w > 0:
                csr_vals.append(csr)
                psr_vals.append(psr)
                osr_vals.append(osr)
                weights.append(w)

        if not csr_vals:
            continue

        w_arr = np.array(weights)
        w_norm = w_arr / w_arr.sum()
        national[year] = {
            'CSR_weighted_avg': round(np.sum(np.array(csr_vals) * w_norm), 2),
            'CSR_median': round(np.median(csr_vals), 2),
            'PSR_weighted_avg': round(np.sum(np.array(psr_vals) * w_norm), 2),
            'PSR_median': round(np.median(psr_vals), 2),
            'OSR_weighted_avg': round(np.sum(np.array(osr_vals) * w_norm), 2),
            'OSR_median': round(np.median(osr_vals), 2),
        }
    return national


def main():
    """Main entry point."""
    print("=" * 60)
    print("Future CSR/PSR/OSR Calculation (2025-2035)")
    print("=" * 60)

    gas_prices = load_gas_prices()
    electricity_prices = load_electricity_prices()
    print(f"Gas prices: {len(gas_prices)} provinces, Electricity prices: {len(electricity_prices)} provinces")

    results = compute_all(gas_prices, electricity_prices)
    national = compute_national(results)

    # Output summary Excel
    summary_path = OUTPUT_DIR / "province_yearly_CSR_PSR_OSR_summary.xlsx"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        for year in YEARS:
            rows = []
            for province in sorted(results.keys()):
                csr = results[province]['csr'].get(year)
                psr = results[province]['psr'].get(year)
                osr = results[province]['osr'].get(year)
                if any(v is not None for v in [csr, psr, osr]):
                    rows.append({'province': province, 'CSR': csr, 'PSR': psr, 'OSR': osr})
            if rows:
                pd.DataFrame(rows).to_excel(writer, sheet_name=str(year), index=False)

        national_rows = []
        for year in YEARS:
            if year in national:
                row = {'year': year}
                row.update(national[year])
                national_rows.append(row)
        if national_rows:
            pd.DataFrame(national_rows).to_excel(writer, sheet_name='national', index=False)

    print(f"\nSummary saved: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
