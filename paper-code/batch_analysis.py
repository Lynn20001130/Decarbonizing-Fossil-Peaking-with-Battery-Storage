"""
Batch Monte Carlo Storage + 3R Integrated Computation

Workflow:
  For each simulation file (process-level parallelism):
      Read Parquet demand data
      Per-province storage optimization -> compute 3R metrics
      Weighted aggregation to national 3R
  Output 3R results CSV per grid point

Performance optimizations:
  1. Numba JIT (nogil=True) for storage simulation
  2. All data pre-converted to numpy arrays
  3. Parquet input format
  4. No intermediate file output; storage -> 3R computed in-process
"""

import os
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config import CONFIG, GAS_POWER_PER_M3, PARALLEL_CONFIG
from utils import (
    load_curtailment_data,
    load_electricity_prices,
    load_gas_prices,
    get_price,
    calculate_gas_unit_cost,
)
from storage_simulator import (
    StorageSimulator,
    simulate_single_year_numba,
    check_should_stop_numba,
)
from metrics import compute_province_3r, aggregate_province_3r


# ---------------------------------------------------------------------------
# Worker shared state (module-level dict for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_worker_shared = {}


def _init_worker(common_provinces, common_month_days, curtailment_dict,
                 province_elec_prices, province_gas_prices, config):
    """Initialize worker process with shared data.

    Parameters
    ----------
    common_provinces : list[str]
        Province names present in both simulation data and config.
    common_month_days : list[str]
        Month-day labels (e.g. ['01-01', '01-02', ...]).
    curtailment_dict : dict
        Mapping province -> numpy array of daily curtailment (kWh).
    province_elec_prices : dict
        Mapping province -> electricity price (Yuan/kWh).
    province_gas_prices : dict
        Mapping province -> gas price (Yuan/m^3).
    config : dict
        Full configuration dictionary.
    """
    _worker_shared['common_provinces'] = common_provinces
    _worker_shared['common_month_days'] = common_month_days
    _worker_shared['curtailment_dict'] = curtailment_dict
    _worker_shared['province_elec_prices'] = province_elec_prices
    _worker_shared['province_gas_prices'] = province_gas_prices
    _worker_shared['config'] = config
    _worker_shared['simulator'] = StorageSimulator(config)
    _worker_shared['gas_power_per_m3'] = (
        config['gas']['heat_value']
        * config['gas']['generation_efficiency']
        / 3600.0
    )
    _worker_shared['generator_cost'] = config['gas']['generator_cost']


def _compute_one_file(sim_file_str):
    """Process one simulation file.

    Steps:
        1. Read Parquet (provinces as index, month-days as columns)
        2. Filter to INCLUDED_PROVINCES
        3. Per-province: storage optimization -> compute 3R -> release memory
        4. Aggregate to weighted national 3R

    Parameters
    ----------
    sim_file_str : str
        Path to the simulation Parquet file.

    Returns
    -------
    dict or None
        ``{'CSR': val, 'PSR': val, 'OSR': val}`` on success, ``None`` on
        failure.
    """
    common_provinces = _worker_shared['common_provinces']
    common_month_days = _worker_shared['common_month_days']
    curtailment_dict = _worker_shared['curtailment_dict']
    province_elec_prices = _worker_shared['province_elec_prices']
    province_gas_prices = _worker_shared['province_gas_prices']
    simulator = _worker_shared['simulator']
    gas_power_per_m3 = _worker_shared['gas_power_per_m3']
    generator_cost = _worker_shared['generator_cost']

    try:
        # 1. Read Parquet
        df = pd.read_parquet(sim_file_str)

        # Provinces as index, month-days as columns
        available_provinces = [p for p in common_provinces if p in df.index]
        available_month_days = [md for md in common_month_days
                                if md in df.columns]

        if not available_provinces or not available_month_days:
            return None

        # 2. Per-province storage optimization and 3R computation
        province_3r_list = []

        for province in available_provinces:
            # Extract demand array for this province
            demand_arr = df.loc[province, available_month_days].values.astype(
                np.float64
            )

            # Get curtailment array for this province
            curtailment_arr = curtailment_dict.get(province)
            if curtailment_arr is None:
                continue

            # Ensure arrays have the same length
            n = min(len(demand_arr), len(curtailment_arr))
            demand_arr = demand_arr[:n]
            curtailment_arr = curtailment_arr[:n]

            # Replace NaN with 0
            demand_arr = np.nan_to_num(demand_arr, nan=0.0)
            curtailment_arr = np.nan_to_num(curtailment_arr, nan=0.0)

            # Skip if no demand
            if np.sum(demand_arr) < 1e-6:
                continue

            # Look up prices for this province
            try:
                elec_price = get_price(province_elec_prices, province)
                gas_price = get_price(province_gas_prices, province)
            except KeyError:
                continue

            # Storage optimization (returns list of dicts)
            results = simulator.search_optimal_storage(
                demand_arr, curtailment_arr,
                elec_price, gas_price,
                gas_power_per_m3, generator_cost,
            )

            # Compute 3R for this province
            province_3r = compute_province_3r(
                results, gas_price, elec_price,
                gas_power_per_m3, generator_cost,
            )

            if province_3r is not None:
                province_3r_list.append(province_3r)

            # Release memory for this province
            del demand_arr, curtailment_arr, results

        if not province_3r_list:
            return None

        # 3. Aggregate to weighted national 3R
        national_3r = aggregate_province_3r(province_3r_list)
        return national_3r

    except Exception:
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    """Main entry point.

    Steps:
        1. Load shared data (curtailment, prices)
        2. Discover grid directories and simulation files
        3. Determine common provinces and month-days from first sim file
        4. Pre-convert curtailment to numpy arrays
        5. Numba JIT warmup
        6. Build flat task list: [(sim_file, grid_name, sub_dir_name), ...]
        7. Process-level parallel computation with ProcessPoolExecutor
           - Use batch_size = num_workers * 2
           - Submit tasks incrementally (fill queue as tasks complete)
           - Progress logging every 10 files
        8. Output per-grid 3R results CSV
           - Columns: avg_run_length, sim_id, CSR, PSR, OSR
           - Sort by (avg_run_length, sim_id)
           - File: {grid_name}_3R_results.csv
    """
    start_time = time.time()

    # ---- Paths ----
    data_dir = CONFIG['paths']['DATA_DIR']
    output_dir = CONFIG['paths']['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)

    # ==================================================================
    # Stage 1: Load shared data
    # ==================================================================
    print("=" * 60)
    print("Stage 1: Loading shared data (curtailment, prices)")
    print("=" * 60)

    # Load electricity and gas prices
    elec_price_path = CONFIG['paths']['electricity_price_xlsx']
    gas_price_path = CONFIG['paths']['gas_price_xlsx']

    province_elec_prices = load_electricity_prices(elec_price_path)
    province_gas_prices = load_gas_prices(gas_price_path)

    # Load curtailment data per province
    included_provinces = CONFIG['provinces']
    curtailment_raw = {}
    for province in included_provinces:
        curt_path = CONFIG['paths']['curtailment_csv'].format(
            province=province
        )
        if os.path.exists(curt_path):
            curt_df = load_curtailment_data(curt_path)
            curtailment_raw[province] = curt_df
        else:
            print(f"  [Warning] Curtailment file not found for {province}: "
                  f"{curt_path}")

    print(f"  Loaded curtailment data for {len(curtailment_raw)} provinces")

    # ==================================================================
    # Stage 2: Discover grid directories and simulation files
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 2: Discovering grid directories and simulation files")
    print("=" * 60)

    simulation_dir = os.path.join(output_dir, "simulations")
    if not os.path.isdir(simulation_dir):
        print(f"  [Error] Simulation directory not found: {simulation_dir}")
        sys.exit(1)

    # Each sub-directory under simulations/ is a "grid"
    grid_dirs = sorted([
        d for d in os.listdir(simulation_dir)
        if os.path.isdir(os.path.join(simulation_dir, d))
    ])

    if not grid_dirs:
        print("  [Error] No grid directories found")
        sys.exit(1)

    print(f"  Found {len(grid_dirs)} grid directories")

    # Build task list: (sim_file_path, grid_name, sub_dir_name)
    task_list = []
    for grid_name in grid_dirs:
        grid_path = os.path.join(simulation_dir, grid_name)
        sub_dirs = sorted([
            sd for sd in os.listdir(grid_path)
            if os.path.isdir(os.path.join(grid_path, sd))
        ])
        for sub_dir_name in sub_dirs:
            sub_dir_path = os.path.join(grid_path, sub_dir_name)
            parquet_files = sorted([
                f for f in os.listdir(sub_dir_path)
                if f.endswith('.parquet')
            ])
            for pf in parquet_files:
                sim_file = os.path.join(sub_dir_path, pf)
                task_list.append((sim_file, grid_name, sub_dir_name))

    print(f"  Total simulation files to process: {len(task_list)}")

    if not task_list:
        print("  [Error] No simulation Parquet files found")
        sys.exit(1)

    # ==================================================================
    # Stage 3: Determine common provinces and month-days
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 3: Determining common provinces and month-days")
    print("=" * 60)

    first_file = task_list[0][0]
    first_df = pd.read_parquet(first_file)

    # Provinces = intersection of simulation index and included provinces
    common_provinces = [
        p for p in included_provinces if p in first_df.index
    ]
    common_month_days = list(first_df.columns)

    print(f"  Common provinces: {len(common_provinces)}")
    print(f"  Month-day columns: {len(common_month_days)}")
    del first_df

    # ==================================================================
    # Stage 4: Pre-convert curtailment to numpy arrays
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 4: Pre-converting curtailment data to numpy arrays")
    print("=" * 60)

    curtailment_dict = {}
    for province in common_provinces:
        if province in curtailment_raw:
            curt_df = curtailment_raw[province]
            # Convert to a 1D numpy array of daily curtailment values
            # Use numeric columns only
            numeric_cols = curt_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                arr = curt_df[numeric_cols].values.flatten().astype(np.float64)
                curtailment_dict[province] = np.nan_to_num(arr, nan=0.0)

    print(f"  Converted curtailment arrays for {len(curtailment_dict)} "
          f"provinces")

    # Release raw data
    del curtailment_raw

    # ==================================================================
    # Stage 5: Numba JIT warmup
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 5: Numba JIT warmup")
    print("=" * 60)

    warmup_demand = np.array([100.0, 200.0, 150.0], dtype=np.float64)
    warmup_curtail = np.array([300.0, 100.0, 200.0], dtype=np.float64)
    _, _, ws, wsp = simulate_single_year_numba(
        warmup_demand, warmup_curtail,
        1000.0, 0.5,
        CONFIG['storage']['charge_eff'],
        CONFIG['storage']['discharge_eff'],
        CONFIG['storage']['soc_min'],
        CONFIG['storage']['soc_max'],
    )
    _ = check_should_stop_numba(warmup_demand, ws, wsp)
    print("  Numba JIT warmup complete")

    # ==================================================================
    # Stage 6-7: Parallel computation
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 6-7: Processing simulation files in parallel")
    print("=" * 60)

    num_workers = PARALLEL_CONFIG.get('num_workers', os.cpu_count() or 4)
    batch_size = num_workers * 2

    print(f"  Workers: {num_workers}, Batch size: {batch_size}")
    print(f"  Total files: {len(task_list)}")

    # Group tasks by grid
    grid_results = {}  # grid_name -> list of (sub_dir_name, sim_id, result)

    # Extract sim_file paths and metadata
    sim_files = [t[0] for t in task_list]
    sim_meta = [(t[1], t[2], os.path.splitext(os.path.basename(t[0]))[0])
                for t in task_list]

    completed_count = 0
    error_count = 0
    compute_start = time.time()

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(
            common_provinces,
            common_month_days,
            curtailment_dict,
            province_elec_prices,
            province_gas_prices,
            CONFIG,
        ),
    ) as executor:

        # Submit tasks incrementally
        futures = {}
        task_iter = iter(range(len(task_list)))

        # Fill initial batch
        for _ in range(min(batch_size, len(task_list))):
            idx = next(task_iter, None)
            if idx is None:
                break
            future = executor.submit(_compute_one_file, sim_files[idx])
            futures[future] = idx

        # Process completed futures and submit new ones
        for future in as_completed(futures):
            idx = futures[future]
            grid_name, sub_dir_name, sim_id = sim_meta[idx]

            try:
                result = future.result()
            except Exception:
                traceback.print_exc()
                print(f"  [Error] Failed to process: {sim_files[idx]}")
                result = None
                error_count += 1

            if result is not None:
                if grid_name not in grid_results:
                    grid_results[grid_name] = []

                # Parse avg_run_length from sub_dir_name
                if sub_dir_name.startswith("avg_run_length_"):
                    avg_run_length_str = sub_dir_name[len("avg_run_length_"):]
                else:
                    avg_run_length_str = sub_dir_name

                try:
                    avg_run_length = float(avg_run_length_str)
                except ValueError:
                    avg_run_length = 0.0

                grid_results[grid_name].append({
                    'avg_run_length': avg_run_length,
                    'sim_id': sim_id,
                    'CSR': result['CSR'],
                    'PSR': result['PSR'],
                    'OSR': result['OSR'],
                })

            completed_count += 1

            # Progress logging every 10 files
            if completed_count % 10 == 0 or completed_count == len(task_list):
                elapsed = time.time() - compute_start
                rate = completed_count / elapsed if elapsed > 0 else 0
                remaining = len(task_list) - completed_count
                eta = remaining / rate if rate > 0 else 0
                print(f"  Progress: {completed_count}/{len(task_list)} files "
                      f"({rate:.1f} files/s, ETA: {eta:.0f}s)")

            # Submit next task
            next_idx = next(task_iter, None)
            if next_idx is not None:
                new_future = executor.submit(
                    _compute_one_file, sim_files[next_idx]
                )
                futures[new_future] = next_idx

    print(f"  Completed: {completed_count}, Errors: {error_count}")

    # ==================================================================
    # Stage 8: Output per-grid 3R results CSV
    # ==================================================================
    print()
    print("=" * 60)
    print("Stage 8: Writing 3R results CSV files")
    print("=" * 60)

    results_dir = os.path.join(output_dir, "3R_results")
    os.makedirs(results_dir, exist_ok=True)

    for grid_name, records in grid_results.items():
        if not records:
            print(f"  [Warning] No valid results for grid: {grid_name}")
            continue

        results_df = pd.DataFrame(records)
        results_df = results_df.sort_values(
            by=['avg_run_length', 'sim_id']
        ).reset_index(drop=True)

        # Columns: avg_run_length, sim_id, CSR, PSR, OSR
        results_df = results_df[['avg_run_length', 'sim_id',
                                  'CSR', 'PSR', 'OSR']]

        out_path = os.path.join(results_dir,
                                f"{grid_name}_3R_results.csv")
        results_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"  Written: {out_path} ({len(results_df)} rows)")

    # ---- Summary ----
    total_elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Batch analysis complete. Total time: {total_elapsed:.1f}s")
    print(f"  Grids processed: {len(grid_results)}")
    print(f"  Files processed: {completed_count}")
    print(f"  Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
