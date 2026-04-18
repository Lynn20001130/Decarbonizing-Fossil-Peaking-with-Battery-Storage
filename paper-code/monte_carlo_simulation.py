"""
Peak-shaving demand Monte Carlo simulation using three-state Markov chains.

Three states:
  State 0: No peak-shaving (demand = 0)
  State 1: Low peak-shaving (0 < x <= Q75)
  State 2: High peak-shaving (x > Q75)
where Q75 is the 75th percentile of non-zero values per province per season.

Transition matrix 3x3: T[i][j] = P(state i -> state j)

Frequency adjustment (decay_factor): Only modifies row 0 of the transition matrix.
  P(0->0) is multiplied by decay_factor, released probability redistributed
  to P(0->1) and P(0->2) proportionally.
  decay=1.0: baseline, decay=0.5: P(0->0) halved, decay=0.0: immediate transition

Output: Parquet files organized by grid/run-length subdirectories.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.linalg import null_space

from config import CONFIG, MC_CONFIG

# ---------------------------------------------------------------------------
# Grid constants for parameter sweep
# ---------------------------------------------------------------------------
DECAY_START = 1.0
DECAY_END = 0.0
DECAY_STEP = -0.05

INTENSITY_START = 0.5
INTENSITY_END = 2.0
INTENSITY_STEP = 0.1


# ===================================================================
# Data loading
# ===================================================================

def load_peak_demand(filepath):
    """Load peak demand Excel file.

    Read with index_col=0, transpose, sort by datetime index,
    add 'month_day' column (MM-DD format), remove Feb 29.
    Returns DataFrame.
    """
    df = pd.read_excel(filepath, index_col=0)
    df = df.T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df["month_day"] = df.index.strftime("%m-%d")
    # Remove Feb 29 entries
    df = df[df["month_day"] != "02-29"].copy()
    return df


# ===================================================================
# Season helpers
# ===================================================================

def get_season_for_month(month, seasons_dict):
    """Return season name for a given month number."""
    for season_name, months in seasons_dict.items():
        if month in months:
            return season_name
    raise ValueError(f"Month {month} not found in any season definition")


# ===================================================================
# State classification and transition matrix
# ===================================================================

def classify_states(season_data, q75):
    """Classify demand data into 3 states.

    State 0: demand == 0 (no peak-shaving)
    State 1: 0 < demand <= q75 (low peak-shaving)
    State 2: demand > q75 (high peak-shaving)

    Returns numpy int array.
    """
    values = np.asarray(season_data, dtype=float)
    states = np.zeros(len(values), dtype=int)
    states[(values > 0) & (values <= q75)] = 1
    states[values > q75] = 2
    return states


def estimate_transition_matrix(states, n_states=3):
    """Estimate n_states x n_states transition probability matrix from state sequence.

    T[i][j] = count(i->j) / sum(count(i->*)).
    Default to state 0 for unseen states (row = [1, 0, ..., 0]).
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for t in range(len(states) - 1):
        i, j = states[t], states[t + 1]
        counts[i, j] += 1.0

    T = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            T[i] = counts[i] / row_sum
        else:
            # Unseen state defaults to staying at state 0
            T[i, 0] = 1.0
    return T


def compute_steady_state(T):
    """Compute stationary distribution of transition matrix.

    Solves (T^T - I) pi = 0 with sum(pi) = 1.
    Returns numpy array.
    """
    n = T.shape[0]
    A = T.T - np.eye(n)
    ns = null_space(A)
    if ns.shape[1] == 0:
        # Fallback: uniform distribution
        return np.ones(n) / n
    pi = ns[:, 0].real
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


# ===================================================================
# Distribution fitting
# ===================================================================

def fit_distribution(values, candidate_dists):
    """Fit best distribution to positive values using AIC.

    candidate_dists: list of scipy.stats distribution names.
    Forces loc=0 (floc=0).
    Falls back to 'empirical' if <5 values or all fits fail.
    Returns (dist_name, dist_params) or ('empirical', None).
    """
    values = np.asarray(values, dtype=float)
    values = values[values > 0]

    if len(values) < 5:
        return ("empirical", None)

    best_aic = np.inf
    best_name = None
    best_params = None

    for name in candidate_dists:
        try:
            dist = getattr(st, name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = dist.fit(values, floc=0)
            log_lik = np.sum(dist.logpdf(values, *params))
            if not np.isfinite(log_lik):
                continue
            k = len(params)
            aic = 2 * k - 2 * log_lik
            if aic < best_aic:
                best_aic = aic
                best_name = name
                best_params = params
        except Exception:
            continue

    if best_name is None:
        return ("empirical", None)
    return (best_name, best_params)


def fit_season_distributions(df, config):
    """For each province and season, fit state model and intensity distributions.

    Returns nested dict: season_params[province][season] = {
        'frequency', 'q75', 'transition_matrix', 'total_days', 'nonzero_count',
        'small_dist_name', 'small_dist_params', 'small_values', 'small_count',
        'large_dist_name', 'large_dist_params', 'large_values', 'large_count'
    }
    """
    mc = MC_CONFIG
    seasons = mc["seasons"]
    candidate_dists = mc["candidate_distributions"]
    quantile = mc["intensity_quantile"]
    provinces = config["provinces"]

    # Assign season to each row
    df["_season"] = df.index.month.map(lambda m: get_season_for_month(m, seasons))

    season_params = {}
    for province in provinces:
        if province not in df.columns:
            continue
        season_params[province] = {}
        for season_name, month_list in seasons.items():
            mask = df["_season"] == season_name
            season_values = df.loc[mask, province].values.astype(float)
            total_days = len(season_values)
            nonzero_vals = season_values[season_values > 0]
            nonzero_count = len(nonzero_vals)
            frequency = nonzero_count / total_days if total_days > 0 else 0.0

            # Q75 of non-zero values
            q75 = float(np.percentile(nonzero_vals, quantile * 100)) if nonzero_count > 0 else 0.0

            # Classify and estimate transition matrix
            states = classify_states(season_values, q75)
            T = estimate_transition_matrix(states)

            # Split non-zero values into low and high
            small_values = nonzero_vals[nonzero_vals <= q75]
            large_values = nonzero_vals[nonzero_vals > q75]

            small_dist_name, small_dist_params = fit_distribution(small_values, candidate_dists)
            large_dist_name, large_dist_params = fit_distribution(large_values, candidate_dists)

            season_params[province][season_name] = {
                "frequency": frequency,
                "q75": q75,
                "transition_matrix": T,
                "total_days": total_days,
                "nonzero_count": nonzero_count,
                "small_dist_name": small_dist_name,
                "small_dist_params": small_dist_params,
                "small_values": small_values,
                "small_count": len(small_values),
                "large_dist_name": large_dist_name,
                "large_dist_params": large_dist_params,
                "large_values": large_values,
                "large_count": len(large_values),
            }

    # Clean up temporary column
    df.drop(columns=["_season"], inplace=True)
    return season_params


# ===================================================================
# Transition matrix adjustment
# ===================================================================

def adjust_transition_matrix(T_orig, decay_factor):
    """Adjust row 0 of transition matrix.

    P(0->0) *= decay_factor
    Released probability distributed to P(0->1), P(0->2) proportionally.
    Returns adjusted copy.
    """
    T = T_orig.copy()
    p00_old = T[0, 0]
    p00_new = p00_old * decay_factor
    released = p00_old - p00_new
    T[0, 0] = p00_new

    # Distribute released probability proportionally to P(0->1) and P(0->2)
    remaining = T[0, 1] + T[0, 2]
    if remaining > 0:
        T[0, 1] += released * (T[0, 1] / remaining)
        T[0, 2] += released * (T[0, 2] / remaining)
    else:
        # If both are zero, split equally
        T[0, 1] += released / 2.0
        T[0, 2] += released / 2.0

    return T


# ===================================================================
# Day-index precomputation
# ===================================================================

def precompute_season_day_indices(month_day_list, seasons):
    """Map each season to its day indices in month_day_list.

    Returns dict: {season_name: numpy array of indices}
    """
    months_array = np.array([int(md.split("-")[0]) for md in month_day_list])
    season_day_indices = {}
    for season_name, month_list in seasons.items():
        mask = np.isin(months_array, month_list)
        season_day_indices[season_name] = np.where(mask)[0]
    return season_day_indices


# ===================================================================
# Intensity sampling
# ===================================================================

def sample_intensity(dist_name, dist_params, empirical_values, intensity_factor, size, rng):
    """Sample intensity values from fitted distribution or empirical data.

    Apply intensity_factor to scale parameter. Clamp to >= 0.
    """
    if dist_name == "empirical" or dist_params is None:
        if empirical_values is not None and len(empirical_values) > 0:
            samples = rng.choice(empirical_values, size=size, replace=True)
            samples = samples * intensity_factor
        else:
            samples = np.zeros(size)
    else:
        dist = getattr(st, dist_name)
        # Modify scale parameter (last positional param) by intensity_factor
        params = list(dist_params)
        # scipy convention: last param is scale, second-to-last is loc
        params[-1] *= intensity_factor  # scale
        samples = dist.rvs(*params, size=size, random_state=rng)

    return np.maximum(samples, 0.0)


# ===================================================================
# Vectorized Markov chain Monte Carlo
# ===================================================================

def run_markov_monte_carlo_batch(season_params, month_day_list, season_day_indices,
                                  decay_factor, intensity_factor, config, rng):
    """Run vectorized Markov chain MC simulation for all provinces and simulations.

    For each (province, season):
    - Adjust transition matrix with decay_factor
    - Initialize states from stationary distribution
    - Vectorized state transitions using cumulative probabilities
    - Sample intensities for state 1 (low) and state 2 (high)

    Returns (stacked, province_list) where stacked.shape = (n_provinces, n_sim, n_days)
    """
    n_sim = MC_CONFIG["num_simulations"]
    n_days = len(month_day_list)
    provinces = config["provinces"]
    seasons = MC_CONFIG["seasons"]

    # Filter to provinces that have fitted parameters
    province_list = [p for p in provinces if p in season_params]
    n_prov = len(province_list)

    # Output array: (n_provinces, n_sim, n_days)
    stacked = np.zeros((n_prov, n_sim, n_days), dtype=float)

    for p_idx, province in enumerate(province_list):
        for season_name, day_indices in season_day_indices.items():
            if len(day_indices) == 0:
                continue

            params = season_params[province][season_name]
            T_adj = adjust_transition_matrix(params["transition_matrix"], decay_factor)

            # Cumulative transition probabilities for vectorized lookup
            cum_T = np.cumsum(T_adj, axis=1)

            # Stationary distribution for initial state
            pi = compute_steady_state(T_adj)
            cum_pi = np.cumsum(pi)

            n_season_days = len(day_indices)

            # Initialize states from stationary distribution: shape (n_sim,)
            init_rand = rng.random(n_sim)
            init_states = np.zeros(n_sim, dtype=int)
            init_states[init_rand >= cum_pi[0]] = 1
            init_states[init_rand >= cum_pi[1]] = 2

            # State array: (n_sim, n_season_days)
            state_array = np.zeros((n_sim, n_season_days), dtype=int)
            state_array[:, 0] = init_states

            # Vectorized state transitions
            rand_matrix = rng.random((n_sim, n_season_days - 1))
            for t in range(n_season_days - 1):
                prev_states = state_array[:, t]
                cum0 = cum_T[prev_states, 0]
                cum1 = cum_T[prev_states, 1]
                r = rand_matrix[:, t]
                new_states = np.where(r < cum0, 0, np.where(r < cum1, 1, 2))
                state_array[:, t + 1] = new_states

            # Sample intensities for low (state 1) and high (state 2)
            mask_low = (state_array == 1)
            mask_high = (state_array == 2)
            n_low = int(mask_low.sum())
            n_high = int(mask_high.sum())

            intensity_array = np.zeros((n_sim, n_season_days), dtype=float)

            if n_low > 0:
                low_samples = sample_intensity(
                    params["small_dist_name"], params["small_dist_params"],
                    params["small_values"], intensity_factor, n_low, rng
                )
                intensity_array[mask_low] = low_samples

            if n_high > 0:
                high_samples = sample_intensity(
                    params["large_dist_name"], params["large_dist_params"],
                    params["large_values"], intensity_factor, n_high, rng
                )
                intensity_array[mask_high] = high_samples

            # Place into output array at correct day positions
            stacked[p_idx, :, day_indices] = intensity_array

    return stacked, province_list


# ===================================================================
# Run length calculation
# ===================================================================

def calc_run_length_1d(day_values):
    """Calculate mean run length for one province.

    run_length = peak_days / num_runs. Returns 0.0 if no peaks.
    """
    is_peak = (np.asarray(day_values, dtype=float) > 0).astype(int)
    peak_days = is_peak.sum()
    if peak_days == 0:
        return 0.0

    # Count number of runs (contiguous groups of peaks)
    diff = np.diff(is_peak)
    num_starts = int((diff == 1).sum())
    # If the first day is a peak, that's an additional run start
    if is_peak[0] == 1:
        num_starts += 1

    if num_starts == 0:
        return 0.0
    return peak_days / num_starts


def calc_avg_run_length(sim_data_2d):
    """Average run length across all provinces, rounded to 3 decimals."""
    n_prov = sim_data_2d.shape[0]
    run_lengths = [calc_run_length_1d(sim_data_2d[p]) for p in range(n_prov)]
    avg = np.mean(run_lengths)
    return round(avg, 3)


# ===================================================================
# Export utilities
# ===================================================================

def export_one_simulation(sim_data_2d, month_day_list, province_list, output_path):
    """Export single simulation as Parquet file."""
    df = pd.DataFrame(sim_data_2d.T, columns=province_list)
    df.insert(0, "month_day", month_day_list)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)


def export_grid_simulations(stacked, month_day_list, province_list, grid_dir, config):
    """Export all simulations for a grid point.

    Group by average run length into subdirectories.
    Use ThreadPoolExecutor for parallel writing.
    """
    n_sim = stacked.shape[1]
    write_workers = MC_CONFIG.get("write_workers", 4)

    tasks = []
    for sim_idx in range(n_sim):
        sim_2d = stacked[:, sim_idx, :]  # (n_provinces, n_days)
        avg_len = calc_avg_run_length(sim_2d)
        sub_dir = os.path.join(grid_dir, f"avg_run_length_{avg_len}")
        filename = f"sim_{sim_idx + 1:04d}.parquet"
        output_path = os.path.join(sub_dir, filename)
        tasks.append((sim_2d, month_day_list, province_list, output_path))

    with ThreadPoolExecutor(max_workers=write_workers) as executor:
        futures = [
            executor.submit(export_one_simulation, t[0], t[1], t[2], t[3])
            for t in tasks
        ]
        for f in futures:
            f.result()  # Raise any exceptions


# ===================================================================
# Main entry point
# ===================================================================

def main():
    """Main entry point.

    1. Load peak demand data
    2. Fit seasonal distributions
    3. Generate decay_factor and intensity_factor grids
    4. For each grid point: run MC simulation, export results
    """
    print("Loading peak demand data...")
    filepath = CONFIG["paths"]["peak_demand_xlsx"]
    df = load_peak_demand(filepath)
    month_day_list = df["month_day"].tolist()

    print("Fitting seasonal distributions...")
    season_params = fit_season_distributions(df, CONFIG)

    # Precompute season-day index mapping
    seasons = MC_CONFIG["seasons"]
    season_day_indices = precompute_season_day_indices(month_day_list, seasons)

    # Build parameter grid
    decay_values = np.arange(DECAY_START, DECAY_END + DECAY_STEP / 2, DECAY_STEP)
    decay_values = np.clip(decay_values, 0.0, 1.0)
    intensity_values = np.arange(INTENSITY_START, INTENSITY_END + INTENSITY_STEP / 2, INTENSITY_STEP)

    output_dir = CONFIG["paths"]["OUTPUT_DIR"]
    mc_output_dir = os.path.join(output_dir, "monte_carlo")
    os.makedirs(mc_output_dir, exist_ok=True)

    total_grid = len(decay_values) * len(intensity_values)
    print(f"Starting Monte Carlo grid search: {len(decay_values)} decay x "
          f"{len(intensity_values)} intensity = {total_grid} grid points")

    seed = MC_CONFIG["random_seed"]
    grid_count = 0

    for decay in decay_values:
        for intensity in intensity_values:
            grid_count += 1
            grid_name = f"decay_{decay:.2f}_int_{intensity:.2f}"
            grid_dir = os.path.join(mc_output_dir, grid_name)

            print(f"  [{grid_count}/{total_grid}] Grid point: {grid_name}")

            rng = np.random.default_rng(seed)

            stacked, province_list = run_markov_monte_carlo_batch(
                season_params, month_day_list, season_day_indices,
                decay, intensity, CONFIG, rng
            )

            export_grid_simulations(stacked, month_day_list, province_list, grid_dir, CONFIG)

    print(f"Monte Carlo simulation complete. Results saved to: {mc_output_dir}")


if __name__ == "__main__":
    main()
