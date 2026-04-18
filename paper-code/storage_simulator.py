"""
Storage simulation module for natural gas peak-shaving with energy storage.

Provides Numba-accelerated simulation kernels for batch processing and a
StorageSimulator class that supports both numpy arrays and pandas Series.
"""

import numpy as np
from numba import njit

from config import CONFIG, CRF, GAS_POWER_PER_M3
from utils import calculate_gas_unit_cost

# ---------------------------------------------------------------------------
# Numba-accelerated simulation kernels
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def simulate_single_year_numba(demand_arr, curtailment_arr, capacity_kwh,
                                initial_soc_ratio, charge_eff, discharge_eff,
                                soc_min, soc_max):
    """Simulate one year of storage operation (Numba-accelerated).

    Parameters
    ----------
    demand_arr : ndarray (float64)
        Daily peak-shaving demand in kWh.
    curtailment_arr : ndarray (float64)
        Daily available curtailed energy in kWh.
    capacity_kwh : float
        Rated storage capacity in kWh.
    initial_soc_ratio : float
        Initial state of charge as a fraction of capacity.
    charge_eff : float
        Round-trip charging efficiency (0-1).
    discharge_eff : float
        Discharging efficiency (0-1).
    soc_min : float
        Minimum allowable SOC fraction.
    soc_max : float
        Maximum allowable SOC fraction.

    Returns
    -------
    total_peak_supply : float
        Total energy supplied for peak-shaving over the year (kWh).
    final_soc : float
        Final SOC as a fraction of capacity.
    daily_supply : ndarray (float64)
        Energy supplied each day (kWh).
    daily_spillage : ndarray (float64)
        Curtailed energy that could not be stored each day (kWh).
    """
    n_days = demand_arr.shape[0]
    daily_supply = np.empty(n_days, dtype=np.float64)
    daily_spillage = np.empty(n_days, dtype=np.float64)

    soc = initial_soc_ratio * capacity_kwh  # absolute SOC in kWh
    total_peak_supply = 0.0
    e_min = soc_min * capacity_kwh
    e_max = soc_max * capacity_kwh

    for i in range(n_days):
        demand = demand_arr[i]
        curtailment = curtailment_arr[i]

        # --- Discharge to meet demand ---
        available_discharge = (soc - e_min) * discharge_eff
        if available_discharge < 0.0:
            available_discharge = 0.0
        supply = min(demand, available_discharge)
        if supply > 0.0:
            soc -= supply / discharge_eff
        total_peak_supply += supply
        daily_supply[i] = supply

        # --- Charge from curtailment ---
        available_space = e_max - soc
        if available_space < 0.0:
            available_space = 0.0
        charge_energy = min(curtailment * charge_eff, available_space)
        spillage = curtailment - charge_energy / charge_eff if charge_eff > 0.0 else curtailment
        if spillage < 0.0:
            spillage = 0.0
        soc += charge_energy
        daily_spillage[i] = spillage

    final_soc = soc / capacity_kwh if capacity_kwh > 0.0 else 0.0
    return total_peak_supply, final_soc, daily_supply, daily_spillage


@njit(cache=True, nogil=True)
def check_should_stop_numba(demand_arr, daily_supply, daily_spillage):
    """Check whether increasing storage capacity would improve peak supply.

    For each day where demand was not fully met, check whether any cumulative
    spillage existed before that day.  If no prior spillage exists before *any*
    unsatisfied day, adding more capacity cannot help and we should stop.

    Parameters
    ----------
    demand_arr : ndarray (float64)
        Daily demand.
    daily_supply : ndarray (float64)
        Daily supply from storage.
    daily_spillage : ndarray (float64)
        Daily spillage (curtailment that could not be stored).

    Returns
    -------
    bool
        True if the search should stop (more capacity will not help).
    """
    n_days = demand_arr.shape[0]
    cumulative_spillage = 0.0

    for i in range(n_days):
        # Accumulate spillage *before* checking this day's deficit
        cumulative_spillage += daily_spillage[i]

        deficit = demand_arr[i] - daily_supply[i]
        if deficit > 1e-6:
            # There is unsatisfied demand on this day
            if cumulative_spillage > 1e-6:
                # Prior spillage exists -> more capacity could help
                return False

    # No unsatisfied day had prior spillage available
    return True


# ---------------------------------------------------------------------------
# StorageSimulator class
# ---------------------------------------------------------------------------


class StorageSimulator:
    """High-level storage simulation driver.

    Wraps both Numba-accelerated and plain-Python simulation routines and
    provides capacity-search methods for batch and detailed analysis.

    Parameters
    ----------
    config : dict
        Configuration dictionary (typically ``CONFIG`` from config.py).
    """

    def __init__(self, config):
        storage = config["storage"]
        convergence = config["convergence"]
        cost = config["cost"]

        self.charge_eff = storage["charge_eff"]
        self.discharge_eff = storage["discharge_eff"]
        self.soc_min = storage["soc_min"]
        self.soc_max = storage["soc_max"]

        self.convergence_ratio = convergence["threshold_ratio"]
        self.max_iterations = convergence["max_iterations"]

        self.step_kwh = storage["step"] * 1000.0  # MWh -> kWh
        self.cost_per_kwh = cost["storage_cost_per_kwh"]
        self.lifetime = cost["project_lifetime"]
        self.crf = CRF

    # ----- Numba-backed steady-state simulation ----------------------------

    def simulate_steady_state(self, demand_arr, curtailment_arr, capacity_kwh):
        """Run the storage simulation until SOC converges (Numba version).

        Parameters
        ----------
        demand_arr : ndarray (float64)
            Daily demand.
        curtailment_arr : ndarray (float64)
            Daily curtailment.
        capacity_kwh : float
            Rated storage capacity in kWh.

        Returns
        -------
        total_peak_supply : float
        daily_supply : ndarray
        daily_spillage : ndarray
        """
        initial_soc = 0.5  # start at 50 %
        threshold = self.convergence_ratio * capacity_kwh

        for _ in range(self.max_iterations):
            total_peak_supply, final_soc, daily_supply, daily_spillage = \
                simulate_single_year_numba(
                    demand_arr, curtailment_arr, capacity_kwh,
                    initial_soc, self.charge_eff, self.discharge_eff,
                    self.soc_min, self.soc_max,
                )
            if abs(final_soc - initial_soc) * capacity_kwh < threshold:
                break
            initial_soc = final_soc

        return total_peak_supply, daily_supply, daily_spillage

    # ----- Plain-Python single-year simulation -----------------------------

    def simulate_single_year(self, daily_demand, daily_curtailment,
                             capacity_kwh, initial_soc_ratio):
        """Simulate one year of storage operation (plain Python).

        Handles pandas Series (via ``.iloc``) as well as plain lists/arrays.

        Returns
        -------
        total_peak_supply : float
        final_soc : float
            Final SOC as a fraction of capacity.
        daily_supply_list : list[float]
        daily_spillage_list : list[float]
        """
        import pandas as pd

        n_days = len(daily_demand)
        daily_supply_list = []
        daily_spillage_list = []

        soc = initial_soc_ratio * capacity_kwh
        total_peak_supply = 0.0
        e_min = self.soc_min * capacity_kwh
        e_max = self.soc_max * capacity_kwh

        for i in range(n_days):
            # Support pandas Series
            if hasattr(daily_demand, 'iloc'):
                demand = daily_demand.iloc[i]
            else:
                demand = daily_demand[i]
            if hasattr(daily_curtailment, 'iloc'):
                curtailment = daily_curtailment.iloc[i]
            else:
                curtailment = daily_curtailment[i]

            # Handle NaN values
            if pd.notna(demand) is False:
                demand = 0.0
            if pd.notna(curtailment) is False:
                curtailment = 0.0

            demand = float(demand)
            curtailment = float(curtailment)

            # --- Discharge ---
            available_discharge = (soc - e_min) * self.discharge_eff
            if available_discharge < 0.0:
                available_discharge = 0.0
            supply = min(demand, available_discharge)
            if supply > 0.0:
                soc -= supply / self.discharge_eff
            total_peak_supply += supply
            daily_supply_list.append(supply)

            # --- Charge ---
            available_space = e_max - soc
            if available_space < 0.0:
                available_space = 0.0
            charge_energy = min(curtailment * self.charge_eff, available_space)
            if self.charge_eff > 0.0:
                spillage = curtailment - charge_energy / self.charge_eff
            else:
                spillage = curtailment
            if spillage < 0.0:
                spillage = 0.0
            soc += charge_energy
            daily_spillage_list.append(spillage)

        final_soc = soc / capacity_kwh if capacity_kwh > 0.0 else 0.0
        return total_peak_supply, final_soc, daily_supply_list, daily_spillage_list

    # ----- Plain-Python steady-state simulation ----------------------------

    def simulate_steady_state_plain(self, daily_demand, daily_curtailment,
                                    capacity_kwh):
        """Run the storage simulation until SOC converges (plain Python).

        Returns
        -------
        total_peak_supply : float
        converged : bool
        iterations : int
        daily_supply_list : list[float]
        daily_spillage_list : list[float]
        """
        initial_soc = 0.5
        threshold = self.convergence_ratio * capacity_kwh
        converged = False
        iterations = 0

        daily_supply_list = []
        daily_spillage_list = []

        for it in range(1, self.max_iterations + 1):
            total_peak_supply, final_soc, daily_supply_list, daily_spillage_list = \
                self.simulate_single_year(
                    daily_demand, daily_curtailment,
                    capacity_kwh, initial_soc,
                )
            iterations = it
            if abs(final_soc - initial_soc) * capacity_kwh < threshold:
                converged = True
                break
            initial_soc = final_soc

        return (total_peak_supply, converged, iterations,
                daily_supply_list, daily_spillage_list)

    # ----- Plain-Python should-stop check ----------------------------------

    def check_should_stop(self, daily_demand, daily_supply_list,
                          daily_spillage_list):
        """Check whether increasing capacity can improve peak supply.

        Uses a prefix sum of spillage. For every day with unmet demand, if
        cumulative spillage up to that point is positive, more capacity could
        help and we should *not* stop.

        Returns
        -------
        bool
            True if the search should stop.
        """
        import pandas as pd

        n_days = len(daily_demand)
        cumulative_spillage = 0.0

        for i in range(n_days):
            cumulative_spillage += daily_spillage_list[i]

            if hasattr(daily_demand, 'iloc'):
                demand = float(daily_demand.iloc[i])
            else:
                demand = float(daily_demand[i])

            if pd.notna(demand) is False:
                demand = 0.0

            deficit = demand - daily_supply_list[i]
            if deficit > 1e-6:
                if cumulative_spillage > 1e-6:
                    return False

        return True

    # ----- Batch optimal-storage search (Numba) ----------------------------

    def search_optimal_storage(self, demand_arr, curtailment_arr,
                               electricity_price, gas_price,
                               gas_power_per_m3, generator_cost):
        """Search for the optimal storage capacity using Numba kernels.

        Capacity is incremented by ``step_kwh`` each iteration.  The search
        uses a *3R early-stopping* heuristic: once the cost-satisfy-rate (CSR),
        profit-satisfy-rate (PSR) and optimal-satisfy-rate (OSR) crossings have
        all been observed, the search continues for 2 additional steps and then
        terminates.

        Parameters
        ----------
        demand_arr : ndarray (float64)
            Daily demand.
        curtailment_arr : ndarray (float64)
            Daily curtailment.
        electricity_price : float
            Yuan/kWh.
        gas_price : float
            Yuan/m^3.
        gas_power_per_m3 : float
            kWh produced per m^3 of natural gas.
        generator_cost : float
            Yuan/kWh generator O&M cost.

        Returns
        -------
        list[dict]
            Each dict contains: capacity_mwh, total_demand_kwh,
            peak_supply_kwh, satisfy_rate_pct, annual_cost, unit_profit.
        """
        gas_unit_cost = gas_price / gas_power_per_m3 + generator_cost
        total_demand = float(np.sum(demand_arr))
        results = []

        # 3R early-stopping flags
        csr_crossed = False
        psr_crossed = False
        osr_crossed = False
        extra_steps = 0

        capacity = self.step_kwh
        prev_unit_profit = None

        while True:
            total_peak_supply, daily_supply, daily_spillage = \
                self.simulate_steady_state(demand_arr, curtailment_arr, capacity)

            satisfy_rate = (total_peak_supply / total_demand * 100.0
                           if total_demand > 0.0 else 0.0)

            annual_storage_cost = capacity * self.cost_per_kwh * self.crf
            annual_revenue = total_peak_supply * (gas_unit_cost - electricity_price)
            annual_cost = annual_storage_cost
            unit_profit = ((annual_revenue - annual_cost) / capacity
                           if capacity > 0.0 else 0.0)

            results.append({
                'capacity_mwh': capacity / 1000.0,
                'total_demand_kwh': total_demand,
                'peak_supply_kwh': total_peak_supply,
                'satisfy_rate_pct': satisfy_rate,
                'annual_cost': annual_cost,
                'unit_profit': unit_profit,
            })

            # --- Check stopping conditions ---
            # Demand fully satisfied
            if satisfy_rate >= 100.0 - 1e-9:
                break

            # No benefit from more capacity
            should_stop = check_should_stop_numba(
                demand_arr, daily_supply, daily_spillage)
            if should_stop:
                break

            # Exceeds reasonable upper bound
            if capacity > total_demand * 10.0:
                break

            # --- 3R early-stopping logic ---
            if not csr_crossed and annual_cost > annual_revenue:
                csr_crossed = True
            if prev_unit_profit is not None:
                if not psr_crossed and unit_profit < prev_unit_profit:
                    psr_crossed = True
                if not osr_crossed and unit_profit < 0.0:
                    osr_crossed = True

            if csr_crossed and psr_crossed and osr_crossed:
                extra_steps += 1
                if extra_steps > 2:
                    break

            prev_unit_profit = unit_profit
            capacity += self.step_kwh

        return results

    # ----- Detailed optimal-storage search (plain Python) ------------------

    def search_optimal_storage_detailed(self, daily_demand, daily_curtailment,
                                        province_name, electricity_price,
                                        config):
        """Search for optimal storage with detailed per-step diagnostics.

        This version uses the plain-Python simulation path and additionally
        tracks daily first-satisfy capacity, IRR, equivalent full cycles (EFC),
        and the economic zero point.

        Parameters
        ----------
        daily_demand : array-like or pandas Series
            Daily peak-shaving demand (kWh).
        daily_curtailment : array-like or pandas Series
            Daily available curtailed energy (kWh).
        province_name : str
            Province identifier (for logging / output).
        electricity_price : float
            Yuan/kWh.
        config : dict
            Full configuration dict (``CONFIG``).

        Returns
        -------
        results : list[dict]
            Each dict contains: capacity_mwh, total_demand_kwh,
            peak_supply_kwh, satisfy_rate_pct, efc, initial_cost,
            annual_cost, annual_revenue, annual_net_profit, unit_profit,
            irr_pct.
        optimal_capacity : float
            Capacity (kWh) that maximises unit profit.
        max_satisfy_rate : float
            Maximum achievable satisfy rate (%).
        is_fully_satisfied : bool
            Whether demand was fully satisfied.
        zero_point_info : dict or None
            Contains capacity_mwh and satisfy_rate_pct at the economic zero
            point (unit_profit crosses zero), or None.
        daily_first_satisfy_capacity : list[float]
            For each day, the first capacity at which that day's demand was
            fully met.
        """
        import numpy_financial as npf

        gas_cfg = config["gas"]
        cost_cfg = config["cost"]
        gas_unit_cost = calculate_gas_unit_cost(config)

        total_demand = float(sum(
            float(daily_demand.iloc[i]) if hasattr(daily_demand, 'iloc')
            else float(daily_demand[i])
            for i in range(len(daily_demand))
        ))

        n_days = len(daily_demand)
        # Track the first capacity at which each day's demand is fully met
        daily_first_satisfy_capacity = [0.0] * n_days

        results = []
        optimal_capacity = 0.0
        max_unit_profit = -np.inf
        max_satisfy_rate = 0.0
        is_fully_satisfied = False
        zero_point_info = None
        prev_unit_profit = None

        capacity = self.step_kwh

        while True:
            (total_peak_supply, converged, iterations,
             daily_supply_list, daily_spillage_list) = \
                self.simulate_steady_state_plain(
                    daily_demand, daily_curtailment, capacity)

            satisfy_rate = (total_peak_supply / total_demand * 100.0
                           if total_demand > 0.0 else 0.0)
            if satisfy_rate > max_satisfy_rate:
                max_satisfy_rate = satisfy_rate

            # Equivalent full cycles
            efc = (total_peak_supply / capacity) if capacity > 0.0 else 0.0

            # Cost and revenue
            initial_cost = capacity * self.cost_per_kwh
            annual_storage_cost = initial_cost * self.crf
            annual_revenue = total_peak_supply * (gas_unit_cost - electricity_price)
            annual_net_profit = annual_revenue - annual_storage_cost
            unit_profit = annual_net_profit / capacity if capacity > 0.0 else 0.0

            # IRR calculation
            irr_pct = np.nan
            try:
                cash_flows = [-initial_cost] + [annual_revenue - annual_storage_cost
                                                 ] * self.lifetime
                # Use annual_revenue as the positive cash flow, not net
                cash_flows_irr = ([-initial_cost]
                                  + [annual_revenue] * self.lifetime)
                irr_val = npf.irr(np.array(cash_flows_irr, dtype=np.float64))
                if np.isfinite(irr_val):
                    irr_pct = irr_val * 100.0
            except Exception:
                pass

            # Track daily first-satisfy capacity
            for d in range(n_days):
                if hasattr(daily_demand, 'iloc'):
                    d_demand = float(daily_demand.iloc[d])
                else:
                    d_demand = float(daily_demand[d])
                if (daily_first_satisfy_capacity[d] == 0.0
                        and d_demand > 1e-6
                        and daily_supply_list[d] >= d_demand - 1e-6):
                    daily_first_satisfy_capacity[d] = capacity

            results.append({
                'capacity_mwh': capacity / 1000.0,
                'total_demand_kwh': total_demand,
                'peak_supply_kwh': total_peak_supply,
                'satisfy_rate_pct': satisfy_rate,
                'efc': efc,
                'initial_cost': initial_cost,
                'annual_cost': annual_storage_cost,
                'annual_revenue': annual_revenue,
                'annual_net_profit': annual_net_profit,
                'unit_profit': unit_profit,
                'irr_pct': irr_pct,
            })

            # Track optimal (max unit_profit) capacity
            if unit_profit > max_unit_profit:
                max_unit_profit = unit_profit
                optimal_capacity = capacity

            # Detect economic zero point (unit_profit crosses from + to -)
            if (prev_unit_profit is not None
                    and zero_point_info is None
                    and prev_unit_profit >= 0.0
                    and unit_profit < 0.0):
                zero_point_info = {
                    'capacity_mwh': capacity / 1000.0,
                    'satisfy_rate_pct': satisfy_rate,
                }

            prev_unit_profit = unit_profit

            # --- Stopping conditions ---
            if satisfy_rate >= 100.0 - 1e-9:
                is_fully_satisfied = True
                break

            should_stop = self.check_should_stop(
                daily_demand, daily_supply_list, daily_spillage_list)
            if should_stop:
                break

            if capacity > total_demand * 10.0:
                break

            capacity += self.step_kwh

        return (results, optimal_capacity, max_satisfy_rate,
                is_fully_satisfied, zero_point_info,
                daily_first_satisfy_capacity)
