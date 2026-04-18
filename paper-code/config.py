"""
Global configuration for the natural gas peak-shaving analysis pipeline.
All parameters, paths, and constants used across scripts are defined here.
"""

import os

# ---------------------------------------------------------------------------
# Capital Recovery Factor
# ---------------------------------------------------------------------------

def calculate_crf(discount_rate: float, years: int) -> float:
    """Calculate Capital Recovery Factor (CRF)."""
    return (discount_rate * (1 + discount_rate) ** years) / \
           ((1 + discount_rate) ** years - 1)


# ---------------------------------------------------------------------------
# Core configuration
# ---------------------------------------------------------------------------

CONFIG = {
    # --- Storage parameters ---
    "storage": {
        "step": 25,                # MWh, capacity search step size
        "charge_eff": 0.95,        # Charging efficiency
        "discharge_eff": 0.95,     # Discharging efficiency
        "soc_min": 0.05,           # Minimum state of charge
        "soc_max": 0.95,           # Maximum state of charge
    },

    # --- Convergence parameters ---
    "convergence": {
        "threshold_ratio": 0.001,  # Relative convergence threshold
        "max_iterations": 20,      # Maximum iteration count
    },

    # --- Cost parameters ---
    "cost": {
        "storage_cost_per_kwh": 450,   # Yuan/kWh installed storage cost
        "project_lifetime": 25,        # Years
        "discount_rate": 0.036,        # Annual discount rate
    },

    # --- Gas parameters ---
    "gas": {
        "heat_value": 35544,               # kJ/m^3, natural gas calorific value
        "generation_efficiency": 0.536,    # Gas-to-electricity conversion efficiency
        "generator_cost": 0.43,            # Yuan/kWh, gas generator O&M cost
    },

    # --- Year-specific storage cost projections (Yuan/kWh) ---
    "year_costs": {
        2025: 450.0,
        2026: 414.3434719,
        2027: 385.5892527,
        2028: 362.585247,
        2029: 344.0692098,
        2030: 329.0214579,
        2031: 316.6179759,
        2032: 306.2657605,
        2033: 297.4925303,
        2034: 289.9450449,
        2035: 283.7692641,
    },

    # --- Province list (Chinese administrative names kept as data) ---
    "provinces": [
        '内蒙古自治区', '辽宁省', '吉林省', '新疆维吾尔自治区', '甘肃省',
        '北京市', '天津市', '青海省', '宁夏回族自治区', '陕西省',
        '山西省', '河北省', '山东省', '四川省', '湖北省',
        '河南省', '安徽省', '江苏省', '云南省', '贵州省',
        '湖南省', '江西省', '浙江省', '广西壮族自治区', '广东省',
        '福建省',
    ],

    # --- Target satisfy rates (%) ---
    "target_satisfy_rates": [
        5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
        55, 60, 65, 70, 75, 80, 85, 90, 95,
        96, 97, 98, 99, 100,
    ],

    # --- Path templates (use placeholders; override per environment) ---
    "paths": {
        "DATA_DIR": os.path.join("data"),
        "OUTPUT_DIR": os.path.join("output"),
        "curtailment_csv": os.path.join("data", "{province}", "curtailment.csv"),
        "electricity_price_xlsx": os.path.join("data", "electricity_prices.xlsx"),
        "gas_price_xlsx": os.path.join("data", "gas_prices.xlsx"),
        "peak_demand_xlsx": os.path.join("data", "peak_demand.xlsx"),
        "simulation_output": os.path.join("output", "{province}", "simulation_results.csv"),
        "capacity_output": os.path.join("output", "{province}", "optimal_capacity.csv"),
    },
}

# ---------------------------------------------------------------------------
# Monte Carlo configuration
# ---------------------------------------------------------------------------

MC_CONFIG = {
    "num_simulations": 100,        # Number of Monte Carlo simulation runs
    "random_seed": 33,             # Random seed for reproducibility

    # Decay rate sampling range
    "decay_range": {
        "min": 0.01,
        "max": 0.15,
    },

    # Intensity sampling range
    "intensity_range": {
        "min": 0.5,
        "max": 2.0,
    },

    # Season definitions (month boundaries)
    "seasons": {
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
        "winter": [12, 1, 2],
    },

    # Candidate probability distributions for fitting
    "candidate_distributions": ["lognorm", "weibull_min", "gamma"],

    # Parallel write workers for file I/O
    "write_workers": 14,

    # Quantile for intensity threshold
    "intensity_quantile": 0.75,
}

# ---------------------------------------------------------------------------
# Parallel execution configuration
# ---------------------------------------------------------------------------

PARALLEL_CONFIG = {
    "num_workers": 16,
}

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Gas power output per cubic metre (kWh/m^3)
#   heat_value (kJ/m^3) * efficiency / 3600 (kJ/kWh)
GAS_POWER_PER_M3 = (
    CONFIG["gas"]["heat_value"]
    * CONFIG["gas"]["generation_efficiency"]
    / 3600.0
)

# Pre-computed CRF using default cost parameters
CRF = calculate_crf(
    CONFIG["cost"]["discount_rate"],
    CONFIG["cost"]["project_lifetime"],
)
