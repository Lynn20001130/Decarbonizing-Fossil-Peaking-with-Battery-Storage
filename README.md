# Decarbonizing Fossil Peaking with Battery Storage

This repository contains the code and data for the paper on assessing the feasibility of replacing natural gas peaking power plants with battery energy storage systems across Chinese provinces.

## Overview

We develop a simulation-optimization framework to evaluate battery storage as a substitute for natural gas-fired peaking generation. The analysis covers 26 provinces in China and projects optimal storage deployment from 2025 to 2035. Key components include:

- **Storage Simulation**: Numba-accelerated daily dispatch simulation of battery storage systems, modeling charging/discharging with efficiency losses and SOC constraints.
- **3R Metrics**: Three storage replacement ratios — CSR (Cost Storage Ratio), PSR (Profit Storage Ratio), and OSR (Optimal Storage Ratio) — to quantify the economic tipping points for storage adoption.
- **Monte Carlo Analysis**: Three-state Markov chain simulation of future peak-shaving demand with configurable decay and intensity parameters, enabling uncertainty quantification.
- **Future Projections**: Year-by-year (2025–2035) computation of 3R metrics under declining storage cost trajectories.

## Repository Structure

```
├── paper-code/
│   ├── config.py                  # Global parameters (storage, cost, gas, provinces)
│   ├── utils.py                   # Shared utilities (data loading, cost calculations)
│   ├── storage_simulator.py       # Numba-JIT storage dispatch simulation
│   ├── metrics.py                 # CSR / PSR / OSR calculation
│   ├── yearly_analysis.py         # Province-level yearly optimization
│   ├── monte_carlo_simulation.py  # Markov-chain demand scenario generation
│   ├── batch_analysis.py          # Batch Monte Carlo + 3R integration
│   └── future_3r.py               # 2025–2035 future 3R projections
├── data/
│   └── Daily Natural Gas Peaking Power Generation by Province in China.xlsx
└── README.md
```

## Data

`data/` contains daily natural gas peaking power generation data (kWh) by province in China, which serves as the primary input for the storage replacement analysis.

## Requirements

- Python 3.8+
- NumPy, Pandas, SciPy
- Numba
- openpyxl (for Excel I/O)

## Usage

1. Configure paths and parameters in `paper-code/config.py`.
2. Run province-level storage optimization:
   ```bash
   python paper-code/yearly_analysis.py
   ```
3. Run Monte Carlo demand simulations:
   ```bash
   python paper-code/monte_carlo_simulation.py
   ```
4. Batch 3R analysis across simulated scenarios:
   ```bash
   python paper-code/batch_analysis.py
   ```
5. Compute future 3R projections (2025–2035):
   ```bash
   python paper-code/future_3r.py
   ```

## License

This project is for academic research purposes.
