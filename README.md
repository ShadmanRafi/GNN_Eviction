# Eviction Rate Prediction Using Multi-Relational Graph Neural Networks

Code for:
> **Structuring Neighborhood Relationships for Eviction Rate Prediction: A Multi-Relational Graph Learning Approach**

## Overview

This repository contains two notebooks:

| Notebook | Description | Environment |
|----------|-------------|-------------|
| `notebooks/gsl_baselines.ipynb` | 12 GSL methods benchmarked along with GCN as a baseline | UTSA ARC (A100 40 GB PCIe) |
| `notebooks/rgcn_experiments.ipynb` | Multi-relational R-GCN experiments | Google Colab (A100 80 GB SXM4) |

## Data

Data files can be obtained from:

- **ACS 5-Year Estimates:** [NHGIS](https://nhgis.org) — census tract level 5-year estimate tables for 2022 and 2023 releases
- **Eviction filing data:** [Eviction Lab Eviction Tracking System](https://evictionlab.org/eviction-tracking)
- **TIGER/Line shapefiles:** [U.S. Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — census tract level files for each state

Place files in the `data/` directory:

```
data/
├── ACS_clean/
│   ├── acs_aligned_2022_5_tract.csv
│   └── acs_aligned_2023_5_tract.csv
├── Eviction/
│   ├── eviction_filings_2022.csv
│   ├── eviction_filings_2023.csv
│   ├── eviction_filings_2024.csv
│   └── eviction_filings_2025.csv
└── shapefiles/
    ├── tl_2022_01_tract.shp  (one per state FIPS)
    └── ...
```

## Setup

**For `gsl_baselines.ipynb`:**
```bash
pip install -r requirements_opengsl.txt
```

**For `rgcn_experiments.ipynb`:**
```bash
pip install -r requirements_rgcn.txt
```

> Note: The two notebooks use different PyTorch versions due to OpenGSL dependency requirements. Use separate environments.

## Reproducibility

All random seeds are fixed to 42. Results reported in the paper use:
- `USE_LAGGED = False` (ACS features only, no lagged eviction rate)
- `SIM_K = 1`, `SIM_THRESHOLD = 0.3` (selected by grid search on validation set)
- `HIDDEN_DIM = 64`, `LR = 1e-3`, `EPOCHS = 200`, `PATIENCE = None`