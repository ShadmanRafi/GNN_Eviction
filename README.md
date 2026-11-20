# Eviction Rate Prediction via Multi-Relational Graph Neural Networks

Code for:
> **Structuring Neighborhood Relationships for Eviction Rate Prediction: A Multi-Relational Graph Learning Approach**

## Overview

This repository contains two notebooks:

| Notebook | Description | Environment |
|----------|-------------|-------------|
| `notebooks/01_gsl_baselines.ipynb` | 12 GSL methods benchmarked along with GCN as a baseline | UTSA ARC (A100 40 GB PCIe) |
| `notebooks/02_rgcn_experiments.ipynb` | Multi-relational R-GCN experiments | Google Colab (A100 80 GB SXM4) |

## Data

Data files can be obtained from:

- **ACS 5-Year Estimates:** [NHGIS](https://nhgis.org) — census tract-level 5-year estimate tables for 2022 and 2023 releases
- **Eviction filing data:** [Eviction Lab Eviction Tracking System](https://evictionlab.org/eviction-tracking)
- **TIGER/Line shapefiles:** [U.S. Census Bureau](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — tract-level files for each state

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

> Note: The two notebooks use different PyTorch versions due to OpenGSL
> compatibility requirements. Use separate environments.

## Experimental Set-Up
All experiments (GSL baselines and R-GCN variants) share the same training protocol to ensure a fair comparison. 

All models are trained with the Adam optimizer ([Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)) at learning rate of `1e-3` and weight decay of `5e-4`. Training runs for up to 200 epochs, and the model weights at the epoch of lowest validation Mean Squared Error (MSE) are restored before test evaluation. MSE is the training loss for all methods. All graphs are built separately for the training and test node sets, using only the features or spatial geometries of tracts in that split. The validation graph is built similarly from validation-only tracts, with no edges crossing split boundaries.

All methods (GSL baselines as well as R-GCN variants) use a two-layer Graph Convolutional Network ([Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907)) as the GNN backbone, with hidden dimension 64, dropout p = 0.5 between layers, and a scalar linear output layer for regression. The layer-l update rule is 

$$
H^{(\ell+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(\ell)} W^{(\ell)}\right), \qquad  \tilde{A} = A + I
$$

GSL baselines are implemented using the OpenGSL library(https://github.com/OpenGSL/OpenGSL), which approximates each method's graph-learning mechanism through a unified three-component interface (metric, post-processing, fusion). This ensures architectural and training-protocol equivalence across all baselines. Each GSL baseline wraps a graph-learning module around the shared GCN backbone. The module receives the input cosine-kNN adjacency and produces a refined adjacency, which is symmetrically normalized and passed to the GCN. The GCN and graph-learner parameters are optimized jointly. The table below summarizes the learner configuration for each method using the OpenGSL modular interface. 

We  report the eight OpenGSL methods whose learner components are directly compatible with a continuous target. Additionally, GGSL is not one of the methods evaluated in the original OpenGSL benchmark, but it was added later to the OpenGSL library as a general purpose implementation of graph structure learning rather than a reproduction of a peer-reviewed algorithm. We report it as an additional graph structure learning method with our used configuration. 

Each configuration is trained with five random seeds, each controlling model initialization, dropout, and the stratified validation split. All reported metrics are mean $\pm$ standard deviation over these five runs.

**Graph-learner configurations for each GSL method using the OpenGSL modular interface.**

| Method     | Encoder      | Metric          | Post             | Fusion           |
|------------|--------------|-----------------|------------------|------------------|
| GRCN       | GCNDiag (2L) | Cosine          | KNN (k=10)       | Interp(1.0, 1.0) |
| IDGL       | -            | WtdCosine (4 h) | EpsilonNN (0.65) | Interp(0.8, 0.2) |
| GAug       | GCN (2L)     | InnerProduct    | EpsilonNN (0.5)  | Interp(0.8, 0.2) |
| STABLE     | -            | WtdCosine (1 h) | KNN (k=10)       | Interp(0.5, 0.5) |
| NodeFormer | -            | WtdCosine (4 h) | KNN (k=10)       | Replace          |
| WSGNN      | -            | WtdCosine (8 h) | EpsilonNN (0.5)  | Interp(0.5, 0.5) |
| GLCN       | -            | OneLayerNN      | -                | Interp(0.8, 0.2) |
| GGSL       | GCN (2L)     | Cosine          | KNN (k=10)       | Interp(0.5, 0.5) |
 
*Interp($\alpha, \beta$) = Interpolate($\alpha \times$ learned, $\beta \times$ original).*


## Reproducibility

All random seeds are fixed to 42. Results reported in the paper use:
- `USE_LAGGED = False` (ACS features only, no lagged eviction rate)
- `SIM_K = 1`, `SIM_THRESHOLD = 0.3` (selected by grid search on validation set)
- `HIDDEN_DIM = 64`, `LR = 1e-3`, `EPOCHS = 200`, `PATIENCE = None`