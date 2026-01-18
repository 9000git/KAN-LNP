# KAN-LNP: Kolmogorov-Arnold Networks for Ionizable Lipid Design

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the implementation of a **small-data-driven computational framework** for the rational design of siloxane-based ionizable lipids in lipid nanoparticles (LNPs), as described in our paper published in the *Journal of Chemical Information and Modeling*.

### Key Features

- **Kolmogorov-Arnold Networks (KANs)**: Symbolic regression-based AI for interpretable predictions
- **Small-data efficiency**: Achieves high accuracy with only 36 training samples
- **Explicit mathematical formulas**: Generates human-readable structure-property relationships
- **Multi-scale integration**: Combines molecular descriptors with nanoparticle features
- **Validation pipeline**: Includes molecular dynamics simulation validation

### Framework Architecture

![Framework Overview](fig_1.jpg)

The framework consists of four sequential components:

1. **Dataset Construction**: Multi-scale feature engineering incorporating molecular descriptors, nanoparticle properties, and experimental delivery efficiency
2. **KAN Model Development**: Symbolic regression to establish quantitative structure-property relationships
3. **Virtual Screening**: Large-scale candidate generation and ranking using trained models
4. **MD Validation**: Umbrella sampling simulations to validate binding affinity predictions

## Abstract

Ionizable lipids are fundamental to the efficacy of lipid nanoparticles (LNPs) in pivotal areas including mRNA vaccines. Their development, however, is hindered by intricate structure-property relationships and limited experimental data. To address these challenges, this study proposed a small-data-driven framework that pioneered the use of Kolmogorov-Arnold networks (KANs)—a symbolic regression-based artificial intelligence (AI) approach—to accelerate the discovery of novel siloxane-based ionizable lipids.

Using only 36 training samples, the resulting KAN model demonstrated high predictive accuracy for mRNA delivery efficiency (Q² = 0.710), outperforming conventional machine learning models by an average absolute improvement of 0.627 in cross-validation and yielding explicit mathematical formulas. Combined with virtual screening and umbrella sampling simulations, the framework identified three candidate lipids with superior predicted performance.

**Keywords**: Lipid nanoparticle, ionizable lipid, machine learning, Kolmogorov-Arnold network, molecular dynamics

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster training)

### Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies:
- `torch >= 1.10.0`
- `pykan >= 0.0.1`
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `optuna >= 3.0.0`
- `matplotlib >= 3.4.0`
- `seaborn >= 0.11.0`
- `openpyxl >= 3.0.0`

## Quick Start

### 1. Prepare Your Data

Organize your data in Excel format with the following structure:

```
input_data.xlsx:
- Column 1: label (target variable - delivery efficiency)
- Columns 2-N: molecular descriptors and nanoparticle features

nami.xlsx:
- Fixed features that should always be included
- Same structure as input_data.xlsx
```

### 2. Run the KAN Model

```bash
python kan_featsel.py
```

This will:
- Perform feature selection (standard deviation, correlation, p-value filtering)
- Execute k-fold cross-validation
- Optimize hyperparameters using Optuna (optional)
- Generate symbolic formulas
- Save results to `./output/` directory

### 3. Customize Parameters

Edit the configuration in `kan_featsel.py`:

```python
# Feature selection parameters
top_k = 100              # Top k features by standard deviation
corr_threshold = 0.9     # Correlation threshold for feature removal
alpha = 0.05             # P-value threshold

# Cross-validation
k_folds = 5              # Number of folds

# Data transformation
transform_features = True   # Apply quantile transformation to features
transform_label = False     # Apply quantile transformation to labels

# Hyperparameter optimization
enable_optuna = True     # Enable Optuna optimization
n_trials = 500           # Number of optimization trials
```

## Methodology

### Feature Selection Pipeline

The framework implements a three-stage feature selection strategy:

1. **Standard Deviation Filtering**: Selects top-k features with highest variance
2. **Correlation Analysis**: Removes highly correlated features (threshold: 0.9)
3. **Statistical Significance**: Filters features based on p-value (α = 0.05)

### KAN Model Architecture

- **Input Layer**: Selected molecular and nanoparticle features
- **Hidden Layer**: Learnable activation functions on edges
- **Output Layer**: Delivery efficiency prediction
- **Symbolic Regression**: Automatic conversion to mathematical formulas

### Cross-Validation Strategy

- **Method**: Random k-fold cross-validation (default: 5 folds)
- **Metric**: Q² (cross-validated R²)
- **Optimization**: Optuna-based Bayesian optimization

### Hyperparameter Search Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| `hid_dim` | [1, 10] | Hidden layer dimension |
| `grid` | [2, 8] | Grid size for spline functions |
| `k` | [1, 5] | Order of spline functions |
| `steps_ori` | [10, 30] | Initial training steps |
| `steps_prune` | [5, 25] | Pruning refinement steps |


## Citation

If you use this code in your research, please cite our paper:

```bibtex

```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyKAN library developers for the KAN implementation
- Optuna team for hyperparameter optimization framework
- All contributors and collaborators

---

**Last Updated**: January 2026
**Version**: 1.0.0
