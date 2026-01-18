#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KAN-LNP: Kolmogorov-Arnold Networks for Ionizable Lipid Design

This module implements a small-data-driven framework for the rational design
of siloxane-based ionizable lipids in lipid nanoparticles (LNPs) using
Kolmogorov-Arnold Networks (KANs).

Key Features:
    - Feature selection pipeline (SD, correlation, p-value filtering)
    - KAN model training with symbolic regression
    - Cross-validation with hyperparameter optimization
    - Automatic formula extraction and visualization

Author: [juntao wang]
Institution: [Dalian University of Technology]
Date: January 2026
License: MIT

    
Example:
    Basic usage:
        $ python kan_featsel.py
        
    Or import as module:
        >>> from kan_featsel import feature_selection, kan_train_and_prune
        >>> selected_df, names = feature_selection(df, top_k=10)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pickle
import sympy as sp
import optuna
from functools import partial
from typing import Tuple, List, Optional, Dict, Any

import torch
from kan import KAN, SYMBOLIC_LIB
from kan.utils import ex_round
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from scipy import stats

# Configuration
torch.set_default_dtype(torch.float64)

# Global variables for tracking best model
best_q2 = -float('inf')
current_best_file = None

# Version
__version__ = '1.0.0'
__author__ = '[JUNTAO WANG]'


def random_kfold(X: np.ndarray, y: np.ndarray, n_splits: int = 5, 
                 random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, valid_idx in kf.split(X):
        splits.append((train_idx, valid_idx))
    
    return splits


def print_split_statistics(y, train_idx, valid_idx, fold_idx):
    y_flat = y.flatten() if y.ndim > 1 else y
    y_train = y_flat[train_idx]
    y_valid = y_flat[valid_idx]
    
    train_min, train_max = np.min(y_train), np.max(y_train)
    valid_min, valid_max = np.min(y_valid), np.max(y_valid)
    
    in_range_count = np.sum((y_valid >= train_min) & (y_valid <= train_max))
    in_range_ratio = in_range_count / len(y_valid) * 100


def formula_to_ascii(formula_str: Optional[Any]) -> str:

    if formula_str is None:
        return "None"
    
    formula_str = str(formula_str)
    
    superscript_map = {
        '⁰': '^0', '¹': '^1', '²': '^2', '³': '^3', '⁴': '^4',
        '⁵': '^5', '⁶': '^6', '⁷': '^7', '⁸': '^8', '⁹': '^9',
        '⁺': '^+', '⁻': '^-', '⁼': '^=', '⁽': '^(', '⁾': '^)',
        'ⁿ': '^n', 'ⁱ': '^i'
    }
    
    subscript_map = {
        '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
        '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
        '₊': '_+', '₋': '_-', '₌': '_=', '₍': '_(', '₎': '_)'
    }
    
    special_map = {
        '×': '*', '÷': '/', '−': '-', '√': 'sqrt',
        'π': 'pi', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma',
        'δ': 'delta', 'ε': 'epsilon', 'θ': 'theta', 'λ': 'lambda',
        'μ': 'mu', 'σ': 'sigma', 'φ': 'phi', 'ω': 'omega'
    }
    
    for old, new in {**superscript_map, **subscript_map, **special_map}.items():
        formula_str = formula_str.replace(old, new)
    
    return formula_str


def feature_cleaning(df, label_col='label', verbose=True):
    removed_features = []
    
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    
    label_data = df[[label_col]].copy()
    feature_df = df.drop(columns=[label_col])
    
    if verbose:
    
    feature_df = feature_df.select_dtypes(include=[float, int, 'float64', 'int64', 'float32', 'int32'])
    
    feature_df = feature_df.dropna(axis=1)
    
    feature_df = feature_df.loc[:, feature_df.nunique() > 1]
    
    threshold = 0.6 * len(feature_df)
    columns_to_keep = []
    for col in feature_df.columns:
        if len(feature_df[col].value_counts()) > 0:
            most_common_count = feature_df[col].value_counts().iloc[0]
            if most_common_count < threshold:
                columns_to_keep.append(col)
    feature_df = feature_df[columns_to_keep]
    
    feature_df = feature_df.loc[:, (feature_df != 0).any()]
    
    if verbose:
    
    cleaned_df = pd.concat([label_data.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    return cleaned_df


def sd_selection(x, feature_names, top_k, output_dir=None):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    sd = np.std(x_scaled, axis=0)
    sel_id = np.argsort(sd)[::-1][:top_k]
    
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(len(sd)) + 1, sd, color='gray', alpha=0.3, label='All Features')
        plt.bar(sel_id + 1, sd[sel_id], color="red", alpha=0.8, label=f'Top {top_k} Features')
        plt.xlabel('Feature index')
        plt.ylabel('Standard deviation')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_dir, 'sd_top_features.png'), dpi=600)
        plt.close()
    
    selected_names = [feature_names[i] for i in sel_id]
    return x[:, sel_id], selected_names


def corr_selection(x, feature_names, corr_threshold, output_dir=None):
    df = pd.DataFrame(x, columns=feature_names)
    corr_mat = df.corr(method='pearson')
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col].abs() > corr_threshold)]
    sel_id = [i for i, col in enumerate(feature_names) if col not in drop_cols]
    selected_names = [feature_names[i] for i in sel_id]
    
    if output_dir and len(sel_id) > 0:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_mat.iloc[sel_id, sel_id], annot=False, cmap='coolwarm', cbar=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=600)
        plt.close()
    
    return x[:, sel_id], selected_names


def pvalue_selection(x, y, feature_names, alpha=0.05, output_dir=None):
    y_flat = y.flatten()
    p_values = []
    for i in range(x.shape[1]):
        try:
            _, p = stats.pearsonr(x[:, i], y_flat)
        except:
            p = 1.0
        p_values.append(p)
    
    p_values = np.array(p_values)
    sel_id = np.where(p_values < alpha)[0]
    selected_names = [feature_names[i] for i in sel_id]
    
    if output_dir:
        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(p_values)) + 1, p_values, color='b')
        plt.axhline(y=alpha, color='r', linestyle='--', label=f'α = {alpha}')
        plt.xlabel('Feature index')
        plt.ylabel('P-value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pvalue_distribution.png'), dpi=600)
        plt.close()
    
    return x[:, sel_id], selected_names


def feature_selection(df, label_col='label', top_k=-1, corr_threshold=-1, alpha=0.05, output_dir=None, verbose=True):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cleaned_df = feature_cleaning(df, label_col, verbose=verbose)
    
    y = cleaned_df[label_col].values.reshape(-1, 1)
    feature_cols = [col for col in cleaned_df.columns if col != label_col]
    x = cleaned_df[feature_cols].values
    
    current_x, current_names = x.copy(), list(feature_cols)
    
    actual_top_k = current_x.shape[1] if top_k == -1 else min(top_k, current_x.shape[1])
    current_x, current_names = sd_selection(current_x, current_names, actual_top_k, output_dir)
    
    actual_corr = 1.0 if corr_threshold == -1 else corr_threshold
    current_x, current_names = corr_selection(current_x, current_names, actual_corr, output_dir)
    
    actual_alpha = 1.01 if alpha == -1 else alpha
    current_x, current_names = pvalue_selection(current_x, y, current_names, actual_alpha, output_dir)
    
    if verbose:
    
    result_df = pd.DataFrame(current_x, columns=current_names)
    result_df.insert(0, label_col, y.flatten())
    
    return result_df, current_names


def apply_feature_selection(df, selected_names, label_col='label'):
    y = df[label_col].values.reshape(-1, 1)
    cols_exist = [col for col in selected_names if col in df.columns]
    x = df[cols_exist].values
    
    result_df = pd.DataFrame(x, columns=cols_exist)
    result_df.insert(0, label_col, y.flatten())
    return result_df


def combine_features(selected_df, fixed_df, label_col='label'):
    y = selected_df[label_col].values.reshape(-1, 1)
    
    fixed_feature_names = [col for col in fixed_df.columns if col != label_col]
    fixed_x = fixed_df[fixed_feature_names].values
    
    selected_feature_names = [col for col in selected_df.columns if col != label_col]
    selected_x = selected_df[selected_feature_names].values
    
    combined_x = np.hstack([fixed_x, selected_x])
    combined_names = fixed_feature_names + selected_feature_names
    
    combined_df = pd.DataFrame(combined_x, columns=combined_names)
    combined_df.insert(0, label_col, y.flatten())
    
    return combined_df, combined_names


def compute_buffer_values(data, n_samples):
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_features = data.shape[1]
    buffer_max = np.zeros(n_features)
    buffer_min = np.zeros(n_features)
    
    for i in range(n_features):
        col_max, col_min = np.max(data[:, i]), np.min(data[:, i])
        data_range = col_max - col_min
        
        buffer_max[i] = col_max * 1.1 if col_max > 0 else (col_max * 0.9 if col_max < 0 else data_range / n_samples)
        buffer_min[i] = col_min * 0.9 if col_min > 0 else (col_min * 1.1 if col_min < 0 else -data_range / n_samples)
    
    return buffer_max, buffer_min


def transform_data(x_train, y_train, x_test, y_test, transform_features=True, transform_label=False):
    n_samples = len(x_train)
    
    if transform_features:
        x_buffer_max, x_buffer_min = compute_buffer_values(x_train, n_samples)
        x_buffer_rows = np.vstack([x_buffer_max.reshape(1, -1), x_buffer_min.reshape(1, -1)])
        x_train_with_buffer = np.vstack([x_train, x_buffer_rows])
        
        qt_x = QuantileTransformer(output_distribution='uniform', random_state=42,
                                   n_quantiles=min(1000, len(x_train_with_buffer)))
        x_train_transform = qt_x.fit_transform(x_train_with_buffer)[:-2, :]
        x_test_transform = qt_x.transform(x_test)
    else:
        x_train_transform, x_test_transform, qt_x = x_train, x_test, None
    
    if transform_label:
        y_buffer_max, y_buffer_min = compute_buffer_values(y_train, n_samples)
        y_buffer_rows = np.vstack([y_buffer_max.reshape(1, -1), y_buffer_min.reshape(1, -1)])
        y_train_with_buffer = np.vstack([y_train, y_buffer_rows])
        
        qt_y = QuantileTransformer(output_distribution='uniform', random_state=42,
                                   n_quantiles=min(1000, len(y_train_with_buffer)))
        y_train_transform = qt_y.fit_transform(y_train_with_buffer)[:-2, :]
        y_test_transform = qt_y.transform(y_test)
    else:
        y_train_transform, y_test_transform, qt_y = y_train, y_test, None
    
    return x_train_transform, y_train_transform, x_test_transform, y_test_transform, qt_x, qt_y


def kan_train_and_prune(x_train, y_train, x_test, y_test,
                        hid_dim=5, grid=5, k=3, seed=0, opt='LBFGS',
                        steps_ori=30, steps_prune=5, lamb=0.001, lamb_entropy=2.0,
                        lib=None, transform_features=True, transform_label=False):
    if lib is None:
        lib = list(SYMBOLIC_LIB.keys())
    
    device = 'cpu'
    n_features = x_train.shape[1]
    
    x_train_t, y_train_t, x_test_t, y_test_t, qt_x, qt_y = transform_data(
        x_train, y_train, x_test, y_test, transform_features, transform_label
    )
    
    data = {
        'train_input': torch.from_numpy(x_train_t).to(device),
        'train_label': torch.from_numpy(y_train_t).to(device),
        'test_input': torch.from_numpy(x_test_t).to(device),
        'test_label': torch.from_numpy(y_test_t).to(device)
    }
    
    model = KAN(width=[n_features, hid_dim, 1], grid=grid, k=k, seed=seed, device=device)
    model.fit(data, opt=opt, steps=steps_ori, lamb=lamb, lamb_entropy=lamb_entropy)
    
    model = model.prune(node_th=3e-2)
    model.auto_symbolic(lib=lib, weight_simple=0.0)
    
    model_r2 = model.copy()
    model_r2.fit(data, opt='LBFGS', steps=steps_prune)
    
    try:
        formula = ex_round(model_r2.symbolic_formula()[0][0], 99)
    except:
        formula = None
    
    y_train_pred = model_r2(data['train_input']).detach().cpu().numpy()
    y_test_pred = model_r2(data['test_input']).detach().cpu().numpy()
    
    if transform_label and qt_y is not None:
        y_train_pred = qt_y.inverse_transform(y_train_pred)
        y_test_pred = qt_y.inverse_transform(y_test_pred)
    
    r2_train = r2_score(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    for f in ['./copy_temp_cache_data', './copy_temp_config.yml', './copy_temp_state']:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
    if os.path.exists('./model'):
        try:
            shutil.rmtree('./model')
        except:
            pass
    
    return formula, r2_train, mae_train, r2_test, mae_test, y_train_pred, y_test_pred


def cross_validation(cv_data_list, hid_dim, grid, k, steps_ori, steps_prune,
                     transform_features, transform_label, output_dir):
    n_total = sum(len(item[4]) for item in cv_data_list)
    y_pred_all = np.empty([n_total, 1])
    y_true_all = np.empty([n_total, 1])
    cv_formulas = []
    
    for fold_idx, (x_train_cv, y_train_cv, x_valid_cv, y_valid_cv, valid_indices) in enumerate(cv_data_list):
        
        try:
            formula, r2_train, mae_train, r2_valid, mae_valid, _, y_valid_pred = kan_train_and_prune(
                x_train_cv, y_train_cv, x_valid_cv, y_valid_cv,
                hid_dim=hid_dim, grid=grid, k=k,
                steps_ori=steps_ori, steps_prune=steps_prune,
                transform_features=transform_features, transform_label=transform_label
            )
            
            cv_formulas.append(formula_to_ascii(formula) if formula else "Formula extraction failed")
            y_pred_all[valid_indices, :] = y_valid_pred
            y_true_all[valid_indices, :] = y_valid_cv
            
            print(f"    R²_train={r2_train:.4f}, R²_valid={r2_valid:.4f}")
            
        except Exception as e:
            print(f"    Fold {fold_idx + 1} failed: {str(e)}")
            cv_formulas.append(f"Training failed: {str(e)}")
            y_pred_all[valid_indices, :] = np.nan
            y_true_all[valid_indices, :] = y_valid_cv
    
    valid_mask = ~np.isnan(y_pred_all).flatten()
    if np.sum(valid_mask) > 0:
        q2 = r2_score(y_true_all[valid_mask], y_pred_all[valid_mask])
    else:
        q2 = -1
    
    cv_formula_dir = os.path.join(output_dir, 'cv_formulas')
    if not os.path.exists(cv_formula_dir):
        os.makedirs(cv_formula_dir)
    with open(os.path.join(cv_formula_dir, 'all_folds.txt'), 'w', encoding='utf-8') as f:
        for i, formula in enumerate(cv_formulas):
            f.write(f'Fold {i + 1}:\n{formula}\n\n')
    
    return q2, cv_formulas, y_pred_all, y_true_all



def objective(trial, x_train_full, y_train_full, cv_data_list, obj_dict,
              transform_features, transform_label, output_dir):
    global best_q2
    
    hid_dim = trial.suggest_int('hid_dim', obj_dict['hid_dim_lower'], obj_dict['hid_dim_upper'])
    grid = trial.suggest_int('grid', obj_dict['grid_lower'], obj_dict['grid_upper'])
    k = trial.suggest_int('k', obj_dict['k_lower'], obj_dict['k_upper'])
    steps_ori = trial.suggest_int('steps_ori', obj_dict['steps_ori_lower'], obj_dict['steps_ori_upper'])
    steps_prune = trial.suggest_int('steps_prune', obj_dict['steps_prune_lower'], obj_dict['steps_prune_upper'])
    
    try:
        formula_full, r2_train_full, mae_train_full, _, _, _, _ = kan_train_and_prune(
            x_train_full, y_train_full, x_train_full, y_train_full,
            hid_dim=hid_dim, grid=grid, k=k,
            steps_ori=steps_ori, steps_prune=steps_prune,
            transform_features=transform_features, transform_label=transform_label
        )
        
        q2, cv_formulas, y_pred_all, y_true_all = cross_validation(
            cv_data_list, hid_dim, grid, k, steps_ori, steps_prune,
            transform_features, transform_label, output_dir
        )
        print(f"  Q2={q2:.4f}")
        
        trial_result_dir = os.path.join(output_dir, 'trial_results')
        if not os.path.exists(trial_result_dir):
            os.makedirs(trial_result_dir)
        
        with open(os.path.join(trial_result_dir, f'trial_{trial.number}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Trial {trial.number}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model parameters:\n")
            f.write(f"  hid_dim: {hid_dim}\n")
            f.write(f"  grid: {grid}\n")
            f.write(f"  k: {k}\n")
            f.write(f"  steps_ori: {steps_ori}\n")
            f.write(f"  steps_prune: {steps_prune}\n\n")
            f.write(f"Complete training set results:\n")
            f.write(f"  formula: y = {formula_to_ascii(formula_full)}\n")
            f.write(f"  R2_train: {r2_train_full:.4f}\n")
            f.write(f"  MAE_train: {mae_train_full:.4f}\n\n")
            f.write(f"Cross-validation results:\n")
            f.write(f"  Q2: {q2:.4f}\n\n")
            f.write(f"Each fold formula:\n")
            for i, cv_formula in enumerate(cv_formulas):
                f.write(f"  Fold {i + 1}: {cv_formula}\n")
        
        result = f'Trial {trial.number}: hid_dim={hid_dim}, grid={grid}, k={k}, '
        result += f'steps_ori={steps_ori}, steps_prune={steps_prune}, '
        result += f'R2_train={r2_train_full:.4f}, MAE_train={mae_train_full:.4f}, Q2={q2:.4f}\n'
        
        with open(os.path.join(output_dir, 'optuna_log.txt'), 'a', encoding='utf-8') as f:
            f.write(result)
        
        if q2 > best_q2:
            best_q2 = q2
            with open(os.path.join(output_dir, 'best_params.txt'), 'w', encoding='utf-8') as f:
                f.write(f'Best Trial: {trial.number}\n')
                f.write(f'Best Q2: {q2:.4f}\n')
                f.write(f'R2_train: {r2_train_full:.4f}\n')
                f.write(f'MAE_train: {mae_train_full:.4f}\n\n')
                f.write(f'公式: y = {formula_to_ascii(formula_full)}\n\n')
                f.write(f'Model parameters:\n')
                f.write(f'hid_dim: {hid_dim}\ngrid: {grid}\nk: {k}\n')
                f.write(f'steps_ori: {steps_ori}\nsteps_prune: {steps_prune}\n')
        
        return q2
    
    except Exception as e:
        print(f"Trial {trial.number} failed: {str(e)}")
        return -1


def plot_results(y_train, y_train_pred, y_test, y_test_pred, r2_train, r2_test, q2, output_path):
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(y_train, y_train_pred, c='#1f77b4', alpha=0.6, s=80, label='Train')
    plt.scatter(y_test, y_test_pred, c='#ff0000', alpha=0.6, s=80, label='Test')
    
    all_vals = np.concatenate([y_train.flatten(), y_test.flatten()])
    min_val, max_val = np.min(all_vals) * 0.9, np.max(all_vals) * 1.1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual Value', fontsize=12)
    plt.ylabel('Predicted Value', fontsize=12)
    plt.title(f'R²_train={r2_train:.4f}, R²_test={r2_test:.4f}, Q²={q2:.4f}', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    input_excel = './input_data.xlsx'
    fixed_excel = './nami.xlsx'
    label_col = 'label'
    
    top_k = 100
    corr_threshold = 0.9
    alpha = 0.05
    
    k_folds = 5
    
    transform_features = True
    transform_label = False
    
    enable_optuna = True
    n_trials = 500
    obj_dict = {
        'hid_dim_lower': 1, 'hid_dim_upper': 10,
        'grid_lower': 2, 'grid_upper': 8,
        'k_lower': 1, 'k_upper': 5,
        'steps_ori_lower': 10, 'steps_ori_upper': 30,
        'steps_prune_lower': 5, 'steps_prune_upper': 25
    }
    
    hid_dim, grid, k_param, steps_ori, steps_prune = 5, 5, 3, 20, 15
    
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    original_df = pd.read_excel(input_excel)
    fixed_df = pd.read_excel(fixed_excel)
    
    fixed_feature_names = [col for col in fixed_df.columns if col != label_col]
    
    n_samples = len(original_df)
    if k_folds == 'all':
        k_folds = n_samples
    
    
    cv_split_dir = os.path.join(output_dir, 'cv_splits')
    if not os.path.exists(cv_split_dir):
        os.makedirs(cv_split_dir)
    
    y_for_split = original_df[label_col].values
    X_for_split = original_df.drop(columns=[label_col]).select_dtypes(include=[np.number]).values
    
    random_splits = random_kfold(X_for_split, y_for_split, n_splits=k_folds, random_state=42)
    
    cv_splits = []
    
    for fold_idx, (train_index, valid_index) in enumerate(random_splits):
        fold_dir = os.path.join(cv_split_dir, f'fold_{fold_idx + 1}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        
        print_split_statistics(y_for_split, train_index, valid_index, fold_idx)
        
        train_df = original_df.iloc[train_index].reset_index(drop=True)
        valid_df = original_df.iloc[valid_index].reset_index(drop=True)
        
        train_fixed_df = fixed_df.iloc[train_index].reset_index(drop=True)
        valid_fixed_df = fixed_df.iloc[valid_index].reset_index(drop=True)
        
        train_df.to_excel(os.path.join(fold_dir, 'train_original.xlsx'), index=False)
        valid_df.to_excel(os.path.join(fold_dir, 'valid_original.xlsx'), index=False)
        train_fixed_df.to_excel(os.path.join(fold_dir, 'train_fixed.xlsx'), index=False)
        valid_fixed_df.to_excel(os.path.join(fold_dir, 'valid_fixed.xlsx'), index=False)
        
        cv_splits.append((train_index, valid_index, train_df, valid_df, train_fixed_df, valid_fixed_df, fold_dir))
    
    
    featsel_full_dir = os.path.join(output_dir, 'featsel_full')
    selected_full_df, selected_full_names = feature_selection(
        original_df, label_col=label_col, top_k=top_k, corr_threshold=corr_threshold,
        alpha=alpha, output_dir=featsel_full_dir
    )
    selected_full_df.to_excel(os.path.join(featsel_full_dir, 'selected_features.xlsx'), index=False)
    
    combined_full_df, combined_full_names = combine_features(selected_full_df, fixed_df, label_col)
    combined_full_df.to_excel(os.path.join(featsel_full_dir, 'combined_features.xlsx'), index=False)
    
    cv_data_list = []
    
    for fold_idx, (train_index, valid_index, train_df, valid_df, train_fixed_df, valid_fixed_df, fold_dir) in enumerate(cv_splits):
        print(f"\n--- Fold {fold_idx + 1} ---")
        
        featsel_cv_dir = os.path.join(fold_dir, 'featsel')
        
        selected_train_df, selected_names = feature_selection(
            train_df, label_col=label_col, top_k=top_k, corr_threshold=corr_threshold,
            alpha=alpha, output_dir=featsel_cv_dir, verbose=False
        )
        selected_train_df.to_excel(os.path.join(fold_dir, 'train_selected.xlsx'), index=False)
        
        selected_valid_df = apply_feature_selection(valid_df, selected_names, label_col)
        selected_valid_df.to_excel(os.path.join(fold_dir, 'valid_selected.xlsx'), index=False)
        
        combined_train_df, combined_names = combine_features(selected_train_df, train_fixed_df, label_col)
        combined_valid_df, _ = combine_features(selected_valid_df, valid_fixed_df, label_col)
        
        combined_train_df.to_excel(os.path.join(fold_dir, 'train_combined.xlsx'), index=False)
        combined_valid_df.to_excel(os.path.join(fold_dir, 'valid_combined.xlsx'), index=False)
        
        
        y_train = combined_train_df[label_col].values.reshape(-1, 1)
        x_train = combined_train_df.drop(columns=[label_col]).values
        y_valid = combined_valid_df[label_col].values.reshape(-1, 1)
        x_valid = combined_valid_df.drop(columns=[label_col]).values
        
        cv_data_list.append((x_train, y_train, x_valid, y_valid, valid_index))

    y_full = combined_full_df[label_col].values.reshape(-1, 1)
    x_full = combined_full_df.drop(columns=[label_col]).values
    
    if enable_optuna:
        study = optuna.create_study(direction='maximize')
        study.optimize(
            partial(objective,
                    x_train_full=x_full, y_train_full=y_full,
                    cv_data_list=cv_data_list, obj_dict=obj_dict,
                    transform_features=transform_features, transform_label=transform_label,
                    output_dir=output_dir),
            n_trials=n_trials
        )
        
        print(f'\nBest trial: {study.best_trial.number}')
        print(f'Best Q²: {study.best_trial.value:.4f}')
        print(f'Best params: {study.best_trial.params}')
        
        best_params = study.best_trial.params
        hid_dim = best_params['hid_dim']
        grid = best_params['grid']
        k_param = best_params['k']
        steps_ori = best_params['steps_ori']
        steps_prune = best_params['steps_prune']
        q2 = study.best_trial.value
    else:
        q2, cv_formulas, y_pred_all, y_true_all = cross_validation(
            cv_data_list, hid_dim, grid, k_param, steps_ori, steps_prune,
            transform_features, transform_label, output_dir
        )
        print(f'\nQ²: {q2:.4f}')
    
    
    formula, r2_train, mae_train, r2_test, mae_test, y_train_pred, y_test_pred = kan_train_and_prune(
        x_full, y_full, x_full, y_full,
        hid_dim=hid_dim, grid=grid, k=k_param,
        steps_ori=steps_ori, steps_prune=steps_prune,
        transform_features=transform_features, transform_label=transform_label
    )
    
    
    with open(os.path.join(output_dir, 'final_result.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Final Formula: y = {formula_to_ascii(formula)}\n\n")
        f.write(f"R²_train: {r2_train:.4f}\n")
        f.write(f"MAE_train: {mae_train:.4f}\n")
        f.write(f"Q²_cv: {q2:.4f}\n\n")
        f.write(f"Model parameters:\n")
        f.write(f"hid_dim: {hid_dim}\ngrid: {grid}\nk: {k_param}\n")
        f.write(f"steps_ori: {steps_ori}\nsteps_prune: {steps_prune}\n")
    
    with open(os.path.join(output_dir, 'formula.pickle'), 'wb') as f:
        pickle.dump(formula, f)
    
    plot_results(y_full, y_train_pred, y_full, y_test_pred, r2_train, r2_test, q2,
                 os.path.join(output_dir, 'prediction_plot.png'))
    


if __name__ == '__main__':
    main()
