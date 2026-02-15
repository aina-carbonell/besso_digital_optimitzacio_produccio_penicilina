#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 4: SCRIPT 3 - OPTIMITZACIÓ D'HIPERPARÀMETRES
Grid Search + Bayesian Optimization
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import time
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    print("scikit-optimize available")
except ImportError:
    print("WARNING: pip install scikit-optimize (opcional)")
    BayesSearchCV = None

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"

print("=" * 80)
print("FASE 4.3 - HYPERPARAMETER OPTIMIZATION")
print("=" * 80)

# Carregar dades
print("\n[1/6] Carregant dades...")
df_train = pd.read_csv(FASE3_OUT / "train_data.csv")
df_test = pd.read_csv(FASE3_OUT / "test_data.csv")

feature_cols = [c for c in df_train.columns if c not in ['batch_id', 'penicillin', 'time']]
X_train = df_train[feature_cols].values
y_train = df_train['penicillin'].values
X_test = df_test[feature_cols].values
y_test = df_test['penicillin'].values

# Sample per accelerar (opcional)
if len(X_train) > 20000:
    idx = np.random.choice(len(X_train), 20000, replace=False)
    X_train_sample = X_train[idx]
    y_train_sample = y_train[idx]
    print(f"   Usant sample: {len(X_train_sample):,}")
else:
    X_train_sample = X_train
    y_train_sample = y_train

# =============================================================================
# RANDOM FOREST - GRID SEARCH
# =============================================================================
print("\n[2/6] Grid Search - Random Forest...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [15, 20, 25],
    'min_samples_split': [5, 10],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
rf_grid.fit(X_train_sample, y_train_sample)
rf_time = time.time() - start_time

print(f"\n   Best RF params: {rf_grid.best_params_}")
print(f"   Best CV MAE: {-rf_grid.best_score_:.4f}")
print(f"   Time: {rf_time:.1f}s")

# =============================================================================
# XGBOOST - GRID SEARCH
# =============================================================================
print("\n[3/6] Grid Search - XGBoost...")

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9]
}

xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, n_jobs=-1),
    xgb_param_grid,
    cv=3,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    verbose=1
)

start_time = time.time()
xgb_grid.fit(X_train_sample, y_train_sample)
xgb_time = time.time() - start_time

print(f"\n   Best XGB params: {xgb_grid.best_params_}")
print(f"   Best CV MAE: {-xgb_grid.best_score_:.4f}")
print(f"   Time: {xgb_time:.1f}s")

# =============================================================================
# BAYESIAN OPTIMIZATION - XGBOOST
# =============================================================================
if BayesSearchCV:
    print("\n[4/6] Bayesian Optimization - XGBoost...")
    
    xgb_bayes_space = {
        'n_estimators': Integer(100, 500),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'gamma': Real(0, 1.0)
    }
    
    xgb_bayes = BayesSearchCV(
        xgb.XGBRegressor(random_state=42, n_jobs=-1),
        xgb_bayes_space,
        n_iter=30,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    start_time = time.time()
    xgb_bayes.fit(X_train_sample, y_train_sample)
    bayes_time = time.time() - start_time
    
    print(f"\n   Best Bayes params: {xgb_bayes.best_params_}")
    print(f"   Best CV MAE: {-xgb_bayes.best_score_:.4f}")
    print(f"   Time: {bayes_time:.1f}s")
else:
    print("\n[4/6] Bayesian Opt skipped (scikit-optimize not installed)")
    xgb_bayes = None

# =============================================================================
# AVALUAR MODELS OPTIMITZATS
# =============================================================================
print("\n[5/6] Avaluant models optimitzats...")

from sklearn.metrics import mean_absolute_error, r2_score

# RF optimitzat
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# XGB Grid
xgb_best_grid = xgb_grid.best_estimator_
y_pred_xgb_grid = xgb_best_grid.predict(X_test)
mae_xgb_grid = mean_absolute_error(y_test, y_pred_xgb_grid)
r2_xgb_grid = r2_score(y_test, y_pred_xgb_grid)

print(f"\n   RF optimitzat - MAE: {mae_rf:.4f}, R²: {r2_rf:.4f}")
print(f"   XGB Grid - MAE: {mae_xgb_grid:.4f}, R²: {r2_xgb_grid:.4f}")

if xgb_bayes:
    xgb_best_bayes = xgb_bayes.best_estimator_
    y_pred_xgb_bayes = xgb_best_bayes.predict(X_test)
    mae_xgb_bayes = mean_absolute_error(y_test, y_pred_xgb_bayes)
    r2_xgb_bayes = r2_score(y_test, y_pred_xgb_bayes)
    print(f"   XGB Bayes - MAE: {mae_xgb_bayes:.4f}, R²: {r2_xgb_bayes:.4f}")

# =============================================================================
# GUARDAR RESULTATS
# =============================================================================
print("\n[6/6] Guardant resultats...")

# Grid search results
results = pd.DataFrame({
    'model': ['RF_grid', 'XGB_grid'],
    'best_mae_cv': [-rf_grid.best_score_, -xgb_grid.best_score_],
    'test_mae': [mae_rf, mae_xgb_grid],
    'test_r2': [r2_rf, r2_xgb_grid],
    'time_s': [rf_time, xgb_time]
})

if xgb_bayes:
    results = pd.concat([results, pd.DataFrame({
        'model': ['XGB_bayes'],
        'best_mae_cv': [-xgb_bayes.best_score_],
        'test_mae': [mae_xgb_bayes],
        'test_r2': [r2_xgb_bayes],
        'time_s': [bayes_time]
    })], ignore_index=True)

results.to_csv(OUTPUT_DIR / "03_optimization_results.csv", index=False)

# Best hyperparameters
best_params = {
    'rf': rf_grid.best_params_,
    'xgb_grid': xgb_grid.best_params_,
}
if xgb_bayes:
    best_params['xgb_bayes'] = xgb_bayes.best_params_

with open(OUTPUT_DIR / "03_best_hyperparameters.json", 'w') as f:
    json.dump(best_params, f, indent=2, default=str)

# Guardar millor model
joblib.dump({'model': xgb_best_grid, 'features': feature_cols},
           OUTPUT_DIR / "03_xgboost_optimized.pkl")

# Visualització
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
models_names = results['model'].values
ax.barh(models_names, results['test_mae'], color=['green', 'blue', 'orange'][:len(models_names)],
       edgecolor='black', alpha=0.7)
ax.set_xlabel('Test MAE (g/L)')
ax.set_title('MAE - Models Optimitzats', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

ax = axes[1]
ax.barh(models_names, results['test_r2'], color=['green', 'blue', 'orange'][:len(models_names)],
       edgecolor='black', alpha=0.7)
ax.set_xlabel('Test R²')
ax.set_title('R² - Models Optimitzats', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_optimization_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("HYPERPARAMETER OPTIMIZATION COMPLETAT")
print("=" * 80)
print("\nMillor model:")
best_idx = results['test_mae'].idxmin()
best_model = results.iloc[best_idx]
print(f"   {best_model['model']}: MAE={best_model['test_mae']:.4f}, R²={best_model['test_r2']:.4f}")
print("\nSegüent: python 04_optimal_conditions.py")
print("=" * 80 + "\n")