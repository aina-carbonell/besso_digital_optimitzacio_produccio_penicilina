#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 4: SCRIPT 2 - ANÀLISI D'INCERTESA
Quantificar variabilitat i prediction intervals
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"

print("=" * 80)
print("FASE 4.2 - ANÀLISI D'INCERTESA")
print("=" * 80)

# Carregar dades
print("\n[1/5] Carregant dades i models...")
df_test = pd.read_csv(FASE3_OUT / "test_data.csv")
feature_cols = [c for c in df_test.columns if c not in ['batch_id', 'penicillin', 'time']]

X_test = df_test[feature_cols].values
y_test = df_test['penicillin'].values

ridge_data = joblib.load(FASE3_OUT / "02_ridge_model.pkl")
rf_model = joblib.load(FASE3_OUT / "03_random_forest_model.pkl")['model']
xgb_model = joblib.load(FASE3_OUT / "03_xgboost_model.pkl")['model']

ridge_model = ridge_data['model']
ridge_scaler = ridge_data['scaler']
X_test_scaled = ridge_scaler.transform(X_test)

# Prediccions
print("\n[2/5] Generant prediccions...")
y_pred_ridge = ridge_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Ensemble mean i std
y_pred_mean = (y_pred_ridge + y_pred_rf + y_pred_xgb) / 3
y_pred_std = np.std([y_pred_ridge, y_pred_rf, y_pred_xgb], axis=0)

# Prediction intervals (ensemble std)
print("\n[3/5] Calculant prediction intervals...")
z_score = 1.96  # 95% confidence
lower_bound = y_pred_mean - z_score * y_pred_std
upper_bound = y_pred_mean + z_score * y_pred_std

# Afegir a dataframe
df_results = df_test.copy()
df_results['pred_mean'] = y_pred_mean
df_results['pred_std'] = y_pred_std
df_results['lower_95'] = lower_bound
df_results['upper_95'] = upper_bound
df_results['in_interval'] = (y_test >= lower_bound) & (y_test <= upper_bound)

coverage = df_results['in_interval'].mean()
print(f"   Coverage 95%: {coverage*100:.1f}%")

# Variabilitat per batch
print("\n[4/5] Analitzant variabilitat per batch...")
batch_stats = df_results.groupby('batch_id').agg({
    'penicillin': ['mean', 'std'],
    'pred_mean': 'mean',
    'pred_std': 'mean',
    'in_interval': 'mean'
}).round(4)
batch_stats.columns = ['pen_mean', 'pen_std', 'pred_mean', 'uncertainty', 'coverage']
batch_stats.to_csv(OUTPUT_DIR / "02_batch_variability.csv")

# Visualitzacions
print("\n[5/5] Generant visualitzacions...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Prediction intervals (batch 95)
ax = axes[0, 0]
batch_95 = df_results[df_results['batch_id'] == 95].sort_values('time')
if len(batch_95) > 0:
    ax.plot(batch_95['time'], batch_95['penicillin'], 'o-', color='black', 
           label='Real', linewidth=2, markersize=4)
    ax.plot(batch_95['time'], batch_95['pred_mean'], 's-', color='blue',
           label='Predicció', linewidth=2, markersize=4, alpha=0.7)
    ax.fill_between(batch_95['time'], batch_95['lower_95'], batch_95['upper_95'],
                    alpha=0.3, color='blue', label='95% CI')
    ax.set_xlabel('Temps (h)')
    ax.set_ylabel('Penicil·lina (g/L)')
    ax.set_title('Batch 95 - Prediction Intervals', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Plot 2: Uncertainty distribution
ax = axes[0, 1]
ax.hist(y_pred_std, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(y_pred_std.mean(), color='red', linestyle='--', lw=2, 
          label=f'Mean: {y_pred_std.mean():.3f}')
ax.set_xlabel('Prediction Std Dev (g/L)')
ax.set_ylabel('Freqüència')
ax.set_title('Distribució Incertesa', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Variabilitat per batch
ax = axes[1, 0]
batches = batch_stats.index
x = np.arange(len(batches))
ax.bar(x, batch_stats['uncertainty'], color='orange', edgecolor='black', alpha=0.7)
ax.set_xlabel('Batch ID')
ax.set_ylabel('Uncertainty (std)')
ax.set_title('Incertesa per Batch', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(batches)
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter uncertainty vs error
ax = axes[1, 1]
errors = np.abs(y_test - y_pred_mean)
ax.scatter(y_pred_std, errors, alpha=0.5, s=20, color='green', edgecolor='black', lw=0.5)
ax.set_xlabel('Prediction Std (g/L)')
ax.set_ylabel('Absolute Error (g/L)')
ax.set_title('Incertesa vs Error', fontweight='bold')
ax.grid(True, alpha=0.3)

# Correlation
corr = np.corrcoef(y_pred_std, errors)[0, 1]
ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes,
       fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_uncertainty_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("UNCERTAINTY ANALYSIS COMPLETAT")
print("=" * 80)
print(f"\nCoverage 95% CI: {coverage*100:.1f}%")
print(f"Mean uncertainty: {y_pred_std.mean():.3f} g/L")
print(f"Correlation uncertainty-error: {corr:.3f}")
print("\nSegüent: python 03_hyperparameter_optimization.py")
print("=" * 80 + "\n")