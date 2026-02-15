#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: MODELS ENSEMBLE - RANDOM FOREST + XGBOOST
Models intermedis amb anàlisi de feature importance
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

print("=" * 80)
print("FASE 3 - MODELS ENSEMBLE: RANDOM FOREST + XGBOOST")
print("=" * 80)

# Carregar dades
print("\n[1/8] Carregant datasets...")

df_train = pd.read_csv(OUTPUT_DIR / "train_data.csv")
df_test = pd.read_csv(OUTPUT_DIR / "test_data.csv")

print(f"   Train: {len(df_train):,} mostres")
print(f"   Test:  {len(df_test):,} mostres")

# Preparar features
feature_cols = [col for col in df_train.columns 
                if col not in ['batch_id', 'penicillin', 'time']]
target_col = 'penicillin'

X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"\n   Features: {len(feature_cols)}")
print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")

# =============================================================================
# RANDOM FOREST
# =============================================================================
print("\n[2/8] Entrenant Random Forest...")

start_time = time.time()

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)

rf_time = time.time() - start_time
print(f"   OK: Entrenat en {rf_time:.1f} segons")

# Prediccions RF
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Mètriques RF
rf_metrics = {
    'train': {
        'mse': mean_squared_error(y_train, y_train_pred_rf),
        'mae': mean_absolute_error(y_train, y_train_pred_rf),
        'r2': r2_score(y_train, y_train_pred_rf)
    },
    'test': {
        'mse': mean_squared_error(y_test, y_test_pred_rf),
        'mae': mean_absolute_error(y_test, y_test_pred_rf),
        'r2': r2_score(y_test, y_test_pred_rf)
    }
}

print(f"\n   RANDOM FOREST - Train R²: {rf_metrics['train']['r2']:.4f}, MAE: {rf_metrics['train']['mae']:.4f}")
print(f"   RANDOM FOREST - Test R²:  {rf_metrics['test']['r2']:.4f}, MAE: {rf_metrics['test']['mae']:.4f}")

# Feature importance
rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 features més importants (RF):")
for idx, row in rf_importance.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}")

# =============================================================================
# XGBOOST
# =============================================================================
print("\n[3/8] Entrenant XGBoost...")

start_time = time.time()

xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

xgb_model.fit(X_train, y_train)

xgb_time = time.time() - start_time
print(f"   OK: Entrenat en {xgb_time:.1f} segons")

# Prediccions XGB
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Mètriques XGB
xgb_metrics = {
    'train': {
        'mse': mean_squared_error(y_train, y_train_pred_xgb),
        'mae': mean_absolute_error(y_train, y_train_pred_xgb),
        'r2': r2_score(y_train, y_train_pred_xgb)
    },
    'test': {
        'mse': mean_squared_error(y_test, y_test_pred_xgb),
        'mae': mean_absolute_error(y_test, y_test_pred_xgb),
        'r2': r2_score(y_test, y_test_pred_xgb)
    }
}

print(f"\n   XGBOOST - Train R²: {xgb_metrics['train']['r2']:.4f}, MAE: {xgb_metrics['train']['mae']:.4f}")
print(f"   XGBOOST - Test R²:  {xgb_metrics['test']['r2']:.4f}, MAE: {xgb_metrics['test']['mae']:.4f}")

# Feature importance
xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 features més importants (XGB):")
for idx, row in xgb_importance.head(5).iterrows():
    print(f"      {row['feature']}: {row['importance']:.4f}")

# =============================================================================
# VISUALITZACIONS - RESULTATS
# =============================================================================
print("\n[4/8] Generant visualitzacions de resultats...")

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: RF Scatter Train
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_train_pred_rf, alpha=0.3, s=15, color='green', edgecolor='none')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax1.set_xlabel('Real (g/L)', fontsize=10)
ax1.set_ylabel('Predicció (g/L)', fontsize=10)
ax1.set_title(f'Random Forest - TRAIN\nR² = {rf_metrics["train"]["r2"]:.4f}', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: RF Scatter Test
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_test_pred_rf, alpha=0.5, s=25, color='green', edgecolor='black', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Real (g/L)', fontsize=10)
ax2.set_ylabel('Predicció (g/L)', fontsize=10)
ax2.set_title(f'Random Forest - TEST\nR² = {rf_metrics["test"]["r2"]:.4f}', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: XGB Scatter Train
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_train, y_train_pred_xgb, alpha=0.3, s=15, color='blue', edgecolor='none')
ax3.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
ax3.set_xlabel('Real (g/L)', fontsize=10)
ax3.set_ylabel('Predicció (g/L)', fontsize=10)
ax3.set_title(f'XGBoost - TRAIN\nR² = {xgb_metrics["train"]["r2"]:.4f}', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: XGB Scatter Test
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test, y_test_pred_xgb, alpha=0.5, s=25, color='blue', edgecolor='black', linewidth=0.5)
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax4.set_xlabel('Real (g/L)', fontsize=10)
ax4.set_ylabel('Predicció (g/L)', fontsize=10)
ax4.set_title(f'XGBoost - TEST\nR² = {xgb_metrics["test"]["r2"]:.4f}', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals RF
ax5 = fig.add_subplot(gs[1, 1])
residuals_rf = y_test - y_test_pred_rf
ax5.scatter(y_test_pred_rf, residuals_rf, alpha=0.5, s=25, color='green', edgecolor='black', linewidth=0.5)
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicció (g/L)', fontsize=10)
ax5.set_ylabel('Residuals (g/L)', fontsize=10)
ax5.set_title('Residuals RF - TEST', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Residuals XGB
ax6 = fig.add_subplot(gs[1, 2])
residuals_xgb = y_test - y_test_pred_xgb
ax6.scatter(y_test_pred_xgb, residuals_xgb, alpha=0.5, s=25, color='blue', edgecolor='black', linewidth=0.5)
ax6.axhline(y=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Predicció (g/L)', fontsize=10)
ax6.set_ylabel('Residuals (g/L)', fontsize=10)
ax6.set_title('Residuals XGB - TEST', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Comparació mètriques
ax7 = fig.add_subplot(gs[2, :2])
models = ['RF Train', 'RF Test', 'XGB Train', 'XGB Test']
r2_vals = [rf_metrics['train']['r2'], rf_metrics['test']['r2'], 
           xgb_metrics['train']['r2'], xgb_metrics['test']['r2']]
mae_vals = [rf_metrics['train']['mae'], rf_metrics['test']['mae'],
            xgb_metrics['train']['mae'], xgb_metrics['test']['mae']]

x = np.arange(len(models))
width = 0.35

ax7.bar(x - width/2, r2_vals, width, label='R²', color='lightblue', edgecolor='black')
ax7_twin = ax7.twinx()
ax7_twin.bar(x + width/2, mae_vals, width, label='MAE', color='lightcoral', edgecolor='black')

ax7.set_ylabel('R²', fontsize=11, color='blue')
ax7_twin.set_ylabel('MAE (g/L)', fontsize=11, color='red')
ax7.set_title('Comparació Mètriques - RF vs XGBoost', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(models, fontsize=10)
ax7.tick_params(axis='y', labelcolor='blue')
ax7_twin.tick_params(axis='y', labelcolor='red')
ax7.legend(loc='upper left')
ax7_twin.legend(loc='upper right')
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Batch 95 temporal
ax8 = fig.add_subplot(gs[2, 2])
batch_95 = df_test[df_test['batch_id'] == 95].copy()
if len(batch_95) > 0:
    batch_95_X = batch_95[feature_cols].values
    batch_95_pred_rf = rf_model.predict(batch_95_X)
    batch_95_pred_xgb = xgb_model.predict(batch_95_X)
    
    ax8.plot(batch_95['time'], batch_95['penicillin'], 'o-', 
            color='black', label='Real', linewidth=2, markersize=3)
    ax8.plot(batch_95['time'], batch_95_pred_rf, 's-', 
            color='green', alpha=0.7, label='RF', linewidth=1.5, markersize=3)
    ax8.plot(batch_95['time'], batch_95_pred_xgb, '^-', 
            color='blue', alpha=0.7, label='XGB', linewidth=1.5, markersize=3)
    ax8.set_xlabel('Temps (h)', fontsize=10)
    ax8.set_ylabel('Penicil·lina (g/L)', fontsize=10)
    ax8.set_title('Batch 95 (falla)', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

plt.suptitle('ENSEMBLE MODELS - Resultats', fontsize=16, fontweight='bold')

results_path = OUTPUT_DIR / "03_ensemble_results.png"
plt.savefig(results_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {results_path.name}")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================
print("\n[5/8] Generant visualització feature importance...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RF importance
ax1 = axes[0]
colors_rf = plt.cm.Greens(np.linspace(0.4, 0.8, len(rf_importance)))
ax1.barh(range(len(rf_importance)), rf_importance['importance'], color=colors_rf, edgecolor='black')
ax1.set_yticks(range(len(rf_importance)))
ax1.set_yticklabels(rf_importance['feature'], fontsize=10)
ax1.set_xlabel('Importance', fontsize=11)
ax1.set_title('Random Forest - Feature Importance', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# XGB importance
ax2 = axes[1]
colors_xgb = plt.cm.Blues(np.linspace(0.4, 0.8, len(xgb_importance)))
ax2.barh(range(len(xgb_importance)), xgb_importance['importance'], color=colors_xgb, edgecolor='black')
ax2.set_yticks(range(len(xgb_importance)))
ax2.set_yticklabels(xgb_importance['feature'], fontsize=10)
ax2.set_xlabel('Importance', fontsize=11)
ax2.set_title('XGBoost - Feature Importance', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()

importance_path = OUTPUT_DIR / "03_feature_importance.png"
plt.savefig(importance_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {importance_path.name}")

# =============================================================================
# GUARDAR MODELS
# =============================================================================
print("\n[6/8] Guardant models...")

rf_path = OUTPUT_DIR / "03_random_forest_model.pkl"
xgb_path = OUTPUT_DIR / "03_xgboost_model.pkl"

joblib.dump({'model': rf_model, 'features': feature_cols}, rf_path)
joblib.dump({'model': xgb_model, 'features': feature_cols}, xgb_path)

print(f"   RF:  {rf_path.name}")
print(f"   XGB: {xgb_path.name}")

# =============================================================================
# GUARDAR MÈTRIQUES
# =============================================================================
print("\n[7/8] Guardant mètriques...")

metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'Random Forest', 'XGBoost', 'XGBoost'],
    'Dataset': ['Train', 'Test', 'Train', 'Test'],
    'MSE': [rf_metrics['train']['mse'], rf_metrics['test']['mse'],
            xgb_metrics['train']['mse'], xgb_metrics['test']['mse']],
    'MAE': [rf_metrics['train']['mae'], rf_metrics['test']['mae'],
            xgb_metrics['train']['mae'], xgb_metrics['test']['mae']],
    'R2': [rf_metrics['train']['r2'], rf_metrics['test']['r2'],
           xgb_metrics['train']['r2'], xgb_metrics['test']['r2']],
    'Training_Time_s': [rf_time, np.nan, xgb_time, np.nan]
})

metrics_path = OUTPUT_DIR / "03_ensemble_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print(f"   Mètriques: {metrics_path.name}")

# Feature importance
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance': rf_model.feature_importances_,
    'XGB_Importance': xgb_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

importance_csv_path = OUTPUT_DIR / "03_feature_importance.csv"
importance_df.to_csv(importance_csv_path, index=False)

print(f"   Importance: {importance_csv_path.name}")

# =============================================================================
# RESUM
# =============================================================================
print("\n[8/8] Resum final...")

print("\n" + "=" * 80)
print("ENSEMBLE MODELS COMPLETATS")
print("=" * 80)

print(f"\nRANDOM FOREST:")
print(f"   Train - R²: {rf_metrics['train']['r2']:.4f}, MAE: {rf_metrics['train']['mae']:.4f} g/L")
print(f"   Test  - R²: {rf_metrics['test']['r2']:.4f}, MAE: {rf_metrics['test']['mae']:.4f} g/L")
print(f"   Temps: {rf_time:.1f}s")

print(f"\nXGBOOST:")
print(f"   Train - R²: {xgb_metrics['train']['r2']:.4f}, MAE: {xgb_metrics['train']['mae']:.4f} g/L")
print(f"   Test  - R²: {xgb_metrics['test']['r2']:.4f}, MAE: {xgb_metrics['test']['mae']:.4f} g/L")
print(f"   Temps: {xgb_time:.1f}s")

print(f"\nFitxers generats:")
print(f"   • {rf_path.name}")
print(f"   • {xgb_path.name}")
print(f"   • {results_path.name}")
print(f"   • {importance_path.name}")
print(f"   • {metrics_path.name}")

print(f"\nSegüent pas: python 04_lstm_model.py")
print("=" * 80 + "\n")