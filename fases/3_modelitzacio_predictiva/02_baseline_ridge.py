#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: MODEL BÀSIC - RIDGE REGRESSION
Establir baseline amb model lineal regularitzat
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

print("=" * 80)
print("FASE 3 - MODEL BÀSIC: RIDGE REGRESSION")
print("=" * 80)

# Carregar dades
print("\n[1/7] Carregant datasets...")

train_file = OUTPUT_DIR / "train_data.csv"
test_file = OUTPUT_DIR / "test_data.csv"

if not train_file.exists() or not test_file.exists():
    print("ERROR: Executa primer 01_data_preparation.py")
    exit(1)

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print(f"   Train: {len(df_train):,} mostres")
print(f"   Test:  {len(df_test):,} mostres")

# =============================================================================
# PREPARACIÓ FEATURES
# =============================================================================
print("\n[2/7] Preparant features...")

# Identificar columnes
feature_cols = [col for col in df_train.columns 
                if col not in ['batch_id', 'penicillin', 'time']]
target_col = 'penicillin'

print(f"   Features: {len(feature_cols)}")
print(f"   Target: {target_col}")

# Extreure X i y
X_train = df_train[feature_cols].values
y_train = df_train[target_col].values
X_test = df_test[feature_cols].values
y_test = df_test[target_col].values

print(f"\n   X_train shape: {X_train.shape}")
print(f"   y_train shape: {y_train.shape}")
print(f"   X_test shape:  {X_test.shape}")
print(f"   y_test shape:  {y_test.shape}")

# =============================================================================
# NORMALITZACIÓ
# =============================================================================
print("\n[3/7] Normalitzant features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   Scaler fitejat amb train set")
print(f"   Mean: {scaler.mean_[:3].round(3)}...")
print(f"   Std:  {scaler.scale_[:3].round(3)}...")

# =============================================================================
# ENTRENAMENT MODEL
# =============================================================================
print("\n[4/7] Entrenant Ridge Regression...")

# Hiperparàmetres
alpha = 1.0  # Coeficient de regularització L2

model = Ridge(alpha=alpha, random_state=42)

print(f"   Hiperparàmetres:")
print(f"      alpha (L2): {alpha}")

# Entrenar
model.fit(X_train_scaled, y_train)

print(f"   Model entrenat!")
print(f"   Coeficients: {model.coef_[:3].round(3)}...")
print(f"   Intercept: {model.intercept_:.3f}")

# =============================================================================
# PREDICCIONS
# =============================================================================
print("\n[5/7] Generant prediccions...")

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print(f"   Prediccions train: {len(y_train_pred):,}")
print(f"   Prediccions test:  {len(y_test_pred):,}")

# =============================================================================
# AVALUACIÓ
# =============================================================================
print("\n[6/7] Avaluant model...")

# Mètriques train
mse_train = mean_squared_error(y_train, y_train_pred)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)

# Mètriques test
mse_test = mean_squared_error(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)

print(f"\n   === TRAIN SET ===")
print(f"   MSE:  {mse_train:.4f} (g/L)²")
print(f"   RMSE: {rmse_train:.4f} g/L")
print(f"   MAE:  {mae_train:.4f} g/L")
print(f"   R²:   {r2_train:.4f}")

print(f"\n   === TEST SET (FALLES) ===")
print(f"   MSE:  {mse_test:.4f} (g/L)²")
print(f"   RMSE: {rmse_test:.4f} g/L")
print(f"   MAE:  {mae_test:.4f} g/L")
print(f"   R²:   {r2_test:.4f}")

# =============================================================================
# VISUALITZACIÓ
# =============================================================================
print("\n[7/7] Generant visualitzacions...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Scatter Train
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_train_pred, alpha=0.3, s=20, color='blue', edgecolor='none')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('Real (g/L)', fontsize=11)
ax1.set_ylabel('Predicció (g/L)', fontsize=11)
ax1.set_title(f'TRAIN SET\nR² = {r2_train:.4f}', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter Test
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test, y_test_pred, alpha=0.5, s=30, color='red', edgecolor='black', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect prediction')
ax2.set_xlabel('Real (g/L)', fontsize=11)
ax2.set_ylabel('Predicció (g/L)', fontsize=11)
ax2.set_title(f'TEST SET (Falles)\nR² = {r2_test:.4f}', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals Train
ax3 = fig.add_subplot(gs[0, 2])
residuals_train = y_train - y_train_pred
ax3.scatter(y_train_pred, residuals_train, alpha=0.3, s=20, color='blue', edgecolor='none')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicció (g/L)', fontsize=11)
ax3.set_ylabel('Residuals (g/L)', fontsize=11)
ax3.set_title('Residuals TRAIN', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals Test
ax4 = fig.add_subplot(gs[1, 0])
residuals_test = y_test - y_test_pred
ax4.scatter(y_test_pred, residuals_test, alpha=0.5, s=30, color='red', edgecolor='black', linewidth=0.5)
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicció (g/L)', fontsize=11)
ax4.set_ylabel('Residuals (g/L)', fontsize=11)
ax4.set_title('Residuals TEST (Falles)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Distribució residuals
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(residuals_train, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
ax5.hist(residuals_test, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
ax5.axvline(x=0, color='black', linestyle='--', lw=2)
ax5.set_xlabel('Residuals (g/L)', fontsize=11)
ax5.set_ylabel('Freqüència', fontsize=11)
ax5.set_title('Distribució de Residuals', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Coeficients
ax6 = fig.add_subplot(gs[1, 2])
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

colors_coef = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
ax6.barh(range(len(coef_df)), coef_df['Coefficient'], color=colors_coef, alpha=0.7, edgecolor='black')
ax6.set_yticks(range(len(coef_df)))
ax6.set_yticklabels(coef_df['Feature'], fontsize=9)
ax6.set_xlabel('Coeficient', fontsize=11)
ax6.set_title('Coeficients Ridge', fontsize=12, fontweight='bold')
ax6.axvline(x=0, color='black', linestyle='-', lw=1)
ax6.grid(True, alpha=0.3, axis='x')

# Plot 7: Mètriques comparatives
ax7 = fig.add_subplot(gs[2, 0])
metrics = ['R²', 'MAE', 'RMSE']
train_vals = [r2_train, mae_train, rmse_train]
test_vals = [r2_test, mae_test, rmse_test]

x = np.arange(len(metrics))
width = 0.35

ax7.bar(x - width/2, train_vals, width, label='Train', color='lightblue', edgecolor='black')
ax7.bar(x + width/2, test_vals, width, label='Test', color='lightcoral', edgecolor='black')
ax7.set_ylabel('Valor', fontsize=11)
ax7.set_title('Comparació Mètriques', fontsize=12, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(metrics)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Evolució temporal (batch 95 - amb falla)
ax8 = fig.add_subplot(gs[2, 1:])
batch_95 = df_test[df_test['batch_id'] == 95].copy()
if len(batch_95) > 0:
    batch_95_X = scaler.transform(batch_95[feature_cols].values)
    batch_95_pred = model.predict(batch_95_X)
    
    ax8.plot(batch_95['time'], batch_95['penicillin'], 'o-', 
            color='black', label='Real', linewidth=2, markersize=4)
    ax8.plot(batch_95['time'], batch_95_pred, 's-', 
            color='red', alpha=0.7, label='Ridge prediction', linewidth=2, markersize=4)
    ax8.set_xlabel('Temps (h)', fontsize=11)
    ax8.set_ylabel('Penicil·lina (g/L)', fontsize=11)
    ax8.set_title('Batch 95 (amb falla) - Real vs Predicció', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

plt.suptitle('RIDGE REGRESSION - Resultats', fontsize=16, fontweight='bold', y=0.995)

results_path = OUTPUT_DIR / "02_ridge_results.png"
plt.savefig(results_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {results_path.name}")

# =============================================================================
# GUARDAR MODEL I RESULTATS
# =============================================================================
print("\n[8/8] Guardant model i resultats...")

# Guardar model
model_path = OUTPUT_DIR / "02_ridge_model.pkl"
joblib.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, model_path)
print(f"   Model: {model_path.name}")

# Guardar mètriques
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Train': [mse_train, rmse_train, mae_train, r2_train],
    'Test': [mse_test, rmse_test, mae_test, r2_test]
})

metrics_path = OUTPUT_DIR / "02_ridge_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Mètriques: {metrics_path.name}")

# =============================================================================
# RESUM
# =============================================================================
print("\n" + "=" * 80)
print("RIDGE REGRESSION COMPLETAT")
print("=" * 80)
print(f"\nRESULTATS:")
print(f"   Train R²: {r2_train:.4f} | MAE: {mae_train:.4f} g/L")
print(f"   Test R²:  {r2_test:.4f} | MAE: {mae_test:.4f} g/L")
print(f"\nFitxers generats:")
print(f"   • {model_path.name}")
print(f"   • {results_path.name}")
print(f"   • {metrics_path.name}")
print(f"\nSegüent pas: python 03_ensemble_models.py")
print("=" * 80 + "\n")