#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: DETECCIÓ DE FALLES (FAULT DETECTION)
Anàlisi d'anomalies en batches 91-100 utilitzant els models entrenats
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

print("=" * 80)
print("FASE 3 - DETECCIÓ DE FALLES (Batches 91-100)")
print("=" * 80)

# =============================================================================
# CARREGAR DADES I MODELS
# =============================================================================
print("\n[1/6] Carregant dades i models...")

df_train = pd.read_csv(OUTPUT_DIR / "train_data.csv")
df_test = pd.read_csv(OUTPUT_DIR / "test_data.csv")

print(f"   Train (normal): {len(df_train):,} mostres, batches 1-90")
print(f"   Test (falles):  {len(df_test):,} mostres, batches 91-100")

# Carregar models
ridge_data = joblib.load(OUTPUT_DIR / "02_ridge_model.pkl")
rf_data = joblib.load(OUTPUT_DIR / "03_random_forest_model.pkl")
xgb_data = joblib.load(OUTPUT_DIR / "03_xgboost_model.pkl")

ridge_model = ridge_data['model']
ridge_scaler = ridge_data['scaler']
rf_model = rf_data['model']
xgb_model = xgb_data['model']
feature_cols = ridge_data['features']

print(f"\n   Models carregats:")
print(f"      • Ridge Regression")
print(f"      • Random Forest")
print(f"      • XGBoost")

# =============================================================================
# PREPARAR DADES
# =============================================================================
print("\n[2/6] Preparant features...")

X_train = df_train[feature_cols].values
y_train = df_train['penicillin'].values
X_test = df_test[feature_cols].values
y_test = df_test['penicillin'].values

# Normalitzar per Ridge
X_train_scaled = ridge_scaler.transform(X_train)
X_test_scaled = ridge_scaler.transform(X_test)

# =============================================================================
# GENERAR PREDICCIONS DE TOTS ELS MODELS
# =============================================================================
print("\n[3/6] Generant prediccions per detectar anomalies...")

# Prediccions train (per establir threshold)
y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_train_pred_rf = rf_model.predict(X_train)
y_train_pred_xgb = xgb_model.predict(X_train)

# Prediccions test
y_test_pred_ridge = ridge_model.predict(X_test_scaled)
y_test_pred_rf = rf_model.predict(X_test)
y_test_pred_xgb = xgb_model.predict(X_test)

print(f"   Prediccions generades per {len(y_test):,} mostres test")

# =============================================================================
# CÀLCUL D'ERRORS I DETECCIÓ D'ANOMALIES
# =============================================================================
print("\n[4/6] Detectant anomalies...")

# Errors absoluts
errors_train_ridge = np.abs(y_train - y_train_pred_ridge)
errors_train_rf = np.abs(y_train - y_train_pred_rf)
errors_train_xgb = np.abs(y_train - y_train_pred_xgb)

errors_test_ridge = np.abs(y_test - y_test_pred_ridge)
errors_test_rf = np.abs(y_test - y_test_pred_rf)
errors_test_xgb = np.abs(y_test - y_test_pred_xgb)

# Thresholds basats en percentil 95 del train
threshold_ridge = np.percentile(errors_train_ridge, 95)
threshold_rf = np.percentile(errors_train_rf, 95)
threshold_xgb = np.percentile(errors_train_xgb, 95)

print(f"\n   Thresholds de detecció (P95 train):")
print(f"      Ridge: {threshold_ridge:.4f} g/L")
print(f"      RF:    {threshold_rf:.4f} g/L")
print(f"      XGB:   {threshold_xgb:.4f} g/L")

# Detecció d'anomalies
anomalies_ridge = errors_test_ridge > threshold_ridge
anomalies_rf = errors_test_rf > threshold_rf
anomalies_xgb = errors_test_xgb > threshold_xgb

# Consensus: anomalia si almenys 2 models coincideixen
anomalies_consensus = (anomalies_ridge.astype(int) + 
                      anomalies_rf.astype(int) + 
                      anomalies_xgb.astype(int)) >= 2

print(f"\n   Anomalies detectades:")
print(f"      Ridge: {anomalies_ridge.sum():,} ({100*anomalies_ridge.mean():.1f}%)")
print(f"      RF:    {anomalies_rf.sum():,} ({100*anomalies_rf.mean():.1f}%)")
print(f"      XGB:   {anomalies_xgb.sum():,} ({100*anomalies_xgb.mean():.1f}%)")
print(f"      Consensus (≥2): {anomalies_consensus.sum():,} ({100*anomalies_consensus.mean():.1f}%)")

# =============================================================================
# ANÀLISI PER BATCH
# =============================================================================
print("\n[5/6] Analitzant anomalies per batch...")

# Crear DataFrame amb resultats
df_test_results = df_test.copy()
df_test_results['error_ridge'] = errors_test_ridge
df_test_results['error_rf'] = errors_test_rf
df_test_results['error_xgb'] = errors_test_xgb
df_test_results['anomaly_ridge'] = anomalies_ridge
df_test_results['anomaly_rf'] = anomalies_rf
df_test_results['anomaly_xgb'] = anomalies_xgb
df_test_results['anomaly_consensus'] = anomalies_consensus
df_test_results['pred_ridge'] = y_test_pred_ridge
df_test_results['pred_rf'] = y_test_pred_rf
df_test_results['pred_xgb'] = y_test_pred_xgb

# Anomaly score per batch
batch_scores = df_test_results.groupby('batch_id').agg({
    'anomaly_ridge': 'mean',
    'anomaly_rf': 'mean',
    'anomaly_xgb': 'mean',
    'anomaly_consensus': 'mean',
    'error_ridge': 'mean',
    'error_rf': 'mean',
    'error_xgb': 'mean'
}).round(4)

batch_scores.columns = ['Ridge_Anomaly_%', 'RF_Anomaly_%', 'XGB_Anomaly_%', 
                       'Consensus_Anomaly_%', 'Ridge_MAE', 'RF_MAE', 'XGB_MAE']

batch_scores *= 100  # Convertir a percentatge (columnes anomaly)
batch_scores[['Ridge_MAE', 'RF_MAE', 'XGB_MAE']] /= 100  # Mantenir MAE en g/L

print(f"\n   Scores per batch (91-100):")
print(batch_scores.to_string())

# =============================================================================
# VISUALITZACIONS
# =============================================================================
print("\n[6/6] Generant visualitzacions...")

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)

# Plot 1: Distribució errors train vs test (Ridge)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(errors_train_ridge, bins=50, alpha=0.7, label='Train (normal)', 
        color='blue', edgecolor='black')
ax1.hist(errors_test_ridge, bins=30, alpha=0.7, label='Test (falles)', 
        color='red', edgecolor='black')
ax1.axvline(threshold_ridge, color='green', linestyle='--', lw=2, label=f'Threshold P95')
ax1.set_xlabel('Error absolut (g/L)', fontsize=10)
ax1.set_ylabel('Freqüència', fontsize=10)
ax1.set_title('Ridge - Distribució Errors', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Distribució errors RF
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(errors_train_rf, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
ax2.hist(errors_test_rf, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
ax2.axvline(threshold_rf, color='green', linestyle='--', lw=2, label=f'Threshold P95')
ax2.set_xlabel('Error absolut (g/L)', fontsize=10)
ax2.set_ylabel('Freqüència', fontsize=10)
ax2.set_title('Random Forest - Distribució Errors', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Distribució errors XGB
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(errors_train_xgb, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
ax3.hist(errors_test_xgb, bins=30, alpha=0.7, label='Test', color='red', edgecolor='black')
ax3.axvline(threshold_xgb, color='green', linestyle='--', lw=2, label=f'Threshold P95')
ax3.set_xlabel('Error absolut (g/L)', fontsize=10)
ax3.set_ylabel('Freqüència', fontsize=10)
ax3.set_title('XGBoost - Distribució Errors', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Heatmap anomalies per batch
ax4 = fig.add_subplot(gs[1, :])
heatmap_data = batch_scores[['Ridge_Anomaly_%', 'RF_Anomaly_%', 'XGB_Anomaly_%', 'Consensus_Anomaly_%']].T
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
           cbar_kws={'label': '% Anomalies'}, ax=ax4, vmin=0, vmax=100)
ax4.set_xlabel('Batch ID', fontsize=11)
ax4.set_ylabel('Model', fontsize=11)
ax4.set_title('Heatmap % Anomalies per Batch', fontsize=13, fontweight='bold')

# Plot 5-7: Timeline batches amb anomalies
for idx, (batch_id, model_name, errors, anomalies, color) in enumerate([
    (95, 'Ridge', errors_test_ridge, anomalies_ridge, 'purple'),
    (96, 'Random Forest', errors_test_rf, anomalies_rf, 'green'),
    (98, 'XGBoost', errors_test_xgb, anomalies_xgb, 'blue')
]):
    ax = fig.add_subplot(gs[2, idx])
    
    batch_data = df_test_results[df_test_results['batch_id'] == batch_id]
    
    if len(batch_data) > 0:
        # Plot penicillin real
        ax.plot(batch_data['time'], batch_data['penicillin'], 
               'o-', color='black', label='Real', linewidth=2, markersize=3)
        
        # Plot predicció
        pred_col = f'pred_{model_name.lower().split()[0]}'
        if pred_col in batch_data.columns:
            ax.plot(batch_data['time'], batch_data[pred_col], 
                   's-', color=color, alpha=0.7, label='Predicció', linewidth=2, markersize=3)
        
        # Marcar anomalies
        anom_col = f'anomaly_{model_name.lower().split()[0]}'
        if anom_col in batch_data.columns:
            anomalies_batch = batch_data[batch_data[anom_col]]
            if len(anomalies_batch) > 0:
                ax.scatter(anomalies_batch['time'], anomalies_batch['penicillin'],
                          color='red', s=100, marker='x', linewidth=3, 
                          label='Anomalia', zorder=5)
        
        ax.set_xlabel('Temps (h)', fontsize=10)
        ax.set_ylabel('Penicil·lina (g/L)', fontsize=10)
        ax.set_title(f'Batch {batch_id} - {model_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

# Plot 8: Comparació MAE per batch
ax8 = fig.add_subplot(gs[3, 0])
batch_ids = batch_scores.index
x = np.arange(len(batch_ids))
width = 0.25

ax8.bar(x - width, batch_scores['Ridge_MAE'], width, label='Ridge', 
       color='purple', edgecolor='black', alpha=0.7)
ax8.bar(x, batch_scores['RF_MAE'], width, label='RF', 
       color='green', edgecolor='black', alpha=0.7)
ax8.bar(x + width, batch_scores['XGB_MAE'], width, label='XGB', 
       color='blue', edgecolor='black', alpha=0.7)

ax8.set_ylabel('MAE (g/L)', fontsize=10)
ax8.set_xlabel('Batch ID', fontsize=10)
ax8.set_title('MAE per Batch i Model', fontsize=11, fontweight='bold')
ax8.set_xticks(x)
ax8.set_xticklabels(batch_ids)
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: % Anomalies per model
ax9 = fig.add_subplot(gs[3, 1])
models = ['Ridge', 'RF', 'XGB', 'Consensus']
anomaly_pcts = [
    100 * anomalies_ridge.mean(),
    100 * anomalies_rf.mean(),
    100 * anomalies_xgb.mean(),
    100 * anomalies_consensus.mean()
]
colors_bar = ['purple', 'green', 'blue', 'orange']

bars = ax9.bar(models, anomaly_pcts, color=colors_bar, edgecolor='black', alpha=0.7)
ax9.set_ylabel('% Anomalies', fontsize=10)
ax9.set_title('% Total Anomalies Detectades', fontsize=11, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# Afegir valors sobre barres
for bar, val in zip(bars, anomaly_pcts):
    height = bar.get_height()
    ax9.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 10: Batch més problemàtic
ax10 = fig.add_subplot(gs[3, 2])

worst_batch_id = batch_scores['Consensus_Anomaly_%'].idxmax()
worst_batch = df_test_results[df_test_results['batch_id'] == worst_batch_id]

if len(worst_batch) > 0:
    ax10.plot(worst_batch['time'], worst_batch['penicillin'], 
             'o-', color='black', label='Real', linewidth=2, markersize=4)
    ax10.plot(worst_batch['time'], worst_batch['pred_xgb'], 
             's-', color='blue', alpha=0.7, label='Pred XGB', linewidth=2, markersize=4)
    
    # Marcar totes les anomalies consensus
    anomalies_worst = worst_batch[worst_batch['anomaly_consensus']]
    if len(anomalies_worst) > 0:
        ax10.scatter(anomalies_worst['time'], anomalies_worst['penicillin'],
                    color='red', s=150, marker='x', linewidth=3, 
                    label=f'Anomalies ({len(anomalies_worst)})', zorder=5)
    
    ax10.set_xlabel('Temps (h)', fontsize=10)
    ax10.set_ylabel('Penicil·lina (g/L)', fontsize=10)
    ax10.set_title(f'Batch {worst_batch_id} (Pitjor cas)\n{batch_scores.loc[worst_batch_id, "Consensus_Anomaly_%"]:.1f}% anomalies', 
                  fontsize=11, fontweight='bold')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)

plt.suptitle('FAULT DETECTION - Detecció d\'Anomalies en Batches 91-100', 
             fontsize=16, fontweight='bold')

results_path = OUTPUT_DIR / "05_fault_detection_results.png"
plt.savefig(results_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {results_path.name}")

# =============================================================================
# GUARDAR RESULTATS
# =============================================================================
print("\n[7/7] Guardant resultats...")

# Guardar scores per batch
scores_path = OUTPUT_DIR / "05_fault_detection_scores.csv"
batch_scores.to_csv(scores_path)
print(f"   Scores: {scores_path.name}")

# Guardar detalls per mostra
details_path = OUTPUT_DIR / "05_fault_detection_details.csv"
df_test_results[['batch_id', 'time', 'penicillin', 
                'pred_ridge', 'pred_rf', 'pred_xgb',
                'error_ridge', 'error_rf', 'error_xgb',
                'anomaly_ridge', 'anomaly_rf', 'anomaly_xgb', 
                'anomaly_consensus']].to_csv(details_path, index=False)
print(f"   Detalls: {details_path.name}")

# =============================================================================
# RESUM
# =============================================================================
print("\n" + "=" * 80)
print("FAULT DETECTION COMPLETAT")
print("=" * 80)

print(f"\nRESULTATS:")
print(f"   Batches analitzats: 91-100 (10 batches amb falles)")
print(f"   Mostres totals: {len(df_test):,}")

print(f"\n   Anomalies detectades:")
print(f"      Ridge:     {anomalies_ridge.sum():,} ({100*anomalies_ridge.mean():.1f}%)")
print(f"      RF:        {anomalies_rf.sum():,} ({100*anomalies_rf.mean():.1f}%)")
print(f"      XGBoost:   {anomalies_xgb.sum():,} ({100*anomalies_xgb.mean():.1f}%)")
print(f"      Consensus: {anomalies_consensus.sum():,} ({100*anomalies_consensus.mean():.1f}%)")

print(f"\n   Batch més problemàtic: {worst_batch_id}")
print(f"      {batch_scores.loc[worst_batch_id, 'Consensus_Anomaly_%']:.1f}% anomalies")

print(f"\nFitxers generats:")
print(f"   • {results_path.name}")
print(f"   • {scores_path.name}")
print(f"   • {details_path.name}")

print(f"\nSegüent pas: python 06_model_comparison.py")
print("=" * 80 + "\n")