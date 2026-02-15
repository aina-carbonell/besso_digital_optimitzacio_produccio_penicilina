#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: COMPARACIÓ FINAL DE MODELS
Resum i visualització comparativa de tots els models
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

print("=" * 80)
print("FASE 3 - COMPARACIÓ FINAL DE MODELS")
print("=" * 80)

# =============================================================================
# CARREGAR MÈTRIQUES DE TOTS ELS MODELS
# =============================================================================
print("\n[1/4] Carregant mètriques...")

# Crear taula final
comparison_data = []

# Ridge
ridge_df = pd.read_csv(OUTPUT_DIR / "02_ridge_metrics.csv")
for dataset in ['Train', 'Test']:
    comparison_data.append({
        'Model': 'Ridge',
        'Dataset': dataset,
        'MSE': ridge_df[ridge_df['Metric'] == 'MSE'][dataset].values[0],
        'RMSE': ridge_df[ridge_df['Metric'] == 'RMSE'][dataset].values[0],
        'MAE': ridge_df[ridge_df['Metric'] == 'MAE'][dataset].values[0],
        'R2': ridge_df[ridge_df['Metric'] == 'R²'][dataset].values[0]
    })

# Ensemble
ensemble_df = pd.read_csv(OUTPUT_DIR / "03_ensemble_metrics.csv")
for _, row in ensemble_df.iterrows():
    comparison_data.append({
        'Model': row['Model'],
        'Dataset': row['Dataset'],
        'MSE': row['MSE'],
        'RMSE': np.sqrt(row['MSE']),
        'MAE': row['MAE'],
        'R2': row['R2']
    })

# LSTM
print("   Carregant LSTM metrics...")
lstm_df = pd.read_csv(OUTPUT_DIR / "04_lstm_metrics.csv")

# Comprovar estructura
print(f"   Estructura LSTM: {lstm_df.shape} - Columnes: {list(lstm_df.columns)}")

# Processar LSTM - ARA SÍ QUE FUNCIONARÀ
for _, row in lstm_df.iterrows():
    metric = row['Metric']
    train_val = row['Train']
    test_val = row['Test']
    
    # Afegir Train
    comparison_data.append({
        'Model': 'LSTM',
        'Dataset': 'Train',
        'MSE': train_val if metric == 'MSE' else None,
        'RMSE': train_val if metric == 'RMSE' else None,
        'MAE': train_val if metric == 'MAE' else None,
        'R2': train_val if metric == 'R²' else None
    })
    
    # Afegir Test
    comparison_data.append({
        'Model': 'LSTM',
        'Dataset': 'Test',
        'MSE': test_val if metric == 'MSE' else None,
        'RMSE': test_val if metric == 'RMSE' else None,
        'MAE': test_val if metric == 'MAE' else None,
        'R2': test_val if metric == 'R²' else None
    })

# Després de crear df_comparison, necessitem "fusionar" les files per tenir
# una fila per Model/Dataset amb totes les mètriques
df_comparison = pd.DataFrame(comparison_data)

# Ara "pivotem" per tenir una fila per cada combinació Model/Dataset
df_pivot = df_comparison.groupby(['Model', 'Dataset']).agg({
    'MSE': 'first',
    'RMSE': 'first', 
    'MAE': 'first',
    'R2': 'first'
}).reset_index()

# Eliminar files amb None
df_pivot = df_pivot.dropna(subset=['MSE', 'RMSE', 'MAE', 'R2'])

print(f"\n   Taula de comparació final:")
print(df_pivot.to_string(index=False))

# Utilitzar df_pivot en lloc de df_comparison per a la resta del codi
df_comparison = df_pivot

# =============================================================================
# VISUALITZACIONS COMPARATIVES
# =============================================================================
print("\n[2/4] Generant visualitzacions comparatives...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: R² Comparison
ax1 = fig.add_subplot(gs[0, :])

models_order = ['Ridge', 'Random Forest', 'XGBoost', 'LSTM']
colors_models = ['purple', 'green', 'blue', 'orange']

train_r2 = []
test_r2 = []

for model in models_order:
    train_val = df_comparison[(df_comparison['Model'] == model) & 
                             (df_comparison['Dataset'] == 'Train')]['R2'].values[0]
    test_val = df_comparison[(df_comparison['Model'] == model) & 
                            (df_comparison['Dataset'] == 'Test')]['R2'].values[0]
    train_r2.append(train_val)
    test_r2.append(test_val)

x = np.arange(len(models_order))
width = 0.35

bars1 = ax1.bar(x - width/2, train_r2, width, label='Train', 
               color='lightblue', edgecolor='black', linewidth=2)
bars2 = ax1.bar(x + width/2, test_r2, width, label='Test', 
               color='lightcoral', edgecolor='black', linewidth=2)

ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('Comparació R² - Tots els Models', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models_order, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.05)

# Afegir valors
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: MAE Comparison
ax2 = fig.add_subplot(gs[1, 0])

train_mae = []
test_mae = []

for model in models_order:
    train_val = df_comparison[(df_comparison['Model'] == model) & 
                             (df_comparison['Dataset'] == 'Train')]['MAE'].values[0]
    test_val = df_comparison[(df_comparison['Model'] == model) & 
                            (df_comparison['Dataset'] == 'Test')]['MAE'].values[0]
    train_mae.append(train_val)
    test_mae.append(test_val)

x = np.arange(len(models_order))
bars1 = ax2.bar(x - width/2, train_mae, width, label='Train', 
               color='lightblue', edgecolor='black')
bars2 = ax2.bar(x + width/2, test_mae, width, label='Test', 
               color='lightcoral', edgecolor='black')

ax2.set_ylabel('MAE (g/L)', fontsize=11, fontweight='bold')
ax2.set_title('Comparació MAE', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models_order, rotation=45, ha='right', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: RMSE Comparison
ax3 = fig.add_subplot(gs[1, 1])

train_rmse = []
test_rmse = []

for model in models_order:
    train_val = df_comparison[(df_comparison['Model'] == model) & 
                             (df_comparison['Dataset'] == 'Train')]['RMSE'].values[0]
    test_val = df_comparison[(df_comparison['Model'] == model) & 
                            (df_comparison['Dataset'] == 'Test')]['RMSE'].values[0]
    train_rmse.append(train_val)
    test_rmse.append(test_val)

bars1 = ax3.bar(x - width/2, train_rmse, width, label='Train', 
               color='lightblue', edgecolor='black')
bars2 = ax3.bar(x + width/2, test_rmse, width, label='Test', 
               color='lightcoral', edgecolor='black')

ax3.set_ylabel('RMSE (g/L)', fontsize=11, fontweight='bold')
ax3.set_title('Comparació RMSE', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(models_order, rotation=45, ha='right', fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Test Performance Radar
ax4 = fig.add_subplot(gs[1, 2], projection='polar')

# Normalitzar mètriques per radar
test_data = df_comparison[df_comparison['Dataset'] == 'Test']

metrics_radar = ['R2', 'MAE_inv', 'RMSE_inv']  # Invertir MAE i RMSE per visualització
angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

for i, model in enumerate(models_order):
    model_data = test_data[test_data['Model'] == model]
    
    r2 = model_data['R2'].values[0]
    mae = model_data['MAE'].values[0]
    rmse = model_data['RMSE'].values[0]
    
    # Normalitzar (invertir MAE i RMSE: menys és millor → més alt al radar)
    mae_inv = 1 / (mae + 0.1)  # Invertir
    rmse_inv = 1 / (rmse + 0.1)
    
    # Normalitzar a 0-1
    values = [r2, mae_inv / 2, rmse_inv / 2]  # Ajustar escala
    values += values[:1]
    
    ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_models[i])
    ax4.fill(angles, values, alpha=0.15, color=colors_models[i])

ax4.set_xticks(angles[:-1])
ax4.set_xticklabels(['R²', '1/MAE', '1/RMSE'])
ax4.set_ylim(0, 1)
ax4.set_title('Performance Test (Radar)', fontsize=12, fontweight='bold', pad=20)
ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax4.grid(True)

# Plot 5: Ranking table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

# Crear ranking per test
test_only = df_comparison[df_comparison['Dataset'] == 'Test'].copy()
test_only['Rank_R2'] = test_only['R2'].rank(ascending=False)
test_only['Rank_MAE'] = test_only['MAE'].rank(ascending=True)
test_only['Rank_RMSE'] = test_only['RMSE'].rank(ascending=True)
test_only['Avg_Rank'] = test_only[['Rank_R2', 'Rank_MAE', 'Rank_RMSE']].mean(axis=1)
test_only = test_only.sort_values('Avg_Rank')

# Crear taula
table_data = []
for _, row in test_only.iterrows():
    table_data.append([
        row['Model'],
        f"{row['R2']:.4f}",
        f"{row['MAE']:.4f}",
        f"{row['RMSE']:.4f}",
        f"{row['Avg_Rank']:.2f}"
    ])

table = ax5.table(cellText=table_data,
                 colLabels=['Model', 'R² Test', 'MAE Test', 'RMSE Test', 'Rank Mitjà'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Colorar header
for i in range(5):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Colorar millor model
best_row = 1  # Primera fila després del header (millor rank)
for i in range(5):
    table[(best_row, i)].set_facecolor('#FFD700')
    table[(best_row, i)].set_text_props(weight='bold')

ax5.set_title('RANKING FINAL (Test Set)', fontsize=14, fontweight='bold', pad=10)

plt.suptitle('COMPARACIÓ FINAL DE MODELS - Fase 3', fontsize=16, fontweight='bold')

comparison_path = OUTPUT_DIR / "06_model_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {comparison_path.name}")

# =============================================================================
# GUARDAR COMPARACIÓ
# =============================================================================
print("\n[3/4] Guardant taules de comparació...")

summary_path = OUTPUT_DIR / "06_metrics_summary.csv"
df_comparison.to_csv(summary_path, index=False)
print(f"   Summary: {summary_path.name}")

# Ranking
ranking_path = OUTPUT_DIR / "06_model_ranking.csv"
test_only[['Model', 'R2', 'MAE', 'RMSE', 'Avg_Rank']].to_csv(ranking_path, index=False)
print(f"   Ranking: {ranking_path.name}")

# =============================================================================
# RESUM FINAL
# =============================================================================
print("\n[4/4] Generant resum final...")

best_model = test_only.iloc[0]

resum_file = OUTPUT_DIR / "FASE3_RESUM_FINAL.txt"

with open(resum_file, 'w', encoding='utf-8', errors='replace') as f:
    f.write("=" * 80 + "\n")
    f.write("FASE 3: MODELITZACIO PREDICTIVA - RESUM FINAL\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OBJECTIU:\n")
    f.write("   Desenvolupar models predictius per concentracio de penicil.lina\n")
    f.write("   amb 3 nivells de complexitat i deteccio de falles.\n\n")
    
    f.write("DATASET:\n")
    f.write("   Train: Batches 1-90 (operacio normal)\n")
    f.write("   Test:  Batches 91-100 (amb falles)\n")
    f.write("   Features: 9 variables top (Fase 2)\n")
    f.write("   Target: penicillin (g/L)\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("RESULTATS PER MODEL\n")
    f.write("=" * 80 + "\n\n")
    
    for model in models_order:
        model_data = df_comparison[df_comparison['Model'] == model]
        f.write(f"{model}:\n")
        
        train_data = model_data[model_data['Dataset'] == 'Train'].iloc[0]
        test_data = model_data[model_data['Dataset'] == 'Test'].iloc[0]
        
        f.write(f"   Train - R2: {train_data['R2']:.4f}, MAE: {train_data['MAE']:.4f} g/L\n")
        f.write(f"   Test  - R2: {test_data['R2']:.4f}, MAE: {test_data['MAE']:.4f} g/L\n")
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("RANKING FINAL (Test Set)\n")
    f.write("=" * 80 + "\n\n")
    
    for idx, (_, row) in enumerate(test_only.iterrows(), 1):
        f.write(f"{idx}. {row['Model']}\n")
        f.write(f"   R2: {row['R2']:.4f}, MAE: {row['MAE']:.4f}, RMSE: {row['RMSE']:.4f}\n")
        f.write(f"   Ranking mitja: {row['Avg_Rank']:.2f}\n")
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write("MILLOR MODEL\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Model: {best_model['Model']}\n")
    f.write(f"   R2 Test:   {best_model['R2']:.4f}\n")
    f.write(f"   MAE Test:  {best_model['MAE']:.4f} g/L\n")
    f.write(f"   RMSE Test: {best_model['RMSE']:.4f} g/L\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("FAULT DETECTION\n")
    f.write("=" * 80 + "\n\n")
    
    fault_scores = pd.read_csv(OUTPUT_DIR / "05_fault_detection_scores.csv")
    
    f.write(f"Batches analitzats: 91-100 (10 batches amb falles)\n")
    f.write(f"Anomalies detectades (Consensus): {fault_scores['Consensus_Anomaly_%'].mean():.1f}%\n")
    f.write(f"Batch mes problematic: {fault_scores['Consensus_Anomaly_%'].idxmax()}\n")
    f.write(f"   {fault_scores['Consensus_Anomaly_%'].max():.1f}% anomalies\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("FITXERS GENERATS\n")
    f.write("=" * 80 + "\n\n")
    
    outputs = [
        "01_train_test_split_visualization.png",
        "02_ridge_model.pkl",
        "02_ridge_results.png",
        "03_random_forest_model.pkl",
        "03_xgboost_model.pkl",
        "03_ensemble_results.png",
        "03_feature_importance.png",
        "04_lstm_model.h5",
        "04_lstm_results.png",
        "05_fault_detection_results.png",
        "05_fault_detection_scores.csv",
        "06_model_comparison.png",
        "06_metrics_summary.csv"
    ]
    
    for output in outputs:
        if (OUTPUT_DIR / output).exists():
            f.write(f"   * {output}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("FASE 3 COMPLETADA\n")
    f.write("=" * 80 + "\n")

print(f"   Resum: {resum_file.name}")

# =============================================================================
# MISSATGE FINAL
# =============================================================================
print("\n" + "=" * 80)
print("COMPARACIO FINAL COMPLETADA")
print("=" * 80)

print(f"\nMILLOR MODEL: {best_model['Model']}")
print(f"   R² Test:  {best_model['R2']:.4f}")
print(f"   MAE Test: {best_model['MAE']:.4f} g/L")

print(f"\nRANKING (Test Set):")
for idx, (_, row) in enumerate(test_only.iterrows(), 1):
    print(f"   {idx}. {row['Model']:15s} - R²: {row['R2']:.4f}, MAE: {row['MAE']:.4f}")

print(f"\nFitxers generats:")
print(f"   • {comparison_path.name}")
print(f"   • {summary_path.name}")
print(f"   • {ranking_path.name}")
print(f"   • {resum_file.name}")

print(f"\n🎉 FASE 3 COMPLETADA AMB ÈXIT!")
print("=" * 80 + "\n")