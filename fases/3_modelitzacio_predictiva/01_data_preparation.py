#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: PREPARACIÓ DE DADES PER MODELITZACIÓ
Split Train/Test: Batches 1-90 (train) vs 91-100 (test amb falles)
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
DATA_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 3: PREPARACIÓ DE DADES PER MODELITZACIÓ")
print("=" * 80)

# Carregar dataset amb top 9 features
print("\n[1/5] Carregant dataset amb Top 9 features...")

data_file = DATA_DIR / "04_penicillin_dataset_top9_features.csv"

if not data_file.exists():
    print(f"ERROR: No s'ha trobat {data_file}")
    print("Executa primer els scripts de la Fase 2")
    exit(1)

df = pd.read_csv(data_file)
print(f"OK: {len(df):,} files, {len(df.columns)} columnes")

# Identificar columna de batch
batch_col = 'batch_id'
if batch_col not in df.columns:
    print(f"ERROR: Columna '{batch_col}' no trobada")
    exit(1)

print(f"\n   Columnes disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"      {i:2d}. {col}")

# Verificar batches
unique_batches = sorted(df[batch_col].unique())
print(f"\n   Total batches: {len(unique_batches)}")
print(f"   Rang: {min(unique_batches)} - {max(unique_batches)}")

# =============================================================================
# SPLIT TRAIN/TEST
# =============================================================================
print("\n[2/5] Creant split Train/Test...")

# Batches 1-90 → Train (operació normal)
# Batches 91-100 → Test (amb falles) → FAULT DETECTION

train_batches = list(range(1, 91))
test_batches = list(range(91, 101))

df_train = df[df[batch_col].isin(train_batches)].copy()
df_test = df[df[batch_col].isin(test_batches)].copy()

print(f"\n   TRAIN SET:")
print(f"      Batches: 1-90 ({len(train_batches)} batches)")
print(f"      Mostres: {len(df_train):,}")
print(f"      Batches únics: {df_train[batch_col].nunique()}")

print(f"\n   TEST SET (amb falles):")
print(f"      Batches: 91-100 ({len(test_batches)} batches)")
print(f"      Mostres: {len(df_test):,}")
print(f"      Batches únics: {df_test[batch_col].nunique()}")

# =============================================================================
# PREPARACIÓ FEATURES
# =============================================================================
print("\n[3/5] Preparant features i target...")

# Identificar features i target
feature_cols = [col for col in df.columns 
                if col not in [batch_col, 'penicillin', 'time']]

target_col = 'penicillin'

print(f"\n   Features ({len(feature_cols)}):")
for i, feat in enumerate(feature_cols, 1):
    print(f"      {i}. {feat}")

print(f"\n   Target: {target_col}")

# Verificar valors perduts
print(f"\n   Verificant valors perduts...")
missing_train = df_train[feature_cols + [target_col]].isnull().sum()
missing_test = df_test[feature_cols + [target_col]].isnull().sum()

if missing_train.sum() > 0:
    print(f"      TRAIN: {missing_train.sum()} valors perduts")
    for col, count in missing_train[missing_train > 0].items():
        print(f"         {col}: {count}")
else:
    print(f"      TRAIN: Sense valors perduts ✓")

if missing_test.sum() > 0:
    print(f"      TEST: {missing_test.sum()} valors perduts")
    for col, count in missing_test[missing_test > 0].items():
        print(f"         {col}: {count}")
else:
    print(f"      TEST: Sense valors perduts ✓")

# Imputar si cal (forward fill + backward fill)
if missing_train.sum() > 0 or missing_test.sum() > 0:
    print("\n   Imputant valors perduts...")
    df_train = df_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
    df_test = df_test.fillna(method='ffill').fillna(method='bfill').fillna(0)
    print("      OK: Valors imputats")

# =============================================================================
# ESTADÍSTIQUES DESCRIPTIVES
# =============================================================================
print("\n[4/5] Generant estadístiques...")

# Train set
stats_train = df_train[feature_cols + [target_col]].describe()
print(f"\n   Estadístiques TRAIN:")
print(stats_train.T[['mean', 'std', 'min', 'max']].round(3))

# Test set
stats_test = df_test[feature_cols + [target_col]].describe()
print(f"\n   Estadístiques TEST:")
print(stats_test.T[['mean', 'std', 'min', 'max']].round(3))

# Comparació distribucions
print(f"\n   Comparació distribucions (target):")
print(f"      TRAIN - Mean: {df_train[target_col].mean():.3f}, Std: {df_train[target_col].std():.3f}")
print(f"      TEST  - Mean: {df_test[target_col].mean():.3f}, Std: {df_test[target_col].std():.3f}")

# =============================================================================
# VISUALITZACIÓ
# =============================================================================
print("\n[5/5] Generant visualitzacions...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Distribució target
ax = axes[0, 0]
ax.hist(df_train[target_col], bins=50, alpha=0.7, label='Train (1-90)', color='blue', edgecolor='black')
ax.hist(df_test[target_col], bins=50, alpha=0.7, label='Test (91-100)', color='red', edgecolor='black')
ax.set_xlabel('Penicil·lina (g/L)', fontsize=11)
ax.set_ylabel('Freqüència', fontsize=11)
ax.set_title('Distribució de Penicil·lina - Train vs Test', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Boxplot per dataset
ax = axes[0, 1]
data_box = [df_train[target_col], df_test[target_col]]
bp = ax.boxplot(data_box, labels=['Train\n(1-90)', 'Test\n(91-100)'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Penicil·lina (g/L)', fontsize=11)
ax.set_title('Comparació Train vs Test', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Evolució temporal mitjana
ax = axes[1, 0]

# Agrupar per batch i calcular mitjana
train_temporal = df_train.groupby(batch_col)[target_col].mean()
test_temporal = df_test.groupby(batch_col)[target_col].mean()

ax.plot(train_temporal.index, train_temporal.values, 'o-', 
        color='blue', alpha=0.6, label='Train (1-90)')
ax.plot(test_temporal.index, test_temporal.values, 'o-', 
        color='red', alpha=0.8, label='Test (91-100)', linewidth=2)
ax.axvline(x=90.5, color='green', linestyle='--', linewidth=2, label='Split Train/Test')
ax.set_xlabel('Batch ID', fontsize=11)
ax.set_ylabel('Penicil·lina mitjana (g/L)', fontsize=11)
ax.set_title('Producció Mitjana per Batch', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Mides dels datasets
ax = axes[1, 1]
categories = ['Train\n(1-90)', 'Test\n(91-100)']
sizes = [len(df_train), len(df_test)]
colors_bar = ['lightblue', 'lightcoral']

bars = ax.bar(categories, sizes, color=colors_bar, edgecolor='black', linewidth=2)
ax.set_ylabel('Nombre de mostres', fontsize=11)
ax.set_title('Mida dels Datasets', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Afegir valors sobre les barres
for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{size:,}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
viz_path = OUTPUT_DIR / "01_train_test_split_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   OK: {viz_path.name}")

# =============================================================================
# GUARDAR DATASETS
# =============================================================================
print("\n[6/6] Guardant datasets...")

# Guardar train i test
train_file = OUTPUT_DIR / "train_data.csv"
test_file = OUTPUT_DIR / "test_data.csv"

df_train.to_csv(train_file, index=False)
df_test.to_csv(test_file, index=False)

print(f"   Train: {train_file.name} ({train_file.stat().st_size / 1024:.1f} KB)")
print(f"   Test:  {test_file.name} ({test_file.stat().st_size / 1024:.1f} KB)")

# Guardar metadata
metadata_file = OUTPUT_DIR / "01_train_test_split_info.txt"

with open(metadata_file, 'w', encoding='utf-8', errors='replace') as f:
    f.write("=" * 80 + "\n")
    f.write("INFORMACIO DEL SPLIT TRAIN/TEST\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("TRAIN SET:\n")
    f.write(f"   Batches: 1-90\n")
    f.write(f"   Total mostres: {len(df_train):,}\n")
    f.write(f"   Batches unics: {df_train[batch_col].nunique()}\n")
    f.write(f"   Target mean: {df_train[target_col].mean():.3f} g/L\n")
    f.write(f"   Target std: {df_train[target_col].std():.3f} g/L\n\n")
    
    f.write("TEST SET (AMB FALLES):\n")
    f.write(f"   Batches: 91-100\n")
    f.write(f"   Total mostres: {len(df_test):,}\n")
    f.write(f"   Batches unics: {df_test[batch_col].nunique()}\n")
    f.write(f"   Target mean: {df_test[target_col].mean():.3f} g/L\n")
    f.write(f"   Target std: {df_test[target_col].std():.3f} g/L\n\n")
    
    f.write("FEATURES ({}):\n".format(len(feature_cols)))
    for i, feat in enumerate(feature_cols, 1):
        f.write(f"   {i}. {feat}\n")
    
    f.write(f"\nTARGET: {target_col}\n")
    f.write("\n" + "=" * 80 + "\n")

print(f"   Metadata: {metadata_file.name}")

# =============================================================================
# RESUM FINAL
# =============================================================================
print("\n" + "=" * 80)
print("PREPARACIO COMPLETADA")
print("=" * 80)
print(f"\nFitxers generats:")
print(f"   • {train_file.name} - Dataset entrenament")
print(f"   • {test_file.name} - Dataset test (falles)")
print(f"   • {viz_path.name} - Visualitzacions")
print(f"   • {metadata_file.name} - Informacio split")
print(f"\nLocalitzacio: {OUTPUT_DIR}")
print(f"\nSegüent pas: python 02_baseline_ridge.py")
print("=" * 80 + "\n")