#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: PREPARACIÓ DE DADES PER MODELITZACIÓ
Split Train/Test estratificat: 80% batches normals + 80% batches amb falles per train
                              20% batches normals + 20% batches amb falles per test
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
print("\n[1/6] Carregant dataset amb Top 9 features...")

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
# IDENTIFICAR TIPUS DE BATCHES
# =============================================================================
print("\n[2/6] Identificant batches normals vs amb falles...")

# Assumim que els batches 91-100 són els que tenen falles (segons Fase 2)
normal_batches = [b for b in unique_batches if b < 91]
fault_batches = [b for b in unique_batches if b >= 91]

print(f"\n   Batches normals (1-90): {len(normal_batches)} batches")
print(f"   Batches amb falles (91-100): {len(fault_batches)} batches")

# =============================================================================
# SPLIT ESTRATIFICAT PER TIPUS DE BATCH
# =============================================================================
print("\n[3/6] Creant split estratificat 80/20...")

# Split per batches normals
train_norm, test_norm = train_test_split(
    normal_batches, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

# Split per batches amb falles
train_fault, test_fault = train_test_split(
    fault_batches, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

# Combinar
train_batches = sorted(train_norm + train_fault)
test_batches = sorted(test_norm + test_fault)

print(f"\n   TRAIN SET:")
print(f"      Batches normals: {len(train_norm)} (80% de {len(normal_batches)})")
print(f"      Batches amb falles: {len(train_fault)} (80% de {len(fault_batches)})")
print(f"      TOTAL: {len(train_batches)} batches")
print(f"      IDs: {train_batches[:10]}...")

print(f"\n   TEST SET:")
print(f"      Batches normals: {len(test_norm)} (20% de {len(normal_batches)})")
print(f"      Batches amb falles: {len(test_fault)} (20% de {len(fault_batches)})")
print(f"      TOTAL: {len(test_batches)} batches")
print(f"      IDs: {test_batches}")

# Crear datasets
df_train = df[df[batch_col].isin(train_batches)].copy()
df_test = df[df[batch_col].isin(test_batches)].copy()

print(f"\n   MOSTRES:")
print(f"      Train: {len(df_train):,} files")
print(f"      Test:  {len(df_test):,} files")

# Verificar proporcions
train_normal_pct = len(df_train[df_train[batch_col].isin(normal_batches)]) / len(df_train) * 100
train_fault_pct = len(df_train[df_train[batch_col].isin(fault_batches)]) / len(df_train) * 100
test_normal_pct = len(df_test[df_test[batch_col].isin(normal_batches)]) / len(df_test) * 100
test_fault_pct = len(df_test[df_test[batch_col].isin(fault_batches)]) / len(df_test) * 100

print(f"\n   PROPORCIONS:")
print(f"      Train - Normals: {train_normal_pct:.1f}%, Amb falles: {train_fault_pct:.1f}%")
print(f"      Test  - Normals: {test_normal_pct:.1f}%, Amb falles: {test_fault_pct:.1f}%")

# =============================================================================
# PREPARACIÓ FEATURES
# =============================================================================
print("\n[4/6] Preparant features i target...")

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
else:
    print(f"      TRAIN: Sense valors perduts ✓")

if missing_test.sum() > 0:
    print(f"      TEST: {missing_test.sum()} valors perduts")
else:
    print(f"      TEST: Sense valors perduts ✓")

# Imputar si cal
if missing_train.sum() > 0 or missing_test.sum() > 0:
    print("\n   Imputant valors perduts...")
    df_train = df_train.fillna(method='ffill').fillna(method='bfill').fillna(0)
    df_test = df_test.fillna(method='ffill').fillna(method='bfill').fillna(0)
    print("      OK: Valors imputats")

# =============================================================================
# ESTADÍSTIQUES DESCRIPTIVES
# =============================================================================
print("\n[5/6] Generant estadístiques...")

# Train set
stats_train = df_train[feature_cols + [target_col]].describe()
print(f"\n   Estadístiques TRAIN:")
print(stats_train.T[['mean', 'std', 'min', 'max']].round(3))

# Test set
stats_test = df_test[feature_cols + [target_col]].describe()
print(f"\n   Estadístiques TEST:")
print(stats_test.T[['mean', 'std', 'min', 'max']].round(3))

# =============================================================================
# VISUALITZACIÓ
# =============================================================================
print("\n[6/6] Generant visualitzacions...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Distribució target
ax = axes[0, 0]
ax.hist(df_train[target_col], bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
ax.hist(df_test[target_col], bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
ax.set_xlabel('Penicil·lina (g/L)', fontsize=11)
ax.set_ylabel('Freqüència', fontsize=11)
ax.set_title('Distribució de Penicil·lina', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Boxplot per dataset
ax = axes[0, 1]
data_box = [df_train[target_col], df_test[target_col]]
bp = ax.boxplot(data_box, labels=['Train', 'Test'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Penicil·lina (g/L)', fontsize=11)
ax.set_title('Comparació Train vs Test', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Proporció tipus de batches
ax = axes[0, 2]
categories = ['Train\nNormals', 'Train\nFalles', 'Test\nNormals', 'Test\nFalles']
sizes = [
    len(df_train[df_train[batch_col].isin(normal_batches)]),
    len(df_train[df_train[batch_col].isin(fault_batches)]),
    len(df_test[df_test[batch_col].isin(normal_batches)]),
    len(df_test[df_test[batch_col].isin(fault_batches)])
]
colors_bar = ['lightblue', 'salmon', 'lightblue', 'salmon']

bars = ax.bar(categories, sizes, color=colors_bar, edgecolor='black', linewidth=2)
ax.set_ylabel('Nombre de mostres', fontsize=11)
ax.set_title('Distribució per Tipus de Batch', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Afegir valors sobre les barres
for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{size:,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Evolució temporal mitjana per batch
ax = axes[1, 0]

# Agrupar per batch i calcular mitjana
train_means = df_train.groupby(batch_col)[target_col].mean()
test_means = df_test.groupby(batch_col)[target_col].mean()

# Pintar normals vs falles
for batch, mean in train_means.items():
    color = 'blue' if batch in normal_batches else 'red'
    ax.scatter(batch, mean, color=color, alpha=0.7, s=50, edgecolor='black', linewidth=1)

for batch, mean in test_means.items():
    color = 'cyan' if batch in normal_batches else 'orange'
    ax.scatter(batch, mean, color=color, alpha=0.9, s=80, edgecolor='black', linewidth=1, marker='s')

ax.axhline(y=df_train[target_col].mean(), color='blue', linestyle='--', alpha=0.5, label='Train mean')
ax.axhline(y=df_test[target_col].mean(), color='red', linestyle='--', alpha=0.5, label='Test mean')
ax.set_xlabel('Batch ID', fontsize=11)
ax.set_ylabel('Penicil·lina mitjana (g/L)', fontsize=11)
ax.set_title('Producció Mitjana per Batch', fontsize=13, fontweight='bold')
ax.legend(['Train (normals)', 'Train (falles)', 'Test (normals)', 'Test (falles)', 'Train mean', 'Test mean'])
ax.grid(True, alpha=0.3)

# Plot 5: Mides dels datasets
ax = axes[1, 1]
categories = ['Train', 'Test']
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

# Plot 6: Informació del split
ax = axes[1, 2]
ax.axis('off')

split_info = f"""
SPLIT ESTRATIFICAT 80/20

BATCHES NORMALS (1-90):
• Train: {len(train_norm)} batches (80%)
• Test:  {len(test_norm)} batches (20%)

BATCHES AMB FALLES (91-100):
• Train: {len(train_fault)} batches (80%)
• Test:  {len(test_fault)} batches (20%)

TOTAL:
• Train: {len(train_batches)} batches
• Test:  {len(test_batches)} batches

PROPORCIÓ MOSTRES:
Train - Normals: {train_normal_pct:.1f}%
Train - Falles:  {train_fault_pct:.1f}%
Test  - Normals: {test_normal_pct:.1f}%
Test  - Falles:  {test_fault_pct:.1f}%
"""

ax.text(0.05, 0.95, split_info, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
viz_path = OUTPUT_DIR / "01_train_test_split_visualization.png"
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   OK: {viz_path.name}")

# =============================================================================
# GUARDAR DATASETS
# =============================================================================
print("\n[7/6] Guardant datasets...")

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
    f.write("INFORMACIO DEL SPLIT ESTRATIFICAT TRAIN/TEST\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ESTRATÈGIA:\n")
    f.write("   Split 80/20 independent per cada tipus de batch:\n")
    f.write("   - Batches normals (1-90): 80% train, 20% test\n")
    f.write("   - Batches amb falles (91-100): 80% train, 20% test\n\n")
    
    f.write("TRAIN SET:\n")
    f.write(f"   Batches normals: {train_norm}\n")
    f.write(f"   Batches amb falles: {train_fault}\n")
    f.write(f"   Total mostres: {len(df_train):,}\n")
    f.write(f"   Target mean: {df_train[target_col].mean():.3f} g/L\n\n")
    
    f.write("TEST SET:\n")
    f.write(f"   Batches normals: {test_norm}\n")
    f.write(f"   Batches amb falles: {test_fault}\n")
    f.write(f"   Total mostres: {len(df_test):,}\n")
    f.write(f"   Target mean: {df_test[target_col].mean():.3f} g/L\n\n")
    
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
print(f"   • {test_file.name} - Dataset test")
print(f"   • {viz_path.name} - Visualitzacions")
print(f"   • {metadata_file.name} - Informacio split")
print(f"\nLocalitzacio: {OUTPUT_DIR}")
print(f"\nSegüent pas: python 02_baseline_ridge.py")
print("=" * 80 + "\n")