#!/usr/bin/env python3
"""
==================================================================================
FASE 2: EXPLORACIÓ I PREPARACIÓ DE DADES
Anàlisi Exploratòria del Dataset IndPenSim Complet
==================================================================================
Dataset: 100_Batches_IndPenSim_V3.csv
- 113,935 files (observacions temporals)
- 2,239 columnes (variables de procés + espectroscòpia Raman)
- ~2 GB de dades
==================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuració de visualització
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

# Configuració de paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 2: ANÀLISI EXPLORATÒRIA DE DADES")
print("Dataset IndPenSim - Producció de Penicil·lina")
print("=" * 80)

# =============================================================================
# 1. CÀRREGA DE DADES
# =============================================================================
print("\n[1/8] Carregant dataset complet...")

# Carregar el dataset complet
data_file = DATA_DIR / "100_Batches_IndPenSim_V3.csv"

if not data_file.exists():
    print(f"❌ ERROR: No s'ha trobat el fitxer {data_file}")
    print("Si us plau, assegura't que el fitxer està a la carpeta 'data/'")
    exit(1)

# Carregar amb chunks per gestionar la memòria
print(f"📂 Carregant {data_file.name}...")
print("   (Això pot trigar uns segons degut a la mida del fitxer...)")

df = pd.read_csv(data_file, low_memory=False)

print(f"\n✅ Dataset carregat exitosament!")
print(f"   📊 Dimensions: {df.shape[0]:,} files × {df.shape[1]:,} columnes")
print(f"   💾 Memòria utilitzada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# =============================================================================
# 2. INSPECCIÓ INICIAL
# =============================================================================
print("\n[2/8] Inspecció inicial de l'estructura...")

# Identificar les variables clau del procés (primeres 39 columnes)
process_cols = df.columns[:39].tolist()
raman_cols = df.columns[39:].tolist()

print(f"\n📋 Variables de procés: {len(process_cols)}")
print(f"📊 Columnes d'espectroscòpia Raman: {len(raman_cols)}")

# Mostrar les primeres variables de procés
print("\n🔍 Variables clau del procés:")
for i, col in enumerate(process_cols[:20], 1):
    print(f"   {i:2d}. {col}")
if len(process_cols) > 20:
    print(f"   ... i {len(process_cols) - 20} més")

# Tipus de dades
print(f"\n📊 Tipus de dades:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"   {dtype}: {count} columnes")

# Valors perduts
print(f"\n🔍 Anàlisi de valors perduts:")
missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df) * 100)
missing_df = pd.DataFrame({
    'Columna': missing_counts.index,
    'Valors perduts': missing_counts.values,
    'Percentatge': missing_pct.values
})
missing_df = missing_df[missing_df['Valors perduts'] > 0].sort_values('Valors perduts', ascending=False)

if len(missing_df) > 0:
    print(f"   ⚠️  {len(missing_df)} columnes amb valors perduts")
    print(f"\n   Top 10 columnes amb més valors perduts:")
    for idx, row in missing_df.head(10).iterrows():
        print(f"      {row['Columna'][:50]:50s}: {row['Valors perduts']:8,} ({row['Percentatge']:5.2f}%)")
else:
    print("   ✅ No hi ha valors perduts!")

# =============================================================================
# 3. IDENTIFICACIÓ DE BATCHES
# =============================================================================
print("\n[3/8] Identificant batches individuals...")

# Identificar la columna de Batch ID
batch_col = None
for col in ['Batch_ref:Batch ref', 'Batch ID', 'Batch ref', 'Batch reference']:
    if col in df.columns:
        batch_col = col
        break

if batch_col:
    print(f"✅ Columna de batch identificada: '{batch_col}'")
    
    # Analitzar batches
    unique_batches = df[batch_col].unique()
    print(f"   📦 Nombre total de batches: {len(unique_batches)}")
    print(f"   📦 IDs de batches: {sorted(unique_batches)[:20]}...")
    
    # Estadístiques per batch
    batch_stats = df.groupby(batch_col).agg({
        'Time (h)': ['count', 'min', 'max']
    })
    batch_stats.columns = ['N_samples', 'Time_min', 'Time_max']
    batch_stats['Duration'] = batch_stats['Time_max'] - batch_stats['Time_min']
    
    print(f"\n   📊 Estadístiques per batch:")
    print(f"      Mostres per batch - Mitjana: {batch_stats['N_samples'].mean():.0f}")
    print(f"                         Min: {batch_stats['N_samples'].min():.0f}")
    print(f"                         Max: {batch_stats['N_samples'].max():.0f}")
    print(f"      Durada (h) - Mitjana: {batch_stats['Duration'].mean():.1f} h")
    print(f"                   Min: {batch_stats['Duration'].min():.1f} h")
    print(f"                   Max: {batch_stats['Duration'].max():.1f} h")
else:
    print("⚠️  No s'ha trobat la columna de Batch ID")
    print("   Assumint que les dades estan ordenades cronològicament")

# =============================================================================
# 4. IDENTIFICACIÓ D'ESTRATÈGIES DE CONTROL
# =============================================================================
print("\n[4/8] Analitzant estratègies de control...")

# Buscar la columna de control
control_col = None
for col in ['Control_ref:Control ref', '0 - Recipe driven 1 - Operator controlled(Control_ref:Control ref)']:
    if col in df.columns:
        control_col = col
        break

# Buscar la columna de PAT (Advanced Control)
pat_col = None
for col in ['PAT_ref:PAT ref', '2-PAT control(PAT_ref:PAT ref)']:
    if col in df.columns:
        pat_col = col
        break

# Buscar la columna de fault
fault_col = None
for col in ['Fault_ref:Fault ref', 'Fault reference(Fault_ref:Fault ref)', 'Fault flag']:
    if col in df.columns:
        fault_col = col
        break

if batch_col and control_col:
    control_strategy = df.groupby(batch_col)[control_col].first()
    print(f"✅ Estratègies de control identificades:")
    print(f"   📊 Distribució per estratègia:")
    strategy_counts = control_strategy.value_counts()
    for strategy, count in strategy_counts.items():
        print(f"      Estratègia {strategy}: {count} batches")

if batch_col and fault_col:
    fault_status = df.groupby(batch_col)[fault_col].first()
    print(f"\n✅ Estat de falles identificat:")
    fault_counts = fault_status.value_counts()
    for status, count in fault_counts.items():
        if status == 0:
            print(f"   ✅ Batches normals: {count}")
        else:
            print(f"   ⚠️  Batches amb falla: {count}")

# =============================================================================
# 5. ESTADÍSTIQUES DESCRIPTIVES DE VARIABLES CLAU
# =============================================================================
print("\n[5/8] Generant estadístiques descriptives...")

# Variables clau per analitzar
key_variables = {
    'Time (h)': 'Temps',
    'Substrate concentration(S:g/L)': 'Substrat',
    'Dissolved oxygen concentration(DO2:mg/L)': 'Oxigen Dissolt',
    'Penicillin concentration(P:g/L)': 'Penicil·lina',
    'Vessel Volume(V:L)': 'Volumen',
    'pH(pH:pH)': 'pH',
    'Temperature(T:K)': 'Temperatura',
    'Agitator RPM(RPM:RPM)': 'Agitació',
    'Aeration rate(Fg:L/h)': 'Aeració',
    'Oxygen Uptake Rate(OUR:(g min^{-1}))': 'OUR',
    'Carbon evolution rate(CER:g/h)': 'CER'
}

# Seleccionar les que existeixen
available_key_vars = {k: v for k, v in key_variables.items() if k in df.columns}

print(f"📊 Estadístiques de {len(available_key_vars)} variables clau:")
stats_summary = pd.DataFrame()

for col, name in available_key_vars.items():
    stats = df[col].describe()
    print(f"\n   {name} ({col}):")
    print(f"      Mitjana: {stats['mean']:.4f}")
    print(f"      Desv.Std: {stats['std']:.4f}")
    print(f"      Min: {stats['min']:.4f}")
    print(f"      Max: {stats['max']:.4f}")
    print(f"      Mediana: {stats['50%']:.4f}")

# =============================================================================
# 6. GUARDAR RESUM DE L'ANÀLISI
# =============================================================================
print("\n[6/8] Guardant resum de l'anàlisi...")

# Crear resum
summary_file = OUTPUT_DIR / "01_dataset_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESUM DE L'ANÀLISI EXPLORATÒRIA\n")
    f.write("Dataset IndPenSim - Producció de Penicil·lina\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"📊 Dimensions del dataset: {df.shape[0]:,} files × {df.shape[1]:,} columnes\n")
    f.write(f"💾 Memòria utilitzada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
    
    f.write(f"📋 Variables de procés: {len(process_cols)}\n")
    f.write(f"📊 Columnes d'espectroscòpia Raman: {len(raman_cols)}\n\n")
    
    if batch_col:
        f.write(f"📦 Nombre total de batches: {len(unique_batches)}\n")
        f.write(f"📊 Mostres per batch (mitjana): {batch_stats['N_samples'].mean():.0f}\n")
        f.write(f"⏱️  Durada mitjana per batch: {batch_stats['Duration'].mean():.1f} hores\n\n")
    
    f.write("🔍 Variables clau identificades:\n")
    for col, name in available_key_vars.items():
        f.write(f"   • {name}: {col}\n")
    
    f.write("\n" + "=" * 80 + "\n")

print(f"✅ Resum guardat a: {summary_file}")

# =============================================================================
# 7. EXPORTAR METADATA
# =============================================================================
print("\n[7/8] Exportant metadades...")

# Crear DataFrame amb metadata de totes les columnes
metadata = pd.DataFrame({
    'Column_Name': df.columns,
    'Data_Type': df.dtypes.values,
    'Non_Null_Count': df.count().values,
    'Null_Count': df.isnull().sum().values,
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).values,
    'Unique_Values': [df[col].nunique() for col in df.columns],
    'Is_Process_Variable': ['Yes' if col in process_cols else 'No' for col in df.columns],
    'Is_Raman': ['Yes' if col in raman_cols else 'No' for col in df.columns]
})

# Afegir estadístiques per columnes numèriques
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in metadata['Column_Name']:
    if col in numeric_cols:
        metadata.loc[metadata['Column_Name'] == col, 'Mean'] = df[col].mean()
        metadata.loc[metadata['Column_Name'] == col, 'Std'] = df[col].std()
        metadata.loc[metadata['Column_Name'] == col, 'Min'] = df[col].min()
        metadata.loc[metadata['Column_Name'] == col, 'Max'] = df[col].max()

metadata_file = OUTPUT_DIR / "01_columns_metadata.csv"
metadata.to_csv(metadata_file, index=False)
print(f"✅ Metadades guardades a: {metadata_file}")

# =============================================================================
# 8. MISSATGE FINAL
# =============================================================================
print("\n[8/8] Anàlisi exploratòria inicial completada!")
print("\n" + "=" * 80)
print("✅ FASE 2 - ETAPA 1 COMPLETADA")
print("=" * 80)
print(f"\n📁 Fitxers generats:")
print(f"   • {summary_file.name}")
print(f"   • {metadata_file.name}")
print(f"\n📂 Localització: {OUTPUT_DIR}")
print(f"\n🚀 Següent pas: Executar '02_temporal_visualization.py'")
print("=" * 80 + "\n")
