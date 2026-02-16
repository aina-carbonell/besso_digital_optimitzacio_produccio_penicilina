#!/usr/bin/env python3
"""
==================================================================================
FASE 2: NETEJA DE DADES I ENGINYERIA DE CARACTERÍSTIQUES
Generació del dataset final amb 33 columnes i visualitzacions
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuració de visualització
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

print("=" * 80)
print("FASE 2: NETEJA DE DADES I ENGINYERIA DE CARACTERÍSTIQUES")
print("=" * 80)

# Carregar dades
print("\n[1/7] Carregant dataset...")
df = pd.read_csv(DATA_DIR / "100_Batches_IndPenSim_V3.csv", low_memory=False)
print(f"✅ Dataset carregat: {len(df):,} files")

# Identificar columnes
batch_col = ' 1-Raman spec recorded'
time_col = 'Time (h)'

print(f"\n[2/7] Netejant i processant dades...")

# Crear dataset base amb variables originals
df_clean = pd.DataFrame()

# Columnes originals necessàries (renomenant per claredat)
column_mapping = {
    'Time (h)': 'time',
    'Substrate concentration(S:g/L)': 'substrate',
    'Dissolved oxygen concentration(DO2:mg/L)': 'DO',
    'Penicillin concentration(P:g/L)': 'penicillin',
    'Vessel Volume(V:L)': 'volume',
    'pH(pH:pH)': 'pH',
    'Temperature(T:K)': 'temperature',
    'Agitator RPM(RPM:RPM)': 'agitation',
    'Aeration rate(Fg:L/h)': 'airflow',
    'Sugar feed rate(Fs:L/h)': 'substrate_feed',
    'Acid flow rate(Fa:L/h)': 'acid_flow',
    'Base flow rate(Fb:L/h)': 'base_flow',
    'Oxygen Uptake Rate(OUR:(g min^{-1}))': 'OUR',
    'Carbon evolution rate(CER:g/h)': 'CER',
    'Offline Biomass concentratio(X_offline:X(g L^{-1}))': 'biomass',
    'Viscosity(Viscosity_offline:centPoise)': 'viscosity',
    batch_col: 'batch_id'
}

# Extreure i renombrar columnes disponibles
for old_col, new_col in column_mapping.items():
    if old_col in df.columns:
        df_clean[new_col] = df[old_col]
    else:
        print(f"⚠️ Columna no trobada: {old_col}")

print(f"✅ Columnes bàsiques extretes: {len(df_clean.columns)}")

print(f"\n[3/7] Generant variables derivades...")

# Convertir temperatura de Kelvin a Celsius si és necessari
if 'temperature' in df_clean.columns:
    if df_clean['temperature'].mean() > 100:
        df_clean['temperature'] = df_clean['temperature'] - 273.15
        print("   ✅ Temperatura convertida de K a °C")

# 16. RQ (Quocient Respiratori)
if 'CER' in df_clean.columns and 'OUR' in df_clean.columns:
    df_clean['RQ'] = df_clean['CER'] / (df_clean['OUR'] * 60 + 1e-10)
    df_clean['RQ'] = df_clean['RQ'].clip(0, 3)
    print("   ✅ RQ calculat")

# 17. kLa (estimat - correlació empírica simplificada)
if 'agitation' in df_clean.columns and 'airflow' in df_clean.columns and 'volume' in df_clean.columns:
    N = df_clean['agitation'] / 60  # rpm to rps
    Q_V = df_clean['airflow'] / df_clean['volume']
    df_clean['kLa'] = 0.05 * (N ** 0.7) * (Q_V ** 0.4)
    print("   ✅ kLa estimat")

# Si no tenim viscositat, estimar-la
if 'viscosity' not in df_clean.columns and 'biomass' in df_clean.columns:
    df_clean['viscosity'] = 1.0 + 0.5 * df_clean['biomass'].fillna(0)
    print("   ✅ Viscositat estimada")

# 19-21. Taxes de canvi (derivades temporals) per batch
print(f"\n[4/7] Calculant taxes de canvi temporal...")

rate_vars = []
if 'biomass' in df_clean.columns:
    rate_vars.append(('biomass', 'biomass_rate'))
if 'penicillin' in df_clean.columns:
    rate_vars.append(('penicillin', 'penicillin_rate'))
if 'substrate' in df_clean.columns:
    rate_vars.append(('substrate', 'substrate_rate'))

for var, rate_name in rate_vars:
    df_clean[rate_name] = 0.0
    
    for batch_id in df_clean['batch_id'].unique():
        mask = df_clean['batch_id'] == batch_id
        batch_data = df_clean.loc[mask, [var, 'time']].copy()
        
        dt = batch_data['time'].diff().fillna(1.0)
        rate = batch_data[var].diff() / dt
        
        # Suavitzar amb rolling window
        rate = rate.rolling(window=5, center=True, min_periods=1).mean()
        rate = rate.fillna(0)
        
        df_clean.loc[mask, rate_name] = rate.values
    
    print(f"   ✅ {rate_name} calculat")

# 22-23. Velocitats específiques
if 'biomass_rate' in df_clean.columns and 'biomass' in df_clean.columns:
    df_clean['specific_growth_rate'] = df_clean['biomass_rate'] / (df_clean['biomass'] + 1e-10)
    df_clean['specific_growth_rate'] = df_clean['specific_growth_rate'].clip(-0.05, 0.2)
    print("   ✅ Velocitat específica de creixement calculada")

if 'penicillin_rate' in df_clean.columns and 'biomass' in df_clean.columns:
    df_clean['specific_production_rate'] = df_clean['penicillin_rate'] / (df_clean['biomass'] + 1e-10)
    df_clean['specific_production_rate'] = df_clean['specific_production_rate'].clip(-0.01, 0.1)
    print("   ✅ Velocitat específica de producció calculada")

# 24-25. Rendiments acumulats per batch
print(f"\n[5/7] Calculant rendiments...")

if 'penicillin' in df_clean.columns and 'biomass' in df_clean.columns:
    df_clean['yield_PX'] = df_clean['penicillin'] / (df_clean['biomass'] + 1e-10)
    df_clean['yield_PX'] = df_clean['yield_PX'].clip(0, 5)
    print("   ✅ Rendiment P/X calculat")

if 'penicillin' in df_clean.columns and 'substrate' in df_clean.columns:
    df_clean['yield_PS'] = 0.0
    
    for batch_id in df_clean['batch_id'].unique():
        mask = df_clean['batch_id'] == batch_id
        batch_data = df_clean.loc[mask].copy()
        
        if len(batch_data) > 0 and 'substrate' in batch_data.columns:
            S0 = batch_data['substrate'].iloc[0]
            substrate_consumed = S0 - batch_data['substrate']
            substrate_consumed = substrate_consumed.clip(lower=0.1)
            
            yield_ps = batch_data['penicillin'] / substrate_consumed
            yield_ps = yield_ps.clip(0, 1)
            
            df_clean.loc[mask, 'yield_PS'] = yield_ps.values
    
    print("   ✅ Rendiment P/S calculat")

# 26-27. Variables acumulatives
if 'substrate' in df_clean.columns and 'volume' in df_clean.columns:
    df_clean['cumulative_substrate'] = 0.0
    
    for batch_id in df_clean['batch_id'].unique():
        mask = df_clean['batch_id'] == batch_id
        batch_data = df_clean.loc[mask].copy()
        
        if len(batch_data) > 0:
            S0 = batch_data['substrate'].iloc[0]
            substrate_consumed = (S0 - batch_data['substrate']) * batch_data['volume']
            substrate_consumed = substrate_consumed.clip(lower=0)
            
            df_clean.loc[mask, 'cumulative_substrate'] = substrate_consumed.values
    
    print("   ✅ Substrat acumulat calculat")

if 'penicillin' in df_clean.columns and 'volume' in df_clean.columns:
    df_clean['cumulative_penicillin'] = df_clean['penicillin'] * df_clean['volume']
    print("   ✅ Penicil·lina acumulada calculada")

# 28-29. Energia
if 'agitation' in df_clean.columns:
    Np = 5.0
    rho = 1000
    D = 2.0
    N_rps = df_clean['agitation'] / 60
    df_clean['power_input'] = Np * rho * (N_rps ** 3) * (D ** 5) / 1000
    print("   ✅ Potència d'agitació calculada")

if 'biomass_rate' in df_clean.columns and 'volume' in df_clean.columns:
    df_clean['heat_generation'] = 12.0 * df_clean['biomass_rate'].clip(lower=0) * df_clean['volume'] / 1000
    print("   ✅ Generació de calor calculada")

# 30-32. Components principals de Raman (placeholder - necessita processament separat)
df_clean['raman_PC1'] = 0.0
df_clean['raman_PC2'] = 0.0
df_clean['raman_PC3'] = 0.0
print("   ℹ️  Components Raman (placeholder - requereix PCA separat)")

print(f"\n[6/7] Finalitzant dataset...")

# Ordenar columnes en l'ordre desitjat
final_columns_order = [
    'time', 'biomass', 'penicillin', 'substrate', 'DO', 'temperature', 'pH',
    'agitation', 'airflow', 'volume', 'substrate_feed', 'acid_flow', 'base_flow',
    'OUR', 'CER', 'RQ', 'kLa', 'viscosity',
    'biomass_rate', 'penicillin_rate', 'substrate_rate',
    'specific_growth_rate', 'specific_production_rate',
    'yield_PX', 'yield_PS',
    'cumulative_substrate', 'cumulative_penicillin',
    'power_input', 'heat_generation',
    'raman_PC1', 'raman_PC2', 'raman_PC3',
    'batch_id'
]

# Seleccionar només columnes que existeixen
final_columns = [col for col in final_columns_order if col in df_clean.columns]
df_final = df_clean[final_columns].copy()

# Omplir valors perduts finals
df_final = df_final.fillna(method='ffill').fillna(method='bfill').fillna(0)

# Guardar dataset final
output_file = OUTPUT_DIR / "03_penicillin_dataset_33_columns.csv"
df_final.to_csv(output_file, index=False)

print(f"\n✅ Dataset final generat!")
print(f"   📊 Dimensions: {len(df_final):,} files × {len(df_final.columns)} columnes")
print(f"   💾 Guardat a: {output_file}")

# Generar resum estadístic
print(f"\n📊 Resum del dataset final:")
print(f"   Total de batches: {df_final['batch_id'].nunique()}")
print(f"   Variables incloses: {len(df_final.columns)}")
print(f"\n   Columnes generades:")
for i, col in enumerate(df_final.columns, 1):
    print(f"      {i:2d}. {col}")

# Guardar resum estadístic
stats_file = OUTPUT_DIR / "03_dataset_statistics.csv"
df_stats = df_final.describe().T
df_stats.to_csv(stats_file)
print(f"\n   📈 Estadístiques guardades: {stats_file.name}")

# =============================================================================
# GENERAR BOXPLOT DE TOTES LES VARIABLES NUMÈRIQUES
# =============================================================================
print(f"\n[7/7] Generant boxplot de variables numèriques...")

# Seleccionar variables numèriques (excloent batch_id i time)
numeric_vars = [col for col in df_final.columns if col not in ['batch_id', 'time']]

# Crear figura amb subplots (múltiples boxplots en una graella)
n_vars = len(numeric_vars)
n_cols = 4  # 4 columnes
n_rows = (n_vars + n_cols - 1) // n_cols  # Calcular files necessàries

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axes = axes.flatten()  # Aplanar per iterar fàcilment

# Normalitzar cada variable per visualitzar-les juntes (z-score)
df_normalized = df_final[numeric_vars].copy()
for col in df_normalized.columns:
    if df_normalized[col].std() > 0:  # Evitar divisió per zero
        df_normalized[col] = (df_normalized[col] - df_normalized[col].mean()) / df_normalized[col].std()

# Crear boxplot per cada variable
for i, var in enumerate(numeric_vars):
    ax = axes[i]
    
    # Dades originals (sense normalitzar) per mostrar valors reals
    data = df_final[var].dropna()
    
    # Boxplot amb estil millorat
    bp = ax.boxplot(data, patch_artist=True, showmeans=True, meanline=True,
                    meanprops=dict(color='red', linewidth=2, linestyle='--'),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color='gray', linewidth=1.5),
                    capprops=dict(color='gray', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='orange', 
                                    markersize=3, alpha=0.5, markeredgecolor='orange'))
    
    # Color del box
    bp['boxes'][0].set_facecolor(plt.cm.viridis(i / len(numeric_vars)))
    bp['boxes'][0].set_alpha(0.7)
    
    # Afegir estadístiques com a text
    stats = data.describe()
    ax.text(0.95, 0.95, f"μ={stats['mean']:.2f}\nσ={stats['std']:.2f}",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(var, fontweight='bold', fontsize=10)
    ax.set_ylabel('Valor', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])  # Treure etiquetes de l'eix X

# Ocultar subplots sobrants
for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('BOXPLOTS DE TOTES LES VARIABLES NUMÈRIQUES', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Guardar boxplot
boxplot_path = OUTPUT_DIR / "03_all_variables_boxplot.png"
plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Boxplot guardat: {boxplot_path.name}")

# Opcional: Boxplot compacte amb totes les variables normalitzades
print(f"\n   Generant boxplot compacte (variables normalitzades)...")

fig, ax = plt.subplots(figsize=(16, 10))

# Ordenar variables per mediana per millor visualització
order = df_normalized.median().sort_values().index

# Boxplot de totes juntes (normalitzades)
df_normalized[order].boxplot(ax=ax, rot=90, grid=False, patch_artist=True,
                             boxprops=dict(alpha=0.7, color='blue'),
                             medianprops=dict(color='red', linewidth=2),
                             flierprops=dict(marker='o', markersize=2, alpha=0.3))

ax.set_title('Distribució de Variables (Normalitzades)', fontsize=14, fontweight='bold')
ax.set_ylabel('Z-Score', fontsize=12)
ax.set_xlabel('Variable', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)

plt.tight_layout()
compact_path = OUTPUT_DIR / "03_variables_boxplot_compact.png"
plt.savefig(compact_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Boxplot compacte guardat: {compact_path.name}")

print("\n" + "=" * 80)
print("✅ NETEJA I ENGINYERIA DE CARACTERÍSTIQUES COMPLETADA")
print("=" * 80)
print(f"\n📁 Fitxers generats:")
print(f"   • 03_penicillin_dataset_33_columns.csv")
print(f"   • 03_dataset_statistics.csv")
print(f"   • 03_all_variables_boxplot.png")
print(f"   • 03_variables_boxplot_compact.png")
print(f"\n🚀 Següent pas: Executar '04_correlation_analysis.py'")
print("=" * 80 + "\n")