#!/usr/bin/env python3
"""
==================================================================================
FASE 2: NETEJA DE DADES I ENGINYERIA DE CARACTERÍSTIQUES
Generació del dataset final amb 33 columnes
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 2: NETEJA DE DADES I ENGINYERIA DE CARACTERÍSTIQUES")
print("=" * 80)

# Carregar dades
print("\n[1/6] Carregant dataset...")
df = pd.read_csv(DATA_DIR / "100_Batches_IndPenSim_V3.csv", low_memory=False)
print(f"✅ Dataset carregat: {len(df):,} files")

# Identificar columnes
batch_col = ' 1-Raman spec recorded'
time_col = 'Time (h)'

print(f"\n[2/6] Netejant i processant dades...")

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

print(f"\n[3/6] Generant variables derivades...")

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
print(f"\n[4/6] Calculant taxes de canvi temporal...")

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
print(f"\n[5/6] Calculant rendiments...")

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

print(f"\n[6/6] Finalitzant dataset...")

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

print("\n" + "=" * 80)
print("✅ NETEJA I ENGINYERIA DE CARACTERÍSTIQUES COMPLETADA")
print("=" * 80)
print(f"\n🚀 Següent pas: Executar '04_correlation_analysis.py'")
print("=" * 80 + "\n")
