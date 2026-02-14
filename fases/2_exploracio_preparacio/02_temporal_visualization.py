#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 2: VISUALITZACIÓ TEMPORAL DE BATCHES - VERSIÓ COMPLETAMENT CORREGIDA
Generació de perfils temporals de variables clau
==================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuració matplotlib per suportar UTF-8
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 2: VISUALITZACIO TEMPORAL DE BATCHES")
print("=" * 80)

# Carregar dades
print("\n[1/4] Carregant dataset...")
try:
    df = pd.read_csv(DATA_DIR / "100_Batches_IndPenSim_V3.csv", low_memory=False)
    print(f"OK Dataset carregat: {len(df):,} files, {len(df.columns):,} columnes")
except Exception as e:
    print(f"ERROR carregant dataset: {e}")
    exit(1)

# Identificar columnes amb cerca robusta
batch_col = None
for possible in ['Batch reference(Batch_ref:Batch ref)', 'Batch_ref:Batch ref', 'Batch ID']:
    if possible in df.columns:
        batch_col = possible
        break

if not batch_col:
    print("ERROR: No s'ha trobat columna de Batch!")
    exit(1)

print(f"   Columna batch: {batch_col}")

time_col = 'Time (h)'

# Funció per trobar columnes
def find_col(patterns):
    for pattern in patterns:
        for col in df.columns:
            if pattern.lower() in col.lower():
                return col
    return None

# Mapeig robust
mapping = {}
mapping[time_col] = 'time'
mapping[batch_col] = 'batch'

vars_to_find = {
    'substrate': ['Substrate concentration', 'S:g/L'],
    'DO': ['Dissolved oxygen', 'DO2:mg/L'],
    'penicillin': ['Penicillin concentration', 'P:g/L'],
    'volume': ['Vessel Volume', 'V:L'],
    'pH': ['pH(pH:pH)'],
    'temperature': ['Temperature', 'T:K'],
    'agitation': ['Agitator RPM'],
    'aeration': ['Aeration rate', 'Fg:L/h'],
    'OUR': ['Oxygen Uptake Rate'],
    'CER': ['Carbon evolution rate'],
    'biomass': ['Offline Biomass']
}

for var_name, patterns in vars_to_find.items():
    col = find_col(patterns)
    if col:
        mapping[col] = var_name

print(f"   Variables trobades: {len(mapping) - 2}")

# Crear dataframe
cols_to_use = list(mapping.keys())
df_plot = df[cols_to_use].copy()
df_plot = df_plot.rename(columns=mapping)

# Funció per crear perfils
def plot_batch_profile(batch_id, df_batch, save_path):
    if len(df_batch) == 0:
        return None
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # Plot 1: Biomassa
    if 'biomass' in df_batch.columns:
        ax = fig.add_subplot(gs[0, 0])
        data = df_batch['biomass'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'b-', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('Biomassa (g/L)')
            ax.set_title('Evolucio de Biomassa', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 2: Penicil·lina
    if 'penicillin' in df_batch.columns:
        ax = fig.add_subplot(gs[0, 1])
        data = df_batch['penicillin'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'r-', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('Penicil.lina (g/L)')
            ax.set_title('Produccio de Penicil.lina', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 3: Substrat
    if 'substrate' in df_batch.columns:
        ax = fig.add_subplot(gs[0, 2])
        data = df_batch['substrate'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'g-', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('Substrat (g/L)')
            ax.set_title('Concentracio de Substrat', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 4: DO
    if 'DO' in df_batch.columns:
        ax = fig.add_subplot(gs[1, 0])
        data = df_batch['DO'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'm-', linewidth=2)
            ax.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Limit critic')
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('DO (mg/L)')
            ax.set_title('Oxigen Dissolt', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Plot 5: pH i Temperatura
    if 'pH' in df_batch.columns and 'temperature' in df_batch.columns:
        ax = fig.add_subplot(gs[1, 1])
        ax2 = ax.twinx()
        
        pH_data = df_batch['pH'].dropna()
        temp_data = df_batch['temperature'].dropna()
        
        if len(pH_data) > 0:
            ax.plot(df_batch.loc[pH_data.index, 'time'], pH_data, 'b-', linewidth=2, label='pH')
            ax.set_ylabel('pH', color='b')
            ax.tick_params(axis='y', labelcolor='b')
        
        if len(temp_data) > 0:
            temp_vals = temp_data.copy()
            if temp_vals.mean() > 100:
                temp_vals = temp_vals - 273.15
            ax2.plot(df_batch.loc[temp_data.index, 'time'], temp_vals, 'r-', linewidth=2, label='Temp')
            ax2.set_ylabel('Temperatura (C)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('Temps (h)')
        ax.set_title('pH i Temperatura', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Agitació i Aeració
    if 'agitation' in df_batch.columns and 'aeration' in df_batch.columns:
        ax = fig.add_subplot(gs[1, 2])
        ax2 = ax.twinx()
        
        agit_data = df_batch['agitation'].dropna()
        aer_data = df_batch['aeration'].dropna()
        
        if len(agit_data) > 0:
            ax.plot(df_batch.loc[agit_data.index, 'time'], agit_data, 'c-', linewidth=2)
            ax.set_ylabel('Agitacio (rpm)', color='c')
            ax.tick_params(axis='y', labelcolor='c')
        
        if len(aer_data) > 0:
            ax2.plot(df_batch.loc[aer_data.index, 'time'], aer_data, 'orange', linewidth=2)
            ax2.set_ylabel('Aeracio (L/h)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        ax.set_xlabel('Temps (h)')
        ax.set_title('Agitacio i Aeracio', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Plot 7: Volume
    if 'volume' in df_batch.columns:
        ax = fig.add_subplot(gs[2, 0])
        data = df_batch['volume'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'purple', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('Volumen (L)')
            ax.set_title('Volumen del Reactor', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 8: OUR
    if 'OUR' in df_batch.columns:
        ax = fig.add_subplot(gs[2, 1])
        data = df_batch['OUR'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'brown', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('OUR (g/min)')
            ax.set_title('Oxygen Uptake Rate', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # Plot 9: CER
    if 'CER' in df_batch.columns:
        ax = fig.add_subplot(gs[2, 2])
        data = df_batch['CER'].dropna()
        if len(data) > 0:
            ax.plot(df_batch.loc[data.index, 'time'], data, 'olive', linewidth=2)
            ax.set_xlabel('Temps (h)')
            ax.set_ylabel('CER (g/h)')
            ax.set_title('Carbon Evolution Rate', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Perfil Temporal - Batch {batch_id}', fontsize=18, fontweight='bold')
    
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except Exception as e:
        print(f"      Error guardant: {e}")
        plt.close()
        return None

# Generar perfils
print("\n[2/4] Generant perfils temporals...")

batches = {15: "Recipe-Driven", 45: "Operator", 75: "APC", 95: "Falla"}

for bid, name in batches.items():
    print(f"   Batch {bid} ({name})...")
    df_b = df_plot[df_plot['batch'] == bid].copy()
    if len(df_b) > 0:
        path = OUTPUT_DIR / f"02_batch_{bid:03d}_profile.png"
        result = plot_batch_profile(bid, df_b, path)
        if result:
            print(f"      OK: {path.name}")

# Comparació d'estratègies
print("\n[3/4] Generant comparacio entre estrategies...")

if 'penicillin' in df_plot.columns:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    ranges = {
        'Recipe-Driven (1-30)': range(1, 31),
        'Operator Control (31-60)': range(31, 61),
        'APC amb Raman (61-90)': range(61, 91),
        'Batches amb Falles (91-100)': range(91, 101)
    }
    
    colors = ['blue', 'green', 'cyan', 'red']
    
    for idx, (name, rng) in enumerate(ranges.items()):
        ax = axes[idx // 2, idx % 2]
        count = 0
        all_data = []
        
        for bid in rng:
            df_b = df_plot[df_plot['batch'] == bid]
            if len(df_b) > 0:
                pen_data = df_b['penicillin'].dropna()
                if len(pen_data) > 0:
                    ax.plot(df_b.loc[pen_data.index, 'time'], pen_data,
                           alpha=0.3, linewidth=1, color=colors[idx])
                    count += 1
                    
                    for i, row in df_b.iterrows():
                        if not pd.isna(row['penicillin']):
                            all_data.append({'time': row['time'], 'pen': row['penicillin']})
        
        # Mitjana
        if all_data:
            dfa = pd.DataFrame(all_data)
            times = np.linspace(dfa['time'].min(), dfa['time'].max(), 100)
            means = []
            mean_t = []
            
            for i in range(len(times)-1):
                mask = (dfa['time'] >= times[i]) & (dfa['time'] < times[i+1])
                if mask.sum() > 0:
                    means.append(dfa.loc[mask, 'pen'].mean())
                    mean_t.append((times[i] + times[i+1])/2)
            
            if mean_t:
                ax.plot(mean_t, means, linewidth=3, color=colors[idx],
                       label=f'Mitjana ({count} batches)', linestyle='--')
        
        ax.set_xlabel('Temps (h)')
        ax.set_ylabel('Penicil.lina (g/L)')
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if count > 0:
            ax.legend()
        
        print(f"      {name}: {count} batches")
    
    plt.suptitle('Comparacio de Produccio entre Estrategies', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    comp_path = OUTPUT_DIR / "02_strategies_comparison.png"
    try:
        plt.savefig(comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   OK: {comp_path.name}")
    except Exception as e:
        print(f"   ERROR: {e}")
        plt.close()
else:
    print("   No es pot crear comparacio (no hi ha penicillin)")

print("\n[4/4] Completat!")
print("=" * 80)
print("OK VISUALITZACIO TEMPORAL COMPLETADA")
print("=" * 80)
print(f"\nFitxers a: {OUTPUT_DIR}")
print("\nSegüent: 03_data_cleaning_and_feature_engineering.py")
print("=" * 80 + "\n")