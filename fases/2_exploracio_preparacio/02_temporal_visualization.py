#!/usr/bin/env python3
"""
==================================================================================
FASE 2: VISUALITZACIÓ TEMPORAL DE BATCHES
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

# Configuració
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 2: VISUALITZACIÓ TEMPORAL DE BATCHES")
print("=" * 80)

# Carregar dades
print("\n[1/4] Carregant dataset...")
df = pd.read_csv(DATA_DIR / "100_Batches_IndPenSim_V3.csv", low_memory=False)

# Identificar columnes clau
batch_col = 'Batch_ref:Batch ref' if 'Batch_ref:Batch ref' in df.columns else 'Batch ID'
time_col = 'Time (h)'

# Mapeig de columnes
col_mapping = {
    'Time (h)': 'time',
    'Substrate concentration(S:g/L)': 'substrate',
    'Dissolved oxygen concentration(DO2:mg/L)': 'DO',
    'Penicillin concentration(P:g/L)': 'penicillin',
    'Vessel Volume(V:L)': 'volume',
    'pH(pH:pH)': 'pH',
    'Temperature(T:K)': 'temperature',
    'Agitator RPM(RPM:RPM)': 'agitation',
    'Aeration rate(Fg:L/h)': 'aeration',
    'Oxygen Uptake Rate(OUR:(g min^{-1}))': 'OUR',
    'Carbon evolution rate(CER:g/h)': 'CER',
    'Offline Biomass concentratio(X_offline:X(g L^{-1}))': 'biomass'
}

# Seleccionar columnes disponibles
available_cols = {k: v for k, v in col_mapping.items() if k in df.columns}
df_plot = df[[batch_col, time_col] + list(available_cols.keys())].copy()
df_plot = df_plot.rename(columns={batch_col: 'batch', time_col: 'time'})
df_plot = df_plot.rename(columns=available_cols)

print(f"✅ Dataset carregat: {len(df)} files, {len(df.columns)} columnes")
print(f"   Variables per visualitzar: {len(available_cols)}")

# =============================================================================
# FUNCIÓ PER PLOTEAR PERFILS DE BATCH
# =============================================================================

def plot_batch_profile(batch_id, df_batch, save_path):
    """Crea un perfil temporal complet d'un batch"""
    
    fig = plt.figure(figsize=(20, 14))
    
    # Crear grid de subplots
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Biomassa
    if 'biomass' in df_batch.columns:
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df_batch['time'], df_batch['biomass'], 'b-', linewidth=2)
        ax1.set_xlabel('Temps (h)', fontsize=11)
        ax1.set_ylabel('Biomassa (g/L)', fontsize=11)
        ax1.set_title('Evolució de Biomassa', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Penicil·lina
    if 'penicillin' in df_batch.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df_batch['time'], df_batch['penicillin'], 'r-', linewidth=2)
        ax2.set_xlabel('Temps (h)', fontsize=11)
        ax2.set_ylabel('Penicil·lina (g/L)', fontsize=11)
        ax2.set_title('Producció de Penicil·lina', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Substrat
    if 'substrate' in df_batch.columns:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(df_batch['time'], df_batch['substrate'], 'g-', linewidth=2)
        ax3.set_xlabel('Temps (h)', fontsize=11)
        ax3.set_ylabel('Substrat (g/L)', fontsize=11)
        ax3.set_title('Concentració de Substrat', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Oxigen Dissolt
    if 'DO' in df_batch.columns:
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(df_batch['time'], df_batch['DO'], 'm-', linewidth=2)
        ax4.axhline(y=30, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Límit crític')
        ax4.set_xlabel('Temps (h)', fontsize=11)
        ax4.set_ylabel('DO (mg/L)', fontsize=11)
        ax4.set_title('Oxigen Dissolt', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    # Plot 5: pH i Temperatura
    if 'pH' in df_batch.columns and 'temperature' in df_batch.columns:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5_twin = ax5.twinx()
        
        line1 = ax5.plot(df_batch['time'], df_batch['pH'], 'b-', linewidth=2, label='pH')
        ax5.axhline(y=6.0, color='b', linestyle='--', alpha=0.3)
        ax5.axhline(y=6.5, color='b', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Temps (h)', fontsize=11)
        ax5.set_ylabel('pH', fontsize=11, color='b')
        ax5.tick_params(axis='y', labelcolor='b')
        
        # Convertir K a °C si és necessari
        temp_data = df_batch['temperature']
        if temp_data.mean() > 100:  # Està en Kelvin
            temp_data = temp_data - 273.15
        
        line2 = ax5_twin.plot(df_batch['time'], temp_data, 'r-', linewidth=2, label='Temperatura')
        ax5_twin.set_ylabel('Temperatura (°C)', fontsize=11, color='r')
        ax5_twin.tick_params(axis='y', labelcolor='r')
        
        ax5.set_title('pH i Temperatura', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
    
    # Plot 6: Agitació i Aeració
    if 'agitation' in df_batch.columns and 'aeration' in df_batch.columns:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6_twin = ax6.twinx()
        
        line1 = ax6.plot(df_batch['time'], df_batch['agitation'], 'c-', linewidth=2, label='Agitació')
        ax6.set_xlabel('Temps (h)', fontsize=11)
        ax6.set_ylabel('Agitació (rpm)', fontsize=11, color='c')
        ax6.tick_params(axis='y', labelcolor='c')
        
        line2 = ax6_twin.plot(df_batch['time'], df_batch['aeration'], 'orange', linewidth=2, label='Aeració')
        ax6_twin.set_ylabel('Aeració (L/h)', fontsize=11, color='orange')
        ax6_twin.tick_params(axis='y', labelcolor='orange')
        
        ax6.set_title('Agitació i Aeració', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='upper left')
    
    # Plot 7: Volumen
    if 'volume' in df_batch.columns:
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(df_batch['time'], df_batch['volume'], 'purple', linewidth=2)
        ax7.set_xlabel('Temps (h)', fontsize=11)
        ax7.set_ylabel('Volumen (L)', fontsize=11)
        ax7.set_title('Volumen del Reactor', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
    
    # Plot 8: OUR
    if 'OUR' in df_batch.columns:
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(df_batch['time'], df_batch['OUR'], 'brown', linewidth=2)
        ax8.set_xlabel('Temps (h)', fontsize=11)
        ax8.set_ylabel('OUR (g/min)', fontsize=11)
        ax8.set_title('Oxygen Uptake Rate', fontsize=13, fontweight='bold')
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: CER
    if 'CER' in df_batch.columns:
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(df_batch['time'], df_batch['CER'], 'olive', linewidth=2)
        ax9.set_xlabel('Temps (h)', fontsize=11)
        ax9.set_ylabel('CER (g/h)', fontsize=11)
        ax9.set_title('Carbon Evolution Rate', fontsize=13, fontweight='bold')
        ax9.grid(True, alpha=0.3)
    
    # Plot 10: Rendiment acumulat P/S
    if 'penicillin' in df_batch.columns and 'substrate' in df_batch.columns:
        ax10 = fig.add_subplot(gs[3, 0])
        substrate_consumed = df_batch['substrate'].iloc[0] - df_batch['substrate']
        yield_ps = df_batch['penicillin'] / (substrate_consumed + 0.1)
        ax10.plot(df_batch['time'], yield_ps, 'darkgreen', linewidth=2)
        ax10.set_xlabel('Temps (h)', fontsize=11)
        ax10.set_ylabel('Rendiment P/S (g/g)', fontsize=11)
        ax10.set_title('Rendiment Penicil·lina/Substrat', fontsize=13, fontweight='bold')
        ax10.grid(True, alpha=0.3)
    
    # Plot 11: Taxa de producció
    if 'penicillin' in df_batch.columns:
        ax11 = fig.add_subplot(gs[3, 1])
        production_rate = df_batch['penicillin'].diff() / df_batch['time'].diff()
        ax11.plot(df_batch['time'], production_rate, 'crimson', linewidth=2)
        ax11.set_xlabel('Temps (h)', fontsize=11)
        ax11.set_ylabel('dP/dt (g/L/h)', fontsize=11)
        ax11.set_title('Taxa de Producció de Penicil·lina', fontsize=13, fontweight='bold')
        ax11.grid(True, alpha=0.3)
    
    # Plot 12: RQ (Quocient Respiratori)
    if 'CER' in df_batch.columns and 'OUR' in df_batch.columns:
        ax12 = fig.add_subplot(gs[3, 2])
        RQ = df_batch['CER'] / (df_batch['OUR'] * 60 + 0.001)  # OUR en g/min, CER en g/h
        RQ = RQ.clip(0, 3)  # Limitar valors raonables
        ax12.plot(df_batch['time'], RQ, 'navy', linewidth=2)
        ax12.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='RQ = 1.0')
        ax12.set_xlabel('Temps (h)', fontsize=11)
        ax12.set_ylabel('RQ (CER/OUR)', fontsize=11)
        ax12.set_title('Quocient Respiratori', fontsize=13, fontweight='bold')
        ax12.grid(True, alpha=0.3)
        ax12.legend()
    
    plt.suptitle(f'Perfil Temporal Complet - Batch {batch_id}', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

# =============================================================================
# GENERAR PERFILS PER BATCHES REPRESENTATIUS
# =============================================================================
print("\n[2/4] Generant perfils temporals...")

# Seleccionar batches representatius de cada estratègia
representative_batches = {
    15: "Recipe-Driven",
    45: "Operator Control",
    75: "APC amb Raman",
    95: "Batch amb Falla"
}

for batch_id, strategy in representative_batches.items():
    print(f"   Generant perfil per Batch {batch_id} ({strategy})...")
    
    df_batch = df_plot[df_plot['batch'] == batch_id].copy()
    
    if len(df_batch) > 0:
        save_path = OUTPUT_DIR / f"02_batch_{batch_id:03d}_profile.png"
        plot_batch_profile(batch_id, df_batch, save_path)
        print(f"      ✅ Guardat: {save_path.name}")
    else:
        print(f"      ⚠️  No s'han trobat dades per al Batch {batch_id}")

# =============================================================================
# COMPARACIÓ D'ESTRATÈGIES
# =============================================================================
print("\n[3/4] Generant comparació entre estratègies...")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

strategies_ranges = {
    'Recipe-Driven (1-30)': range(1, 31),
    'Operator Control (31-60)': range(31, 61),
    'APC amb Raman (61-90)': range(61, 91),
    'Batches amb Falles (91-100)': range(91, 101)
}

colors = ['blue', 'green', 'cyan', 'red']

for idx, (strategy_name, batch_range) in enumerate(strategies_ranges.items()):
    ax = axes[idx // 2, idx % 2]
    
    for batch_id in batch_range:
        df_batch = df_plot[df_plot['batch'] == batch_id]
        if len(df_batch) > 0 and 'penicillin' in df_batch.columns:
            ax.plot(df_batch['time'], df_batch['penicillin'], 
                   alpha=0.3, linewidth=1, color=colors[idx])
    
    # Calcular i plotar la mitjana
    all_times = []
    all_penicillin = []
    
    for batch_id in batch_range:
        df_batch = df_plot[df_plot['batch'] == batch_id]
        if len(df_batch) > 0 and 'penicillin' in df_batch.columns:
            all_times.extend(df_batch['time'].values)
            all_penicillin.extend(df_batch['penicillin'].values)
    
    if all_times:
        # Crear bins de temps per calcular mitjana
        time_bins = np.linspace(min(all_times), max(all_times), 100)
        mean_penicillin = []
        mean_times = []
        
        for i in range(len(time_bins) - 1):
            mask = (np.array(all_times) >= time_bins[i]) & (np.array(all_times) < time_bins[i+1])
            if mask.any():
                mean_penicillin.append(np.mean(np.array(all_penicillin)[mask]))
                mean_times.append((time_bins[i] + time_bins[i+1]) / 2)
        
        ax.plot(mean_times, mean_penicillin, linewidth=3, color=colors[idx], 
               label='Mitjana', linestyle='--')
    
    ax.set_xlabel('Temps (h)', fontsize=12)
    ax.set_ylabel('Penicil·lina (g/L)', fontsize=12)
    ax.set_title(strategy_name, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('Comparació de Producció de Penicil·lina entre Estratègies de Control', 
             fontsize=16, fontweight='bold')
plt.tight_layout()

comparison_path = OUTPUT_DIR / "02_strategies_comparison.png"
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Comparació guardada: {comparison_path.name}")

# =============================================================================
# MISSATGE FINAL
# =============================================================================
print("\n[4/4] Visualització temporal completada!")
print("\n" + "=" * 80)
print("✅ VISUALITZACIÓ TEMPORAL COMPLETADA")
print("=" * 80)
print(f"\n📁 Fitxers generats:")
print(f"   • Perfils de batches representatius")
print(f"   • Comparació entre estratègies")
print(f"\n📂 Localització: {OUTPUT_DIR}")
print(f"\n🚀 Següent pas: Executar '03_data_cleaning.py'")
print("=" * 80 + "\n")
