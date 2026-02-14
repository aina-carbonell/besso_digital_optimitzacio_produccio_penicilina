#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 2: CORRELACIO I SELECCIO - VERSIO COMPLETAMENT CORREGIDA
==================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import io
import sys
warnings.filterwarnings('ignore')

# Force UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("FASE 2: ANALISI DE CORRELACIO I SELECCIO")
print("=" * 80)

print("\n[1/4] Carregant dataset...")
data_file = OUTPUT_DIR / "03_penicillin_dataset_33_columns.csv"

if not data_file.exists():
    print(f"ERROR: {data_file} no existeix")
    print("Executa primer: 03_data_cleaning_and_feature_engineering.py")
    exit(1)

df = pd.read_csv(data_file)
print(f"OK: {len(df):,} files x {len(df.columns)} columnes")

# Numeric only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'batch_id' in numeric_cols:
    numeric_cols.remove('batch_id')

df_num = df[numeric_cols].copy()
print(f"   Variables numeriques: {len(numeric_cols)}")

if 'penicillin' not in df_num.columns:
    print("ERROR: No hi ha columna 'penicillin'")
    exit(1)

# Neteja de NaN
df_num = df_num.dropna(subset=['penicillin'])
print(f"   Files valides: {len(df_num):,}")

# CORRELACIONS
print("\n[2/4] Calculant correlacions...")

corr_p = df_num.corr()['penicillin'].sort_values(ascending=False)

corr_s = {}
for col in df_num.columns:
    if col != 'penicillin':
        try:
            c, _ = spearmanr(df_num[col].dropna(),
                            df_num.loc[df_num[col].notna(), 'penicillin'],
                            nan_policy='omit')
            corr_s[col] = c if not np.isnan(c) else 0
        except:
            corr_s[col] = 0

corr_s_series = pd.Series(corr_s)

df_corr = pd.DataFrame({
    'Variable': corr_p.index,
    'Pearson': corr_p.values,
    'Spearman': [corr_s.get(v, 0) for v in corr_p.index],
    'Abs_Pearson': np.abs(corr_p.values)
})

df_corr = df_corr[df_corr['Variable'] != 'penicillin']
df_corr = df_corr.sort_values('Abs_Pearson', ascending=False)

print("\nTop 15 correlacions:")
for _, row in df_corr.head(15).iterrows():
    print(f"   {row['Variable']:30s}: P={row['Pearson']:+.3f}, S={row['Spearman']:+.3f}")

# Guardar
corr_file = OUTPUT_DIR / "04_correlations_with_penicillin.csv"
df_corr.to_csv(corr_file, index=False, encoding='utf-8')
print(f"\nOK: {corr_file.name}")

# Grafics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

top15 = df_corr.head(15)
colors = ['green' if x > 0 else 'red' for x in top15['Pearson']]

ax1.barh(range(len(top15)), top15['Pearson'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top15)))
ax1.set_yticklabels(top15['Variable'])
ax1.set_xlabel('Correlacio Pearson')
ax1.set_title('Top 15 Variables - Correlacio amb Penicil.lina', fontweight='bold')
ax1.axvline(x=0, color='black', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='x')

ax2.scatter(top15['Pearson'], top15['Spearman'], s=150, alpha=0.6,
           c=range(len(top15)), cmap='viridis', edgecolors='black', linewidths=2)

for i, var in enumerate(top15['Variable']):
    ax2.annotate(var, (top15.iloc[i]['Pearson'], top15.iloc[i]['Spearman']),
                fontsize=9, ha='right', alpha=0.7)

ax2.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=2)
ax2.set_xlabel('Pearson')
ax2.set_ylabel('Spearman')
ax2.set_title('Pearson vs Spearman', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)

plt.tight_layout()
corr_plot = OUTPUT_DIR / "04_correlation_analysis.png"
plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"OK: {corr_plot.name}")

# SELECCIO TOP 9
print("\n[3/4] Seleccionant Top 9...")

selected = []
candidates = df_corr.head(20)['Variable'].tolist()

print("\n   Seleccio iterativa:")

for i in range(min(9, len(candidates))):
    best = None
    min_vif = float('inf')
    
    for cand in candidates:
        test = selected + [cand]
        X = df_num[test].dropna()
        
        if len(X) > len(test) + 10 and X.shape[1] > 1:
            try:
                vifs = []
                for j in range(X.shape[1]):
                    v = variance_inflation_factor(X.values, j)
                    if not np.isnan(v) and not np.isinf(v):
                        vifs.append(v)
                
                if vifs:
                    max_v = max(vifs)
                    if max_v < min_vif:
                        min_vif = max_v
                        best = cand
            except:
                if best is None:
                    best = cand
                    min_vif = 0
        elif best is None:
            best = cand
            min_vif = 0
    
    if best:
        selected.append(best)
        candidates.remove(best)
        corr_val = df_corr[df_corr['Variable'] == best]['Pearson'].values[0]
        print(f"      {i+1}. {best:30s} (r={corr_val:+.3f}, VIF={min_vif:.2f})")

print(f"\nOK: {len(selected)} variables")

# Dataset reduit
df_red = df[selected + ['penicillin', 'batch_id', 'time']].copy()
red_file = OUTPUT_DIR / "04_penicillin_dataset_top9_features.csv"
df_red.to_csv(red_file, index=False, encoding='utf-8')
print(f"   Guardat: {red_file.name}")

# MATRIU
print("\n[4/4] Matriu de correlacio...")

vars_matrix = [v for v in selected if v in df_num.columns]
if 'penicillin' in df_num.columns:
    vars_matrix.append('penicillin')

if len(vars_matrix) > 1:
    corr_mat = df_num[vars_matrix].corr()
    
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title('Matriu Correlacio - Top 9 + Penicil.lina', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    
    mat_plot = OUTPUT_DIR / "04_correlation_matrix_top9.png"
    plt.savefig(mat_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK: {mat_plot.name}")

# RESUM (amb encoding UTF-8 forçat)
summary_file = OUTPUT_DIR / "04_selected_features_summary.txt"

just = {
    'biomass': 'Relacionada amb produccio (qP * X)',
    'time': 'Fase del proces',
    'substrate': 'Control limitacio substrat',
    'DO': 'Metabolisme aerobic',
    'pH': 'Activitat enzimatica',
    'temperature': 'Cinetica enzimatica',
    'volume': 'Estrategia fed-batch',
    'specific_production_rate': 'Velocitat biosintesi',
    'cumulative_penicillin': 'Produccio acumulada',
    'yield_PX': 'Eficiencia biomassa-producte',
    'OUR': 'Activitat metabolica',
    'CER': 'Activitat metabolica',
    'RQ': 'Estat metabolic',
    'kLa': 'Transferencia massa',
    'penicillin_rate': 'Taxa produccio',
    'biomass_rate': 'Taxa creixement',
    'airflow': 'Subministrament oxigen',
    'agitation': 'Homogeneitzacio',
    'yield_PS': 'Eficiencia substrat-producte'
}

try:
    with open(summary_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUM SELECCIO DE CARACTERISTIQUES\n")
        f.write("=" * 80 + "\n\n")
        f.write("9 VARIABLES SELECCIONADES:\n\n")
        
        for i, feat in enumerate(selected, 1):
            corr = df_corr[df_corr['Variable'] == feat]['Pearson'].values[0]
            f.write(f"   {i}. {feat:30s} (r = {corr:+.3f})\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("\nJUSTIFICACIO TEORICA:\n\n")
        
        for feat in selected:
            if feat in just:
                f.write(f"   * {feat}: {just[feat]}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("\nESTADISTIQUES:\n\n")
        
        f.write("Correlacio positiva (>0.5):\n")
        for feat in selected:
            corr = df_corr[df_corr['Variable'] == feat]['Pearson'].values[0]
            if corr > 0.5:
                f.write(f"   * {feat}: r = {corr:.3f}\n")
        
        f.write("\nCorrelacio negativa (<-0.5):\n")
        found = False
        for feat in selected:
            corr = df_corr[df_corr['Variable'] == feat]['Pearson'].values[0]
            if corr < -0.5:
                f.write(f"   * {feat}: r = {corr:.3f}\n")
                found = True
        if not found:
            f.write("   (Cap)\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"OK: {summary_file.name}")
    
    # Verificar que el fitxer no està buit
    if summary_file.stat().st_size > 0:
        print(f"   Fitxer OK ({summary_file.stat().st_size} bytes)")
    else:
        print("   WARNING: Fitxer buit!")

except Exception as e:
    print(f"ERROR escrivint resum: {e}")
    # Intent de fallback
    try:
        with open(summary_file, 'w', encoding='ascii', errors='replace') as f:
            f.write("SELECTED FEATURES:\n\n")
            for i, feat in enumerate(selected, 1):
                corr = df_corr[df_corr['Variable'] == feat]['Pearson'].values[0]
                f.write(f"{i}. {feat} (r = {corr:+.3f})\n")
        print(f"   Versio ASCII guardada")
    except:
        print(f"   No s'ha pogut guardar")

print("\n" + "=" * 80)
print("OK CORRELACIO I SELECCIO COMPLETADA")
print("=" * 80)
print(f"\nFitxers generats:")
print(f"   * {corr_file.name}")
print(f"   * {corr_plot.name}")
print(f"   * {red_file.name}")
if (OUTPUT_DIR / "04_correlation_matrix_top9.png").exists():
    print(f"   * 04_correlation_matrix_top9.png")
if summary_file.exists() and summary_file.stat().st_size > 0:
    print(f"   * {summary_file.name}")
print(f"\nTots a: {OUTPUT_DIR}")
print("\nFASE 2 COMPLETADA!")
print("=" * 80 + "\n")