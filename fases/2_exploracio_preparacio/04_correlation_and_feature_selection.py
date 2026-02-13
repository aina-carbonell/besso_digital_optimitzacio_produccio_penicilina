#!/usr/bin/env python3
"""
==================================================================================
FASE 2: ANÀLISI DE CORRELACIÓ I SELECCIÓ DE CARACTERÍSTIQUES
Identificació de les 9 variables més predictives per a penicil·lina
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
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "2_exploracio_preparacio" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("FASE 2: ANÀLISI DE CORRELACIÓ I SELECCIÓ DE CARACTERÍSTIQUES")
print("=" * 80)

# Carregar dataset processat
print("\n[1/4] Carregant dataset processat...")
data_file = OUTPUT_DIR / "03_penicillin_dataset_33_columns.csv"

if not data_file.exists():
    print(f"❌ ERROR: Primer has d'executar 03_data_cleaning_and_feature_engineering.py")
    exit(1)

df = pd.read_csv(data_file)
print(f"✅ Dataset carregat: {len(df):,} files × {len(df.columns)} columnes")

# Eliminar columnes no numèriques
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'batch_id' in numeric_cols:
    numeric_cols.remove('batch_id')

df_numeric = df[numeric_cols].copy()
print(f"   Variables numèriques: {len(numeric_cols)}")

# =============================================================================
# ANÀLISI DE CORRELACIÓ
# =============================================================================
print("\n[2/4] Calculant correlacions amb penicil·lina...")

if 'penicillin' not in df_numeric.columns:
    print("❌ ERROR: La columna 'penicillin' no existeix al dataset")
    exit(1)

# Correlació de Pearson
corr_pearson = df_numeric.corr()['penicillin'].sort_values(ascending=False)

# Correlació de Spearman
corr_spearman = {}
for col in df_numeric.columns:
    if col != 'penicillin':
        corr, _ = spearmanr(df_numeric[col], df_numeric['penicillin'], nan_policy='omit')
        corr_spearman[col] = corr

corr_spearman = pd.Series(corr_spearman).sort_values(ascending=False)

# Crear DataFrame comparatiu
df_correlations = pd.DataFrame({
    'Variable': corr_pearson.index,
    'Pearson': corr_pearson.values,
    'Spearman': [corr_spearman.get(var, 0) for var in corr_pearson.index],
    'Abs_Pearson': np.abs(corr_pearson.values)
})

df_correlations = df_correlations[df_correlations['Variable'] != 'penicillin']
df_correlations = df_correlations.sort_values('Abs_Pearson', ascending=False)

print(f"\n📊 Top 15 variables més correlacionades:")
for idx, row in df_correlations.head(15).iterrows():
    print(f"   {row['Variable']:30s}: Pearson={row['Pearson']:+.3f}, Spearman={row['Spearman']:+.3f}")

# Guardar correlacions
corr_file = OUTPUT_DIR / "04_correlations_with_penicillin.csv"
df_correlations.to_csv(corr_file, index=False)
print(f"\n✅ Correlacions guardades: {corr_file.name}")

# Visualització de correlacions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Gràfic 1: Top 15 Pearson
top_15 = df_correlations.head(15)
colors = ['green' if x > 0 else 'red' for x in top_15['Pearson']]

ax1.barh(range(len(top_15)), top_15['Pearson'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['Variable'])
ax1.set_xlabel('Correlació de Pearson', fontsize=13)
ax1.set_title('Top 15 Variables Correlacionades amb Penicil·lina (Pearson)', 
              fontsize=14, fontweight='bold')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax1.grid(True, alpha=0.3, axis='x')

# Gràfic 2: Pearson vs Spearman
ax2.scatter(top_15['Pearson'], top_15['Spearman'], s=150, alpha=0.6, 
           c=range(len(top_15)), cmap='viridis', edgecolors='black', linewidths=2)

for i, var in enumerate(top_15['Variable']):
    ax2.annotate(var, (top_15.iloc[i]['Pearson'], top_15.iloc[i]['Spearman']),
                fontsize=9, ha='right', alpha=0.7)

ax2.plot([-1, 1], [-1, 1], 'k--', alpha=0.3, linewidth=2)
ax2.set_xlabel('Correlació de Pearson', fontsize=13)
ax2.set_ylabel('Correlació de Spearman', fontsize=13)
ax2.set_title('Comparació Pearson vs Spearman', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)

plt.tight_layout()
corr_plot = OUTPUT_DIR / "04_correlation_analysis.png"
plt.savefig(corr_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Gràfic de correlacions guardat: {corr_plot.name}")

# =============================================================================
# SELECCIÓ DE TOP 9 CARACTERÍSTIQUES
# =============================================================================
print("\n[3/4] Seleccionant les 9 variables més predictives...")

# Algorisme greedy per minimitzar multicolinealitat (VIF)
selected_features = []
remaining_candidates = df_correlations.head(20)['Variable'].tolist()

print("\n   Selecció iterativa (minimitzant VIF):")

for i in range(min(9, len(remaining_candidates))):
    best_feature = None
    min_max_vif = float('inf')
    
    for candidate in remaining_candidates:
        test_features = selected_features + [candidate]
        
        # Calcular VIF per aquesta combinació
        X_test = df_numeric[test_features].dropna()
        
        if len(X_test) > len(test_features) and X_test.shape[1] > 1:
            try:
                vif_values = []
                for j in range(X_test.shape[1]):
                    vif = variance_inflation_factor(X_test.values, j)
                    vif_values.append(vif)
                
                max_vif = max(vif_values)
                
                if max_vif < min_max_vif:
                    min_max_vif = max_vif
                    best_feature = candidate
            except:
                # Si falla el càlcul de VIF, seleccionar per correlació
                if best_feature is None:
                    best_feature = candidate
    
    if best_feature:
        selected_features.append(best_feature)
        remaining_candidates.remove(best_feature)
        corr_val = df_correlations[df_correlations['Variable'] == best_feature]['Pearson'].values[0]
        print(f"      {i+1}. {best_feature:30s} (r={corr_val:+.3f}, VIF_max={min_max_vif:.2f})")

print(f"\n✅ {len(selected_features)} variables seleccionades!")

# Crear dataset reduït
df_reduced = df[selected_features + ['penicillin', 'batch_id', 'time']].copy()
reduced_file = OUTPUT_DIR / "04_penicillin_dataset_top9_features.csv"
df_reduced.to_csv(reduced_file, index=False)
print(f"   💾 Dataset reduït guardat: {reduced_file.name}")

# =============================================================================
# MATRIU DE CORRELACIÓ
# =============================================================================
print("\n[4/4] Generant matriu de correlació...")

# Matriu per les 9 variables seleccionades
corr_matrix = df_numeric[selected_features + ['penicillin']].corr()

plt.figure(figsize=(12, 10))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            vmin=-1, vmax=1)

plt.title('Matriu de Correlació - Top 9 Variables + Penicil·lina', 
          fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

matrix_plot = OUTPUT_DIR / "04_correlation_matrix_top9.png"
plt.savefig(matrix_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Matriu de correlació guardada: {matrix_plot.name}")

# Guardar resum final
summary_file = OUTPUT_DIR / "04_selected_features_summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RESUM DE SELECCIÓ DE CARACTERÍSTIQUES\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("🎯 9 VARIABLES SELECCIONADES PER PREDICCIÓ DE PENICIL·LINA:\n\n")
    
    for i, feat in enumerate(selected_features, 1):
        corr = df_correlations[df_correlations['Variable'] == feat]['Pearson'].values[0]
        f.write(f"   {i}. {feat:30s} (r = {corr:+.3f})\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("\n📊 JUSTIFICACIÓ TEÒRICA:\n\n")
    
    justifications = {
        'biomass': 'Directament relacionada amb producció (q_P · X)',
        'time': 'Fase del procés (producció en fase estacionària)',
        'substrate': 'Control de limitació per substrat',
        'DO': 'Metabolisme aeròbic essencial',
        'pH': 'Afecta activitat enzimàtica de biosíntesi',
        'temperature': 'Afecta cinètica enzimàtica',
        'volume': 'Estratègia fed-batch i dilució',
        'specific_production_rate': 'Velocitat específica de biosíntesi',
        'cumulative_penicillin': 'Producció acumulada total',
        'yield_PX': 'Eficiència de conversió biomassa-producte',
        'OUR': 'Indicador d\'activitat metabòlica',
        'CER': 'Indicador d\'activitat metabòlica',
        'RQ': 'Estat metabòlic del microorganisme'
    }
    
    for feat in selected_features:
        if feat in justifications:
            f.write(f"   • {feat}: {justifications[feat]}\n")
    
    f.write("\n" + "=" * 80 + "\n")

print(f"✅ Resum guardat: {summary_file.name}")

print("\n" + "=" * 80)
print("✅ ANÀLISI DE CORRELACIÓ I SELECCIÓ COMPLETADA")
print("=" * 80)
print(f"\n📁 Fitxers generats:")
print(f"   • {corr_file.name}")
print(f"   • {corr_plot.name}")
print(f"   • {reduced_file.name}")
print(f"   • {matrix_plot.name}")
print(f"   • {summary_file.name}")
print(f"\n📂 Tots els fitxers a: {OUTPUT_DIR}")
print(f"\n🎉 FASE 2 COMPLETADA!")
print("=" * 80 + "\n")
