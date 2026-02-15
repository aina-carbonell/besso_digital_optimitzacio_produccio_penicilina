#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 4: SCRIPT 4 - CONDICIONS ÒPTIMES
Identificar setpoints que maximitzin producció
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 4.4 - CONDICIONS ÒPTIMES")
print("=" * 80)

# =============================================================================
# FUNCIÓ PER PDP MANUAL (alternativa a sklearn)
# =============================================================================
def plot_pdp_manual(model, X, feature_names, feature_name, ax):
    """Calcula PDP manualment per evitar problemes amb sklearn"""
    
    if feature_name not in feature_names:
        print(f"   ⚠ Feature {feature_name} no trobada, saltant...")
        return None, None
    
    feat_idx = feature_names.index(feature_name)
    
    # Crear grid de valors (percentils 5-95 per evitar extrapolació)
    min_val = np.percentile(X[:, feat_idx], 5)
    max_val = np.percentile(X[:, feat_idx], 95)
    values = np.linspace(min_val, max_val, 50)
    
    # Calcular PDP
    avg_predictions = []
    X_baseline = X[:500].copy()  # Usar 500 mostres
    
    for val in values:
        X_temp = X_baseline.copy()
        X_temp[:, feat_idx] = val
        preds = model.predict(X_temp)
        avg_predictions.append(np.mean(preds))
    
    # Plot
    ax.plot(values, avg_predictions, 'b-', linewidth=2)
    
    # Marcar òptim
    optimal_idx = np.argmax(avg_predictions)
    optimal_val = values[optimal_idx]
    optimal_pred = avg_predictions[optimal_idx]
    
    ax.axvline(optimal_val, color='red', linestyle='--', lw=2,
               label=f'Òptim: {optimal_val:.2f}')
    ax.scatter([optimal_val], [optimal_pred], color='red', s=100, zorder=5)
    
    ax.set_xlabel(feature_name, fontsize=11)
    ax.set_ylabel('Penicil·lina parcial (g/L)', fontsize=11)
    ax.set_title(f'Partial Dependence - {feature_name}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return optimal_val, optimal_pred

# =============================================================================
# CARREGAR DADES I MODEL
# =============================================================================
print("\n[1/5] Carregant model optimitzat...")

try:
    df_train = pd.read_csv(FASE3_OUT / "train_data.csv")
    print(f"   ✅ Train dataset carregat: {df_train.shape}")
except Exception as e:
    print(f"   ❌ Error carregant train_data.csv: {e}")
    exit(1)

feature_cols = [c for c in df_train.columns if c not in ['batch_id', 'penicillin', 'time']]
print(f"   📊 Features detectades: {len(feature_cols)}")

# Intentar carregar model optimitzat, sinó usar original
try:
    xgb_data = joblib.load(OUTPUT_DIR / "03_xgboost_optimized.pkl")
    print("   ✅ Model optimitzat carregat")
except:
    try:
        xgb_data = joblib.load(FASE3_OUT / "03_xgboost_model.pkl")
        print("   ✅ Model original carregat")
    except Exception as e:
        print(f"   ❌ Error carregant model: {e}")
        exit(1)

model = xgb_data['model']

# Estadístiques features (per constraints)
print("\n[2/5] Calculant rangs operacionals...")
feature_stats = df_train[feature_cols].describe()
print(f"   ✅ Rangs calculats per a {len(feature_cols)} variables")

# =============================================================================
# PARTIAL DEPENDENCE PLOTS (VERSIÓ MANUAL)
# =============================================================================
print("\n[3/5] Generant Partial Dependence Plots (manual)...")

# Top features per analitzar (excloent cumulative_penicillin)
top_features = ['viscosity', 'DO', 'substrate', 'OUR', 'specific_production_rate', 'substrate_rate']
top_features = [f for f in top_features if f in feature_cols][:4]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

X_sample = df_train[feature_cols].values[:1000]  # Mostra per PDP

for i, feat in enumerate(top_features):
    print(f"   📈 Generant PDP per {feat}...")
    plot_pdp_manual(model, X_sample, feature_cols, feat, axes[i])

plt.tight_layout()
pdp_path = OUTPUT_DIR / "04_partial_dependence_plots.png"
plt.savefig(pdp_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Partial Dependence Plots guardats: {pdp_path.name}")

# =============================================================================
# OPTIMITZACIÓ
# =============================================================================
print("\n[4/5] Optimitzant setpoints...")

# Definir bounds (basat en percentils 5-95 de training data)
bounds = []
feature_bounds = {}

for feat in feature_cols:
    if feat == 'cumulative_penicillin':
        # Fixar a valor mig (no controlable directament)
        median_val = df_train[feat].median()
        bounds.append((median_val, median_val))
        feature_bounds[feat] = {'min': median_val, 'max': median_val}
    else:
        lower = df_train[feat].quantile(0.05)
        upper = df_train[feat].quantile(0.95)
        bounds.append((lower, upper))
        feature_bounds[feat] = {'min': lower, 'max': upper}

print(f"   ✅ Bounds definits per a {len(bounds)} variables")

# Funció objectiu (maximitzar predicció)
def objective(x):
    return -model.predict(x.reshape(1, -1))[0]

# Optimització amb differential evolution
print("   🔄 Executant optimització (pot trigar 1-2 minuts)...")
result = differential_evolution(
    objective,
    bounds,
    maxiter=100,
    seed=42,
    workers=1  # Evitar problemes de multiprocessing a Windows
)

optimal_features = result.x
optimal_production = -result.fun

print(f"\n   ✅ Producció òptima esperada: {optimal_production:.2f} g/L")

# =============================================================================
# SETPOINTS RECOMANATS
# =============================================================================
print("\n[5/5] Generant recomanacions...")

# Comparar amb baseline (mitjanes actuals)
baseline_features = df_train[feature_cols].mean().values
baseline_production = model.predict(baseline_features.reshape(1, -1))[0]

improvement_pct = ((optimal_production - baseline_production) / baseline_production) * 100

# Crear taula setpoints
setpoints = pd.DataFrame({
    'Variable': feature_cols,
    'Baseline': baseline_features,
    'Optimal': optimal_features,
    'Change_%': ((optimal_features - baseline_features) / np.abs(baseline_features)) * 100
})

# Afegir informació de bounds
setpoints['Min_Bound'] = [feature_bounds[f]['min'] for f in feature_cols]
setpoints['Max_Bound'] = [feature_bounds[f]['max'] for f in feature_cols]
setpoints['Within_Bounds'] = (setpoints['Optimal'] >= setpoints['Min_Bound']) & (setpoints['Optimal'] <= setpoints['Max_Bound'])

# Filtrar només variables amb canvi significatiu i dins dels bounds
setpoints['Controllable'] = (np.abs(setpoints['Change_%']) > 5) & setpoints['Within_Bounds']
setpoints_ctrl = setpoints[setpoints['Controllable']].sort_values('Change_%', key=abs, ascending=False)

print("\n   === SETPOINTS RECOMANATS (canvi >5%) ===")
print(setpoints_ctrl[['Variable', 'Baseline', 'Optimal', 'Change_%']].to_string(index=False))

# Guardar
setpoints.to_csv(OUTPUT_DIR / "04_optimal_setpoints.csv", index=False)
print(f"   ✅ Setpoints guardats: 04_optimal_setpoints.csv")

# =============================================================================
# VISUALITZACIÓ
# =============================================================================
print("\n   📊 Generant visualitzacions...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Comparació baseline vs òptim (top variables)
ax = axes[0, 0]
top_vars = setpoints_ctrl.head(6)
if len(top_vars) > 0:
    x = np.arange(len(top_vars))
    width = 0.35
    
    ax.barh(x - width/2, top_vars['Baseline'], width, label='Baseline', 
           color='lightblue', edgecolor='black')
    ax.barh(x + width/2, top_vars['Optimal'], width, label='Òptim',
           color='lightgreen', edgecolor='black')
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_vars['Variable'])
    ax.set_xlabel('Valor')
    ax.set_title('Setpoints: Baseline vs Òptim', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No hi ha variables amb canvi significatiu', 
            ha='center', va='center', transform=ax.transAxes)

# Plot 2: % Change
ax = axes[0, 1]
if len(top_vars) > 0:
    colors = ['green' if c > 0 else 'red' for c in top_vars['Change_%']]
    ax.barh(x, top_vars['Change_%'], color=colors, edgecolor='black', alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(top_vars['Variable'])
    ax.set_xlabel('Canvi (%)')
    ax.set_title('Canvi Recomanat (%)', fontweight='bold')
    ax.axvline(0, color='black', lw=1)
    ax.grid(True, alpha=0.3, axis='x')
else:
    ax.text(0.5, 0.5, 'No hi ha variables amb canvi significatiu', 
            ha='center', va='center', transform=ax.transAxes)

# Plot 3: Expected improvement
ax = axes[1, 0]
categories = ['Baseline', 'Òptim']
productions = [baseline_production, optimal_production]
colors_bar = ['lightblue', 'lightgreen']

bars = ax.bar(categories, productions, color=colors_bar, edgecolor='black', linewidth=2, width=0.6)
ax.set_ylabel('Penicil·lina (g/L)', fontsize=11)
ax.set_title(f'Producció Esperada (+{improvement_pct:.1f}%)', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, productions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{val:.2f} g/L', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Plot 4: Text summary
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
RESUM OPTIMITZACIÓ

Producció Baseline: {baseline_production:.2f} g/L
Producció Òptima:   {optimal_production:.2f} g/L

Millora Esperada:   +{improvement_pct:.1f}%

TOP CANVIS RECOMANATS:
"""

for i, row in top_vars.head(3).iterrows():
    summary_text += f"\n• {row['Variable']}: "
    if row['Change_%'] > 0:
        summary_text += f"↑ +{row['Change_%']:.1f}%"
    else:
        summary_text += f"↓ {row['Change_%']:.1f}%"

summary_text += f"""

INTERPRETACIÓ:
• DO: Control d'oxigen per metabolisme aeròbic
• Substrate: Estratègia fed-batch òptima
• Viscosity: Relacionada amb concentració de biomassa
• OUR: Activitat metabòlica del cultiu
"""

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
summary_path = OUTPUT_DIR / "04_optimization_summary.png"
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✅ Visualització guardada: {summary_path.name}")

# =============================================================================
# RESUM FINAL
# =============================================================================
print("\n" + "=" * 80)
print("OPTIMAL CONDITIONS COMPLETAT")
print("=" * 80)
print(f"\n📊 RESULTATS:")
print(f"   Producció baseline: {baseline_production:.2f} g/L")
print(f"   Producció òptima:   {optimal_production:.2f} g/L")
print(f"   Millora esperada:   +{improvement_pct:.1f}%")

print(f"\n📁 FITXERS GENERATS:")
print(f"   • 04_partial_dependence_plots.png")
print(f"   • 04_optimal_setpoints.csv")
print(f"   • 04_optimization_summary.png")

print(f"\n🚀 Següent: python 05_sensitivity_analysis.py")
print("=" * 80 + "\n")