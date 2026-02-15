#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 4: SCRIPT 5 - ANÀLISI DE SENSIBILITAT
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"

print("="*80)
print("FASE 4.5 - SENSITIVITY ANALYSIS")
print("="*80)

# Carregar
print("\n[1/3] Carregant model...")
df_train = pd.read_csv(FASE3_OUT / "train_data.csv")
feature_cols = [c for c in df_train.columns if c not in ['batch_id', 'penicillin', 'time']]
try:
    model = joblib.load(OUTPUT_DIR / "03_xgboost_optimized.pkl")['model']
except:
    model = joblib.load(FASE3_OUT / "03_xgboost_model.pkl")['model']

# Baseline
baseline = df_train[feature_cols].mean().values
baseline_pred = model.predict(baseline.reshape(1, -1))[0]

# Sensitivity analysis
print("\n[2/3] Calculant sensibilitats...")
sensitivities = []

for i, feat in enumerate(feature_cols):
    # +10%
    X_plus = baseline.copy()
    X_plus[i] *= 1.10
    pred_plus = model.predict(X_plus.reshape(1, -1))[0]
    
    # -10%
    X_minus = baseline.copy()
    X_minus[i] *= 0.90
    pred_minus = model.predict(X_minus.reshape(1, -1))[0]
    
    # Sensitivity
    sensitivity = (pred_plus - pred_minus) / (2 * 0.10 * baseline[i])
    
    sensitivities.append({
        'feature': feat,
        'sensitivity': sensitivity,
        'abs_sensitivity': abs(sensitivity)
    })

sens_df = pd.DataFrame(sensitivities).sort_values('abs_sensitivity', ascending=False)
sens_df.to_csv(OUTPUT_DIR / "05_sensitivity_analysis.csv", index=False)

# Tornado diagram
print("\n[3/3] Generant visualitzacions...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

top_sens = sens_df.head(9)
y_pos = np.arange(len(top_sens))
colors = ['green' if s > 0 else 'red' for s in top_sens['sensitivity']]

ax.barh(y_pos, top_sens['abs_sensitivity'], color=colors, edgecolor='black', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_sens['feature'])
ax.set_xlabel('Sensibilitat (dPen/dX)', fontsize=11)
ax.set_title('Tornado Diagram - Anàlisi de Sensibilitat', fontweight='bold', fontsize=13)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_sensitivity_tornado.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n"+"="*80)
print("SENSITIVITY ANALYSIS COMPLETAT")
print("="*80)
print("\nTop 3 més sensibles:")
for i, row in top_sens.head(3).iterrows():
    print(f"   {row['feature']:30s} {row['sensitivity']:.4f}")
print("\nSegüent: python 06_interpretability_report.py")
print("="*80+"\n")