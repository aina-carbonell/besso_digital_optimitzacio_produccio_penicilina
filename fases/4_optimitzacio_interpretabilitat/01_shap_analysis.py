#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 4: SCRIPT 1 - ANÀLISI SHAP (SHapley Additive exPlanations)
Interpretabilitat: Quines variables impulsen les prediccions
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    print(f"SHAP version: {shap.__version__}")
except ImportError:
    print("ERROR: pip install shap")
    exit(1)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("FASE 4.1 - SHAP ANALYSIS")
print("=" * 80)

# Carregar dades
print("\n[1/6] Carregant dades...")
df_train = pd.read_csv(FASE3_OUT / "train_data.csv")
feature_cols = [c for c in df_train.columns if c not in ['batch_id', 'penicillin', 'time']]
X_train = df_train[feature_cols].values

# Sample per eficiència
n_samples = min(1000, len(X_train))
idx = np.random.choice(len(X_train), n_samples, replace=False)
X_sample = X_train[idx]
print(f"   Mostres: {n_samples}")

# Carregar models
print("\n[2/6] Carregant models...")
rf_model = joblib.load(FASE3_OUT / "03_random_forest_model.pkl")['model']
xgb_model = joblib.load(FASE3_OUT / "03_xgboost_model.pkl")['model']

# SHAP RF
print("\n[3/6] Calculant SHAP - Random Forest...")
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_sample)

# SHAP XGB
print("\n[4/6] Calculant SHAP - XGBoost...")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

# Visualitzacions RF
print("\n[5/6] Generant visualitzacions...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plt.sca(axes[0, 0])
shap.summary_plot(shap_values_rf, X_sample, feature_names=feature_cols, show=False)
axes[0, 0].set_title('RF - SHAP Summary', fontsize=13, fontweight='bold')

plt.sca(axes[0, 1])
shap.summary_plot(shap_values_rf, X_sample, feature_names=feature_cols, 
                 plot_type='bar', show=False)
axes[0, 1].set_title('RF - Feature Importance', fontsize=13, fontweight='bold')

if 'cumulative_penicillin' in feature_cols:
    idx = feature_cols.index('cumulative_penicillin')
    plt.sca(axes[1, 0])
    shap.dependence_plot(idx, shap_values_rf, X_sample, feature_names=feature_cols,
                        show=False, ax=axes[1, 0])
    axes[1, 0].set_title('RF - cumulative_penicillin', fontsize=12, fontweight='bold')

if 'viscosity' in feature_cols:
    idx = feature_cols.index('viscosity')
    plt.sca(axes[1, 1])
    shap.dependence_plot(idx, shap_values_rf, X_sample, feature_names=feature_cols,
                        show=False, ax=axes[1, 1])
    axes[1, 1].set_title('RF - viscosity', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_shap_rf_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# Visualitzacions XGB
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

plt.sca(axes[0, 0])
shap.summary_plot(shap_values_xgb, X_sample, feature_names=feature_cols, show=False)
axes[0, 0].set_title('XGB - SHAP Summary', fontsize=13, fontweight='bold')

plt.sca(axes[0, 1])
shap.summary_plot(shap_values_xgb, X_sample, feature_names=feature_cols,
                 plot_type='bar', show=False)
axes[0, 1].set_title('XGB - Feature Importance', fontsize=13, fontweight='bold')

if 'DO' in feature_cols:
    idx = feature_cols.index('DO')
    plt.sca(axes[1, 0])
    shap.dependence_plot(idx, shap_values_xgb, X_sample, feature_names=feature_cols,
                        show=False, ax=axes[1, 0])
    axes[1, 0].set_title('XGB - DO', fontsize=12, fontweight='bold')

if 'substrate' in feature_cols:
    idx = feature_cols.index('substrate')
    plt.sca(axes[1, 1])
    shap.dependence_plot(idx, shap_values_xgb, X_sample, feature_names=feature_cols,
                        show=False, ax=axes[1, 1])
    axes[1, 1].set_title('XGB - substrate', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_shap_xgb_summary.png", dpi=300, bbox_inches='tight')
plt.close()

# Guardar SHAP values
print("\n[6/6] Guardant resultats...")
shap_df = pd.DataFrame(shap_values_xgb, columns=feature_cols)
shap_df.to_csv(OUTPUT_DIR / "01_shap_values.csv", index=False)

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_rf': np.abs(shap_values_rf).mean(axis=0),
    'importance_xgb': np.abs(shap_values_xgb).mean(axis=0)
})
importance['importance_mean'] = (importance['importance_rf'] + importance['importance_xgb']) / 2
importance = importance.sort_values('importance_mean', ascending=False)
importance.to_csv(OUTPUT_DIR / "01_shap_feature_importance.csv", index=False)

print("\n" + "=" * 80)
print("SHAP COMPLETAT")
print("=" * 80)
print("\nTop 5 features:")
for i, row in importance.head(5).iterrows():
    print(f"   {row['feature']:30s} {row['importance_mean']:.4f}")
print("\nSegüent: python 02_uncertainty_analysis.py")
print("=" * 80 + "\n")