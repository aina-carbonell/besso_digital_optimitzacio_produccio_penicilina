#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 5: DASHBOARD INTERACTIU
Monitorització en temps real amb prediccions i alertes
"""
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
FASE3_OUT = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"
FASE4_OUT = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat" / "outputs"
OUTPUT_DIR = PROJECT_ROOT / "fases" / "5_visualitzacio_conclusions" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("FASE 5.1 - DASHBOARD INTERACTIU")
print("="*80)

# Carregar dades i models
print("\n[1/4] Carregant dades...")
df_test = pd.read_csv(FASE3_OUT / "test_data.csv")
feature_cols = [c for c in df_test.columns if c not in ['batch_id', 'penicillin', 'time']]

try:
    model_data = joblib.load(FASE4_OUT / "03_xgboost_optimized.pkl")
except:
    model_data = joblib.load(FASE3_OUT / "03_xgboost_model.pkl")
model = model_data['model']

# Generar prediccions
X_test = df_test[feature_cols].values
y_test = df_test['penicillin'].values
y_pred = model.predict(X_test)
df_test['prediction'] = y_pred
df_test['error'] = np.abs(y_test - y_pred)

# Calcular threshold anomalies (P95)
threshold = np.percentile(df_test['error'], 95)
df_test['anomaly'] = df_test['error'] > threshold

print("\n[2/4] Creant dashboard HTML...")

# Crear dashboard amb Plotly
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=(
        'Predicció vs Real - Batch 95',
        'Distribució Errors',
        'Timeline amb Anomalies',
        'KPIs per Batch',
        'Feature Importance',
        'Mapa de Calor Variables'
    ),
    specs=[
        [{"type": "scatter"}, {"type": "histogram"}],
        [{"colspan": 2, "type": "scatter"}, None],
        [{"type": "bar"}, {"type": "heatmap"}]
    ]
)

# Plot 1: Predicció vs Real (Batch 95)
batch_95 = df_test[df_test['batch_id'] == 95].sort_values('time')
fig.add_trace(
    go.Scatter(x=batch_95['time'], y=batch_95['penicillin'], 
              name='Real', mode='lines+markers', line=dict(color='black', width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=batch_95['time'], y=batch_95['prediction'],
              name='Predicció', mode='lines+markers', 
              line=dict(color='blue', width=2, dash='dash')),
    row=1, col=1
)

# Marcar anomalies
anomalies_95 = batch_95[batch_95['anomaly']]
if len(anomalies_95) > 0:
    fig.add_trace(
        go.Scatter(x=anomalies_95['time'], y=anomalies_95['penicillin'],
                  name='Anomalia', mode='markers', 
                  marker=dict(color='red', size=12, symbol='x')),
        row=1, col=1
    )

# Plot 2: Distribució errors
fig.add_trace(
    go.Histogram(x=df_test['error'], nbinsx=50, name='Errors',
                marker=dict(color='purple', opacity=0.7)),
    row=1, col=2
)

# Plot 3: Timeline anomalies (tots els batches)
for batch_id in df_test['batch_id'].unique():
    batch_data = df_test[df_test['batch_id'] == batch_id].sort_values('time')
    fig.add_trace(
        go.Scatter(x=batch_data['time'], y=batch_data['penicillin'],
                  name=f'Batch {batch_id}', mode='lines', opacity=0.6),
        row=2, col=1
    )

# Plot 4: KPIs per batch
batch_kpis = df_test.groupby('batch_id').agg({
    'penicillin': 'mean',
    'error': 'mean',
    'anomaly': 'mean'
}).reset_index()

fig.add_trace(
    go.Bar(x=batch_kpis['batch_id'], y=batch_kpis['penicillin'],
          name='Mean Penicillin', marker=dict(color='green')),
    row=3, col=1
)

# Plot 5: Heatmap variables (correlació simulada)
feature_sample = df_test[feature_cols[:6]].corr().values
fig.add_trace(
    go.Heatmap(z=feature_sample, x=feature_cols[:6], y=feature_cols[:6],
              colorscale='RdBu', zmid=0),
    row=3, col=2
)

# Layout
fig.update_layout(
    title_text="Dashboard Monitorització Producció Penicil·lina",
    title_font=dict(size=20, color='darkblue', family='Arial Black'),
    showlegend=True,
    height=1200,
    template='plotly_white'
)

fig.update_xaxes(title_text="Temps (h)", row=1, col=1)
fig.update_yaxes(title_text="Penicil·lina (g/L)", row=1, col=1)
fig.update_xaxes(title_text="Error (g/L)", row=1, col=2)
fig.update_yaxes(title_text="Freqüència", row=1, col=2)
fig.update_xaxes(title_text="Temps (h)", row=2, col=1)
fig.update_yaxes(title_text="Penicil·lina (g/L)", row=2, col=1)

# Guardar HTML interactiu
html_path = OUTPUT_DIR / "01_dashboard_interactiu.html"
fig.write_html(str(html_path))

print(f"   Dashboard: {html_path.name}")

# Crear dashboard KPIs en text
print("\n[3/4] Generant KPIs...")

kpis = {
    'Total_Batches': int(df_test['batch_id'].nunique()),
    'Mean_Production': float(df_test['penicillin'].mean()),
    'Std_Production': float(df_test['penicillin'].std()),
    'Mean_Error': float(df_test['error'].mean()),
    'Anomaly_Rate': float(df_test['anomaly'].mean()),
    'Best_Batch': int(batch_kpis.loc[batch_kpis['penicillin'].idxmax(), 'batch_id']),
    'Worst_Batch': int(batch_kpis.loc[batch_kpis['penicillin'].idxmin(), 'batch_id'])
}

with open(OUTPUT_DIR / "01_dashboard_kpis.json", 'w', encoding='utf-8') as f:
    json.dump(kpis, f, indent=2)

print("\n[4/4] Generant README dashboard...")

readme = f"""
# DASHBOARD INTERACTIU - FASE 5

## 📊 Accés al Dashboard

Obre el fitxer: **01_dashboard_interactiu.html**

El dashboard és completament interactiu amb Plotly:
- Zoom in/out
- Pan (arrossegar)
- Hover per veure valors
- Llegenda clicable per ocultar/mostrar sèries

## 📈 KPIs Principals

- **Batches analitzats**: {kpis['Total_Batches']}
- **Producció mitjana**: {kpis['Mean_Production']:.2f} g/L
- **Desviació estàndard**: {kpis['Std_Production']:.2f} g/L
- **Error mitjà predicció**: {kpis['Mean_Error']:.2f} g/L
- **Taxa d'anomalies**: {kpis['Anomaly_Rate']*100:.1f}%
- **Millor batch**: {kpis['Best_Batch']}
- **Pitjor batch**: {kpis['Worst_Batch']}

## 🎯 Components del Dashboard

1. **Predicció vs Real (Batch 95)**
   - Comparació temporal
   - Anomalies marcades en vermell
   
2. **Distribució d'Errors**
   - Histograma de precisió del model
   
3. **Timeline amb Anomalies**
   - Evolució de tots els batches
   - Identificació visual de problemes
   
4. **KPIs per Batch**
   - Producció mitjana per batch
   
5. **Mapa de Calor Variables**
   - Correlacions entre features

## 💡 Ús en Producció

Aquest dashboard pot ser adaptat per:
- Monitorització en temps real
- Alertes automàtiques
- Integració amb SCADA
- Reporting automàtic
"""

with open(OUTPUT_DIR / "01_dashboard_README.md", 'w', encoding='utf-8') as f:
    f.write(readme)

print("\n"+"="*80)
print("DASHBOARD CREAT AMB ÈXIT")
print("="*80)
print(f"\nObrir: {html_path}")
print(f"\nKPIs principals:")
print(f"   Batches: {kpis['Total_Batches']}")
print(f"   Producció: {kpis['Mean_Production']:.2f} ± {kpis['Std_Production']:.2f} g/L")
print(f"   Anomalies: {kpis['Anomaly_Rate']*100:.1f}%")
print("\nSegüent: python 02_technical_conclusions.py")
print("="*80+"\n")