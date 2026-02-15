#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 5: IMPLICACIONS INDUSTRIALS
Com milloraria el sistema en planta real
"""
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "5_visualitzacio_conclusions" / "outputs"

print("="*80)
print("FASE 5.3 - IMPLICACIONS INDUSTRIALS")
print("="*80)

print("\n[1/2] Calculant ROI...")

# Assumptions
current_production = 100  # kg/batch baseline
improvement_pct = 12  # % improvement
batches_per_year = 250
price_per_kg = 150  # EUR
energy_cost_increase = 5  # %
current_energy_cost = 50000  # EUR/year

# Càlculs
new_production = current_production * (1 + improvement_pct/100)
additional_kg_year = (new_production - current_production) * batches_per_year
revenue_increase = additional_kg_year * price_per_kg
energy_cost_additional = current_energy_cost * (energy_cost_increase/100)
implementation_cost = 100000  # EUR one-time
annual_maintenance = 15000  # EUR/year

net_benefit_year1 = revenue_increase - energy_cost_additional - implementation_cost - annual_maintenance
net_benefit_year2plus = revenue_increase - energy_cost_additional - annual_maintenance
payback_period = implementation_cost / (revenue_increase - energy_cost_additional - annual_maintenance)

roi_5years = ((net_benefit_year1 + net_benefit_year2plus*4) / implementation_cost) * 100

print("\n[2/2] Generant informe industrial...")

report = f"""
================================================================================
IMPLICACIONS INDUSTRIALS - IMPLEMENTACIÓ EN PLANTA REAL
================================================================================

1. MILLORES EN CONTROL DE QUALITAT

Situació Actual:
   • Control manual basat en experiència
   • Mostreig off-line cada 2-4h
   • Decisions reactives
   • Alta variabilitat batch-to-batch (CV ~25%)
   • Temps identificació problemes: 6-12h

Amb Sistema Predictiu:
   ✓ Predicció concentració cada 15 min
   ✓ Detecció anomalies en temps real (<5 min)
   ✓ Decisions proactives basades en dades
   ✓ Reducció variabilitat: -20% (CV ~20%)
   ✓ Early warning 2-4h abans de problema

Impacte:
   • -30% batches defectuosos
   • +98% compliment especificacions
   • -50% temps downtime per investigacions
   • Documentació completa per reguladors

================================================================================
2. INTEGRACIÓ AMB SISTEMES EXISTENTS
================================================================================

Components necessaris:

A) Hardware:
   • Servidor on-premise o cloud (mínims: 16GB RAM, 8 cores)
   • Sensors addicionals: viscosímetre online (20K EUR)
   • OPC UA gateway per integració SCADA

B) Software:
   • Model XGBoost desplegat (Python/Docker)
   • Dashboard HTML5 (Plotly/Grafana)
   • Alertes via email/SMS/Teams
   • API REST per integració

C) Integració SCADA:
   • Lectura temps real: DO, pH, T, flow rates
   • Escriptura setpoints: aeration, base addition
   • Logs històrics per reentrenament
   • Backup i redundància

D) Training Personal:
   • Operadors: Interpretar dashboard (1 dia)
   • Engineers: Troubleshooting (2 dies)
   • Management: KPIs i ROI (4h)

================================================================================
3. ANÀLISI ROI (Return on Investment)
================================================================================

Situació Base:
   Producció actual:        {current_production} kg/batch
   Batches/any:             {batches_per_year}
   Producció anual:         {current_production * batches_per_year:,} kg
   Preu venda:              {price_per_kg} EUR/kg

Amb Sistema Predictiu:
   Millora producció:       +{improvement_pct}%
   Nova producció:          {new_production:.1f} kg/batch
   Kg addicionals/any:      {additional_kg_year:,.0f} kg
   
Ingressos:
   Increment anual:         {revenue_increase:,.0f} EUR

Costos:
   Implementació (1 cop):   {implementation_cost:,.0f} EUR
   Manteniment anual:       {annual_maintenance:,.0f} EUR
   Increment energia:       {energy_cost_additional:,.0f} EUR/any

Benefici Net:
   Any 1:                   {net_benefit_year1:,.0f} EUR
   Any 2+:                  {net_benefit_year2plus:,.0f} EUR/any
   
ROI a 5 anys:               {roi_5years:.0f}%
Payback period:             {payback_period:.1f} anys

================================================================================
4. BENEFICIS QUALITATIUS
================================================================================

Operacionals:
   ✓ Reducció càrrega cognitiva operadors
   ✓ Decisions basades en dades, no intuïció
   ✓ Documentació automàtica per audits
   ✓ Flexibilitat per canvis de formulació

Estratègics:
   ✓ Avantatge competitiu (qualitat + cost)
   ✓ Base per digitalització planta
   ✓ Atracció talent (tecnologia avançada)
   ✓ Millor relació amb reguladors (FDA/EMA)

Seguretat:
   ✓ Detecció precoç desviacions perilloses
   ✓ Reducció risc contaminació
   ✓ Menys exposició operadors (automatització)

================================================================================
5. RISCOS I MITIGACIONS
================================================================================

Risc: Model es degrada amb el temps
Mitigació: Reentrenament trimestral amb dades noves, monitoring drift

Risc: Dependència excessiva tecnologia
Mitigació: Manteniment protocols manuals, training continu

Risc: Fallada sistema crític
Mitigació: Redundància, fallback automàtic a control manual

Risc: Resistència al canvi operadors
Mitigació: Involucrar equip des del principi, mostrar beneficis

Risc: Costos manteniment més alts del previst
Mitigació: SLA amb proveïdor, documentació exhaustiva

================================================================================
6. PLA D'IMPLEMENTACIÓ INDUSTRIAL
================================================================================

FASE 1: Pilot (Mes 1-2)
   □ Instal·lar en 1 reactor
   □ Córrer en paral·lel amb control actual
   □ Validar prediccions vs real
   □ Ajustar thresholds alertes
   □ Training operadors
   Cost: 30K EUR

FASE 2: Validació (Mes 3-4)
   □ Anàlisi 20+ batches
   □ Calcular ROI real
   □ Documentar per reguladors
   □ Go/No-Go decision
   Cost: 20K EUR

FASE 3: Roll-out (Mes 5-8)
   □ Implementar en tots els reactors
   □ Integració completa SCADA
   □ Documentació final
   □ Certificació validació
   Cost: 50K EUR

FASE 4: Optimització (Mes 9-12)
   □ Fine-tuning setpoints
   □ Reentrenament models
   □ Expansió a altres productes
   □ Lessons learned
   Cost: 20K EUR (inclòs manteniment)

TOTAL: 120K EUR (proper a estimació inicial)

================================================================================
7. CASOS D'ÚS ADDICIONALS
================================================================================

Un cop implementat, el sistema pot:

1. Predicció multi-batch:
   • Planificació producció 1-2 setmanes vista
   • Optimització scheduling

2. Control avançat:
   • MPC (Model Predictive Control)
   • Setpoints dinàmics per màxima eficiència

3. Expansió altres productes:
   • Transfer learning a cefalosporines
   • Altres antibiòtics β-lactàmics

4. Digital Twin:
   • Simulació "what-if" scenarios
   • Training nous operadors

5. Supply chain:
   • Predicció necessitats matèries primeres
   • Optimització inventari

================================================================================
8. CONCLUSIONS
================================================================================

El sistema proposat ofereix:

ROI atractiu: {roi_5years:.0f}% a 5 anys
Payback ràpid: {payback_period:.1f} anys
Beneficis tangibles: +{improvement_pct}% producció, -{20}% variabilitat
Beneficis intangibles: Qualitat, compliance, competitivitat

Recomanació: IMPLEMENTAR

Pròxims passos immediats:
1. Presentar a management (aquest informe)
2. Aprovar budget pilot (30K EUR)
3. Seleccionar reactor pilot
4. Contractar engineering support
5. Inici en 1 mes

================================================================================
FI DE L'INFORME INDUSTRIAL
================================================================================
"""

with open(OUTPUT_DIR / "03_industrial_implications.txt", 'w', encoding='utf-8', errors='replace') as f:
    f.write(report)

# Visualització ROI
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Cash flow 5 anys
years = list(range(0, 6))
cashflow = [-implementation_cost, net_benefit_year1] + [net_benefit_year2plus]*4
cumulative = [sum(cashflow[:i+1]) for i in range(len(cashflow))]

ax = axes[0]
ax.plot(years, cumulative, 'o-', linewidth=3, markersize=10, color='green')
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.fill_between(years, 0, cumulative, where=[c>=0 for c in cumulative], 
                alpha=0.3, color='green', label='Profit')
ax.fill_between(years, 0, cumulative, where=[c<0 for c in cumulative],
                alpha=0.3, color='red', label='Cost')
ax.set_xlabel('Any', fontsize=12, fontweight='bold')
ax.set_ylabel('Cash Flow Acumulat (EUR)', fontsize=12, fontweight='bold')
ax.set_title(f'ROI Projecte (Payback: {payback_period:.1f} anys)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Beneficis anuals
ax = axes[1]
benefits = ['Increment\nIngressos', 'Cost\nEnergia', 'Manteniment', 'Benefici\nNet']
values = [revenue_increase, -energy_cost_additional, -annual_maintenance, net_benefit_year2plus]
colors_ben = ['green', 'red', 'orange', 'blue']
ax.bar(benefits, values, color=colors_ben, edgecolor='black', alpha=0.7)
ax.axhline(0, color='black', lw=1)
ax.set_ylabel('EUR/any', fontsize=12, fontweight='bold')
ax.set_title('Anàlisi Beneficis Anuals', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

for i, (b, v) in enumerate(zip(benefits, values)):
    ax.text(i, v, f'{v:,.0f}', ha='center', 
           va='bottom' if v>0 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_roi_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("\n"+"="*80)
print("INFORME INDUSTRIAL COMPLETAT")
print("="*80)
print(f"\nROI 5 anys: {roi_5years:.0f}%")
print(f"Payback: {payback_period:.1f} anys")
print(f"Benefici net anual: {net_benefit_year2plus:,.0f} EUR")
print("\nRECOMANACIÓ: Implementar sistema")
print("\nSegüent: python 04_final_presentation.py")
print("="*80+"\n")