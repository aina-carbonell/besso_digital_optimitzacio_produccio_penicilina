#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE 4: RUN ALL - Executa tots els scripts de la Fase 4
"""
import subprocess
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "fases" / "4_optimitzacio_interpretabilitat"

scripts = [
    "01_shap_analysis.py",
    "02_uncertainty_analysis.py",
    "03_hyperparameter_optimization.py",
    "04_optimal_conditions.py",
    "05_sensitivity_analysis.py",
    "06_interpretability_report.py"
]

print("="*80)
print("FASE 4: EXECUCIÓ COMPLETA")
print("="*80)
print(f"\nScripts: {len(scripts)}")
for i, s in enumerate(scripts, 1):
    print(f"   {i}. {s}")

print("\n"+"="*80)

start_total = time.time()
results = []

for i, script in enumerate(scripts, 1):
    script_path = SCRIPTS_DIR / script
    
    print(f"\n{'='*80}")
    print(f"[{i}/{len(scripts)}] Executant: {script}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SCRIPTS_DIR),
            capture_output=False,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        results.append({'script': script, 'status': '✓ OK', 'time': elapsed})
        print(f"\n✓ {script} completat en {elapsed:.1f}s")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        results.append({'script': script, 'status': '✗ ERROR', 'time': elapsed})
        print(f"\n✗ ERROR en {script}")
        
        response = input("\nContinuar? (s/n): ")
        if response.lower() != 's':
            break

total_time = time.time() - start_total

print("\n"+"="*80)
print("RESUM EXECUCIÓ")
print("="*80)
print(f"\nScripts: {len(results)}/{len(scripts)}")
for i, r in enumerate(results, 1):
    print(f"   {i}. {r['script']:40s} {r['status']:8s} ({r['time']:6.1f}s)")

success = sum(1 for r in results if '✓' in r['status'])
errors = sum(1 for r in results if '✗' in r['status'])

print(f"\n   Èxits: {success}")
print(f"   Errors: {errors}")
print(f"\nTemps total: {total_time:.1f}s ({total_time/60:.1f} min)")

if errors == 0:
    print("\n🎉 FASE 4 COMPLETADA AMB ÈXIT!")
else:
    print(f"\n⚠️  {errors} error(s)")

print("="*80+"\n")

sys.exit(0 if errors == 0 else 1)