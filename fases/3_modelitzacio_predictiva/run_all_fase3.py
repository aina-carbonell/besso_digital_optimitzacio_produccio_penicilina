#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: SCRIPT D'EXECUCIÓ COMPLETA
Executa tots els scripts de modelització predictiva en ordre
==================================================================================
"""

import subprocess
import sys
from pathlib import Path
import time

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva"

scripts = [
    "01_data_preparation.py",
    "02_baseline_ridge.py",
    "03_ensemble_models.py",
    "04_lstm_model.py",
    "05_fault_detection.py",
    "06_model_comparison.py"
]

print("=" * 80)
print("FASE 3: EXECUCIÓ COMPLETA - MODELITZACIÓ PREDICTIVA")
print("=" * 80)
print(f"\nScripts a executar: {len(scripts)}")
for i, script in enumerate(scripts, 1):
    print(f"   {i}. {script}")

print("\n" + "=" * 80)

# Temps total
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
        
        results.append({
            'script': script,
            'status': '✓ OK',
            'time': elapsed
        })
        
        print(f"\n✓ {script} completat en {elapsed:.1f}s")
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        results.append({
            'script': script,
            'status': '✗ ERROR',
            'time': elapsed
        })
        
        print(f"\n✗ ERROR en {script}")
        print(f"   Codi de sortida: {e.returncode}")
        
        # Preguntar si continuar
        response = input("\n   Continuar amb el següent script? (s/n): ")
        if response.lower() != 's':
            print("\n   Execució cancel·lada per l'usuari")
            break

# Temps total
total_time = time.time() - start_total

# =============================================================================
# RESUM FINAL
# =============================================================================
print("\n" + "=" * 80)
print("RESUM D'EXECUCIÓ")
print("=" * 80)

print(f"\nScripts executats: {len(results)}/{len(scripts)}")
print(f"\nResultats:")

for i, res in enumerate(results, 1):
    print(f"   {i}. {res['script']:40s} {res['status']:8s} ({res['time']:6.1f}s)")

# Comptar èxits i errors
success_count = sum(1 for r in results if '✓' in r['status'])
error_count = sum(1 for r in results if '✗' in r['status'])

print(f"\n   Èxits: {success_count}")
print(f"   Errors: {error_count}")

print(f"\nTemps total: {total_time:.1f}s ({total_time/60:.1f} minuts)")

if error_count == 0:
    print("\n🎉 FASE 3 COMPLETADA AMB ÈXIT!")
else:
    print(f"\n⚠️  Hi ha hagut {error_count} error(s)")

print("=" * 80)

# Verificar outputs generats
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

expected_outputs = [
    "train_data.csv",
    "test_data.csv",
    "02_ridge_model.pkl",
    "03_random_forest_model.pkl",
    "03_xgboost_model.pkl",
    "04_lstm_model.h5",
    "06_metrics_summary.csv",
    "FASE3_RESUM_FINAL.txt"
]

print(f"\nVerificant fitxers...")
missing = []
found = []

for output in expected_outputs:
    output_path = OUTPUT_DIR / output
    if output_path.exists():
        found.append(output)
    else:
        missing.append(output)

print(f"   Trobats: {len(found)}/{len(expected_outputs)}")

if missing:
    print(f"\n   ⚠️  Fitxers que falten:")
    for f in missing:
        print(f"      • {f}")
else:
    print(f"\n   ✓ Tots els fitxers principals generats correctament")

print("\n" + "=" * 80 + "\n")

# Exit code
sys.exit(0 if error_count == 0 else 1)