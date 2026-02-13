#!/usr/bin/env python3
"""
==================================================================================
FASE 2: SCRIPT MASTER - EXECUCIÓ COMPLETA
Executa tot el pipeline d'exploració i preparació de dades
==================================================================================
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("\n" + "=" * 80)
print("FASE 2: PIPELINE COMPLET D'EXPLORACIÓ I PREPARACIÓ DE DADES")
print("=" * 80)
print(f"\nInici: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Scripts a executar en ordre
scripts = [
    "01_exploratory_data_analysis.py",
    "02_temporal_visualization.py",
    "03_data_cleaning_and_feature_engineering.py",
    "04_correlation_and_feature_selection.py"
]

PROJECT_ROOT = Path(__file__).parent
failed = False

for i, script in enumerate(scripts, 1):
    script_path = PROJECT_ROOT / script
    
    print(f"\n{'='*80}")
    print(f"[{i}/{len(scripts)}] Executant: {script}")
    print(f"{'='*80}\n")
    
    if not script_path.exists():
        print(f"❌ ERROR: Script no trobat: {script}")
        failed = True
        break
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=False,
            text=True,
            check=True
        )
        
        print(f"\n✅ {script} completat correctament")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR executant {script}")
        print(f"   Codi d'error: {e.returncode}")
        failed = True
        break
    except Exception as e:
        print(f"\n❌ ERROR inesperat: {e}")
        failed = True
        break

print("\n" + "=" * 80)
if not failed:
    print("🎉 PIPELINE COMPLETAT AMB ÈXIT!")
    print("=" * 80)
    print("\n📁 Tots els outputs generats es troben a:")
    print(f"   {PROJECT_ROOT / 'fases' / '2. Exploracio i preparacio' / 'outputs'}")
    print("\n📊 Fitxers principals generats:")
    print("   • Dataset amb 33 columnes")
    print("   • Dataset amb top 9 variables")
    print("   • Visualitzacions de batches")
    print("   • Anàlisi de correlacions")
    print("   • Resum estadístic")
else:
    print("❌ PIPELINE INTERROMPUT PER ERRORS")
    print("=" * 80)
    print("\n⚠️  Revisa els missatges d'error anteriors")

print(f"\nFinalitzat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80 + "\n")

sys.exit(0 if not failed else 1)
