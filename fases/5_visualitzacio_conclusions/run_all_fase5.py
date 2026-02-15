#!/usr/bin/env python3
"""
FASE 5: RUN ALL
"""
import subprocess, sys, time
from pathlib import Path

scripts = [
    "01_dashboard_creation.py",
    "02_technical_conclusions.py",
    "03_industrial_report.py",
    "04_final_presentation.py"
]

print("="*80)
print("FASE 5: EXECUCIÓ COMPLETA")
print("="*80)

start = time.time()
for i, script in enumerate(scripts, 1):
    print(f"\n[{i}/{len(scripts)}] {script}")
    try:
        subprocess.run([sys.executable, script], check=True)
    except:
        print(f"ERROR en {script}")
        break

print(f"\nTemps total: {time.time()-start:.1f}s")
print("="*80+"\n")