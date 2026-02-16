import pandas as pd
from datetime import datetime

NOM_FITXER = "100_Batches_IndPenSim_V3.csv"
NOM_SORTIDA = "analisi_dataset.txt"

print(f"📁 Processant {NOM_FITXER}...")

try:
    if NOM_FITXER.endswith('.csv'):
        df = pd.read_csv(NOM_FITXER)
    elif NOM_FITXER.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(NOM_FITXER)
    else:
        df = pd.read_csv(NOM_FITXER, sep=None, engine='python')
    
    with open(NOM_SORTIDA, 'w', encoding='utf-8') as f:
        
        f.write("=" * 80 + "\n")
        f.write(f"ANÀLISI DE DATASET\n")
        f.write(f"Generat: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fitxer original: {NOM_FITXER}\n")
        f.write("=" * 80 + "\n\n")
        
        # ============================================
        # 1️⃣ INFORMACIÓ GENERAL
        # ============================================
        f.write("📊 INFORMACIÓ GENERAL:\n")
        f.write("-" * 40 + "\n")
        f.write(f"▶ Total files: {df.shape[0]}\n")
        f.write(f"▶ Total columnes: {df.shape[1]}\n")
        f.write(f"▶ Memòria utilitzada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n")
        
        # ============================================
        # 2️⃣ LLISTAT COMPLET DE COLUMNES
        # ============================================
        f.write("📋 LLISTAT COMPLET DE COLUMNES:\n")
        f.write("-" * 40 + "\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:3}. {col}\n")
        f.write(f"\n▶ TOTAL COLUMNES: {len(df.columns)}\n\n")
        
        # ============================================
        # 3️⃣ PRIMERES 5 FILES
        # ============================================
        f.write("👁️ PRIMERES 5 FILES:\n")
        f.write("-" * 40 + "\n")
        
        # Convertir les primeres 5 files a string
        head_str = df.head().to_string()
        f.write(head_str + "\n\n")
        
        # ============================================
        # 4️⃣ TIPUS DE DADES
        # ============================================
        f.write("🔤 TIPUS DE DADES (primeres 10 columnes):\n")
        f.write("-" * 40 + "\n")
        for col in df.columns[:10]:
            f.write(f"  {col}: {df[col].dtype}\n")
        
        # Si hi ha més de 10 columnes, indicar-ho
        if len(df.columns) > 10:
            f.write(f"  ... i {len(df.columns) - 10} columnes més\n")
        
        # ============================================
        # 5️⃣ ESTADÍSTIQUES BÀSIQUES (opcional)
        # ============================================
        f.write("\n📈 ESTADÍSTIQUES BÀSIQUES (columnes numèriques):\n")
        f.write("-" * 40 + "\n")
        
        # Seleccionar columnes numèriques
        num_cols = df.select_dtypes(include=['number']).columns[:5]  # Primeres 5 numèriques
        
        if len(num_cols) > 0:
            for col in num_cols:
                f.write(f"\n{col}:\n")
                f.write(f"  Min: {df[col].min():.2f}\n")
                f.write(f"  Max: {df[col].max():.2f}\n")
                f.write(f"  Mitjana: {df[col].mean():.2f}\n")
                f.write(f"  Mediana: {df[col].median():.2f}\n")
        else:
            f.write("  No hi ha columnes numèriques\n")
        
        # PEU DE PÀGINA
        f.write("\n" + "=" * 80 + "\n")
        f.write("✅ ANÀLISI COMPLETADA\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Fitxer '{NOM_SORTIDA}' generat correctament!")
    print(f"📁 Pots obrir-lo amb qualsevol editor de text")

except FileNotFoundError:
    print(f"\n❌ ERROR: No es troba el fitxer '{NOM_FITXER}'")
    print("📌 Comprova que:")
    print("   - El fitxer està a la mateixa carpeta que aquest script")
    print("   - El nom està escrit correctament (majúscules/minúscules)")
    print("   - L'extensió del fitxer és correcta")
except Exception as e:
    print(f"\n❌ ERROR INESPERAT: {e}")