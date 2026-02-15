#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==================================================================================
FASE 3: MODEL AVANÇAT - LSTM (Long Short-Term Memory)
Xarxa neuronal recurrent per capturar dinàmiques temporals
==================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("ERROR: TensorFlow no està instal·lat")
    print("Instal·la amb: pip install tensorflow")
    exit(1)

# Configuració
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "fases" / "3_modelitzacio_predictiva" / "outputs"

print("=" * 80)
print("FASE 3 - MODEL AVANÇAT: LSTM")
print("=" * 80)

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"\n✓ GPU detectada: {len(gpus)} dispositiu(s)")
else:
    print(f"\n⚠ GPU no disponible. Usant CPU (més lent)")

# Carregar dades
print("\n[1/9] Carregant datasets...")

df_train = pd.read_csv(OUTPUT_DIR / "train_data.csv")
df_test = pd.read_csv(OUTPUT_DIR / "test_data.csv")

print(f"   Train: {len(df_train):,} mostres")
print(f"   Test:  {len(df_test):,} mostres")

feature_cols = [col for col in df_train.columns 
                if col not in ['batch_id', 'penicillin', 'time']]
target_col = 'penicillin'

print(f"   Features: {len(feature_cols)}")

# =============================================================================
# PREPARACIÓ SEQUENCES TEMPORALS
# =============================================================================
print("\n[2/9] Creant sequences temporals...")

def create_sequences(df, feature_cols, target_col, seq_length=20):
    """Crear sequences per LSTM"""
    sequences_X = []
    sequences_y = []
    
    # Processar per batch
    for batch_id in df['batch_id'].unique():
        batch_data = df[df['batch_id'] == batch_id].sort_values('time')
        
        if len(batch_data) < seq_length + 1:
            continue
        
        X_batch = batch_data[feature_cols].values
        y_batch = batch_data[target_col].values
        
        # Crear sequences
        for i in range(len(batch_data) - seq_length):
            sequences_X.append(X_batch[i:i+seq_length])
            sequences_y.append(y_batch[i+seq_length])
    
    return np.array(sequences_X), np.array(sequences_y)

SEQ_LENGTH = 20  # Window de 20 time steps

print(f"   Sequence length: {SEQ_LENGTH}")
print(f"   Creant sequences train...")

X_train_seq, y_train_seq = create_sequences(df_train, feature_cols, target_col, SEQ_LENGTH)

print(f"   Creant sequences test...")
X_test_seq, y_test_seq = create_sequences(df_test, feature_cols, target_col, SEQ_LENGTH)

print(f"\n   X_train shape: {X_train_seq.shape}")
print(f"   y_train shape: {y_train_seq.shape}")
print(f"   X_test shape:  {X_test_seq.shape}")
print(f"   y_test shape:  {y_test_seq.shape}")

# =============================================================================
# NORMALITZACIÓ
# =============================================================================
print("\n[3/9] Normalitzant features...")

# Reshape per normalitzar
n_samples_train = X_train_seq.shape[0]
n_samples_test = X_test_seq.shape[0]
n_features = X_train_seq.shape[2]

X_train_flat = X_train_seq.reshape(-1, n_features)
X_test_flat = X_test_seq.reshape(-1, n_features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# Reshape back
X_train_scaled = X_train_scaled.reshape(n_samples_train, SEQ_LENGTH, n_features)
X_test_scaled = X_test_scaled.reshape(n_samples_test, SEQ_LENGTH, n_features)

print(f"   Scaled train: {X_train_scaled.shape}")
print(f"   Scaled test:  {X_test_scaled.shape}")

# =============================================================================
# CONSTRUCCIÓ MODEL LSTM
# =============================================================================
print("\n[4/9] Construint arquitectura LSTM...")

model = keras.Sequential([
    # Primera capa LSTM
    layers.LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)),
    layers.Dropout(0.2),
    
    # Segona capa LSTM
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    
    # Capes denses
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Output: penicillin
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"\n   Arquitectura del model:")
model.summary()

# =============================================================================
# ENTRENAMENT
# =============================================================================
print("\n[5/9] Entrenant model LSTM...")

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Training
EPOCHS = 50
BATCH_SIZE = 128

print(f"   Epochs: {EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Validation split: 20%")

history = model.fit(
    X_train_scaled, y_train_seq,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(f"\n   ✓ Entrenament completat!")
print(f"   Epochs executats: {len(history.history['loss'])}")

# =============================================================================
# PREDICCIONS
# =============================================================================
print("\n[6/9] Generant prediccions...")

y_train_pred = model.predict(X_train_scaled, verbose=0).flatten()
y_test_pred = model.predict(X_test_scaled, verbose=0).flatten()

print(f"   Prediccions train: {len(y_train_pred):,}")
print(f"   Prediccions test:  {len(y_test_pred):,}")

# =============================================================================
# AVALUACIÓ
# =============================================================================
print("\n[7/9] Avaluant model...")

# Mètriques train
mse_train = mean_squared_error(y_train_seq, y_train_pred)
mae_train = mean_absolute_error(y_train_seq, y_train_pred)
r2_train = r2_score(y_train_seq, y_train_pred)
rmse_train = np.sqrt(mse_train)

# Mètriques test
mse_test = mean_squared_error(y_test_seq, y_test_pred)
mae_test = mean_absolute_error(y_test_seq, y_test_pred)
r2_test = r2_score(y_test_seq, y_test_pred)
rmse_test = np.sqrt(mse_test)

print(f"\n   === TRAIN SET ===")
print(f"   MSE:  {mse_train:.4f} (g/L)²")
print(f"   RMSE: {rmse_train:.4f} g/L")
print(f"   MAE:  {mae_train:.4f} g/L")
print(f"   R²:   {r2_train:.4f}")

print(f"\n   === TEST SET ===")
print(f"   MSE:  {mse_test:.4f} (g/L)²")
print(f"   RMSE: {rmse_test:.4f} g/L")
print(f"   MAE:  {mae_test:.4f} g/L")
print(f"   R²:   {r2_test:.4f}")

# =============================================================================
# VISUALITZACIONS
# =============================================================================
print("\n[8/9] Generant visualitzacions...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

# Plot 1: Learning curves
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontsize=11)
ax1.set_title('Learning Curves - Loss', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: MAE curves
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(history.history['mae'], 'b-', label='Train MAE', linewidth=2)
ax2.plot(history.history['val_mae'], 'r-', label='Val MAE', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('MAE (g/L)', fontsize=11)
ax2.set_title('Learning Curves - MAE', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Scatter Train
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_train_seq, y_train_pred, alpha=0.3, s=10, color='blue', edgecolor='none')
ax3.plot([y_train_seq.min(), y_train_seq.max()], 
         [y_train_seq.min(), y_train_seq.max()], 'r--', lw=2)
ax3.set_xlabel('Real (g/L)', fontsize=10)
ax3.set_ylabel('Predicció (g/L)', fontsize=10)
ax3.set_title(f'LSTM - TRAIN\nR² = {r2_train:.4f}', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Scatter Test
ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(y_test_seq, y_test_pred, alpha=0.5, s=20, color='red', edgecolor='black', linewidth=0.5)
ax4.plot([y_test_seq.min(), y_test_seq.max()], 
         [y_test_seq.min(), y_test_seq.max()], 'r--', lw=2)
ax4.set_xlabel('Real (g/L)', fontsize=10)
ax4.set_ylabel('Predicció (g/L)', fontsize=10)
ax4.set_title(f'LSTM - TEST\nR² = {r2_test:.4f}', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals
ax5 = fig.add_subplot(gs[1, 2])
residuals_test = y_test_seq - y_test_pred
ax5.scatter(y_test_pred, residuals_test, alpha=0.5, s=20, color='purple', edgecolor='black', linewidth=0.5)
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicció (g/L)', fontsize=10)
ax5.set_ylabel('Residuals (g/L)', fontsize=10)
ax5.set_title('Residuals LSTM - TEST', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Distribució residuals
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(residuals_test, bins=50, color='purple', alpha=0.7, edgecolor='black')
ax6.axvline(x=0, color='red', linestyle='--', lw=2)
ax6.set_xlabel('Residuals (g/L)', fontsize=10)
ax6.set_ylabel('Freqüència', fontsize=10)
ax6.set_title('Distribució Residuals', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Mètriques
ax7 = fig.add_subplot(gs[2, 1])
metrics = ['R²', 'MAE', 'RMSE']
train_vals = [r2_train, mae_train, rmse_train]
test_vals = [r2_test, mae_test, rmse_test]

x = np.arange(len(metrics))
width = 0.35

ax7.bar(x - width/2, train_vals, width, label='Train', color='lightblue', edgecolor='black')
ax7.bar(x + width/2, test_vals, width, label='Test', color='lightcoral', edgecolor='black')
ax7.set_ylabel('Valor', fontsize=10)
ax7.set_title('Mètriques LSTM', fontsize=11, fontweight='bold')
ax7.set_xticks(x)
ax7.set_xticklabels(metrics)
ax7.legend()
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Predicció temporal (batch 95)
ax8 = fig.add_subplot(gs[2, 2])

# Reconstruir prediccions per batch 95
batch_95_test = df_test[df_test['batch_id'] == 95].copy()
if len(batch_95_test) >= SEQ_LENGTH:
    # Crear sequences per aquest batch
    X_b95, y_b95 = create_sequences(
        batch_95_test, feature_cols, target_col, SEQ_LENGTH
    )
    
    # Normalitzar
    X_b95_flat = X_b95.reshape(-1, n_features)
    X_b95_scaled = scaler.transform(X_b95_flat).reshape(-1, SEQ_LENGTH, n_features)
    
    # Predir
    y_b95_pred = model.predict(X_b95_scaled, verbose=0).flatten()
    
    # Plot
    time_b95 = batch_95_test['time'].values[SEQ_LENGTH:]
    ax8.plot(time_b95, y_b95, 'o-', color='black', label='Real', linewidth=2, markersize=4)
    ax8.plot(time_b95, y_b95_pred, 's-', color='purple', alpha=0.7, 
            label='LSTM', linewidth=2, markersize=4)
    ax8.set_xlabel('Temps (h)', fontsize=10)
    ax8.set_ylabel('Penicil·lina (g/L)', fontsize=10)
    ax8.set_title('Batch 95 - Predicció Temporal', fontsize=11, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

plt.suptitle('LSTM - Resultats', fontsize=16, fontweight='bold')

results_path = OUTPUT_DIR / "04_lstm_results.png"
plt.savefig(results_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"   OK: {results_path.name}")

# =============================================================================
# GUARDAR MODEL I RESULTATS
# =============================================================================
print("\n[9/9] Guardant model i resultats...")

# Guardar model
model_path = OUTPUT_DIR / "04_lstm_model.h5"
model.save(model_path)
print(f"   Model: {model_path.name}")

# Guardar scaler
import joblib
scaler_path = OUTPUT_DIR / "04_lstm_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"   Scaler: {scaler_path.name}")

# Guardar history
history_df = pd.DataFrame(history.history)
history_path = OUTPUT_DIR / "04_lstm_history.csv"
history_df.to_csv(history_path, index=False)
print(f"   History: {history_path.name}")

# Guardar mètriques
metrics_df = pd.DataFrame({
    'Metric': ['MSE', 'RMSE', 'MAE', 'R²'],
    'Train': [mse_train, rmse_train, mae_train, r2_train],
    'Test': [mse_test, rmse_test, mae_test, r2_test]
})

metrics_path = OUTPUT_DIR / "04_lstm_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"   Mètriques: {metrics_path.name}")

# =============================================================================
# RESUM
# =============================================================================
print("\n" + "=" * 80)
print("LSTM MODEL COMPLETAT")
print("=" * 80)
print(f"\nARQUITECTURA:")
print(f"   Input: ({SEQ_LENGTH}, {n_features})")
print(f"   LSTM layers: 128 → 64 units")
print(f"   Dense layers: 32 → 16 → 1")
print(f"   Total params: {model.count_params():,}")

print(f"\nRESULTATS:")
print(f"   Train - R²: {r2_train:.4f}, MAE: {mae_train:.4f} g/L")
print(f"   Test  - R²: {r2_test:.4f}, MAE: {mae_test:.4f} g/L")

print(f"\nFitxers generats:")
print(f"   • {model_path.name}")
print(f"   • {results_path.name}")
print(f"   • {history_path.name}")
print(f"   • {metrics_path.name}")

print(f"\nSegüent pas: python 05_fault_detection.py")
print("=" * 80 + "\n")