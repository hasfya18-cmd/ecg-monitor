# Train & Save Models untuk Aplikasi ECG Monitoring
# Jalankan script ini SEKALI sebelum menjalankan app.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import joblib
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import wfdb

# ============================================================
# 1. FUNGSI UTILITAS
# ============================================================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

# ============================================================
# 2. LOAD DATA
# ============================================================
data_path = "data/mitdb"
cache_path = "data/mitdb_cache.npz"
models_dir = "models"

os.makedirs(models_dir, exist_ok=True)

if os.path.exists(cache_path):
    print("Loading dataset dari cache...")
    cache = np.load(cache_path)
    X, y = cache['X'], cache['y']
    print("Cache loaded!")
else:
    records = sorted(set(
        f.replace('.dat', '') for f in os.listdir(data_path) if f.endswith('.dat')
    ))
    print(f"Found {len(records)} records: {records}")

    X, y = [], []
    normal_label = ['N']

    for rec in records:
        try:
            record = wfdb.rdrecord(f"{data_path}/{rec}")
            ann = wfdb.rdann(f"{data_path}/{rec}", 'atr')
            signal = record.p_signal[:, 0]
            fs = record.fs
            filtered = bandpass_filter(signal, 0.5, 40, fs)
            norm = filtered / np.max(np.abs(filtered))
            peaks, _ = find_peaks(
                norm,
                distance=int(0.6 * fs),
                height=np.mean(norm) + 0.5 * np.std(norm)
            )
            pre, post = int(0.2 * fs), int(0.4 * fs)
            count = 0
            for r in peaks:
                if r - pre >= 0 and r + post < len(norm):
                    beat = norm[r - pre:r + post]
                    idx = np.argmin(np.abs(ann.sample - r))
                    label = ann.symbol[idx]
                    X.append(beat)
                    y.append(0 if label in normal_label else 1)
                    count += 1
            print(f"  Record {rec}: {count} beats extracted")
        except Exception as e:
            print(f"  Skipping record {rec}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)
    np.savez(cache_path, X=X, y=y)
    print("Cache saved!")

print(f"\nDataset: {len(y)} beats (Normal: {np.sum(y==0)}, Abnormal: {np.sum(y==1)})")

# ============================================================
# 3. SPLIT DATA
# ============================================================
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 4. TRAIN CNN
# ============================================================
print("\n--- Training CNN ---")
Xtr_cnn = Xtr[..., np.newaxis]

cnn = Sequential([
    Input(shape=Xtr_cnn.shape[1:]),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

cnn.fit(Xtr_cnn, ytr, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
cnn.save(os.path.join(models_dir, "cnn_model.keras"))
print("CNN model saved!")

# ============================================================
# 5. TRAIN SVM
# ============================================================
print("\n--- Training SVM ---")
svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])
svm.fit(Xtr, ytr)
joblib.dump(svm, os.path.join(models_dir, "svm_model.pkl"))
print("SVM model saved!")

# ============================================================
# 6. SAVE TEST DATA (untuk streaming demo)
# ============================================================
# Simpan sample beats untuk demo streaming
np.savez(
    os.path.join(models_dir, "demo_data.npz"),
    X_test=Xte,
    y_test=yte,
    X_all=X,
    y_all=y
)
print("Demo data saved!")

print("\n[OK] Semua model dan data tersimpan di folder 'models/'")
print("   - models/cnn_model.keras")
print("   - models/svm_model.pkl")
print("   - models/demo_data.npz")
print("\nSekarang jalankan: python app.py")
