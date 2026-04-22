import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ================= 1. FUNGSI UTAMA =================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def add_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

# ================= 2. LOAD & PREPROCESS DATA =================
data_path = "data/mitdb"
records = ['100', '200', '201', '203', '207']
X_clean, y = [], []
normal_label = ['N']

for rec in records:
    record = wfdb.rdrecord(f"{data_path}/{rec}")
    annotation = wfdb.rdann(f"{data_path}/{rec}", 'atr')
    signal = record.p_signal[:, 0]
    fs = record.fs

    # Filter & Normalisasi
    filtered = bandpass_filter(signal, 0.5, 40, fs)
    norm = filtered / np.max(np.abs(filtered))

    # Deteksi Puncak R
    peaks, _ = find_peaks(norm, distance=int(0.6*fs),
                         height=np.mean(norm)+0.5*np.std(norm))

    pre_r, post_r = int(0.2*fs), int(0.4*fs)
    for r in peaks:
        if r-pre_r >= 0 and r+post_r < len(filtered):
            beat = norm[r-pre_r:r+post_r]
            idx = np.argmin(np.abs(annotation.sample - r))
            label = annotation.symbol[idx]
            X_clean.append(beat)
            y.append(0 if label in normal_label else 1)

X_clean = np.array(X_clean)
y = np.array(y)

# Simulasi Sinyal Nirkabel Berderau (SNR 20dB)
X_noisy = np.array([add_awgn(b, 20) for b in X_clean])

# Split Data (80% Train, 20% Test)
Xtr, Xte_clean, ytr, yte = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)
_, Xte_noisy, _, _ = train_test_split(X_noisy, y, test_size=0.2, random_state=42, stratify=y)

# ================= 3. TRAINING MODEL =================
# CNN Model
Xtr_cnn = Xtr[..., np.newaxis]
cnn_model = Sequential([
    Input(shape=Xtr_cnn.shape[1:]),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(Xtr_cnn, ytr, epochs=5, batch_size=64, verbose=0)

# SVM Model
svm_model = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True))])
svm_model.fit(Xtr, ytr)

# ================= 4. ANALISIS KOMPARATIF & SIMULASI =================
def evaluate_latency_and_accuracy(model, data_test, labels, is_cnn=False):
    if is_cnn: data_test = data_test[..., np.newaxis]
    
    start_time = time.time()
    preds = (model.predict(data_test) > 0.5).astype(int) if is_cnn else model.predict(data_test)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / len(data_test) * 1000 # dalam ms
    acc = accuracy_score(labels, preds)
    return acc, avg_latency

# Menjalankan Evaluasi
print(f"{'Metode':<15} | {'Kondisi':<10} | {'Akurasi':<10} | {'Latency/Beat'}")
print("-" * 55)

for name, model, is_cnn in [("CNN", cnn_model, True), ("SVM", svm_model, False)]:
    # Uji Sinyal Bersih
    acc_c, lat_c = evaluate_latency_and_accuracy(model, Xte_clean, yte, is_cnn)
    print(f"{name:<15} | Clean      | {acc_c*100:>8.2f}% | {lat_c:>8.4f} ms")
    
    # Uji Sinyal Nirkabel (Noisy)
    acc_n, lat_n = evaluate_latency_and_accuracy(model, Xte_noisy, yte, is_cnn)
    print(f"{name:<15} | Noisy 20dB | {acc_n*100:>8.2f}% | {lat_n:>8.4f} ms")