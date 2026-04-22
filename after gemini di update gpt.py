# Analisis Komparatif & Simulasi Klasifikasi EKG Nirkabel
# CNN vs SVM MIT-BIH Arrhythmia Database

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

# 1. FUNGSI
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def add_awgn(signal, snr_db):
    power = np.mean(signal**2)
    snr = 10**(snr_db/10)
    noise_power = power / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

# 2. LOAD DATA
data_path = "data/mitdb"
records = ['100', '200', '201', '203', '207']

X, y = [], []
normal_label = ['N']

print("MULTI-RECORD ECG PROCESSING")

for rec in records:
    record = wfdb.rdrecord(f"{data_path}/{rec}")
    ann = wfdb.rdann(f"{data_path}/{rec}", 'atr')

    signal = record.p_signal[:,0]
    fs = record.fs

    filtered = bandpass_filter(signal, 0.5, 40, fs)
    norm = filtered / np.max(np.abs(filtered))

    peaks, _ = find_peaks(
        norm,
        distance=int(0.6*fs),
        height=np.mean(norm)+0.5*np.std(norm)
    )

    pre, post = int(0.2*fs), int(0.4*fs)

    for r in peaks:
        if r-pre >= 0 and r+post < len(norm):
            beat = norm[r-pre:r+post]
            idx = np.argmin(np.abs(ann.sample - r))
            label = ann.symbol[idx]
            X.append(beat)
            y.append(0 if label in normal_label else 1)

X = np.array(X)
y = np.array(y)

print("\nDATASET SUMMARY")
print("TOTAL BEAT:", len(y))
print("NORMAL:", np.sum(y==0))
print("ABNORMAL:", np.sum(y==1))

# 3. SIMULASI NIRKABEL
X_clean = X
X_noisy = np.array([add_awgn(b, 20) for b in X_clean])

# Visualisasi sinyal
plt.figure(figsize=(10,4))
plt.plot(X_clean[0], label='Clean ECG')
plt.plot(X_noisy[0], label='Noisy ECG (20 dB)', alpha=0.7)
plt.title("ECG Beat: Clean vs Noisy")
plt.legend()
plt.grid()
plt.show()

# 4. SPLIT DATA
Xtr_c, Xte_c, ytr, yte = train_test_split(
    X_clean, y, test_size=0.2, stratify=y, random_state=42
)

_, Xte_n, _, _ = train_test_split(
    X_noisy, y, test_size=0.2, stratify=y, random_state=42
)

# 5. CNN MODEL
Xtr_cnn = Xtr_c[..., np.newaxis]

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

cnn.fit(
    Xtr_cnn, ytr,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# 6. SVM MODEL
svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])
svm.fit(Xtr_c, ytr)

# 7. EVALUASI
def evaluate(model, Xtest, ytest, cnn_mode=False):
    if cnn_mode:
        Xtest = Xtest[..., np.newaxis]
        start = time.time()
        ypred = (model.predict(Xtest) > 0.5).astype(int)
    else:
        start = time.time()
        ypred = model.predict(Xtest)
    latency = (time.time() - start) / len(ytest) * 1000
    acc = accuracy_score(ytest, ypred)
    return acc, latency

print("\nMETODE | KONDISI | AKURASI | LATENCY (ms/beat)")
print("-"*50)

for name, model, iscnn in [
    ("CNN", cnn, True),
    ("SVM", svm, False)
]:
    acc_c, lat_c = evaluate(model, Xte_c, yte, iscnn)
    acc_n, lat_n = evaluate(model, Xte_n, yte, iscnn)

    print(f"{name:4} | Clean | {acc_c*100:6.2f}% | {lat_c:6.3f}")
    print(f"{name:4} | Noisy | {acc_n*100:6.2f}% | {lat_n:6.3f}")
    print("-"*50)

print("\nClassification Report (CNN – Clean ECG)")
print(classification_report(yte, (cnn.predict(Xte_c[...,np.newaxis])>0.5).astype(int)))
