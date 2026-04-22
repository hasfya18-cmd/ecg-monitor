# PIPELINE ANALISIS EKG STEP 1–6

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# STEP 1 — LOAD & VALIDASI DATASET

print("STEP 1: Load & Validasi Dataset")

data_path = "data/mitdb"
record = wfdb.rdrecord(f"{data_path}/100")
annotation = wfdb.rdann(f"{data_path}/100", 'atr')

signal = record.p_signal[:, 0]
fs = record.fs

plt.figure(figsize=(10,4))
plt.plot(signal)
plt.title("Sinyal EKG Mentah")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.show()

# ===============================
# STEP 2 — PREPROCESSING
# ===============================
print("STEP 2: Preprocessing")

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

filtered_signal = bandpass_filter(signal, 0.5, 40, fs)

plt.figure(figsize=(10,4))
plt.plot(filtered_signal)
plt.title("Sinyal EKG Setelah Filtering")
plt.show()

# ===============================
# STEP 3 — DETEKSI R-PEAK
# ===============================
print("STEP 3: Deteksi R-Peak")

peaks, _ = find_peaks(filtered_signal,
                      distance=0.6*fs,
                      height=np.mean(filtered_signal))

plt.figure(figsize=(10,4))
plt.plot(filtered_signal)
plt.plot(peaks, filtered_signal[peaks], "ro")
plt.title("Deteksi R-Peak")
plt.show()

# ===============================
# STEP 4 — SEGMENTASI BEAT
# ===============================
print("STEP 4: Segmentasi Beat")

pre_r = int(0.2 * fs)
post_r = int(0.4 * fs)

X, y = [], []
normal_label = ['N']

for r in peaks:
    if r - pre_r >= 0 and r + post_r < len(filtered_signal):
        beat = filtered_signal[r - pre_r : r + post_r]

        idx = np.argmin(np.abs(annotation.sample - r))
        label = annotation.symbol[idx]

        X.append(beat)
        y.append(0 if label in normal_label else 1)

X = np.array(X)
y = np.array(y)

print("Total beat:", X.shape[0])
print("Panjang tiap beat:", X.shape[1])

# ===============================
# STEP 5 — MACHINE LEARNING (SVM)
# ===============================
print("STEP 5: Machine Learning (SVM)")

X_ml = X
y_ml = y

X_train, X_test, y_train, y_test = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

y_pred_ml = svm.predict(X_test)
print("Hasil Klasifikasi SVM:")
print(classification_report(y_test, y_pred_ml))

# ===============================
# STEP 6 — DEEP LEARNING (CNN 1D)
# ===============================
print("STEP 6: Deep Learning (CNN 1D)")

X_dl = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X_dl, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Conv1D(32, 5, activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

y_pred_dl = (model.predict(X_test) > 0.5).astype(int)
print("Hasil Klasifikasi CNN:")
print(classification_report(y_test, y_pred_dl))
