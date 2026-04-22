import wfdb
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# ===== Load Data =====
data_path = "data/mitdb"
record = wfdb.rdrecord(f"{data_path}/100")
annotation = wfdb.rdann(f"{data_path}/100", 'atr')

signal = record.p_signal[:, 0]
fs = record.fs

# ===== R-peak detection =====
peaks, _ = find_peaks(signal, distance=0.6*fs, height=np.mean(signal)+0.5)

# ===== Segmentasi =====
pre_r = int(0.2 * fs)
post_r = int(0.4 * fs)

X, y = [], []
normal_labels = ['N']

for r in peaks:
    if r - pre_r >= 0 and r + post_r < len(signal):
        beat = signal[r - pre_r : r + post_r]
        idx = np.argmin(np.abs(annotation.sample - r))
        label = annotation.symbol[idx]
        X.append(beat)
        y.append(0 if label in normal_labels else 1)

X = np.array(X)
y = np.array(y)

# ===== Reshape for CNN =====
X = X[..., np.newaxis]  # (samples, timesteps, channels)

# ===== Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== CNN Model =====
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=5, activation='relu'),
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

# ===== Train =====
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# ===== Evaluate =====
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))
