# ================= IMPORT =================
import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
# =========================================


# ================= FILTER =================
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)
# =========================================


# ================= NOISE =================
def add_awgn(signal, snr_db):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise
# =========================================


print("MULTI-RECORD ECG PROCESSING")

# ================= STEP 1–4 =================
data_path = "data/mitdb"
records = ['100', '200', '201', '203', '207']
X, y = [], []
normal_label = ['N']

for rec in records:
    record = wfdb.rdrecord(f"{data_path}/{rec}")
    annotation = wfdb.rdann(f"{data_path}/{rec}", 'atr')

    signal = record.p_signal[:, 0]
    fs = record.fs

    filtered = bandpass_filter(signal, 0.5, 40, fs)
    norm = filtered / np.max(np.abs(filtered))

    peaks, _ = find_peaks(norm, distance=int(0.6*fs),
                          height=np.mean(norm)+0.5*np.std(norm))

    pre_r, post_r = int(0.2*fs), int(0.4*fs)

    for r in peaks:
        if r-pre_r >= 0 and r+post_r < len(filtered):
            beat = filtered[r-pre_r:r+post_r]
            idx = np.argmin(np.abs(annotation.sample - r))
            label = annotation.symbol[idx]

            X.append(beat)
            y.append(0 if label in normal_label else 1)

X = np.array(X)
y = np.array(y)

print("\nDATASET SUMMARY")
print("TOTAL:", len(y), "| NORMAL:", np.sum(y==0), "| ABNORMAL:", np.sum(y==1))

# ================= STEP 5 =================
X = X / np.max(np.abs(X), axis=1, keepdims=True)
X_cnn = X[..., np.newaxis]

# ================= STEP 6A: CNN CLEAN =================
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    X_cnn, y, test_size=0.2, random_state=42, stratify=y)

cnn_clean = Sequential([
    Input(shape=Xtr_c.shape[1:]),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_clean.compile(
    optimizer=Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history_clean = cnn_clean.fit(
    Xtr_c, ytr_c,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

ypred_c = (cnn_clean.predict(Xte_c) > 0.5).astype(int)
yprob_c = cnn_clean.predict(Xte_c)

print("\nCNN CLEAN REPORT")
print(classification_report(yte_c, ypred_c))

# ================= STEP 7: VISUALISASI =================
plt.figure()
plt.plot(history_clean.history['accuracy'], label='Train')
plt.plot(history_clean.history['val_accuracy'], label='Validation')
plt.title('CNN Accuracy (Clean ECG)')
plt.legend(); plt.grid(); plt.show()

plt.figure()
plt.plot(history_clean.history['loss'], label='Train')
plt.plot(history_clean.history['val_loss'], label='Validation')
plt.title('CNN Loss (Clean ECG)')
plt.legend(); plt.grid(); plt.show()

cm = confusion_matrix(yte_c, ypred_c)
plt.figure()
plt.imshow(cm); plt.title('Confusion Matrix (Clean ECG)')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.colorbar(); plt.show()

fpr, tpr, _ = roc_curve(yte_c, yprob_c)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
plt.plot([0,1],[0,1],'--')
plt.title('ROC Curve (Clean ECG)')
plt.legend(); plt.grid(); plt.show()

# ================= STEP 8: SVM =================
X_svm = X.reshape(X.shape[0], -1)
Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
    X_svm, y, test_size=0.2, random_state=42, stratify=y)

svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])

svm.fit(Xtr_s, ytr_s)
ypred_svm = svm.predict(Xte_s)

print("\nSVM REPORT")
print(classification_report(yte_s, ypred_svm))
