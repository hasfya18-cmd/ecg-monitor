import wfdb
import numpy as np
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

X = []
y = []

# Label Normal vs Abnormal
normal_labels = ['N']

for r in peaks:
    if r - pre_r >= 0 and r + post_r < len(signal):
        beat = signal[r - pre_r : r + post_r]
        
        # cari anotasi terdekat
        idx = np.argmin(np.abs(annotation.sample - r))
        label = annotation.symbol[idx]

        X.append(beat)
        y.append(0 if label in normal_labels else 1)

X = np.array(X)
y = np.array(y)

print("Total data:", X.shape)

# ===== Train-Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== SVM Baseline =====
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ===== Evaluasi =====
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
