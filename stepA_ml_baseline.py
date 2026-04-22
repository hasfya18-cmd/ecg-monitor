# STEP A: MACHINE LEARNING BASELINE (SVM)

import os
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# LOAD & PREPROCESS 

DATA_PATH = "data/mitdb"
RECORD = "100"

record = wfdb.rdrecord(os.path.join(DATA_PATH, RECORD))
signal = record.p_signal[:, 0]
fs = record.fs

def bandpass(sig):
    nyq = fs / 2
    b, a = butter(4, [0.5/nyq, 40/nyq], btype='band')
    return filtfilt(b, a, sig)

signal = bandpass(signal)
signal = (signal - np.mean(signal)) / np.std(signal)


# R-PEAK & SEGMENTATION

peaks, _ = find_peaks(signal, distance=0.6*fs, height=np.mean(signal))
segments = []

win = int(0.6 * fs)
for p in peaks:
    s = p - win//2
    e = p + win//2
    if s > 0 and e < len(signal):
        segments.append(signal[s:e])

segments = np.array(segments)
print("Total segments:", segments.shape)


# FEATURE EXTRACTION

features = []
labels = []

for seg in segments:
    features.append([
        np.mean(seg),
        np.std(seg),
        np.max(seg),
        np.min(seg)
    ])
    labels.append(1)  # Normal (dummy, nanti bisa pakai annotation)

X = np.array(features)
y = np.array(labels)

# TRAINING ML (SVM)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Unique labels:", np.unique(y_train))
print("Label counts:", np.bincount(y_train))

model = SVC(kernel='rbf')
model.fit(X_train, y_train)


# EVALUATION

y_pred = model.predict(X_test)

print("Accuracy ML:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
