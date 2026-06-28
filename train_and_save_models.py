# Train & Save Models — Large Scale 12-Lead ECG Database
# Uses 100 arrhythmia + 100 normal patients from PhysioNet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import json, joblib
from scipy.signal import butter, filtfilt, find_peaks, resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import wfdb

# ============================================================
# CONFIG
# ============================================================
ARR_DIR = "data/large_ecg"           # 100 arrhythmia patients
NOR_DIR = "data/large_ecg_normal"    # 100 normal patients
MODELS_DIR = "models"
BEAT_SIZE = 187   # Output beat length (resampled)
FS_TARGET = 360   # Target sample rate equivalent

os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# UTILITY
# ============================================================
def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def extract_beats_from_record(filepath, fs=500):
    """Extract individual heartbeats from a WFDB record, using Lead II."""
    try:
        record = wfdb.rdrecord(filepath)
        # Use Lead II (index 1) if available, else Lead I (index 0)
        lead_idx = 1 if record.p_signal.shape[1] > 1 else 0
        signal = record.p_signal[:, lead_idx].astype(float)
        fs_actual = record.fs

        # Filter
        filtered = bandpass_filter(signal, fs=fs_actual)
        # Normalize
        max_val = np.max(np.abs(filtered))
        if max_val > 0:
            norm = filtered / max_val
        else:
            return np.array([]), fs_actual

        # Detect R-peaks
        pre_samples = int(0.25 * fs_actual)  # 0.25s before R
        post_samples = int(0.45 * fs_actual)  # 0.45s after R
        raw_beat_len = pre_samples + post_samples

        peaks, _ = find_peaks(
            norm,
            distance=int(0.5 * fs_actual),
            height=np.mean(norm) + 0.3 * np.std(norm)
        )

        beats = []
        for r in peaks:
            if r - pre_samples >= 0 and r + post_samples < len(norm):
                beat = norm[r - pre_samples:r + post_samples]
                # Resample to BEAT_SIZE
                beat_resampled = resample(beat, BEAT_SIZE)
                beats.append(beat_resampled)

        return np.array(beats) if beats else np.array([]), fs_actual
    except Exception as e:
        return np.array([]), 500

# ============================================================
# LOAD & PROCESS DATA
# ============================================================
print("=" * 60)
print("  Loading 12-Lead ECG Data (Large Scale Database)")
print("=" * 60)

# Load arrhythmia patient metadata
with open(os.path.join(ARR_DIR, "patients_meta.json")) as f:
    arr_meta = json.load(f)

# --- Construct Normal Beat Template from 12-Lead Normal records ---
print("\nConstructing Normal Beat Template from 12-Lead Normal records...")
normal_records_for_template = sorted(set(f.replace('.hea','') for f in os.listdir(NOR_DIR) if f.endswith('.hea')))
template_beats = []
for r_id in normal_records_for_template[:15]:  # Use first 15 normal patients to construct a robust template
    filepath = os.path.join(NOR_DIR, r_id)
    beats, fs = extract_beats_from_record(filepath)
    if len(beats) > 0:
        template_beats.extend(beats)
if len(template_beats) > 0:
    normal_template = np.mean(template_beats, axis=0)
    print(f"Normal template constructed successfully from {len(template_beats)} normal beats.")
else:
    normal_template = np.zeros(BEAT_SIZE)
    print("WARNING: Could not construct normal template, using zero template.")

per_patient = {}  # For dashboard demo data (max 100 patients, all arrhythmia patients)
X_all, y_all = [], []

# --- Process arrhythmia patients ---
print(f"\nProcessing {len(arr_meta)} arrhythmia patients with dynamic labeling...")
for i, pm in enumerate(arr_meta):
    rec = pm['record_id']
    filepath = os.path.join(ARR_DIR, rec)
    beats, fs = extract_beats_from_record(filepath)

    if len(beats) == 0:
        print(f"  [{i+1}] {rec}: no beats found, skipping")
        continue

    # Dynamically label each beat based on morphologic correlation with normal template
    labels_list = []
    for b in beats:
        corr = np.corrcoef(b, normal_template)[0, 1]
        # Correlation > 0.65 indicates morphologically normal sinus beat, else abnormal
        labels_list.append(0 if corr > 0.65 else 1)
    labels = np.array(labels_list, dtype=int)

    num_normal = int(np.sum(labels == 0))
    num_abnormal = int(np.sum(labels == 1))
    rate = float((num_abnormal / len(beats)) * 100.0) if len(beats) > 0 else 0.0

    # Add all 100 arrhythmia patients to the dashboard directory
    if len(per_patient) < 100:
        per_patient[rec] = {
            'X': beats, 'y': labels,
            'total': len(beats), 'normal': num_normal, 'abnormal': num_abnormal,
            'rate': rate, 'meta': pm
        }

    X_all.extend(beats)
    y_all.extend(labels)
    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(arr_meta)}] {rec}: {len(beats)} beats ({num_normal} normal, {num_abnormal} abnormal)")

# --- Process normal patients ---
print(f"\nProcessing normal patients from {NOR_DIR}...")
normal_records = sorted(set(f.replace('.hea','') for f in os.listdir(NOR_DIR) if f.endswith('.hea')))
import random
# Use a seed for deterministic normal patient metadata generation
rng = random.Random(42)

for i, rec in enumerate(normal_records):
    filepath = os.path.join(NOR_DIR, rec)
    beats, fs = extract_beats_from_record(filepath)

    if len(beats) == 0:
        continue

    labels = np.zeros(len(beats), dtype=int)  # All beats labeled as normal
    X_all.extend(beats)
    y_all.extend(labels)

    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(normal_records)}] {rec}: {len(beats)} beats (normal)")

X_all = np.array(X_all)
y_all = np.array(y_all)
print(f"\nTotal: {len(y_all)} beats (Normal: {np.sum(y_all==0)}, Abnormal: {np.sum(y_all==1)})")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
Xtr, Xte, ytr, yte = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
print(f"Train: {len(ytr)}, Test: {len(yte)}")

# ============================================================
# TRAIN CNN
# ============================================================
print("\n--- Training CNN ---")
Xtr_cnn = Xtr[..., np.newaxis]

cnn = Sequential([
    Input(shape=(BEAT_SIZE, 1)),
    Conv1D(32, 5, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(Xtr_cnn, ytr, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
cnn.save(os.path.join(MODELS_DIR, "cnn_model.keras"))
print("CNN saved!")

# ============================================================
# TRAIN SVM
# ============================================================
print("\n--- Training SVM ---")
svm = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', probability=True))])
svm.fit(Xtr, ytr)
joblib.dump(svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
print("SVM saved!")

# ============================================================
# SAVE PER-PATIENT DEMO DATA (100 arrhythmia patients)
# ============================================================
print("\n--- Saving per-patient demo data ---")
save_dict = {}
patients_meta_out = []

for rank, (rec, info) in enumerate(per_patient.items(), 1):
    save_dict[f'X_{rec}'] = info['X']
    save_dict[f'y_{rec}'] = info['y']
    patients_meta_out.append({
        'record_id': rec,
        'rank': rank,
        'total': info['total'],
        'normal': info['normal'],
        'abnormal': info['abnormal'],
        'rate': info['rate'],
        'age': info['meta'].get('age'),
        'sex': info['meta'].get('sex'),
        'arrhythmia_labels': info['meta'].get('arrhythmia_labels', []),
    })

# Also save global training datasets for thesis benchmarking scripts
save_dict['X_all'] = X_all
save_dict['y_all'] = y_all

np.savez(os.path.join(MODELS_DIR, "demo_data.npz"), **save_dict)
with open(os.path.join(MODELS_DIR, "patients_meta.json"), 'w') as f:
    json.dump(patients_meta_out, f, indent=2)

print(f"Saved {len(per_patient)} patients to demo_data.npz")
print("\n[OK] All models and data saved!")
print("   - models/cnn_model.keras")
print("   - models/svm_model.pkl")
print("   - models/demo_data.npz")
print("   - models/patients_meta.json")
print("\nRun: python app.py")
