# Analisis Komparatif & Simulasi Klasifikasi EKG Nirkabel
# CNN vs SVM MIT-BIH Arrhythmia Database

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info & warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import wfdb
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

total_start = time.time()

# 1. FUNGSI UTILITAS
# ============================================================
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

def simulate_packet_loss(data, loss_rate=0.05):
    """Simulasi packet loss pada transmisi nirkabel BLE.
    Beat yang hilang diganti dengan zeros (sinyal hilang)."""
    data_loss = data.copy()
    n = len(data_loss)
    n_lost = int(n * loss_rate)
    lost_indices = np.random.choice(n, n_lost, replace=False)
    data_loss[lost_indices] = 0  # Beat hilang (packet drop)
    return data_loss, lost_indices

# ============================================================
# 2. LOAD DATA (dengan caching .npz)
# ============================================================
data_path = "data/mitdb"
cache_path = "data/mitdb_cache.npz"

if os.path.exists(cache_path):
    print("Loading dataset dari cache...")
    cache = np.load(cache_path)
    X, y = cache['X'], cache['y']
    print("Cache loaded!\n")
else:
    records = sorted(set(
        f.replace('.dat', '') for f in os.listdir(data_path) if f.endswith('.dat')
    ))
    print(f"Found {len(records)} records: {records}")

    X, y = [], []
    normal_label = ['N']

    print("MULTI-RECORD ECG PROCESSING")

    for rec in records:
        try:
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
            count = 0

            for r in peaks:
                if r-pre >= 0 and r+post < len(norm):
                    beat = norm[r-pre:r+post]
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
    print("\nCache saved ke", cache_path)

print("\nDATASET SUMMARY")
print("TOTAL BEAT:", len(y))
print("NORMAL:", np.sum(y==0))
print("ABNORMAL:", np.sum(y==1))

# ============================================================
# 3. SIMULASI NIRKABEL — MULTI-SNR + PACKET LOSS
# ============================================================
SNR_LEVELS = [10, 15, 20, 25, 30]  # dB
PACKET_LOSS_RATE = 0.05  # 5% packet loss (tipikal BLE)

X_clean = X

# Visualisasi sinyal: Clean vs berbagai SNR
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
axes[0].plot(X_clean[0])
axes[0].set_title("Clean ECG")
axes[0].grid(True)
for i, snr in enumerate(SNR_LEVELS):
    noisy = add_awgn(X_clean[0], snr)
    axes[i+1].plot(noisy, alpha=0.8)
    axes[i+1].set_title(f"Noisy ECG ({snr} dB)")
    axes[i+1].grid(True)
plt.suptitle("ECG Beat: Clean vs Berbagai Level SNR", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("output_snr_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 4. SPLIT DATA
# ============================================================
Xtr_c, Xte_c, ytr, yte = train_test_split(
    X_clean, y, test_size=0.2, stratify=y, random_state=42
)

# ============================================================
# 5. CNN MODEL
# ============================================================
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

# ============================================================
# 6. SVM MODEL
# ============================================================
svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True))
])
svm.fit(Xtr_c, ytr)

# ============================================================
# 7. ANALISIS KOMPLEKSITAS MODEL
# ============================================================
print("\n" + "="*60)
print("ANALISIS KOMPLEKSITAS MODEL")
print("="*60)

# CNN complexity
cnn_total_params = cnn.count_params()
cnn_trainable = sum(
    np.prod(w.shape) for w in cnn.trainable_weights
)
cnn_non_trainable = cnn_total_params - cnn_trainable

# Estimasi ukuran model (float32 = 4 bytes per parameter)
cnn_size_mb = (cnn_total_params * 4) / (1024 * 1024)

print(f"\n--- CNN ---")
print(f"  Total Parameters    : {cnn_total_params:,}")
print(f"  Trainable Parameters: {cnn_trainable:,}")
print(f"  Est. Model Size     : {cnn_size_mb:.2f} MB")
cnn.summary()

# SVM complexity
svm_model = svm.named_steps['svm']
n_sv = svm_model.n_support_
print(f"\n--- SVM ---")
print(f"  Kernel              : {svm_model.kernel}")
print(f"  Support Vectors     : {np.sum(n_sv)} (Normal: {n_sv[0]}, Abnormal: {n_sv[1]})")
print(f"  Feature Dimension   : {svm_model.support_vectors_.shape[1]}")

# Estimasi ukuran SVM (support vectors * features * 8 bytes float64)
svm_size_mb = (svm_model.support_vectors_.nbytes) / (1024 * 1024)
print(f"  Est. Model Size     : {svm_size_mb:.2f} MB")

print(f"\n{'Metrik':<25} {'CNN':>12} {'SVM':>12}")
print("-"*50)
print(f"{'Total Parameters':<25} {cnn_total_params:>12,} {'N/A':>12}")
print(f"{'Support Vectors':<25} {'N/A':>12} {np.sum(n_sv):>12,}")
print(f"{'Est. Model Size (MB)':<25} {cnn_size_mb:>12.2f} {svm_size_mb:>12.2f}")

# ============================================================
# 8. EVALUASI — MULTI-SNR + PACKET LOSS
# ============================================================
def evaluate(model, Xtest, ytest, cnn_mode=False):
    if cnn_mode:
        Xtest_input = Xtest[..., np.newaxis]
        start = time.time()
        y_prob = model.predict(Xtest_input, verbose=0)
        ypred = (y_prob > 0.5).astype(int).flatten()
    else:
        start = time.time()
        ypred = model.predict(Xtest)
        y_prob = model.predict_proba(Xtest)[:, 1]
    latency = (time.time() - start) / len(ytest) * 1000
    acc = accuracy_score(ytest, ypred)
    return acc, latency, ypred, y_prob

print("\n" + "="*60)
print("EVALUASI PERFORMA — MULTI-SNR & PACKET LOSS")
print("="*60)

# Storage untuk hasil
results = {
    'CNN': {'snr': [], 'acc': [], 'lat': []},
    'SVM': {'snr': [], 'acc': [], 'lat': []}
}

# --- 8a. Evaluasi Clean ---
print(f"\n{'METODE':<6} | {'KONDISI':<20} | {'AKURASI':>8} | {'LATENCY (ms/beat)':>18}")
print("-"*60)

for name, model, iscnn in [("CNN", cnn, True), ("SVM", svm, False)]:
    acc_c, lat_c, ypred_c, yprob_c = evaluate(model, Xte_c, yte, iscnn)
    print(f"{name:<6} | {'Clean':.<20} | {acc_c*100:7.2f}% | {lat_c:17.3f}")
    results[name]['snr'].append(('Clean', acc_c, lat_c))

print("-"*60)

# --- 8b. Evaluasi Multi-SNR ---
for snr in SNR_LEVELS:
    X_noisy_snr = np.array([add_awgn(b, snr) for b in Xte_c])
    for name, model, iscnn in [("CNN", cnn, True), ("SVM", svm, False)]:
        acc_n, lat_n, _, _ = evaluate(model, X_noisy_snr, yte, iscnn)
        print(f"{name:<6} | {'SNR ' + str(snr) + ' dB':.<20} | {acc_n*100:7.2f}% | {lat_n:17.3f}")
        results[name]['snr'].append((f'{snr} dB', acc_n, lat_n))
    print("-"*60)

# --- 8c. Evaluasi Packet Loss ---
print(f"\nSimulasi Packet Loss (BLE): {PACKET_LOSS_RATE*100:.0f}% loss rate")
print("-"*60)
Xte_loss, lost_idx = simulate_packet_loss(Xte_c.copy(), PACKET_LOSS_RATE)
# Hapus beat yang hilang (packet drop) dari evaluasi
valid_mask = np.ones(len(Xte_loss), dtype=bool)
valid_mask[lost_idx] = False
Xte_valid = Xte_loss[valid_mask]
yte_valid = yte[valid_mask]

print(f"Total test beats: {len(yte)} | Received: {len(yte_valid)} | Lost: {len(lost_idx)}")
for name, model, iscnn in [("CNN", cnn, True), ("SVM", svm, False)]:
    acc_pl, lat_pl, _, _ = evaluate(model, Xte_valid, yte_valid, iscnn)
    print(f"{name:<6} | {'Packet Loss 5%':.<20} | {acc_pl*100:7.2f}% | {lat_pl:17.3f}")
print("-"*60)

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, (name, model, iscnn) in enumerate([("CNN", cnn, True), ("SVM", svm, False)]):
    _, _, ypred, _ = evaluate(model, Xte_c, yte, iscnn)
    cm = confusion_matrix(yte, ypred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Abnormal'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"Confusion Matrix — {name} (Clean ECG)", fontsize=12)

plt.tight_layout()
plt.savefig("output_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 10. ROC CURVE & AUC
# ============================================================
print("\n" + "="*60)
print("ROC CURVE & AUC")
print("="*60)

fig, ax = plt.subplots(figsize=(8, 6))

colors = {'CNN': '#2196F3', 'SVM': '#FF5722'}

for name, model, iscnn in [("CNN", cnn, True), ("SVM", svm, False)]:
    _, _, _, yprob = evaluate(model, Xte_c, yte, iscnn)
    fpr, tpr, _ = roc_curve(yte, yprob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[name], lw=2,
            label=f'{name} (AUC = {roc_auc:.4f})')
    print(f"  {name} AUC: {roc_auc:.4f}")

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random (AUC = 0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — CNN vs SVM (Clean ECG)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("output_roc_curve.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 11. GRAFIK PERBANDINGAN — BAR CHART & LINE PLOT
# ============================================================

# --- 11a. Bar Chart: Akurasi CNN vs SVM di berbagai kondisi ---
conditions = ['Clean'] + [f'{s} dB' for s in SNR_LEVELS]
cnn_accs = [item[1]*100 for item in results['CNN']['snr']]
svm_accs = [item[1]*100 for item in results['SVM']['snr']]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, cnn_accs, width, label='CNN', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, svm_accs, width, label='SVM', color='#FF5722', edgecolor='white')

# Tambah label nilai di atas bar
for bar in bars1:
    ax.annotate(f'{bar.get_height():.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')
for bar in bars2:
    ax.annotate(f'{bar.get_height():.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Kondisi Sinyal', fontsize=12)
ax.set_ylabel('Akurasi (%)', fontsize=12)
ax.set_title('Perbandingan Akurasi CNN vs SVM — Berbagai Kondisi', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("output_accuracy_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# --- 11b. Line Plot: Akurasi vs SNR ---
snr_accs_cnn = [item[1]*100 for item in results['CNN']['snr'][1:]]  # skip Clean
snr_accs_svm = [item[1]*100 for item in results['SVM']['snr'][1:]]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(SNR_LEVELS, snr_accs_cnn, 'o-', color='#2196F3', lw=2.5,
        markersize=8, label='CNN')
ax.plot(SNR_LEVELS, snr_accs_svm, 's-', color='#FF5722', lw=2.5,
        markersize=8, label='SVM')

for i, snr in enumerate(SNR_LEVELS):
    ax.annotate(f'{snr_accs_cnn[i]:.1f}%', (snr, snr_accs_cnn[i]),
                textcoords='offset points', xytext=(0, 12), ha='center', fontsize=9)
    ax.annotate(f'{snr_accs_svm[i]:.1f}%', (snr, snr_accs_svm[i]),
                textcoords='offset points', xytext=(0, -15), ha='center', fontsize=9)

ax.set_xlabel('SNR (dB)', fontsize=12)
ax.set_ylabel('Akurasi (%)', fontsize=12)
ax.set_title('Degradasi Akurasi vs Level Noise (SNR)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(SNR_LEVELS)
plt.tight_layout()
plt.savefig("output_accuracy_vs_snr.png", dpi=150, bbox_inches='tight')
plt.show()

# --- 11c. Bar Chart: Latency CNN vs SVM ---
cnn_lats = [item[2] for item in results['CNN']['snr']]
svm_lats = [item[2] for item in results['SVM']['snr']]

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, cnn_lats, width, label='CNN', color='#2196F3', edgecolor='white')
bars2 = ax.bar(x + width/2, svm_lats, width, label='SVM', color='#FF5722', edgecolor='white')

for bar in bars1:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=9)
for bar in bars2:
    ax.annotate(f'{bar.get_height():.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=9)

ax.set_xlabel('Kondisi Sinyal', fontsize=12)
ax.set_ylabel('Latency (ms/beat)', fontsize=12)
ax.set_title('Perbandingan Latency CNN vs SVM — Berbagai Kondisi', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("output_latency_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================
print("\n" + "="*60)
print("CLASSIFICATION REPORT — CLEAN ECG")
print("="*60)

_, _, ypred_cnn, _ = evaluate(cnn, Xte_c, yte, cnn_mode=True)
_, _, ypred_svm, _ = evaluate(svm, Xte_c, yte, cnn_mode=False)

print("\n--- CNN ---")
print(classification_report(yte, ypred_cnn, target_names=['Normal', 'Abnormal']))

print("--- SVM ---")
print(classification_report(yte, ypred_svm, target_names=['Normal', 'Abnormal']))

# ============================================================
# 13. RINGKASAN AKHIR
# ============================================================
print("\n" + "="*60)
print("RINGKASAN HASIL ANALISIS KOMPARATIF")
print("="*60)

_, _, _, yprob_cnn = evaluate(cnn, Xte_c, yte, cnn_mode=True)
_, _, _, yprob_svm = evaluate(svm, Xte_c, yte, cnn_mode=False)
fpr_c, tpr_c, _ = roc_curve(yte, yprob_cnn)
fpr_s, tpr_s, _ = roc_curve(yte, yprob_svm)
auc_cnn = auc(fpr_c, tpr_c)
auc_svm = auc(fpr_s, tpr_s)

acc_clean_cnn = results['CNN']['snr'][0][1]*100
acc_clean_svm = results['SVM']['snr'][0][1]*100
lat_clean_cnn = results['CNN']['snr'][0][2]
lat_clean_svm = results['SVM']['snr'][0][2]

print(f"\n{'Metrik':<30} {'CNN':>12} {'SVM':>12}")
print("-"*55)
print(f"{'Akurasi Clean (%)':<30} {acc_clean_cnn:>11.2f}% {acc_clean_svm:>11.2f}%")
print(f"{'AUC Score':<30} {auc_cnn:>12.4f} {auc_svm:>12.4f}")
print(f"{'Latency Clean (ms/beat)':<30} {lat_clean_cnn:>12.3f} {lat_clean_svm:>12.3f}")
print(f"{'Model Size (MB)':<30} {cnn_size_mb:>12.2f} {svm_size_mb:>12.2f}")
print(f"{'Total Parameters':<30} {cnn_total_params:>12,} {'N/A':>12}")
print(f"{'Support Vectors':<30} {'N/A':>12} {np.sum(n_sv):>12,}")

print("\n--- Output Files ---")
print("  output_snr_comparison.png      : Visualisasi sinyal berbagai SNR")
print("  output_confusion_matrix.png    : Confusion Matrix CNN vs SVM")
print("  output_roc_curve.png           : ROC Curve & AUC")
print("  output_accuracy_comparison.png : Bar chart akurasi")
print("  output_accuracy_vs_snr.png     : Line plot akurasi vs SNR")
print("  output_latency_comparison.png  : Bar chart latency")
total_time = time.time() - total_start
print(f"\nTotal running time: {total_time/60:.1f} menit ({total_time:.1f} detik)")
print("\nDone! ✅")
