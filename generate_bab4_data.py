# ============================================================
# BAB 4 — Data Collection Script
# Menghasilkan tabel perbandingan CNN vs SVM untuk Skripsi
# ============================================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import joblib
import time
import csv
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# --- LOAD MODELS & DATA ---
print("=" * 60)
print("  BAB 4 — Pengumpulan Data Perbandingan CNN vs SVM")
print("=" * 60)
print("\nMemuat model dan data...")

cnn_model = load_model("models/cnn_model.keras")
svm_model = joblib.load("models/svm_model.pkl")

demo = np.load("models/demo_data.npz")
X_all = demo['X_all']
y_all = demo['y_all']

print(f"Total beat: {len(y_all)}")
print(f"Normal: {np.sum(y_all == 0)}, Abnormal: {np.sum(y_all == 1)}")

# --- NOISE FUNCTION ---
def add_awgn(signal, snr_db):
    if snr_db >= 100:
        return signal.copy()
    power = np.mean(signal ** 2)
    snr = 10 ** (snr_db / 10)
    noise_power = power / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

# --- TEST CONFIGURATION ---
SNR_LEVELS = [40, 30, 20, 15, 10, 5]
NUM_BEATS = 500  # Jumlah beat per skenario (cukup representatif)

# Randomly sample beats (balanced)
np.random.seed(42)
indices = np.random.choice(len(y_all), NUM_BEATS, replace=False)
X_test = X_all[indices]
y_test = y_all[indices]

print(f"\nSampel uji: {NUM_BEATS} beat")
print(f"  Normal: {np.sum(y_test == 0)}, Abnormal: {np.sum(y_test == 1)}")

# --- RUN BENCHMARKS ---
results = []

print("\n" + "-" * 60)
print(f"{'SNR':>6} | {'Model':>5} | {'Acc':>7} | {'F1':>7} | {'Prec':>7} | {'Rec':>7} | {'Avg ms':>8}")
print("-" * 60)

for snr in SNR_LEVELS:
    # Add noise
    X_noisy = np.array([add_awgn(x, snr) for x in X_test])
    
    # --- CNN ---
    cnn_times = []
    cnn_preds = []
    for i in range(len(X_noisy)):
        beat = X_noisy[i].reshape(1, -1, 1)
        t0 = time.perf_counter()
        prob = float(cnn_model.predict(beat, verbose=0)[0][0])
        t1 = time.perf_counter()
        cnn_preds.append(1 if prob > 0.5 else 0)
        cnn_times.append((t1 - t0) * 1000)
    
    cnn_acc = accuracy_score(y_test, cnn_preds) * 100
    cnn_f1 = f1_score(y_test, cnn_preds, zero_division=0) * 100
    cnn_prec = precision_score(y_test, cnn_preds, zero_division=0) * 100
    cnn_rec = recall_score(y_test, cnn_preds, zero_division=0) * 100
    cnn_avg_ms = np.mean(cnn_times)
    cnn_cm = confusion_matrix(y_test, cnn_preds)
    
    print(f"{snr:>4} dB | {'CNN':>5} | {cnn_acc:>6.1f}% | {cnn_f1:>6.1f}% | {cnn_prec:>6.1f}% | {cnn_rec:>6.1f}% | {cnn_avg_ms:>6.2f}ms")
    
    # --- SVM ---
    svm_times = []
    svm_preds = []
    for i in range(len(X_noisy)):
        beat = X_noisy[i].reshape(1, -1)
        t0 = time.perf_counter()
        pred = int(svm_model.predict(beat)[0])
        t1 = time.perf_counter()
        svm_preds.append(pred)
        svm_times.append((t1 - t0) * 1000)
    
    svm_acc = accuracy_score(y_test, svm_preds) * 100
    svm_f1 = f1_score(y_test, svm_preds, zero_division=0) * 100
    svm_prec = precision_score(y_test, svm_preds, zero_division=0) * 100
    svm_rec = recall_score(y_test, svm_preds, zero_division=0) * 100
    svm_avg_ms = np.mean(svm_times)
    svm_cm = confusion_matrix(y_test, svm_preds)
    
    print(f"{snr:>4} dB | {'SVM':>5} | {svm_acc:>6.1f}% | {svm_f1:>6.1f}% | {svm_prec:>6.1f}% | {svm_rec:>6.1f}% | {svm_avg_ms:>6.2f}ms")
    print("-" * 60)
    
    results.append({
        'snr': snr,
        'cnn_acc': round(cnn_acc, 2), 'cnn_f1': round(cnn_f1, 2),
        'cnn_prec': round(cnn_prec, 2), 'cnn_rec': round(cnn_rec, 2),
        'cnn_ms': round(cnn_avg_ms, 2),
        'cnn_cm': cnn_cm.tolist(),
        'svm_acc': round(svm_acc, 2), 'svm_f1': round(svm_f1, 2),
        'svm_prec': round(svm_prec, 2), 'svm_rec': round(svm_rec, 2),
        'svm_ms': round(svm_avg_ms, 2),
        'svm_cm': svm_cm.tolist()
    })

# --- EXPORT TO CSV ---
csv_path = 'bab4_perbandingan_cnn_svm.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['SNR (dB)', 
                     'CNN Akurasi (%)', 'CNN F1-Score (%)', 'CNN Precision (%)', 'CNN Recall (%)', 'CNN Waktu (ms)',
                     'SVM Akurasi (%)', 'SVM F1-Score (%)', 'SVM Precision (%)', 'SVM Recall (%)', 'SVM Waktu (ms)'])
    for r in results:
        writer.writerow([r['snr'],
                         r['cnn_acc'], r['cnn_f1'], r['cnn_prec'], r['cnn_rec'], r['cnn_ms'],
                         r['svm_acc'], r['svm_f1'], r['svm_prec'], r['svm_rec'], r['svm_ms']])

print(f"\n✅ Data CSV tersimpan: {csv_path}")

# --- PRINT FORMATTED TABLES FOR THESIS ---
print("\n" + "=" * 60)
print("  TABEL SIAP COPAS KE SKRIPSI (Format LaTeX/Word)")
print("=" * 60)

print("\n📊 Tabel 4.1 — Perbandingan Akurasi CNN vs SVM pada Berbagai SNR")
print("-" * 50)
print(f"{'SNR (dB)':>10} | {'CNN (%)':>10} | {'SVM (%)':>10} | {'Selisih':>10}")
print("-" * 50)
for r in results:
    diff = r['cnn_acc'] - r['svm_acc']
    winner = "CNN" if diff > 0 else "SVM"
    print(f"{r['snr']:>8} dB | {r['cnn_acc']:>9.2f} | {r['svm_acc']:>9.2f} | {abs(diff):>7.2f} ({winner})")

print(f"\n📊 Tabel 4.2 — Perbandingan F1-Score CNN vs SVM")
print("-" * 50)
print(f"{'SNR (dB)':>10} | {'CNN (%)':>10} | {'SVM (%)':>10} | {'Selisih':>10}")
print("-" * 50)
for r in results:
    diff = r['cnn_f1'] - r['svm_f1']
    winner = "CNN" if diff > 0 else "SVM"
    print(f"{r['snr']:>8} dB | {r['cnn_f1']:>9.2f} | {r['svm_f1']:>9.2f} | {abs(diff):>7.2f} ({winner})")

print(f"\n📊 Tabel 4.3 — Perbandingan Waktu Eksekusi (Latency)")
print("-" * 50)
print(f"{'SNR (dB)':>10} | {'CNN (ms)':>10} | {'SVM (ms)':>10} | {'Rasio':>10}")
print("-" * 50)
for r in results:
    ratio = r['cnn_ms'] / r['svm_ms'] if r['svm_ms'] > 0 else 0
    print(f"{r['snr']:>8} dB | {r['cnn_ms']:>9.2f} | {r['svm_ms']:>9.2f} | {ratio:>8.1f}x")

print(f"\n📊 Tabel 4.4 — Confusion Matrix pada SNR 30 dB")
r30 = [r for r in results if r['snr'] == 30][0]
print(f"\nCNN (SNR 30 dB):")
print(f"  {'':>15} | {'Pred Normal':>12} | {'Pred Abnormal':>14}")
print(f"  {'Actual Normal':>15} | {r30['cnn_cm'][0][0]:>12} | {r30['cnn_cm'][0][1]:>14}")
print(f"  {'Actual Abnormal':>15} | {r30['cnn_cm'][1][0]:>12} | {r30['cnn_cm'][1][1]:>14}")

print(f"\nSVM (SNR 30 dB):")
print(f"  {'':>15} | {'Pred Normal':>12} | {'Pred Abnormal':>14}")
print(f"  {'Actual Normal':>15} | {r30['svm_cm'][0][0]:>12} | {r30['svm_cm'][0][1]:>14}")
print(f"  {'Actual Abnormal':>15} | {r30['svm_cm'][1][0]:>12} | {r30['svm_cm'][1][1]:>14}")

print("\n" + "=" * 60)
print("  SELESAI! Buka file 'bab4_perbandingan_cnn_svm.csv'")
print("  di Excel untuk membuat grafik batang.")
print("=" * 60)
