# ============================================================
# BAB 4 — Generate Grafik & Tabel untuk Skripsi
# Output: gambar PNG siap masuk Word
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# --- DATA HASIL PENGUJIAN ---
snr_levels = [40, 30, 20, 15, 10, 5]
snr_labels = ['40\n(Bersih)', '30\n(Baik)', '20\n(Sedang)', '15\n(Cukup)', '10\n(Buruk)', '5\n(Sgt Buruk)']

cnn_acc = [99.4, 99.4, 99.4, 99.0, 97.8, 95.4]
svm_acc = [99.4, 99.4, 99.4, 99.4, 99.6, 95.0]

cnn_f1 = [99.1, 99.1, 99.1, 98.5, 96.7, 92.9]
svm_f1 = [99.1, 99.1, 99.1, 99.1, 99.4, 92.8]

cnn_prec = [98.2, 98.2, 98.2, 98.2, 95.9, 95.5]
svm_prec = [98.8, 98.8, 98.8, 98.8, 98.8, 89.4]

cnn_rec = [100.0, 100.0, 100.0, 98.8, 97.6, 90.4]
svm_rec = [99.4, 99.4, 99.4, 99.4, 100.0, 96.4]

cnn_ms = [204.70, 111.83, 113.36, 137.08, 312.02, 328.36]
svm_ms = [5.03, 2.00, 1.99, 6.58, 5.96, 9.40]

# Confusion Matrix SNR 30dB
cnn_cm_30 = np.array([[328, 6], [0, 166]])
svm_cm_30 = np.array([[332, 2], [1, 165]])

# --- OUTPUT DIR ---
out_dir = 'grafik_bab4'
os.makedirs(out_dir, exist_ok=True)

# --- STYLE ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = '#fafafa'
plt.rcParams['figure.facecolor'] = 'white'

CNN_COLOR = '#2563eb'
SVM_COLOR = '#7c3aed'
bar_width = 0.35
x = np.arange(len(snr_levels))

# ============================================================
# GRAFIK 1: Perbandingan Akurasi
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
bars1 = ax.bar(x - bar_width/2, cnn_acc, bar_width, label='CNN', color=CNN_COLOR, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + bar_width/2, svm_acc, bar_width, label='SVM', color=SVM_COLOR, edgecolor='white', linewidth=0.8)

ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=12)
ax.set_ylabel('Akurasi (%)', fontweight='bold', fontsize=12)
ax.set_title('Grafik 4.1 - Perbandingan Akurasi CNN vs SVM\npada Berbagai Tingkat SNR', fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.set_ylim(93, 101)
ax.legend(loc='lower left', fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=CNN_COLOR)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=SVM_COLOR)

plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_1_akurasi.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.1 - Akurasi")

# ============================================================
# GRAFIK 2: Perbandingan F1-Score
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
bars1 = ax.bar(x - bar_width/2, cnn_f1, bar_width, label='CNN', color=CNN_COLOR, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + bar_width/2, svm_f1, bar_width, label='SVM', color=SVM_COLOR, edgecolor='white', linewidth=0.8)

ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=12)
ax.set_ylabel('F1-Score (%)', fontweight='bold', fontsize=12)
ax.set_title('Grafik 4.2 - Perbandingan F1-Score CNN vs SVM\npada Berbagai Tingkat SNR', fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.set_ylim(90, 101)
ax.legend(loc='lower left', fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=CNN_COLOR)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold', color=SVM_COLOR)

plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_2_f1score.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.2 - F1-Score")

# ============================================================
# GRAFIK 3: Perbandingan Waktu Eksekusi
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
bars1 = ax.bar(x - bar_width/2, cnn_ms, bar_width, label='CNN', color=CNN_COLOR, edgecolor='white', linewidth=0.8)
bars2 = ax.bar(x + bar_width/2, svm_ms, bar_width, label='SVM', color=SVM_COLOR, edgecolor='white', linewidth=0.8)

ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=12)
ax.set_ylabel('Waktu Eksekusi (ms)', fontweight='bold', fontsize=12)
ax.set_title('Grafik 4.3 - Perbandingan Waktu Eksekusi CNN vs SVM\npada Berbagai Tingkat SNR', fontweight='bold', fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(snr_labels, fontsize=9)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=CNN_COLOR)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=SVM_COLOR)

plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_3_waktu_eksekusi.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.3 - Waktu Eksekusi")

# ============================================================
# GRAFIK 4: Line Chart - Degradasi Performa vs SNR
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(snr_levels[::-1], cnn_acc[::-1], 'o-', color=CNN_COLOR, linewidth=2.5, markersize=8, label='CNN Akurasi')
ax.plot(snr_levels[::-1], svm_acc[::-1], 's--', color=SVM_COLOR, linewidth=2.5, markersize=8, label='SVM Akurasi')
ax.plot(snr_levels[::-1], cnn_f1[::-1], '^:', color='#059669', linewidth=2, markersize=7, label='CNN F1-Score')
ax.plot(snr_levels[::-1], svm_f1[::-1], 'D-.', color='#dc2626', linewidth=2, markersize=7, label='SVM F1-Score')

ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=12)
ax.set_ylabel('Persentase (%)', fontweight='bold', fontsize=12)
ax.set_title('Grafik 4.4 - Kurva Degradasi Performa\nterhadap Penurunan Kualitas Sinyal', fontweight='bold', fontsize=13, pad=15)
ax.set_ylim(90, 101)
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

ax.axvspan(0, 10, alpha=0.08, color='red', label='_nolegend_')
ax.axvspan(10, 20, alpha=0.05, color='orange', label='_nolegend_')
ax.text(7, 91.5, 'Zona Kritis', fontsize=9, color='red', ha='center', fontstyle='italic')

plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_4_degradasi.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.4 - Degradasi Performa")

# ============================================================
# GRAFIK 5: Precision vs Recall
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(x - bar_width/2, cnn_prec, bar_width, label='CNN', color=CNN_COLOR, edgecolor='white')
ax1.bar(x + bar_width/2, svm_prec, bar_width, label='SVM', color=SVM_COLOR, edgecolor='white')
ax1.set_xlabel('SNR (dB)', fontweight='bold')
ax1.set_ylabel('Precision (%)', fontweight='bold')
ax1.set_title('Grafik 4.5a - Precision', fontweight='bold', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels([str(s) for s in snr_levels], fontsize=9)
ax1.set_ylim(86, 102)
ax1.legend(fontsize=9)
ax1.grid(axis='y', alpha=0.3)

ax2.bar(x - bar_width/2, cnn_rec, bar_width, label='CNN', color=CNN_COLOR, edgecolor='white')
ax2.bar(x + bar_width/2, svm_rec, bar_width, label='SVM', color=SVM_COLOR, edgecolor='white')
ax2.set_xlabel('SNR (dB)', fontweight='bold')
ax2.set_ylabel('Recall (%)', fontweight='bold')
ax2.set_title('Grafik 4.5b - Recall', fontweight='bold', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([str(s) for s in snr_levels], fontsize=9)
ax2.set_ylim(86, 102)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Grafik 4.5 - Perbandingan Precision dan Recall CNN vs SVM', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_5_precision_recall.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.5 - Precision & Recall")

# ============================================================
# GRAFIK 6: Confusion Matrix (SNR 30dB)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

labels_cm = ['Normal', 'Abnormal']

# CNN
im1 = ax1.imshow(cnn_cm_30, cmap='Blues', aspect='auto')
ax1.set_title('CNN (SNR 30 dB)', fontweight='bold', fontsize=12)
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(labels_cm)
ax1.set_yticklabels(labels_cm)
ax1.set_xlabel('Prediksi', fontweight='bold')
ax1.set_ylabel('Aktual', fontweight='bold')
for i in range(2):
    for j in range(2):
        color = 'white' if cnn_cm_30[i, j] > 100 else 'black'
        ax1.text(j, i, str(cnn_cm_30[i, j]), ha='center', va='center', fontsize=20, fontweight='bold', color=color)

# SVM
im2 = ax2.imshow(svm_cm_30, cmap='Purples', aspect='auto')
ax2.set_title('SVM (SNR 30 dB)', fontweight='bold', fontsize=12)
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(labels_cm)
ax2.set_yticklabels(labels_cm)
ax2.set_xlabel('Prediksi', fontweight='bold')
ax2.set_ylabel('Aktual', fontweight='bold')
for i in range(2):
    for j in range(2):
        color = 'white' if svm_cm_30[i, j] > 100 else 'black'
        ax2.text(j, i, str(svm_cm_30[i, j]), ha='center', va='center', fontsize=20, fontweight='bold', color=color)

plt.suptitle('Grafik 4.6 - Confusion Matrix pada SNR 30 dB', fontweight='bold', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_6_confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.6 - Confusion Matrix")

# ============================================================
# GRAFIK 7: Radar Chart - Ringkasan Perbandingan
# ============================================================
categories = ['Akurasi\n(SNR 30)', 'F1-Score\n(SNR 30)', 'Recall\n(SNR 30)', 'Precision\n(SNR 30)', 'Kecepatan\n(1/ms)', 'Ketahanan\nNoise']
N = len(categories)

# Normalize to 0-100 scale
cnn_vals = [99.4, 99.1, 100.0, 98.2, (1/111.83)*1000*5, 95.4]  # speed normalized
svm_vals = [99.4, 99.1, 99.4, 98.8, (1/2.0)*1000*5, 99.6]

# Cap at 100
cnn_vals = [min(v, 100) for v in cnn_vals]
svm_vals = [min(v, 100) for v in svm_vals]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
cnn_vals_r = cnn_vals + [cnn_vals[0]]
svm_vals_r = svm_vals + [svm_vals[0]]
angles += [angles[0]]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax.plot(angles, cnn_vals_r, 'o-', linewidth=2.5, color=CNN_COLOR, label='CNN')
ax.fill(angles, cnn_vals_r, alpha=0.15, color=CNN_COLOR)
ax.plot(angles, svm_vals_r, 's-', linewidth=2.5, color=SVM_COLOR, label='SVM')
ax.fill(angles, svm_vals_r, alpha=0.15, color=SVM_COLOR)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(88, 102)
ax.set_title('Grafik 4.7 - Radar Chart Ringkasan\nPerbandingan CNN vs SVM', fontweight='bold', fontsize=13, pad=25)
ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.05), fontsize=11)

plt.tight_layout()
plt.savefig(f'{out_dir}/grafik_4_7_radar.png', dpi=200, bbox_inches='tight')
plt.close()
print("[OK] Grafik 4.7 - Radar Chart")

print("\n" + "=" * 50)
print(f"  SELESAI! {7} grafik tersimpan di folder: {out_dir}/")
print("=" * 50)
print("\nDaftar file:")
for f in sorted(os.listdir(out_dir)):
    print(f"  - {out_dir}/{f}")
