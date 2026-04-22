import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===== Load Data =====
data_path = "data/mitdb"
record = wfdb.rdrecord(f"{data_path}/100")

signal = record.p_signal[:, 0]   # MLII
fs = record.fs

# ===== Deteksi R-Peak =====
# threshold disesuaikan sinyal (aman untuk trial)
peaks, _ = find_peaks(signal, distance=0.6*fs, height=np.mean(signal)+0.5)

# ===== Hitung RR Interval =====
rr_intervals = np.diff(peaks) / fs  # dalam detik

# ===== Plot =====
plt.figure(figsize=(12,4))
plt.plot(signal, label="ECG Signal")
plt.plot(peaks, signal[peaks], "ro", label="R-peaks")
plt.title("R-Peak Detection (Trial)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# ===== Print RR Interval =====
print("Contoh RR interval (detik):")
print(rr_intervals[:10])
