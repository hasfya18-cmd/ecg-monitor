import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===== Load Data =====
data_path = "data/mitdb"
record = wfdb.rdrecord(f"{data_path}/100")

signal = record.p_signal[:, 0]
fs = record.fs

# ===== Deteksi R-peak (ulang agar mandiri) =====
peaks, _ = find_peaks(signal, distance=0.6*fs, height=np.mean(signal)+0.5)

# ===== Parameter Segmentasi =====
pre_r = int(0.2 * fs)   # 200 ms sebelum R
post_r = int(0.4 * fs)  # 400 ms sesudah R

beats = []

for r in peaks:
    if r - pre_r >= 0 and r + post_r < len(signal):
        beat = signal[r - pre_r : r + post_r]
        beats.append(beat)

beats = np.array(beats)

print("Jumlah beat:", beats.shape[0])
print("Panjang tiap beat:", beats.shape[1])

# ===== Plot contoh beat =====
plt.figure(figsize=(10,4))
for i in range(5):
    plt.plot(beats[i], label=f"Beat {i+1}")

plt.title("Contoh Segmentasi Beat EKG")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
