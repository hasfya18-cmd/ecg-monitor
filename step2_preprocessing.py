import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ===== Load Data =====
data_path = "data/mitdb"
record = wfdb.rdrecord(f"{data_path}/100")

signal = record.p_signal[:, 0]   # MLII
fs = record.fs

# ===== Bandpass Filter =====
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

filtered_signal = bandpass_filter(signal, 0.5, 40, fs)

# ===== Plot =====
plt.figure(figsize=(12,5))

plt.subplot(2,1,1)
plt.plot(signal)
plt.title("Raw ECG Signal")

plt.subplot(2,1,2)
plt.plot(filtered_signal)
plt.title("Filtered ECG Signal (0.5–40 Hz)")

plt.tight_layout()
plt.show()
