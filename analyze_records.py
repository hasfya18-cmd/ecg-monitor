# Quick analysis: rank MIT-BIH records by arrhythmia rate
import os
import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

data_path = "data/mitdb"

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

records = sorted(set(
    f.replace('.dat', '') for f in os.listdir(data_path) if f.endswith('.dat')
))

results = []
for rec in records:
    try:
        record = wfdb.rdrecord(f"{data_path}/{rec}")
        ann = wfdb.rdann(f"{data_path}/{rec}", 'atr')
        signal = record.p_signal[:, 0]
        fs = record.fs
        filtered = bandpass_filter(signal, 0.5, 40, fs)
        norm = filtered / np.max(np.abs(filtered))
        peaks, _ = find_peaks(norm, distance=int(0.6*fs), height=np.mean(norm)+0.5*np.std(norm))
        
        normal_count = 0
        abnormal_count = 0
        pre, post = int(0.2*fs), int(0.4*fs)
        for r in peaks:
            if r - pre >= 0 and r + post < len(norm):
                idx = np.argmin(np.abs(ann.sample - r))
                label = ann.symbol[idx]
                if label == 'N':
                    normal_count += 1
                else:
                    abnormal_count += 1
        
        total = normal_count + abnormal_count
        rate = (abnormal_count / total * 100) if total > 0 else 0
        results.append((rec, total, normal_count, abnormal_count, rate))
        print(f"  Record {rec}: {total} beats, {abnormal_count} abnormal ({rate:.1f}%)")
    except Exception as e:
        print(f"  Skipping {rec}: {e}")

# Sort by arrhythmia rate descending
results.sort(key=lambda x: x[4], reverse=True)
print("\n" + "="*60)
print("TOP 20 RECORDS BY ARRHYTHMIA RATE:")
print("="*60)
print(f"{'Rank':<5} {'Record':<10} {'Total':<8} {'Normal':<8} {'Abnormal':<10} {'Rate%':<8}")
print("-"*60)
for i, (rec, total, norm, abn, rate) in enumerate(results[:20], 1):
    print(f"{i:<5} {rec:<10} {total:<8} {norm:<8} {abn:<10} {rate:<8.1f}")
