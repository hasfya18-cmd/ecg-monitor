import wfdb
import matplotlib.pyplot as plt

data_path = "data/mitdb"

record = wfdb.rdrecord(f"{data_path}/100")
annotation = wfdb.rdann(f"{data_path}/100", 'atr')

signal = record.p_signal[:, 0]

plt.figure(figsize=(12,4))
plt.plot(signal)
plt.scatter(annotation.sample,
            signal[annotation.sample],
            color='red', s=10)
plt.title("Trial EKG Record 100")
plt.show()
