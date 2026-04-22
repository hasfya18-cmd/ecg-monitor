import requests
import numpy as np
import time

URL = "http://localhost:5000/api/ecg_device"

print("Memuat sekumpulan data simulasi (demo_data.npz)...")
demo = np.load("models/demo_data.npz")
X_demo = demo['X_all']
y_demo = demo['y_all']

print("\nMencari sinyal aritmia dari dataset...")
# Cari index yang merupakan aritmia (label 1)
abnormal_indices = np.where(y_demo == 1)[0]
if len(abnormal_indices) == 0:
    print("Tidak ditemukan data abnormal di demo.")
    exit()

# Ambil satu sinyal abnormal
target_idx = abnormal_indices[0]
abnormal_beat = X_demo[target_idx]

print(f"Mengirimkan sinyal detak jantung (Aritmia) ke {URL}...")
payload = {
    "values": abnormal_beat.tolist()
}

try:
    response = requests.post(URL, json=payload, timeout=5)
    print(f"Status Code: {response.status_code}")
    print("Respons Server/AI:")
    print(response.json())
except requests.exceptions.ConnectionError:
    print("Gagal terhubung ke server. Pastikan Flask (app.py) sudah berjalan di port 5000.")
