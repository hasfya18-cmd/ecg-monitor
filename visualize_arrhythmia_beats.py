import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    cache_path = "data/mitdb_cache.npz"
    if not os.path.exists(cache_path):
        print(f"Error: {cache_path} not found. Please run your training script first to generate the cache.")
        return

    print("Memuat dataset dari cache...")
    try:
        data = np.load(cache_path)
        X = data['X']
        y = data['y']
    except Exception as e:
        print(f"Gagal memuat cache numpy: {e}")
        return

    # Pisahkan index kelas Normal (0) dan Abnormal/Aritmia (1)
    normal_idx = np.where(y == 0)[0]
    abnormal_idx = np.where(y == 1)[0]

    print(f"Total Normal Beats: {len(normal_idx)}")
    print(f"Total Abnormal Beats: {len(abnormal_idx)}")

    if len(normal_idx) == 0 or len(abnormal_idx) == 0:
        print("Data tidak cukup untuk menampilkan kedua kelas.")
        return

    # Ambil beberapa sampel acak untuk divisualisasikan
    np.random.seed(42) # Agar hasil random selalu sama
    n_samples = 3
    
    # Pastikan kita tidak mengambil lebih dari jumlah yang tersedia
    n_norm = min(n_samples, len(normal_idx))
    n_abn = min(n_samples, len(abnormal_idx))
    
    sample_normal = np.random.choice(normal_idx, n_norm, replace=False)
    sample_abnormal = np.random.choice(abnormal_idx, n_abn, replace=False)

    plt.figure(figsize=(15, 8))
    plt.suptitle("Perbandingan Sinyal ECG: Normal vs Aritmia (Abnormal)", fontsize=16)

    # Plot sampel Normal
    for i, idx in enumerate(sample_normal):
        plt.subplot(2, max(n_norm, n_abn), i + 1)
        plt.plot(X[idx], color='blue', linewidth=1.5)
        plt.title(f"Normal Beat (Index: {idx})")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', alpha=0.7)

    # Plot sampel Abnormal (Aritmia)
    for i, idx in enumerate(sample_abnormal):
        plt.subplot(2, max(n_norm, n_abn), i + 1 + max(n_norm, n_abn))
        plt.plot(X[idx], color='red', linewidth=1.5)
        plt.title(f"Arrhythmia Beat (Index: {idx})")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9) # Memberi ruang untuk suptitle
    
    output_path = "grafik_hasil_aritmia.png"
    plt.savefig(output_path, dpi=300)
    print(f"Grafik berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    main()
