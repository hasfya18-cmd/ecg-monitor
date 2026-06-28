# Extract 100 Normal (Sinus Rhythm) patients for training balance
import zipfile, os, json, random

ZIP_PATH = r"c:\Users\USER\Downloads\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
OUTPUT_DIR = r"c:\Users\USER\Documents\skripsi_ekg\data\large_ecg_normal"
NORMAL_CODE = '426783006'  # Sinus Rhythm
# Arrhythmia codes to EXCLUDE
ARRHYTHMIA_CODES = {'164889003','164890007','713422000','426761007','233896004','233897008',
    '17338001','284470004','11157007','251173003','251180001','75532003','426995002',
    '251164006','13640000','195060002','74390002','270492004','195042002','54016002',
    '28189009','27885002','233917008','427084000','426177001','427393009'}

print("Scanning for pure-normal patients...")
normals = []
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_files = zf.namelist()
    hea_files = [f for f in all_files if f.endswith('.hea')]
    for i, hea_path in enumerate(hea_files):
        if i % 5000 == 0: print(f"  {i}/{len(hea_files)}...")
        try:
            content = zf.read(hea_path).decode('utf-8', errors='ignore')
            dx_codes = []
            for line in content.strip().split('\n'):
                if line.strip().startswith('#Dx:'):
                    dx_codes = [c.strip() for c in line.split(':')[1].strip().split(',')]
            # Only SR, no arrhythmia codes
            if NORMAL_CODE in dx_codes and not any(c in ARRHYTHMIA_CODES for c in dx_codes):
                normals.append({'record': os.path.basename(hea_path).replace('.hea',''), 'dir': os.path.dirname(hea_path), 'hea': hea_path})
        except: continue

print(f"Found {len(normals)} pure-normal patients")
random.seed(42)
selected = random.sample(normals, min(100, len(normals)))
print(f"Selected {len(selected)} for extraction")

os.makedirs(OUTPUT_DIR, exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for i, p in enumerate(selected, 1):
        prefix = f"{p['dir']}/{p['record']}"
        for f in [x for x in all_files if x.startswith(prefix)]:
            with zf.open(f) as src, open(os.path.join(OUTPUT_DIR, os.path.basename(f)), 'wb') as dst:
                dst.write(src.read())
        if i % 20 == 0: print(f"  Extracted {i}/{len(selected)}...")

print(f"Done! {len(selected)} normal patients extracted to {OUTPUT_DIR}")
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR))
print(f"Total size: {total_size/(1024*1024):.1f} MB")
