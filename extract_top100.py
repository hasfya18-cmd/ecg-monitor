# Extract Top 100 Arrhythmia Patients from Large Scale ECG Database
# Reads .hea files from ZIP to find arrhythmia labels, then extracts only those patients

import zipfile
import os
import re
from collections import Counter

ZIP_PATH = r"c:\Users\USER\Downloads\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0.zip"
OUTPUT_DIR = r"c:\Users\USER\Documents\skripsi_ekg\data\large_ecg"
TOP_N = 100

# Arrhythmia-related SNOMED codes (from ConditionNames_SNOMED-CT.csv)
# We focus on actual rhythm disorders, not structural conditions
ARRHYTHMIA_CODES = {
    '164889003': 'AFIB',    # Atrial Fibrillation
    '164890007': 'AF',      # Atrial Flutter
    '713422000': 'AT',      # Atrial Tachycardia
    '426761007': 'SVT',     # Supraventricular Tachycardia
    '233896004': 'AVNRT',   # AV Node Reentrant Tachycardia
    '233897008': 'AVRT',    # AV Reentrant Tachycardia
    '17338001': 'VPB',      # Ventricular Premature Beat
    '284470004': 'APB',     # Atrial Premature Beats
    '11157007': 'VB',       # Ventricular Bigeminy
    '251173003': 'ABI',     # Atrial Bigeminy
    '251180001': 'VET',     # Ventricular Escape Trigeminy
    '75532003': 'VEB',      # Ventricular Escape Beat
    '426995002': 'JEB',     # Junctional Escape Beat
    '251164006': 'JPT',     # Junctional Premature Beat
    '13640000': 'VFW',      # Ventricular Fusion Wave
    '195060002': 'VPE',     # Ventricular Preexcitation
    '74390002': 'WPW',      # WPW
    '270492004': '1AVB',    # 1st degree AV block
    '195042002': '2AVB',    # 2nd degree AV block
    '54016002': '2AVB1',    # 2nd degree AV block type 1
    '28189009': '2AVB2',    # 2nd degree AV block type 2
    '27885002': '3AVB',     # 3rd degree AV block
    '233917008': 'AVB',     # AV block
    '427084000': 'ST',      # Sinus Tachycardia
    '426177001': 'SB',      # Sinus Bradycardia
    '427393009': 'SA',      # Sinus Irregularity
}

# Normal rhythm code (to exclude pure-normal patients)
NORMAL_CODE = '426783006'  # Sinus Rhythm (SR)

print(f"Opening ZIP: {ZIP_PATH}")
print(f"Looking for top {TOP_N} arrhythmia patients...\n")

# Step 1: Read all .hea files from ZIP and parse diagnoses
patients = []
hea_count = 0

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    all_files = zf.namelist()
    hea_files = [f for f in all_files if f.endswith('.hea')]
    total_hea = len(hea_files)
    print(f"Found {total_hea} .hea files in ZIP")
    
    for i, hea_path in enumerate(hea_files):
        if i % 5000 == 0:
            print(f"  Scanning {i}/{total_hea}...")
        
        try:
            content = zf.read(hea_path).decode('utf-8', errors='ignore')
            lines = content.strip().split('\n')
            
            # Extract patient info from header
            record_name = os.path.basename(hea_path).replace('.hea', '')
            record_dir = os.path.dirname(hea_path)
            
            age = None
            sex = None
            dx_codes = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('#Age:'):
                    try:
                        age = int(line.split(':')[1].strip())
                    except:
                        age = None
                elif line.startswith('#Sex:'):
                    sex = line.split(':')[1].strip()
                elif line.startswith('#Dx:'):
                    dx_str = line.split(':')[1].strip()
                    dx_codes = [c.strip() for c in dx_str.split(',')]
            
            # Count arrhythmia codes for this patient
            arrhythmia_labels = []
            for code in dx_codes:
                if code in ARRHYTHMIA_CODES:
                    arrhythmia_labels.append(ARRHYTHMIA_CODES[code])
            
            # Only include patients that have at least 1 arrhythmia
            if len(arrhythmia_labels) > 0:
                patients.append({
                    'record': record_name,
                    'dir': record_dir,
                    'hea_path': hea_path,
                    'age': age,
                    'sex': sex,
                    'dx_codes': dx_codes,
                    'arrhythmia_labels': arrhythmia_labels,
                    'arrhythmia_count': len(arrhythmia_labels),
                })
            
            hea_count += 1
        except Exception as e:
            continue

print(f"\nScanned {hea_count} patients total")
print(f"Patients with arrhythmia: {len(patients)}")

# Step 2: Sort by number of arrhythmia conditions (most first)
patients.sort(key=lambda x: x['arrhythmia_count'], reverse=True)

# Take top N
top_patients = patients[:TOP_N]

print(f"\n{'='*70}")
print(f"TOP {TOP_N} PATIENTS BY ARRHYTHMIA COUNT:")
print(f"{'='*70}")
print(f"{'#':<4} {'Record':<12} {'Age':<5} {'Sex':<5} {'#Arr':<5} {'Arrhythmia Types'}")
print(f"{'-'*70}")
for i, p in enumerate(top_patients, 1):
    arr_str = ', '.join(p['arrhythmia_labels'])
    age_str = str(p['age']) if p['age'] else '?'
    sex_str = p['sex'] if p['sex'] else '?'
    print(f"{i:<4} {p['record']:<12} {age_str:<5} {sex_str:<5} {p['arrhythmia_count']:<5} {arr_str}")

# Count arrhythmia distribution
all_arr = []
for p in top_patients:
    all_arr.extend(p['arrhythmia_labels'])
print(f"\nArrhythmia distribution in top {TOP_N}:")
for label, count in Counter(all_arr).most_common():
    print(f"  {label}: {count}")

# Step 3: Extract only those patients' files
print(f"\n{'='*70}")
print(f"Extracting {TOP_N} patients to: {OUTPUT_DIR}")
print(f"{'='*70}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
    for i, p in enumerate(top_patients, 1):
        record_dir = p['dir']
        record_name = p['record']
        
        # Find all files for this patient (.hea and .mat)
        prefix = f"{record_dir}/{record_name}"
        patient_files = [f for f in all_files if f.startswith(prefix)]
        
        for f in patient_files:
            # Extract to flat output directory
            basename = os.path.basename(f)
            target = os.path.join(OUTPUT_DIR, basename)
            
            with zf.open(f) as src, open(target, 'wb') as dst:
                dst.write(src.read())
        
        if i % 20 == 0:
            print(f"  Extracted {i}/{TOP_N}...")

# Save metadata
import json
meta = []
for i, p in enumerate(top_patients, 1):
    meta.append({
        'rank': i,
        'record_id': p['record'],
        'age': p['age'],
        'sex': p['sex'],
        'arrhythmia_labels': p['arrhythmia_labels'],
        'arrhythmia_count': p['arrhythmia_count'],
        'all_dx_codes': p['dx_codes'],
    })

meta_path = os.path.join(OUTPUT_DIR, 'patients_meta.json')
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nDone! Extracted {TOP_N} patients")
print(f"Files saved to: {OUTPUT_DIR}")
print(f"Metadata saved to: {meta_path}")

# Show folder size
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR))
print(f"Total size: {total_size / (1024*1024):.1f} MB")
