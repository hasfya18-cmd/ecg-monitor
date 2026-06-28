# ECG Real-Time Monitoring Web App
# Flask Backend with SSE Streaming — Per-Patient Mode

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only mode
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
import threading
import numpy as np
import joblib
import requests
from collections import deque
from scipy.signal import butter, filtfilt
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

# Restrict TensorFlow thread counts to avoid memory bloat on multi-core servers
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure cookies for Hugging Face Spaces iframe compatibility
if os.environ.get('SPACE_ID') or os.environ.get('SPACE_NAME') or os.environ.get('SPACE_AUTHOR'):
    app.config.update(
        SESSION_COOKIE_SAMESITE='None',
        SESSION_COOKIE_SECURE=True
    )

GOOGLE_CLIENT_ID = os.environ.get('GOOGLE_CLIENT_ID', '')

# Admin credentials with secure password hashing (werkzeug pbkdf2)
DEMO_USERNAME = 'admin'
DEMO_PASSWORD_HASH = generate_password_hash('admin123')

def get_wib_time():
    """Returns datetime object in UTC+7 (WIB)."""
    return datetime.now(timezone(timedelta(hours=7)))

# ============================================================
# GLOBAL STATE
# ============================================================
state = {
    'model_choice': 'CNN',      # CNN or SVM
    'snr_db': 30,               # SNR level (dB)
    'packet_loss': 0.05,        # Packet loss rate
    'is_streaming': True,       # Streaming on/off
    'speed': 0.8,               # Seconds between beats
    'source': 'demo',           # 'demo' or 'live' (Smartwatch)
    'telegram_token': os.environ.get('TELEGRAM_BOT_TOKEN', '8789979813:AAGi8ug2yiccyZ_BOqI0aznujhADK2UvcxQ'),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID', ''),
    'groq_api_key': os.environ.get('GROQ_API_KEY', 'ISI_API_KEY_GROQ_KAMU_DI_SINI'),
    'telegram_api_url': os.environ.get('TELEGRAM_API_URL', 'https://plain-truth-129f.hasfya18.workers.dev').rstrip('/'),
    'live_stats': {}            # Dictionary to cache live EKG stats for /status bot command
}

# State synchronization lock for concurrent request handling
state_lock = threading.Lock()

# Fallback alert queue (when Telegram fails, alerts are stored here for web dashboard)
failed_alerts = deque(maxlen=50)
failed_alerts_lock = threading.Lock()

# Telegram outbox — queued messages for browser-side dispatch (bypasses HF outbound blocks)
telegram_outbox = deque(maxlen=100)
telegram_outbox_lock = threading.Lock()

# Global alert store — collects arrhythmia alerts from ALL patients (for Alert Center sidebar)
global_alerts = deque(maxlen=100)
global_alerts_lock = threading.Lock()

# Global Simulation Start Time
global_sim_start = time.time()

# ============================================================
# LIVE STREAMING BUFFERS (Circular Buffers using deque)
# ============================================================
MAX_ECG_BUFFER = 1000          # Max raw values to buffer
MAX_RESULTS_QUEUE = 100        # Max processed results in queue
live_ecg_buffer = deque(maxlen=MAX_ECG_BUFFER)
live_results_queue = deque(maxlen=MAX_RESULTS_QUEUE)
BEAT_SIZE = 187

# ============================================================
# LOAD MODELS & DATA
# ============================================================
print("Loading models...")
cnn_model = load_model("models/cnn_model.keras")
svm_model = joblib.load("models/svm_model.pkl")
print("Models loaded!")

# Load per-patient metadata
print("Loading patient metadata...")
with open("models/patients_meta.json", 'r') as f:
    patients_meta = json.load(f)
print(f"Found {len(patients_meta)} patients")

# Enrich patients with record IDs and fields needed by frontend
for i, p in enumerate(patients_meta):
    p['name'] = f"Pasien {p['record_id']}"
    p['total_beats'] = p.get('total', 0)

# Load per-patient demo data
print("Loading per-patient demo data...")
demo = np.load("models/demo_data.npz")
patient_data = {}
for p in patients_meta:
    rec = p['record_id']
    patient_data[rec] = {
        'X': demo[f'X_{rec}'],
        'y': demo[f'y_{rec}'],
        'meta': p
    }
    print(f"  Patient {rec}: {p['total']} beats ({p['rate']}% aritmia)")

# Set default selected patient to first one
state['selected_patient'] = patients_meta[0]['record_id']
print(f"Default patient: {state['selected_patient']}")
print("All data loaded!")

# ============================================================
# PER-PATIENT TELEGRAM REGISTRATIONS (SQLite Database)
# ============================================================
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ecg_data.db')

def db_init():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS telegram_registrations 
                 (record_id TEXT PRIMARY KEY, chat_id TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 full_name TEXT NOT NULL,
                 email TEXT UNIQUE NOT NULL,
                 password_hash TEXT NOT NULL,
                 role TEXT NOT NULL DEFAULT 'Tenaga Medis',
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()
    count = len(db_get_all_registrations())
    user_count = db_count_users()
    print(f"SQLite DB initialized — {count} Telegram registrations, {user_count} registered users")

# Access code required for registration
REGISTRATION_ACCESS_CODE = 'MEDIS2026'

def db_count_users():
    """Count total registered users."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM users')
    count = c.fetchone()[0]
    conn.close()
    return count

def db_get_user_by_email(email):
    """Get user by email address."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id, full_name, email, password_hash, role FROM users WHERE email = ?', (email.lower(),))
    row = c.fetchone()
    conn.close()
    if row:
        return {'id': row[0], 'full_name': row[1], 'email': row[2], 'password_hash': row[3], 'role': row[4]}
    return None

def db_create_user(full_name, email, password, role):
    """Create a new user account."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (full_name, email, password_hash, role) VALUES (?, ?, ?, ?)',
                  (full_name, email.lower(), generate_password_hash(password), role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def db_get_registration(record_id):
    """Get chat_id for a specific patient record_id."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT chat_id FROM telegram_registrations WHERE record_id = ?', (record_id,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

def db_set_registration(record_id, chat_id):
    """Insert or update a patient's Telegram registration."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO telegram_registrations (record_id, chat_id) VALUES (?, ?)', (record_id, str(chat_id)))
    conn.commit()
    conn.close()

def db_delete_registration(record_id):
    """Remove a patient's Telegram registration."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM telegram_registrations WHERE record_id = ?', (record_id,))
    conn.commit()
    conn.close()

def db_get_all_registrations():
    """Get all registrations as a dict {record_id: chat_id}."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT record_id, chat_id FROM telegram_registrations')
    rows = c.fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

def db_find_record_by_chat_id(chat_id):
    """Find a record_id by its associated chat_id."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT record_id FROM telegram_registrations WHERE chat_id = ?', (str(chat_id),))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None

db_init()

# ============================================================
# INTERACTIVE TELEGRAM AI BOT POLLING
# ============================================================
def send_bot_reply(token, chat_id, text):
    """Utility to send messages back to Telegram."""
    try:
        with state_lock:
            api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')
        url = f"{api_url}/bot{token}/sendMessage"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        requests.get(url, params={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }, headers=headers, timeout=10)
    except Exception as e:
        print(f"Failed to send bot reply: {e}")

def ask_groq_ai(user_message, groq_api_key):
    """Send user message to Groq AI and return the response."""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        # Get current WIB time to inject into prompt for real-time timezone context
        now_wib = get_wib_time()
        current_time_str = now_wib.strftime('%d %B %Y, %H:%M:%S WIB')
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "max_tokens": 500,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah asisten kesehatan jantung AI yang ramah dan profesional "
                        "untuk sistem monitoring EKG bernama LOVECG. "
                        f"Waktu saat ini di perangkat pasien (WIB): {current_time_str}.\n"
                        "PENTING: Sesuaikan ucapan salam hangat Anda (Pagi/Siang/Sore/Malam) "
                        "berdasarkan jam tersebut. Sebagai acuan: 05:00-11:59 adalah Pagi, "
                        "12:00-14:59 adalah Siang, 15:00-17:59 adalah Sore, dan 18:00-04:59 adalah Malam.\n\n"
                        "Sistem ini menggunakan dua algoritma AI secara bersamaan:\n"
                        "- CNN (Deep Learning): akurasi 99.4%, recall 100% pada sinyal bersih\n"
                        "- SVM (Machine Learning): latensi ~3ms, akurasi 99.4%\n\n"
                        "Jenis aritmia yang dipantau: Atrial Fibrillation (AFIB), "
                        "Atrial Flutter (AFL), Atrial Premature Beats (APB), "
                        "Ventricular Premature Beat (VPB).\n\n"
                        "Aturan penting:\n"
                        "1. Jawab dalam Bahasa Indonesia yang hangat dan mudah dipahami\n"
                        "2. Jangan gantikan saran dokter — selalu anjurkan konsultasi dokter\n"
                        "3. Jika pasien panik atau darurat, arahkan ke IGD atau 119\n"
                        "4. Jawaban singkat, maksimal 3 paragraf pendek\n"
                        "5. Jelaskan istilah medis dengan bahasa sederhana\n"
                        "6. Ingatkan pasien bisa ketik /help untuk melihat menu perintah"
                    )
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        }
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code == 200:
            data = res.json()
            return data['choices'][0]['message']['content']
        else:
            print(f"Groq API error: {res.status_code} — {res.text}")
            return None
    except Exception as e:
        print(f"Groq AI error: {e}")
        return None

def process_telegram_message(chat_id, text):
    """Processes a message received from Telegram and returns the reply text (or None)."""
    text = text.strip()
    if not text:
        return None
        
    # Check if this chat_id is registered
    matched_record = db_find_record_by_chat_id(chat_id)
    
    with state_lock:
        admin_chat_id = state.get('telegram_chat_id', '')
    is_admin = (str(chat_id) == str(admin_chat_id)) if admin_chat_id else False

    # Security: restrict bot commands and chat to registered/authorized users only to prevent porn/hijack spam
    if not text.startswith('/start') and not text.startswith('/daftar'):
        if not matched_record and not is_admin:
            return "⚠️ *Akses Ditolak.*\nAnda belum terdaftar di sistem notifikasi EKG ini. Harap daftarkan Record ID Anda terlebih dahulu menggunakan perintah:\n`/daftar [Record_ID]`"
        
    # Handle commands
    if text.startswith('/start'):
        return (
            f"🤖 *LOVECG AI ASSISTANT*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Halo! Selamat datang di asisten AI sistem EKG.\n\n"
            f"Saya terhubung langsung dengan sistem monitoring EKG laboratorium untuk menganalisis aritmia secara interaktif menggunakan kecerdasan buatan (*CNN & SVM*).\n\n"
            f"📌 *Daftar Perintah (Commands):*\n"
            f"• `/daftar [Record_ID]` : Hubungkan akun Telegram Anda dengan database lab (Contoh: `/daftar JS11550`)\n"
            f"• `/status` : Lihat analisis EKG terakhir Anda dari algoritma\n"
            f"• `/algoritma [CNN/SVM]` : Ubah algoritma yang digunakan untuk analisis (Contoh: `/algoritma SVM`)\n"
            f"• `/info` : Lihat edukasi klinis tentang Aritmia dan EKG\n"
            f"• `/batal` : Hapus registrasi Telegram Anda dari sistem"
        )
        
    elif text.startswith('/help'):
        return (
            f"📌 *Bantuan Perintah (Commands):*\n"
            f"• `/daftar [Record_ID]` - Registrasi pasien baru\n"
            f"• `/status` - Cek status jantung & analisis algoritma\n"
            f"• `/algoritma [CNN/SVM]` - Ganti algoritma klasifikasi\n"
            f"• `/info` - Edukasi Aritmia dan EKG\n"
            f"• `/batal` - Hapus registrasi Telegram"
        )
        
    elif text.startswith('/daftar'):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            return "⚠️ *Format Salah.*\nGunakan format: `/daftar [Record_ID]`\nContoh: `/daftar JS11550`"
            
        record_id = parts[1].strip()
        if record_id not in patient_data:
            return f"❌ *Registrasi Gagal.*\nRecord ID `{record_id}` tidak ditemukan dalam database laboratorium. Harap cek kembali Record ID pada lembar laboratorium Anda."
            
        # Save registration to SQLite database
        db_set_registration(record_id, chat_id)
        
        p_info = patient_data[record_id]['meta']
        arr_labels = p_info.get('arrhythmia_labels', [])
        arr_str = ', '.join(arr_labels) if arr_labels else 'Normal (Stabil)'
        
        return (
            f"✅ *REGISTRASI BERHASIL*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Halo **{p_info.get('name', record_id)}**, Anda telah berhasil terdaftar!\n\n"
            f"📋 *Data Lab Anda:*\n"
            f"• Record ID: `{record_id}`\n"
            f"• Usia: {p_info.get('age', '-')} tahun\n"
            f"• Jenis Kelamin: {'Wanita' if p_info.get('sex') == 'Female' else 'Pria'}\n"
            f"• Riwayat Diagnosis: *{arr_str}*\n\n"
            f"Sistem kami akan memantau sinyal EKG Anda secara real-time dan langsung memberi tahu Anda di sini jika ada indikasi bahaya aritmia."
        )
        
    elif text.startswith('/status'):
        # Find patient record linked to this chat_id (SQLite query)
        matched_record = db_find_record_by_chat_id(chat_id)
                
        if not matched_record:
            return "⚠️ *Anda Belum Terdaftar.*\nSilakan hubungkan akun Telegram Anda dengan Record ID laboratorium Anda terlebih dahulu.\n\nKetik: `/daftar [Record_ID]`"
            
        p_info = patient_data[matched_record]['meta']
        
        # Check if we have live simulation stats for this patient
        with state_lock:
            live_stats = state.get('live_stats', {}).get(matched_record)
            current_model = state.get('model_choice', 'CNN')
            
        if live_stats and (time.time() - live_stats.get('last_time', 0)) < 120:
            # We have active live monitoring statistics!
            live_bpm = live_stats.get('last_bpm', 72)
            live_total = live_stats.get('total', 0)
            live_abnormal = live_stats.get('abnormal', 0)
            live_normal = live_stats.get('normal', 0)
            live_rate = (live_abnormal / live_total * 100) if live_total > 0 else 0.0
            
            # Determine real-time alert status
            if live_abnormal >= 2:
                status_str = "⚠️ LIVE ARITMIA / SIAGA"
                status_color = "🔴"
            elif live_rate > 5.0:
                status_str = "⚠️ LIVE HATI-HATI"
                status_color = "🟡"
            else:
                status_str = "✓ LIVE SEHAT / STABIL"
                status_color = "🟢"
                
            now_wib = get_wib_time().strftime('%H:%M:%S WIB')
            return (
                f"📊 *LAPORAN ANALISIS EKG (REAL-TIME)*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"👤 *Pasien:* `{matched_record}`\n"
                f"📋 *Nama:* {p_info.get('name', matched_record)}\n\n"
                f"{status_color} *Status Saat Ini:* *{status_str}*\n"
                f"💓 *Heart Rate:* *{live_bpm} BPM*\n"
                f"📈 *Arrhythmia Rate (Live):* *{live_rate:.1f}%*\n"
                f"📊 *Total Detak Terpantau:* {live_total} beat\n"
                f"   • Normal: {live_normal} beat\n"
                f"   • Abnormal: {live_abnormal} beat\n\n"
                f"🧠 *Algoritma Aktif:* *{current_model} (1D)*\n"
                f" Waktu Update: {now_wib}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 *Sistem sedang melakukan pemantauan EKG secara aktif di laboratorium.*"
            )
        else:
            # Fallback to static lab database record
            rate = p_info.get('rate', 0.0)
            if rate > 20.0:
                status_str = "⚠️ ARITMIA / SIAGA"
                status_color = "🔴"
                status_desc = "Terdapat riwayat aritmia tinggi di lab. Harap rutin lakukan monitoring."
            elif rate > 5.0:
                status_str = "⚠️ HATI-HATI"
                status_color = "🟡"
                status_desc = "Terdapat sedikit riwayat ketidakteraturan ritme jantung."
            else:
                status_str = "✓ SEHAT / STABIL"
                status_color = "🟢"
                status_desc = "Ritme kelistrikan jantung Anda terpantau aman."
                
            return (
                f"📊 *LAPORAN ANALISIS EKG (LAB)*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"👤 *Pasien:* `{matched_record}`\n"
                f"📋 *Nama:* {p_info.get('name', matched_record)}\n\n"
                f"🧠 *Algoritma Aktif:* *{current_model} (1D)*\n"
                f"💓 *Rata-rata Denyut:* {int(p_info.get('rate', 0) * 0.4 + 70)} BPM\n"
                f"📈 *Arrhythmia Rate (Lab):* {rate:.1f}%\n"
                f"{status_color} *Status Terakhir:* *{status_str}*\n\n"
                f"📝 *Catatan Dokter:*\n"
                f"_{status_desc}_\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💡 *Catatan:* Pasien saat ini tidak sedang terpantau di Live Monitor. Menampilkan data lab historis."
            )
        
    elif text.startswith('/model') or text.startswith('/algoritma'):
        parts = text.split(maxsplit=1)
        if len(parts) < 2:
            with state_lock:
                m_choice = state.get('model_choice', 'CNN')
            return f"🧠 *Algoritma Aktif:* *{m_choice}*\n\nUbah dengan mengetik:\n• `/algoritma CNN`\n• `/algoritma SVM`"
            
        new_model = parts[1].strip().upper()
        if new_model not in ['CNN', 'SVM']:
            return "❌ *Gagal.* Algoritma tidak dikenal. Gunakan `CNN` atau `SVM`."
            
        with state_lock:
            state['model_choice'] = new_model
        return f"🔄 *Algoritma Berhasil Diubah!*\n\nSistem monitoring web sekarang menggunakan algoritma *{new_model}* untuk melakukan klasifikasi detak jantung secara real-time."
        
    elif text.startswith('/info'):
        return (
            f"ℹ️ *EDUKASI KLINIS EKG & ARITMIA*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🏥 *Apa itu Aritmia?*\n"
            f"Aritmia adalah gangguan irama jantung, di mana detak jantung bisa terlalu cepat, terlalu lambat, atau tidak teratur.\n\n"
            f"📊 *Cara Membaca Parameter:*\n"
            f"1. *BPM (Beats Per Minute):*\n"
            f"   • Normal: 60 - 100 BPM\n"
            f"   • Bradikardia: < 60 BPM (terlalu lambat)\n"
            f"   • Takikardia: > 100 BPM (terlalu cepat)\n\n"
            f"2. *Arrhythmia Rate (%):*\n"
            f"   Menunjukkan persentase detak abnormal yang terdeteksi dalam seluruh rekaman EKG Anda di laboratorium.\n\n"
            f"🧠 *Algoritma Deteksi (CNN vs SVM):*\n"
            f"• *CNN (1D):* Menggunakan Deep Learning dengan jaringan saraf konvolusi untuk mengenali lekukan bentuk gelombang EKG secara spasial.\n"
            f"• *SVM:* Algoritma cerdas klasik yang mengukur kecocokan morfologis berdasarkan fitur statistik."
        )
        
    elif text.startswith('/batal'):
        matched_record = db_find_record_by_chat_id(chat_id)
        if not matched_record:
            return "❌ Akun Anda belum terdaftar di sistem."
            
        db_delete_registration(matched_record)
        return f"🗑️ *Registrasi Dibatalkan.*\nRecord ID `{matched_record}` berhasil dihapus dari sistem notifikasi Telegram."
        
    else:
        # Default AI conversation flow (Groq AI)
        with state_lock:
            groq_key = state.get('groq_api_key', '')
            token = state.get('telegram_token')
            api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')
            
        # Send typing action (best effort outbound)
        try:
            requests.post(
                f"{api_url}/bot{token}/sendChatAction",
                json={"chat_id": chat_id, "action": "typing"},
                timeout=5
            )
        except Exception:
            pass
            
        if groq_key and groq_key != 'ISI_API_KEY_GROQ_KAMU_DI_SINI':
            ai_response = ask_groq_ai(text, groq_key)
            if ai_response:
                return (
                    f"🤖 *Asisten AI LOVECG:*\n\n"
                    f"{ai_response}\n\n"
                    f"_💡 Jawaban AI — konsultasikan dengan dokter untuk keputusan medis._"
                )
            else:
                return "🤖 Maaf, AI sedang tidak tersedia. Ketik `/help` untuk melihat perintah yang tersedia."
        else:
            return "🤖 Saya tidak memahami perintah tersebut.\n\nKetik `/help` untuk melihat daftar perintah asisten AI yang tersedia."

def telegram_bot_poll():
    """Background thread to poll Telegram for incoming messages (AI Bot Interaction)."""
    import traceback
    print("Telegram Bot Polling thread started.")
    offset = 0
    while True:
        try:
            with state_lock:
                token = state.get('telegram_token')
                api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')
            
            if not token:
                time.sleep(5)
                continue
                
            url = f"{api_url}/bot{token}/getUpdates"
            params = {'offset': offset, 'timeout': 5}
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            res = requests.get(url, params=params, headers=headers, timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                if data.get('ok'):
                    for update in data.get('result', []):
                        offset = update['update_id'] + 1
                        message = update.get('message', {})
                        chat_id = str(message.get('chat', {}).get('id', ''))
                        text = message.get('text', '').strip()
                        
                        if not chat_id or not text:
                            continue
                            
                        reply = process_telegram_message(chat_id, text)
                        if reply:
                            send_bot_reply(token, chat_id, reply)
            elif res.status_code == 409:
                # Conflict: webhook is active, so polling is disabled.
                # Sleep longer to avoid spamming the log.
                time.sleep(15)
                continue
            else:
                print(f"Telegram updates error: {res.status_code}")
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.ConnectionError:
            # Silent fallback to avoid flooding logs on HF Spaces outbound blocks
            time.sleep(15)
        except Exception as e:
            print(f"Telegram polling thread error: {e}")
            traceback.print_exc()
            time.sleep(5)

@app.route('/telegram_webhook', methods=['POST'])
def telegram_webhook():
    """Endpoint for Telegram webhook updates (bypasses outbound API blocks on HF Spaces)."""
    try:
        update = request.get_json()
        print(f"[WEBHOOK] Received update: {str(update)[:300]}")
        if not update:
            return jsonify({"status": "error", "message": "no data"}), 400
            
        message = update.get('message', {})
        chat_id = str(message.get('chat', {}).get('id', ''))
        text = message.get('text', '').strip()
        
        print(f"[WEBHOOK] chat_id={chat_id}, text={text}")
        
        if not chat_id or not text:
            return jsonify({"status": "ok"})
            
        reply = process_telegram_message(chat_id, text)
        if reply:
            print(f"[WEBHOOK] Replying to {chat_id}: {reply[:100]}...")
            return jsonify({
                "method": "sendMessage",
                "chat_id": chat_id,
                "text": reply,
                "parse_mode": "Markdown"
            })
    except Exception as e:
        print(f"[WEBHOOK] Error: {e}")
        import traceback
        traceback.print_exc()
        
    return jsonify({"status": "ok"})

# ============================================================

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=4):
    """Butterworth bandpass filter — matches training pipeline."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)

def add_awgn(signal, snr_db):
    """Add AWGN noise to simulate wireless transmission."""
    if snr_db >= 100:  # Clean signal
        return signal.copy()
    power = np.mean(signal ** 2)
    if power == 0:
        return signal.copy()
    snr = 10 ** (snr_db / 10)
    noise_power = power / snr
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

def classify_beat(beat, model_choice='CNN'):
    """Classify a single beat using CNN or SVM."""
    start = time.perf_counter()
    if model_choice == 'CNN':
        beat_input = beat.reshape(1, -1, 1)
        prob = float(cnn_model.predict(beat_input, verbose=0)[0][0])
        pred = 1 if prob > 0.5 else 0
    else:
        beat_input = beat.reshape(1, -1)
        pred = int(svm_model.predict(beat_input)[0])
        prob = float(svm_model.predict_proba(beat_input)[0][1])
    elapsed_ms = (time.perf_counter() - start) * 1000
    return pred, prob, elapsed_ms

def classify_both(beat):
    """Classify a beat with BOTH CNN and SVM for benchmarking."""
    cnn_pred, cnn_prob, cnn_ms = classify_beat(beat, 'CNN')
    svm_pred, svm_prob, svm_ms = classify_beat(beat, 'SVM')
    return {
        'cnn': {'pred': cnn_pred, 'prob': cnn_prob, 'ms': round(cnn_ms, 2)},
        'svm': {'pred': svm_pred, 'prob': svm_prob, 'ms': round(svm_ms, 2)}
    }

def send_telegram_alert(patient_id, bpm, confidence, arrhythmia_type="Abnormal", model_choice="CNN"):
    """Send a rich, per-patient alert message via Telegram Bot API."""
    with state_lock:
        token = state.get('telegram_token')
        admin_chat_id = state.get('telegram_chat_id')
        api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')
    
    if not token:
        return False
    
    # Per-patient chat_id lookup (SQLite query)
    patient_chat_id = db_get_registration(patient_id)
    
    # If no patient-specific and no admin chat_id, skip
    if not patient_chat_id and not admin_chat_id:
        return False
    
    # Get patient metadata for rich message
    patient_info = patient_data.get(patient_id, {}).get('meta', {})
    age = patient_info.get('age', '-')
    sex = patient_info.get('sex', '-')
    sex_label = 'Wanita' if sex == 'Female' else 'Pria' if sex == 'Male' else sex
    arr_labels = patient_info.get('arrhythmia_labels', [])
    arr_detail = ', '.join(arr_labels) if arr_labels else arrhythmia_type
    
    # Determine HR condition
    if bpm > 100:
        hr_status = "TAKIKARDIA"
        hr_icon = "⚡"
        hr_desc = "Detak jantung terlalu cepat"
    elif bpm < 60:
        hr_status = "BRADIKARDIA"
        hr_icon = "🔽"
        hr_desc = "Detak jantung terlalu lambat"
    else:
        hr_status = "ARITMIA"
        hr_icon = "💔"
        hr_desc = "Terdeteksi gejala aritmia"
    
    now = get_wib_time().strftime('%d %b %Y, %H:%M:%S WIB')
    
    message = (
        f"🏥 *ECG MONITORING SYSTEM*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"⚠️ *PERINGATAN DETEKSI ARITMIA*\n\n"
        f"👤 *Pasien:* `{patient_id}`\n"
        f"     Usia: {age} tahun  |  {sex_label}\n\n"
        f"💓 *Kondisi Terdeteksi:*\n"
        f"     {hr_icon} Status: *{hr_status}*\n"
        f"     • Heart Rate: *{bpm} BPM*\n"
        f"     • {hr_desc}\n"
        f"     • Jenis Aritmia: {arr_detail}\n\n"
        f"📊 *Analisis AI ({model_choice}):*\n"
        f"     • Prediksi: *ABNORMAL*\n"
        f"     • Confidence: *{confidence * 100:.1f}%*\n\n"
        f"🕐 Waktu Deteksi: {now}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔴 *TINDAKAN DIPERLUKAN*\n"
        f"Segera periksa kondisi pasien\n"
        f"dan hubungi dokter jaga.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━"
    )
    
    # Collect all target chat IDs
    targets = set()
    if patient_chat_id:
        targets.add(str(patient_chat_id))
    if admin_chat_id:
        targets.add(str(admin_chat_id))
    try:
        all_regs = db_get_all_registrations()
        for cid in all_regs.values():
            targets.add(str(cid))
    except Exception:
        pass
    
    if not targets:
        print(f"[TELEGRAM] No targets for {patient_id}, skipping.")
        return False
    
    # Queue message for browser-side dispatch (bypasses HF outbound blocks)
    outbox_item = {
        'id': int(time.time() * 1000),
        'targets': list(targets),
        'token': token,
        'api_url': api_url,
        'message': message,
        'patient_id': patient_id,
        'timestamp': get_wib_time().strftime('%H:%M:%S WIB')
    }
    with telegram_outbox_lock:
        telegram_outbox.append(outbox_item)
    print(f"[TELEGRAM] Alert for {patient_id} queued for browser dispatch -> targets: {targets}")
    return True

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        google_jwt = request.form.get('google_jwt')
        if google_jwt:
            try:
                from google.oauth2 import id_token
                from google.auth.transport import requests as google_requests
                
                # Verify Google ID Token
                idinfo = id_token.verify_oauth2_token(google_jwt, google_requests.Request(), GOOGLE_CLIENT_ID)
                
                # Verify issuer
                if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                    raise ValueError('Issuer tidak valid.')
                
                email = idinfo.get('email', '').strip().lower()
                name = idinfo.get('name', 'User Google')
                
                # Check if Google user is registered in our system
                user = db_get_user_by_email(email)
                if user:
                    session['logged_in'] = True
                    session['user_email'] = email
                    session['user_name'] = user['full_name']
                    session['user_role'] = user['role']
                    return redirect(url_for('index'))
                elif email.endswith('.ac.id') or email.endswith('@gmail.com'):
                    session['logged_in'] = True
                    session['user_email'] = email
                    session['user_name'] = name
                    session['user_role'] = 'Google User'
                    return redirect(url_for('index'))
                else:
                    error = 'Akses ditolak: Gunakan email Google (.ac.id atau @gmail.com)!'
            except Exception as e:
                error = f'Verifikasi Akun Google Gagal: {str(e)}'
        else:
            email_or_username = request.form.get('username', '').strip().lower()
            password = request.form.get('password', '')
            
            # 1. Default Admin Login (password hashed with werkzeug)
            if email_or_username == 'admin' and check_password_hash(DEMO_PASSWORD_HASH, password):
                session['logged_in'] = True
                session['user_email'] = 'admin@loveecg.net'
                session['user_name'] = 'System Admin'
                session['user_role'] = 'Pembuat Web'
                return redirect(url_for('index'))
            
            # 2. Check registered users in database
            user = db_get_user_by_email(email_or_username)
            if user:
                if check_password_hash(user['password_hash'], password):
                    session['logged_in'] = True
                    session['user_email'] = user['email']
                    session['user_name'] = user['full_name']
                    session['user_role'] = user['role']
                    return redirect(url_for('index'))
                else:
                    error = 'Password salah! Silakan coba lagi.'
                
            # 3. Google Mock SSO for Gmail/Academic (simulation mode)
            elif email_or_username.endswith('.ac.id') or email_or_username.endswith('@gmail.com'):
                if password == 'google-sso-verified':
                    session['logged_in'] = True
                    session['user_email'] = email_or_username
                    name_part = email_or_username.split('@')[0]
                    session['user_name'] = name_part.replace('.', ' ').title()
                    session['user_role'] = 'Google User'
                    return redirect(url_for('index'))
                else:
                    error = 'Akun belum terdaftar. Silakan daftar terlebih dahulu atau gunakan Google Sign-In.'
            else:
                error = 'Gunakan email akademik (.ac.id), akun Gmail, atau admin default!'
            
    return render_template('login.html', error=error, google_client_id=GOOGLE_CLIENT_ID)


@app.route('/register', methods=['POST'])
def register():
    """Handle user registration with access code verification."""
    full_name = request.form.get('full_name', '').strip()
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    role = request.form.get('role', 'Tenaga Medis')
    if role == 'pembuat_web':
        role = 'Pembuat Web'
    elif role == 'tenaga_medis':
        role = 'Tenaga Medis'
        
    access_code = request.form.get('access_code', '').strip()
    
    # Validate access code
    if access_code != REGISTRATION_ACCESS_CODE:
        return render_template('login.html', 
                             signup_error='Kode akses salah! Hubungi administrator untuk mendapatkan kode akses yang benar.',
                             google_client_id=GOOGLE_CLIENT_ID)
    
    # Validate full name
    if not full_name or len(full_name) < 2:
        return render_template('login.html',
                             signup_error='Nama lengkap harus diisi (minimal 2 karakter).',
                             google_client_id=GOOGLE_CLIENT_ID)
    
    # Validate email
    if not (email.endswith('@gmail.com') or '.ac.id' in email):
        return render_template('login.html',
                             signup_error='Gunakan email @gmail.com atau domain akademik (.ac.id)!',
                             google_client_id=GOOGLE_CLIENT_ID)
    
    # Validate password
    if len(password) < 6:
        return render_template('login.html',
                             signup_error='Password harus minimal 6 karakter.',
                             google_client_id=GOOGLE_CLIENT_ID)
    
    # Validate role
    if role not in ['Pembuat Web', 'Tenaga Medis']:
        return render_template('login.html',
                             signup_error='Role tidak valid.',
                             google_client_id=GOOGLE_CLIENT_ID)
    
    # Try to create user
    success = db_create_user(full_name, email, password, role)
    if success:
        return render_template('login.html',
                             signup_success=f'Akun berhasil dibuat! Silakan login dengan email {email}.',
                             google_client_id=GOOGLE_CLIENT_ID)
    else:
        return render_template('login.html',
                             signup_error='Email sudah terdaftar! Silakan login atau gunakan email lain.',
                             google_client_id=GOOGLE_CLIENT_ID)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================================================
# PATIENT API
# ============================================================
@app.route('/api/patients')
def get_patients():
    """Return list of 20 patients with metadata."""
    with state_lock:
        sel_patient = state['selected_patient']
    return jsonify({
        'patients': patients_meta,
        'selected': sel_patient
    })

@app.route('/api/select_patient', methods=['POST'])
def select_patient():
    """Switch the active patient for streaming."""
    data = request.json
    record_id = data.get('record_id', '')
    if record_id in patient_data:
        with state_lock:
            state['selected_patient'] = record_id
        return jsonify({
            'status': 'ok',
            'selected': record_id,
            'meta': patient_data[record_id]['meta']
        })
    return jsonify({'status': 'error', 'message': 'Patient not found'}), 404

# ============================================================
# STREAMING
# ============================================================
@app.route('/api/stream')
def stream():
    """SSE endpoint — stream ECG beats one at a time for selected patient."""
    def generate():
        with state_lock:
            current_patient = state['selected_patient']
            speed = state['speed']
        
        # Calculate current global tick based on speed
        elapsed = time.time() - global_sim_start
        global_tick = int(elapsed / speed)
        
        # Each patient might have different length, wrap around
        total_beats = len(patient_data[current_patient]['y'])
        beat_index = global_tick % total_beats
        
        stats = {
            'total': 0,
            'normal': 0,
            'abnormal': 0,
            'alerts': []
        }

        consecutive_abnormal = 0
        last_alert_time = 0
        last_valid_beat = None  # For packet loss interpolation (kept outside stats to avoid JSON issues)

        while True:
            with state_lock:
                is_streaming = state['is_streaming']
                sel_patient = state['selected_patient']
                speed = state['speed']
                source = state['source']
                model_choice = state['model_choice']
                snr_db = state['snr_db']
                packet_loss_rate = state['packet_loss']

            if not is_streaming:
                time.sleep(0.5)
                continue

            # Check if patient changed
            if current_patient != sel_patient:
                current_patient = sel_patient
                
                # Re-sync to new patient
                elapsed = time.time() - global_sim_start
                global_tick = int(elapsed / speed)
                total_beats = len(patient_data[current_patient]['y'])
                beat_index = global_tick % total_beats
                
                stats = {
                    'total': 0,
                    'normal': 0,
                    'abnormal': 0,
                    'alerts': []
                }
                consecutive_abnormal = 0
                last_alert_time = 0
                last_valid_beat = None
                # Send a reset event to the frontend
                reset_data = {
                    'type': 'patient_change',
                    'record_id': current_patient,
                    'meta': patient_data[current_patient]['meta']
                }
                yield f"data: {json.dumps(reset_data)}\n\n"
                continue

            if source == 'live':
                # LIVE MODE: read from the queue
                if len(live_results_queue) == 0:
                    time.sleep(0.1)
                    continue

                live_item = live_results_queue.popleft()

                beat_index += 1
                stats['total'] += 1
                if live_item['prediction'] == 0:
                    stats['normal'] += 1
                else:
                    stats['abnormal'] += 1

                # Update live stats in global state for Telegram bot /status command
                with state_lock:
                    if 'live_stats' not in state:
                        state['live_stats'] = {}
                    state['live_stats'][current_patient] = {
                        'total': stats['total'],
                        'normal': stats['normal'],
                        'abnormal': stats['abnormal'],
                        'last_bpm': live_item['heart_rate'],
                        'last_time': time.time()
                    }

                data = {
                    'type': 'beat',
                    'beat_index': beat_index,
                    'beat_data': live_item['beat_data'].tolist(),
                    'clean_data': live_item['clean_data'].tolist(),
                    'prediction': live_item['prediction'],
                    'label': live_item['label'],
                    'confidence': live_item['confidence'],
                    'true_label': -1,
                    'heart_rate': live_item['heart_rate'],
                    'model': model_choice,
                    'snr_db': snr_db,
                    'packet_loss_rate': packet_loss_rate,
                    'record_id': current_patient,
                    'stats': stats.copy()
                }
                if 'benchmark' in live_item:
                    data['benchmark'] = live_item['benchmark']

                if live_item['prediction'] == 1:
                    alert = {
                        'time': get_wib_time().strftime('%H:%M:%S'),
                        'beat': beat_index,
                        'confidence': live_item['confidence'],
                        'heart_rate': live_item['heart_rate']
                    }
                    stats['alerts'].append(alert)
                    if len(stats['alerts']) > 50:
                        stats['alerts'] = stats['alerts'][-50:]
                    consecutive_abnormal += 1
                    
                    # Trigger Telegram Alert if 2 consecutive abnormals and at least 30 seconds since last alert
                    current_time = time.time()
                    if consecutive_abnormal >= 2 and (current_time - last_alert_time) > 30:
                        patient_info = patient_data.get(current_patient, {}).get('meta', {})
                        arrhythmia_labels = patient_info.get('arrhythmia_labels', [])
                        arr_type = ', '.join(arrhythmia_labels) if arrhythmia_labels else "Abnormal Rhythm"
                        
                        send_telegram_alert(
                            patient_id=current_patient,
                            bpm=live_item['heart_rate'],
                            confidence=live_item['confidence'],
                            arrhythmia_type=arr_type,
                            model_choice=model_choice
                        )
                        last_alert_time = current_time
                        consecutive_abnormal = 0 # reset after alert
                else:
                    consecutive_abnormal = 0

                # Push to global alert store for cross-patient Alert Center (independent of active model selection)
                alert_time = get_wib_time().strftime('%H:%M:%S')
                if 'benchmark' in live_item:
                    if live_item['benchmark']['cnn_pred'] == 1:
                        with global_alerts_lock:
                            global_alerts.append({
                                'id': int(time.time() * 1000),
                                'patient_id': current_patient,
                                'time': alert_time,
                                'beat': beat_index,
                                'confidence': live_item['benchmark']['cnn_conf'],
                                'heart_rate': live_item['heart_rate'],
                                'model': 'CNN'
                            })
                    if live_item['benchmark']['svm_pred'] == 1:
                        with global_alerts_lock:
                            global_alerts.append({
                                'id': int(time.time() * 1000) + 1,
                                'patient_id': current_patient,
                                'time': alert_time,
                                'beat': beat_index,
                                'confidence': live_item['benchmark']['svm_conf'],
                                'heart_rate': live_item['heart_rate'],
                                'model': 'SVM'
                            })
                else:
                    if live_item['prediction'] == 1:
                        with global_alerts_lock:
                            global_alerts.append({
                                'id': int(time.time() * 1000),
                                'patient_id': current_patient,
                                'time': alert_time,
                                'beat': beat_index,
                                'confidence': round(live_item['confidence'], 4),
                                'heart_rate': live_item['heart_rate'],
                                'model': model_choice
                            })

                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(0.1)

            else:
                # DEMO MODE: stream from selected patient
                p = patient_data[current_patient]
                total_beats = len(p['y'])

                clean_beat = p['X'][beat_index % total_beats]
                true_label = int(p['y'][beat_index % total_beats])

                # Simulate packet loss
                is_lost = np.random.random() < packet_loss_rate

                if is_lost:
                    # Interpolate from last valid beat if available
                    interpolated_data = None
                    if last_valid_beat is not None:
                        interpolated_data = last_valid_beat.tolist()

                    data = {
                        'type': 'packet_loss',
                        'beat_index': beat_index,
                        'record_id': current_patient,
                        'message': 'Network Packet Lost (IoT)',
                        'interpolated_data': interpolated_data
                    }
                else:
                    # Apply wireless noise
                    noisy_beat = add_awgn(clean_beat, snr_db)

                    # Classify with BOTH models for benchmarking
                    both = classify_both(noisy_beat)

                    # Use the selected model for main prediction
                    chosen = both['cnn'] if model_choice == 'CNN' else both['svm']
                    pred = chosen['pred']
                    prob = chosen['prob']

                    stats['total'] += 1
                    if pred == 0:
                        stats['normal'] += 1
                    else:
                        stats['abnormal'] += 1

                    # Save last valid beat for interpolation on packet loss
                    last_valid_beat = noisy_beat.copy()

                    # Calculate simulated heart rate
                    base_hr = 72
                    hr_variation = np.random.randint(-8, 9)
                    if pred == 1:
                        hr_variation += np.random.randint(5, 20)
                    heart_rate = base_hr + hr_variation

                    # Update live stats in global state for Telegram bot /status command
                    with state_lock:
                        if 'live_stats' not in state:
                            state['live_stats'] = {}
                        state['live_stats'][current_patient] = {
                            'total': stats['total'],
                            'normal': stats['normal'],
                            'abnormal': stats['abnormal'],
                            'last_bpm': heart_rate,
                            'last_time': time.time()
                        }

                    data = {
                        'type': 'beat',
                        'beat_index': beat_index,
                        'beat_data': noisy_beat.tolist(),
                        'clean_data': clean_beat.tolist(),
                        'prediction': pred,
                        'label': 'Abnormal' if pred == 1 else 'Normal',
                        'confidence': round(prob if pred == 1 else 1 - prob, 4),
                        'true_label': true_label,
                        'heart_rate': heart_rate,
                        'model': model_choice,
                        'snr_db': snr_db,
                        'packet_loss_rate': packet_loss_rate,
                        'record_id': current_patient,
                        'stats': stats.copy(),
                        'benchmark': {
                            'cnn_pred': both['cnn']['pred'],
                            'cnn_conf': round(both['cnn']['prob'] if both['cnn']['pred'] == 1 else 1 - both['cnn']['prob'], 4),
                            'cnn_ms': both['cnn']['ms'],
                            'svm_pred': both['svm']['pred'],
                            'svm_conf': round(both['svm']['prob'] if both['svm']['pred'] == 1 else 1 - both['svm']['prob'], 4),
                            'svm_ms': both['svm']['ms']
                        }
                    }

                    # Alert for abnormal
                    if pred == 1:
                        alert = {
                            'time': get_wib_time().strftime('%H:%M:%S'),
                            'beat': beat_index,
                            'confidence': round(prob, 4),
                            'heart_rate': heart_rate
                        }
                        stats['alerts'].append(alert)
                        if len(stats['alerts']) > 50:
                            stats['alerts'] = stats['alerts'][-50:]
                        consecutive_abnormal += 1
                        
                        # Trigger Telegram Alert if 2 consecutive abnormals and at least 30 seconds since last alert
                        current_time = time.time()
                        if consecutive_abnormal >= 2 and (current_time - last_alert_time) > 30:
                            patient_info = patient_data.get(current_patient, {}).get('meta', {})
                            arrhythmia_labels = patient_info.get('arrhythmia_labels', [])
                            arr_type = ', '.join(arrhythmia_labels) if arrhythmia_labels else "Abnormal Rhythm"
                            
                            send_telegram_alert(
                                patient_id=current_patient,
                                bpm=heart_rate,
                                confidence=round(prob, 4),
                                arrhythmia_type=arr_type,
                                model_choice=model_choice
                            )
                            last_alert_time = current_time
                            consecutive_abnormal = 0 # reset after alert
                    else:
                        consecutive_abnormal = 0

                    # Push to global alert store for cross-patient Alert Center (independent of active model selection)
                    alert_time = get_wib_time().strftime('%H:%M:%S')
                    if both['cnn']['pred'] == 1:
                        with global_alerts_lock:
                            global_alerts.append({
                                'id': int(time.time() * 1000),
                                'patient_id': current_patient,
                                'time': alert_time,
                                'beat': beat_index,
                                'confidence': round(both['cnn']['prob'], 4),
                                'heart_rate': heart_rate,
                                'model': 'CNN'
                            })
                    if both['svm']['pred'] == 1:
                        with global_alerts_lock:
                            global_alerts.append({
                                'id': int(time.time() * 1000) + 1,
                                'patient_id': current_patient,
                                'time': alert_time,
                                'beat': beat_index,
                                'confidence': round(both['svm']['prob'], 4),
                                'heart_rate': heart_rate,
                                'model': 'SVM'
                            })

                yield f"data: {json.dumps(data)}\n\n"

                beat_index += 1
                time.sleep(speed)

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update simulation settings."""
    data = request.json
    with state_lock:
        if 'model_choice' in data:
            state['model_choice'] = data['model_choice']
        if 'snr_db' in data:
            state['snr_db'] = int(data['snr_db'])
        if 'packet_loss' in data:
            state['packet_loss'] = float(data['packet_loss'])
        if 'is_streaming' in data:
            state['is_streaming'] = bool(data['is_streaming'])
        if 'speed' in data:
            state['speed'] = float(data['speed'])
        if 'source' in data:
            state['source'] = data['source']
            if data['source'] == 'live':
                global live_ecg_buffer, live_results_queue
                live_ecg_buffer = deque(maxlen=MAX_ECG_BUFFER)
                live_results_queue = deque(maxlen=MAX_RESULTS_QUEUE)
        state_copy = state.copy()
    return jsonify({'status': 'ok', 'state': state_copy})

@app.route('/api/telegram_config', methods=['POST', 'GET'])
def telegram_config():
    if request.method == 'POST':
        data = request.json
        with state_lock:
            if 'telegram_token' in data:
                state['telegram_token'] = data['telegram_token']
            if 'telegram_chat_id' in data:
                state['telegram_chat_id'] = data['telegram_chat_id']
            if 'telegram_api_url' in data:
                state['telegram_api_url'] = data['telegram_api_url'].strip().rstrip('/')
        return jsonify({'status': 'ok'})
    else:
        with state_lock:
            token = state.get('telegram_token', '')
            chat_id = state.get('telegram_chat_id', '')
            api_url = state.get('telegram_api_url', 'https://api.telegram.org')
        return jsonify({
            'telegram_token': token,
            'telegram_chat_id': chat_id,
            'telegram_api_url': api_url
        })

@app.route('/api/debug_telegram')
def debug_telegram():
    env_token = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    env_api_url = os.environ.get('TELEGRAM_API_URL', 'https://api.telegram.org').rstrip('/')
    with state_lock:
        state_token = state.get('telegram_token', '')
        state_chat_id = state.get('telegram_chat_id', '')
        state_api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')

    def mask_token(t):
        if not t: return "None"
        parts = t.split(':')
        if len(parts) != 2: return f"InvalidFormat (length: {len(t)})"
        return f"{parts[0]}:{parts[1][:4]}...{parts[1][-4:]} (length: {len(t)})"

    # Test multiple domains
    results = {}
    test_urls = {
        'httpbin': 'https://httpbin.org/get',
        'cloudflare_worker_get': 'https://plain-truth-129f.hasfya18.workers.dev/',
        'telegram_api': env_api_url + '/bot' + env_token + '/getMe',
        'telegram_direct_getMe': 'https://api.telegram.org/bot' + env_token + '/getMe',
        'telegram_direct_root': 'https://api.telegram.org/'
    }

    for name, config in test_urls.items():
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            r = requests.get(config, headers=headers, timeout=10)
            results[name] = f"Success: {r.status_code} {r.text[:50]}"
        except Exception as e:
            results[name] = f"Error: {e}"

    return jsonify({
        'telegram_dns_ip': '149.154.166.110',
        'env_token_masked': mask_token(env_token),
        'polling_thread_alive': t_bot.is_alive(),
        'test_results': results
    })

@app.route('/api/test_telegram_alert')
def test_telegram_alert():
    patient_id = request.args.get('patient_id', 'JS08073')
    bpm = 105
    confidence = 0.985
    arrhythmia_type = "AF"
    
    # Trigger alert
    sent = send_telegram_alert(patient_id, bpm, confidence, arrhythmia_type, "CNN")
    
    return jsonify({
        "status": "success" if sent else "error",
        "message": f"Telegram alert trigger sent for {patient_id}. Check server logs for delivery status."
    })


# ============================================================
# PER-PATIENT TELEGRAM REGISTRATION
# ============================================================
@app.route('/api/telegram/register', methods=['POST'])
def register_telegram():
    """Register a patient's Telegram Chat ID for personal notifications."""
    data = request.json
    record_id = data.get('record_id', '').strip()
    chat_id = data.get('chat_id', '').strip()
    
    if not record_id or not chat_id:
        return jsonify({'status': 'error', 'message': 'Record ID dan Chat ID wajib diisi'}), 400
    
    # Validate patient exists in database
    if record_id not in patient_data:
        return jsonify({'status': 'error', 'message': f'Pasien {record_id} tidak ditemukan dalam database laboratorium'}), 404
    
    # Validate bot token exists
    with state_lock:
        token = state.get('telegram_token')
        api_url = state.get('telegram_api_url', 'https://api.telegram.org').rstrip('/')
    if not token:
        return jsonify({'status': 'error', 'message': 'Bot Token belum dikonfigurasi di halaman Settings'}), 500
    
    # Send verification message to patient
    patient_info = patient_data[record_id]['meta']
    arr_labels = patient_info.get('arrhythmia_labels', [])
    arr_str = ', '.join(arr_labels) if arr_labels else 'Belum terdeteksi'
    
    verify_msg = (
        f"✅ *REGISTRASI BERHASIL*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"Halo! Anda telah terdaftar untuk\n"
        f"menerima notifikasi EKG.\n\n"
        f"👤 *Pasien:* `{record_id}`\n"
        f"📋 *Nama:* {patient_info.get('name', record_id)}\n"
        f"🎂 *Usia:* {patient_info.get('age', '-')} tahun\n"
        f"⚧ *Jenis Kelamin:* {'Wanita' if patient_info.get('sex') == 'Female' else 'Pria'}\n"
        f"📊 *Riwayat Aritmia:* {arr_str}\n"
        f"💡 *Waktu:* {get_wib_time().strftime('%d %b %Y, %H:%M:%S WIB')}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Anda akan menerima peringatan\n"
        f"otomatis jika sistem mendeteksi\n"
        f"aritmia pada rekaman EKG Anda.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━"
    )
    
    try:
        url = f"{api_url}/bot{token}/sendMessage"
        res = requests.post(url, json={
            'chat_id': chat_id,
            'text': verify_msg,
            'parse_mode': 'Markdown'
        }, timeout=10)
        
        if res.status_code != 200:
            return jsonify({'status': 'error', 'message': 'Chat ID tidak valid atau pasien belum mengirim /start ke bot Telegram'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Gagal mengirim pesan verifikasi: {str(e)}'}), 500
    
    # Save registration to SQLite
    db_set_registration(record_id, chat_id)
    
    return jsonify({
        'status': 'ok',
        'message': f'Pasien {record_id} berhasil didaftarkan untuk notifikasi Telegram',
        'patient': patient_info
    })

@app.route('/api/telegram/registrations')
def get_telegram_registrations():
    """Get all registered patients for Telegram notifications."""
    result = []
    all_regs = db_get_all_registrations()
    for record_id, chat_id in all_regs.items():
        meta = patient_data.get(record_id, {}).get('meta', {})
        result.append({
            'record_id': record_id,
            'chat_id': chat_id,
            'name': meta.get('name', record_id),
            'age': meta.get('age', '-'),
            'sex': meta.get('sex', '-'),
            'rate': meta.get('rate', 0)
        })
    return jsonify({'registrations': result})

@app.route('/api/telegram/unregister', methods=['POST'])
def unregister_telegram():
    """Remove a patient's Telegram registration."""
    data = request.json
    record_id = data.get('record_id', '')
    if db_get_registration(record_id):
        db_delete_registration(record_id)
        return jsonify({'status': 'ok', 'message': f'Pasien {record_id} dihapus dari notifikasi Telegram'})
    return jsonify({'status': 'error', 'message': 'Pasien tidak ditemukan dalam daftar registrasi'}), 404

# ============================================================
# FALLBACK ALERT API (when Telegram fails)
# ============================================================
@app.route('/api/pending_alerts')
def get_pending_alerts():
    """Get alerts that failed to send via Telegram (fallback for web dashboard)."""
    with failed_alerts_lock:
        alerts = list(failed_alerts)
    return jsonify({'alerts': alerts, 'count': len(alerts)})

@app.route('/api/dismiss_alerts', methods=['POST'])
def dismiss_alerts():
    """Clear all pending fallback alerts."""
    with failed_alerts_lock:
        failed_alerts.clear()
    return jsonify({'status': 'ok'})

@app.route('/api/telegram_outbox')
def get_telegram_outbox():
    """Get queued Telegram messages for browser-side dispatch."""
    with telegram_outbox_lock:
        items = list(telegram_outbox)
    return jsonify(items)

@app.route('/api/telegram_outbox/clear', methods=['POST'])
def clear_telegram_outbox():
    """Remove dispatched items from the outbox."""
    data = request.get_json() or {}
    ids_to_clear = set(data.get('ids', []))
    with telegram_outbox_lock:
        # Remove only the items that were successfully dispatched
        remaining = [item for item in telegram_outbox if item['id'] not in ids_to_clear]
        telegram_outbox.clear()
        telegram_outbox.extend(remaining)
    return jsonify({'status': 'ok', 'remaining': len(telegram_outbox)})

@app.route('/api/global_alerts')
def get_global_alerts():
    """Get arrhythmia alerts from ALL patients for the Alert Center sidebar."""
    with global_alerts_lock:
        alerts = list(global_alerts)
    return jsonify({'alerts': alerts, 'count': len(alerts)})


@app.route('/api/ecg_device', methods=['POST'])
def receive_ecg_data():
    """Endpoint for smartwatch to stream IoT data."""
    global live_ecg_buffer, live_results_queue
    data = request.json

    if not data or 'values' not in data:
        return jsonify({'error': 'No data provided'}), 400

    with state_lock:
        state['source'] = 'live'
        snr_db = state['snr_db']
        model_choice = state['model_choice']

    values = data['values']
    live_ecg_buffer.extend(values)

    responses = []

    while len(live_ecg_buffer) >= BEAT_SIZE:
        # Extract one beat from circular buffer
        beat = [live_ecg_buffer.popleft() for _ in range(BEAT_SIZE)]

        clean_beat = np.array(beat)
        # Apply bandpass filter to match training pipeline
        try:
            filtered_beat = bandpass_filter(clean_beat, fs=500)
        except Exception:
            filtered_beat = clean_beat  # Fallback if filter fails
        noisy_beat = add_awgn(filtered_beat, snr_db)

        # Classify with BOTH models for benchmarking/dual vitals cards
        both = classify_both(noisy_beat)
        
        # Use active model selection for the main response/prediction flow
        chosen = both['cnn'] if model_choice == 'CNN' else both['svm']
        pred = chosen['pred']
        prob = chosen['prob']

        base_hr = 72
        hr_variation = np.random.randint(-8, 9)
        if pred == 1:
            hr_variation += np.random.randint(5, 20)
        heart_rate = base_hr + hr_variation

        live_results_queue.append({
            'beat_data': noisy_beat,
            'clean_data': clean_beat,
            'prediction': pred,
            'label': 'Abnormal' if pred == 1 else 'Normal',
            'confidence': round(prob if pred == 1 else 1 - prob, 4),
            'heart_rate': heart_rate,
            'benchmark': {
                'cnn_pred': both['cnn']['pred'],
                'cnn_conf': round(both['cnn']['prob'] if both['cnn']['pred'] == 1 else 1 - both['cnn']['prob'], 4),
                'cnn_ms': both['cnn']['ms'],
                'svm_pred': both['svm']['pred'],
                'svm_conf': round(both['svm']['prob'] if both['svm']['pred'] == 1 else 1 - both['svm']['prob'], 4),
                'svm_ms': both['svm']['ms']
            }
        })

        responses.append({
            'aritmia_terdeteksi': bool(pred == 1),
            'confidence': round(prob, 4),
            'heart_rate': heart_rate
        })

    if responses:
        last_resp = responses[-1]
        return jsonify({
            'status': 'processed',
            'aritmia_terdeteksi': last_resp['aritmia_terdeteksi'],
            'heart_rate': last_resp['heart_rate']
        })

    return jsonify({'status': 'buffered', 'aritmia_terdeteksi': False})

@app.route('/api/status')
def get_status():
    """Get current settings."""
    with state_lock:
        state_copy = state.copy()
    return jsonify(state_copy)

# ============================================================
# MAIN & BOT STARTUP
# ============================================================
# Start Telegram AI Bot polling thread at module level so it runs on both Gunicorn and local Python
t_bot = threading.Thread(target=telegram_bot_poll, daemon=True)
t_bot.start()

if __name__ == '__main__':
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "0.0.0.0"

    # Generate self-signed SSL certificate for HTTPS
    cert_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cert.pem')
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'key.pem')
    ssl_ctx = None

    if not (os.path.exists(cert_file) and os.path.exists(key_file)):
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            import datetime, ipaddress

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, u"ECG Monitoring System"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, u"Skripsi EKG"),
            ])
            cert = (x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.utcnow())
                .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
                .add_extension(x509.SubjectAlternativeName([
                    x509.DNSName(u"localhost"),
                    x509.IPAddress(ipaddress.IPv4Address(u"127.0.0.1")),
                ]), critical=False)
                .sign(key, hashes.SHA256())
            )
            with open(cert_file, 'wb') as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            with open(key_file, 'wb') as f:
                f.write(key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()
                ))
            print("[OK] Self-signed SSL certificate generated (cert.pem, key.pem)")
        except ImportError:
            print("[WARNING] 'cryptography' package not installed. Running without HTTPS.")
            print("   Install with: pip install cryptography")
        except Exception as e:
            print(f"[WARNING] Could not generate SSL certificate: {e}")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_ctx = (cert_file, key_file)
        protocol = "https"
    else:
        protocol = "http"

    print("\n" + "=" * 50)
    print("  ECG Real-Time Monitoring App")
    print(f"  {len(patients_meta)} pasien dimuat (per-pasien)")
    print(f"  Database: SQLite (ecg_data.db)")
    print(f"  Keamanan: {'HTTPS (SSL/TLS)' if ssl_ctx else 'HTTP (tanpa enkripsi)'}")
    print(f"  Akses lokal PC: {protocol}://localhost:5000")
    print(f"  Akses Jaringan (Smartwatch/IoT): {protocol}://{local_ip}:5000")
    print("=" * 50 + "\n")
    
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True, ssl_context=ssl_ctx)

