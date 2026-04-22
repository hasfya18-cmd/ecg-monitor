# ECG Real-Time Monitoring Web App
# Flask Backend with SSE Streaming

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import time
import threading
import numpy as np
import joblib
from flask import Flask, render_template, Response, request, jsonify, session, redirect, url_for
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Simple demo credentials (for thesis demonstration only)
DEMO_USERNAME = 'admin'
DEMO_PASSWORD = 'admin123'

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
}

# ============================================================
# LIVE STREAMING BUFFERS
# ============================================================
live_ecg_buffer = []           # Buffer for incoming raw values
live_results_queue = []        # Queue for processed results to stream to frontend
BEAT_SIZE = 187

# ============================================================
# LOAD MODELS & DATA
# ============================================================
print("Loading models...")
cnn_model = load_model("models/cnn_model.keras")
svm_model = joblib.load("models/svm_model.pkl")
print("Models loaded!")

print("Loading demo data...")
demo = np.load("models/demo_data.npz")
X_demo = demo['X_all']
y_demo = demo['y_all']
print(f"Demo data: {len(y_demo)} beats ready")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def add_awgn(signal, snr_db):
    """Add AWGN noise to simulate wireless transmission."""
    if snr_db >= 100:  # Clean signal
        return signal.copy()
    power = np.mean(signal ** 2)
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
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        if username == DEMO_USERNAME and password == DEMO_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Username atau password salah!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/stream')
def stream():
    """SSE endpoint — stream ECG beats one at a time."""
    def generate():
        # Memulai dari index 1850 agar tidak menunggu 25 menit 
        # (Aritmia yang berhasil dideteksi AI pertama kali ada di sekitar index 1898)
        beat_index = 1850
        total_beats = len(X_demo)
        stats = {
            'total': 0,
            'normal': 0,
            'abnormal': 0,
            'alerts': []
        }

        while True:
            if not state['is_streaming']:
                time.sleep(0.5)
                continue

            if state['source'] == 'live':
                # LIVE MODE: read from the queue
                if len(live_results_queue) == 0:
                    time.sleep(0.1) # Wait for live data
                    continue
                
                # Fetch next live beat from queue
                live_item = live_results_queue.pop(0)
                
                beat_index += 1
                stats['total'] += 1
                if live_item['prediction'] == 0:
                    stats['normal'] += 1
                else:
                    stats['abnormal'] += 1

                data = {
                    'type': 'beat',
                    'beat_index': beat_index,
                    'beat_data': live_item['beat_data'].tolist(),
                    'clean_data': live_item['clean_data'].tolist(),
                    'prediction': live_item['prediction'],
                    'label': live_item['label'],
                    'confidence': live_item['confidence'],
                    'true_label': -1, # Unknown in live mode
                    'heart_rate': live_item['heart_rate'],
                    'model': state['model_choice'],
                    'snr_db': state['snr_db'],
                    'packet_loss_rate': state['packet_loss'],
                    'stats': stats.copy()
                }

                if live_item['prediction'] == 1:
                    alert = {
                        'time': time.strftime('%H:%M:%S'),
                        'beat': beat_index,
                        'confidence': live_item['confidence'],
                        'heart_rate': live_item['heart_rate']
                    }
                    stats['alerts'].append(alert)
                    if len(stats['alerts']) > 50:
                        stats['alerts'] = stats['alerts'][-50:]
                
                yield f"data: {json.dumps(data)}\n\n"
                
                # In live mode the speed is controlled by the rate at which data is sent
                time.sleep(0.1)

            else:
                # DEMO MODE: read from demo dataset
                clean_beat = X_demo[beat_index % total_beats]
                true_label = int(y_demo[beat_index % total_beats])

                # Simulate packet loss
                is_lost = np.random.random() < state['packet_loss']

                if is_lost:
                    # Packet lost — send empty/zero beat
                    data = {
                        'type': 'packet_loss',
                        'beat_index': beat_index,
                        'message': 'Network Packet Lost (IoT)'
                    }
                else:
                    # Apply wireless noise
                    noisy_beat = add_awgn(clean_beat, state['snr_db'])

                    # Classify with BOTH models for benchmarking
                    both = classify_both(noisy_beat)

                    # Use the selected model for main prediction
                    chosen = both['cnn'] if state['model_choice'] == 'CNN' else both['svm']
                    pred = chosen['pred']
                    prob = chosen['prob']

                    stats['total'] += 1
                    if pred == 0:
                        stats['normal'] += 1
                    else:
                        stats['abnormal'] += 1

                    # Calculate simulated heart rate (60-100 normal range, with variation)
                    base_hr = 72
                    hr_variation = np.random.randint(-8, 9)
                    if pred == 1:
                        hr_variation += np.random.randint(5, 20)  # Abnormal = higher HR
                    heart_rate = base_hr + hr_variation

                    # Build data payload
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
                        'model': state['model_choice'],
                        'snr_db': state['snr_db'],
                        'packet_loss_rate': state['packet_loss'],
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
                            'time': time.strftime('%H:%M:%S'),
                            'beat': beat_index,
                            'confidence': round(prob, 4),
                            'heart_rate': heart_rate
                        }
                        stats['alerts'].append(alert)
                        # Keep only last 50 alerts
                        if len(stats['alerts']) > 50:
                            stats['alerts'] = stats['alerts'][-50:]

                yield f"data: {json.dumps(data)}\n\n"

                beat_index += 1
                time.sleep(state['speed'])

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
            # Reset buffers when switching to live mode
            global live_ecg_buffer, live_results_queue
            live_ecg_buffer = []
            live_results_queue = []
    return jsonify({'status': 'ok', 'state': state})

@app.route('/api/ecg_device', methods=['POST'])
def receive_ecg_data():
    """Endpoint for smartwatch to stream IoT data."""
    global live_ecg_buffer, live_results_queue
    data = request.json

    if not data or 'values' not in data:
        return jsonify({'error': 'No data provided'}), 400

    # Ensure source is live
    state['source'] = 'live'
    
    values = data['values']
    
    # Simple rate limiting logic or buffering mechanism
    live_ecg_buffer.extend(values)
    
    responses = []
    
    # Process multiple beats if buffer becomes large enough
    while len(live_ecg_buffer) >= BEAT_SIZE:
        beat = live_ecg_buffer[:BEAT_SIZE]
        live_ecg_buffer = live_ecg_buffer[BEAT_SIZE:]
        
        # Format the beat
        clean_beat = np.array(beat)
        
        # Apply wireless noise according to UI settings (Simulate imperfect connection)
        noisy_beat = add_awgn(clean_beat, state['snr_db'])
        
        # Check packet loss simulation if required or assume live connections might just fail to send
        
        # Classify
        pred, prob, _ = classify_beat(noisy_beat, state['model_choice'])
        
        # Calculate simulated heart rate
        base_hr = 72
        hr_variation = np.random.randint(-8, 9)
        if pred == 1:
            hr_variation += np.random.randint(5, 20)
        heart_rate = base_hr + hr_variation
        
        # Add to result queue for stream endpoint
        live_results_queue.append({
            'beat_data': noisy_beat,
            'clean_data': clean_beat,
            'prediction': pred,
            'label': 'Abnormal' if pred == 1 else 'Normal',
            'confidence': round(prob if pred == 1 else 1 - prob, 4),
            'heart_rate': heart_rate
        })
        
        responses.append({
            'aritmia_terdeteksi': bool(pred == 1),
            'confidence': round(prob, 4),
            'heart_rate': heart_rate
        })
    
    # If a full beat was processed, return the status of the *last* evaluated beat so watch can vibrate
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
    return jsonify(state)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  ECG Real-Time Monitoring App")
    print("  Buka browser: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, port=5000, threaded=True)
