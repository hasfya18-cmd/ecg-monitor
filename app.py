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
from flask import Flask, render_template, Response, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ============================================================
# GLOBAL STATE
# ============================================================
state = {
    'model_choice': 'CNN',      # CNN or SVM
    'snr_db': 30,               # SNR level (dB)
    'packet_loss': 0.05,        # Packet loss rate
    'is_streaming': True,       # Streaming on/off
    'speed': 0.8,               # Seconds between beats
}

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
    if model_choice == 'CNN':
        beat_input = beat.reshape(1, -1, 1)
        prob = float(cnn_model.predict(beat_input, verbose=0)[0][0])
        pred = 1 if prob > 0.5 else 0
    else:
        beat_input = beat.reshape(1, -1)
        pred = int(svm_model.predict(beat_input)[0])
        prob = float(svm_model.predict_proba(beat_input)[0][1])
    return pred, prob

# ============================================================
# ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

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

            # Get current beat
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

                # Classify
                pred, prob = classify_beat(noisy_beat, state['model_choice'])

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
                    'stats': stats.copy()
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
    return jsonify({'status': 'ok', 'state': state})

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
