// ============================================================
// ECG Real-Time Monitor — Frontend Logic
// ============================================================

// --- STATE ---
const appState = {
    isStreaming: true,
    model: 'CNN',
    snrDb: 30,
    packetLoss: 5,
    speed: 0.8,
    packetLossCount: 0,
    alertCount: 0,
};

// --- ECG CANVAS ---
const canvas = document.getElementById('ecgCanvas');
const ctx = canvas.getContext('2d');
let beatBuffer = [];        // All beat points to draw
const MAX_POINTS = 1800;    // How many points visible on screen
let drawX = 0;              // Current draw position

function resizeCanvas() {
    const wrapper = canvas.parentElement;
    canvas.width = wrapper.clientWidth;
    canvas.height = 220;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// --- DRAW ECG WAVEFORM (hospital monitor sweep style) ---
function drawECG() {
    const W = canvas.width;
    const H = canvas.height;
    const midY = H / 2;

    // Draw grid
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.06)';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x < W; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }
    for (let y = 0; y < H; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    if (beatBuffer.length < 2) {
        requestAnimationFrame(drawECG);
        return;
    }

    // Calculate visible points
    const pointsPerPixel = beatBuffer.length / W;
    const scale = H * 0.35;

    // Draw glow effect
    ctx.shadowColor = 'rgba(34, 211, 238, 0.5)';
    ctx.shadowBlur = 8;
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < beatBuffer.length && i < MAX_POINTS; i++) {
        const x = (i / MAX_POINTS) * W;
        const y = midY - beatBuffer[i] * scale;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw sweep line (current position indicator)
    if (beatBuffer.length < MAX_POINTS) {
        const sweepX = (beatBuffer.length / MAX_POINTS) * W;
        const gradient = ctx.createLinearGradient(sweepX - 30, 0, sweepX, 0);
        gradient.addColorStop(0, 'transparent');
        gradient.addColorStop(1, 'rgba(34, 211, 238, 0.6)');
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(sweepX, 0);
        ctx.lineTo(sweepX, H);
        ctx.stroke();
    }

    requestAnimationFrame(drawECG);
}

// Start render loop
requestAnimationFrame(drawECG);

// --- ADD BEAT TO BUFFER ---
function addBeatToBuffer(beatData) {
    // Add beat points with small gap between beats
    for (let i = 0; i < beatData.length; i++) {
        beatBuffer.push(beatData[i]);
    }

    // If buffer is full, remove oldest points (scrolling effect)
    while (beatBuffer.length > MAX_POINTS) {
        beatBuffer.shift();
    }
}

// --- SSE CONNECTION ---
let eventSource = null;

function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/stream');

    eventSource.onopen = () => {
        updateConnectionStatus(true);
    };

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleBeatData(data);
    };

    eventSource.onerror = () => {
        updateConnectionStatus(false);
        // Auto-reconnect after 3 seconds
        setTimeout(connectSSE, 3000);
    };
}

// --- HANDLE INCOMING BEAT DATA ---
function handleBeatData(data) {
    updateConnectionStatus(true);

    if (data.type === 'packet_loss') {
        // Packet was lost
        appState.packetLossCount++;
        document.getElementById('statLoss').textContent = appState.packetLossCount;

        // Add flat line for lost packet
        const flatBeat = new Array(30).fill(0);
        addBeatToBuffer(flatBeat);

        // Show packet loss alert
        addAlert({
            time: new Date().toLocaleTimeString('id-ID'),
            msg: `Packet lost (Beat #${data.beat_index})`,
            type: 'packet_loss'
        });
        return;
    }

    // Normal beat data
    const beat = data.beat_data;
    const pred = data.prediction;
    const label = data.label;
    const confidence = data.confidence;
    const hr = data.heart_rate;
    const stats = data.stats;

    // Update ECG waveform
    addBeatToBuffer(beat);

    // Update heart rate
    document.getElementById('heartRate').textContent = hr;

    // Update classification status
    const classCard = document.getElementById('classCard');
    const classLabel = document.getElementById('classLabel');
    const classConfidence = document.getElementById('classConfidence');
    const classModel = document.getElementById('classModel');

    classLabel.textContent = label;
    classConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    classModel.textContent = `Model: ${data.model}`;

    if (pred === 0) {
        classCard.className = 'card card-metric card-status normal';
        classLabel.innerHTML = '✅ ' + label;
    } else {
        classCard.className = 'card card-metric card-status abnormal';
        classLabel.innerHTML = '⚠️ ' + label;

        // Play alarm sound
        playAlarmSound();

        // Simulate sending alert back to patient device
        simulateDeviceFeedback(hr);

        // Add alert
        addAlert({
            time: new Date().toLocaleTimeString('id-ID'),
            msg: `Aritmia terdeteksi (Beat #${data.beat_index})`,
            confidence: confidence,
            hr: hr,
            type: 'abnormal'
        });
    }

    // Update stats
    document.getElementById('statTotal').textContent = stats.total;
    document.getElementById('statNormal').textContent = stats.normal;
    document.getElementById('statAbnormal').textContent = stats.abnormal;

    // Update meta tags
    document.getElementById('metaModel').textContent = `Model: ${data.model}`;
    document.getElementById('metaSNR').textContent = `SNR: ${data.snr_db} dB`;
    document.getElementById('metaBeat').textContent = `Beat: ${data.beat_index}`;
}

// --- AUDIO ALARM ---
let audioCtx = null;
let isAlarmEnabled = true;

function toggleAlarm() {
    isAlarmEnabled = !isAlarmEnabled;
    const btn = document.getElementById('btnAlarm');
    if (isAlarmEnabled) {
        btn.querySelector('.btn-icon').textContent = '🔊';
        btn.querySelector('.btn-label').textContent = 'Sound: ON';
        btn.className = 'btn-stream';
    } else {
        btn.querySelector('.btn-icon').textContent = '🔇';
        btn.querySelector('.btn-label').textContent = 'Sound: OFF';
        btn.className = 'btn-stream paused';
    }
}

function playAlarmSound() {
    if (!isAlarmEnabled) return;

    // Initialize audio context on first user interaction
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Play a short beep
    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime); // 880Hz (A5)

    // Envelope to avoid clicking but very loud
    // Note: Most browser audio Contexts max out linearly near 1.0 (0dBfs). We set this to 1.5 to push volume.
    gainNode.gain.setValueAtTime(0, audioCtx.currentTime);
    gainNode.gain.linearRampToValueAtTime(1.5, audioCtx.currentTime + 0.05);
    gainNode.gain.setValueAtTime(1.5, audioCtx.currentTime + 0.1);
    gainNode.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.15);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + 0.2);
}

// --- DEVICE SIMULATION (IoT Feedback) ---
let feedbackTimeout = null;

function simulateDeviceFeedback(hr) {
    const screen = document.getElementById('deviceScreen');
    const status = document.getElementById('feedbackStatus');
    const hrVal = document.getElementById('deviceHrValue');
    const timeEl = document.getElementById('deviceTime');

    // Update basic metrics
    hrVal.textContent = hr;

    // Trigger Alert Mode
    screen.className = 'device-screen alert-mode';
    screen.querySelector('.device-status').innerHTML = '⚠️ PERINGATAN<br>ARITMIA!';

    status.className = 'badge-status badge-alerting';
    status.innerHTML = 'Menerima Sinyal Bahaya dari AI...';

    // Vibrate device if supported by browser (for realism if viewed on phone)
    if (navigator.vibrate) {
        navigator.vibrate([200, 100, 200, 100, 500]);
    }

    // Reset back to normal after 3 seconds
    if (feedbackTimeout) clearTimeout(feedbackTimeout);

    feedbackTimeout = setTimeout(() => {
        screen.className = 'device-screen';
        screen.querySelector('.device-status').innerHTML = 'Memantau...';
        status.className = 'badge-status badge-connected';
        status.innerHTML = 'Menunggu Alert / Sinkronisasi...';
    }, 3000);
}

// Update clock routinely
setInterval(() => {
    const timeEl = document.getElementById('deviceTime');
    if (timeEl) timeEl.textContent = new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
}, 1000);

// --- ALERTS ---
function addAlert(alert) {
    const alertsBody = document.getElementById('alertsBody');
    const emptyMsg = alertsBody.querySelector('.alert-empty');
    if (emptyMsg) emptyMsg.remove();

    const alertEl = document.createElement('div');
    alertEl.className = `alert-item ${alert.type === 'packet_loss' ? 'packet-loss' : ''}`;

    if (alert.type === 'packet_loss') {
        alertEl.innerHTML = `
            <span class="alert-time">${alert.time}</span>
            <span class="alert-msg">📡 ${alert.msg}</span>
            <span class="alert-badge">LOST</span>
        `;
    } else {
        alertEl.innerHTML = `
            <span class="alert-time">${alert.time}</span>
            <span class="alert-msg">🚨 ${alert.msg}</span>
            <span class="alert-badge">${(alert.confidence * 100).toFixed(0)}% | ${alert.hr} BPM</span>
        `;
    }

    // Insert at top
    alertsBody.insertBefore(alertEl, alertsBody.firstChild);

    // Update count
    appState.alertCount++;
    document.getElementById('alertCount').textContent = appState.alertCount;

    // Keep only last 30 alerts in DOM
    while (alertsBody.children.length > 30) {
        alertsBody.removeChild(alertsBody.lastChild);
    }
}

// --- CONNECTION STATUS ---
function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    if (connected) {
        el.className = 'connection-status connected';
        el.querySelector('.status-text').textContent = 'Connected';
    } else {
        el.className = 'connection-status disconnected';
        el.querySelector('.status-text').textContent = 'Disconnected';
    }
}

// --- CONTROLS ---
function toggleStream() {
    appState.isStreaming = !appState.isStreaming;
    const btn = document.getElementById('btnStream');

    if (appState.isStreaming) {
        btn.className = 'btn-stream';
        btn.querySelector('.btn-icon').textContent = '⏸';
        btn.querySelector('.btn-label').textContent = 'Pause';
    } else {
        btn.className = 'btn-stream paused';
        btn.querySelector('.btn-icon').textContent = '▶';
        btn.querySelector('.btn-label').textContent = 'Resume';
    }

    updateSettings({ is_streaming: appState.isStreaming });
}

function setModel(model) {
    appState.model = model;
    document.getElementById('btnCNN').className = model === 'CNN' ? 'toggle-btn active' : 'toggle-btn';
    document.getElementById('btnSVM').className = model === 'SVM' ? 'toggle-btn active' : 'toggle-btn';
    updateSettings({ model_choice: model });
}

function updateSNR(value) {
    appState.snrDb = parseInt(value);
    document.getElementById('snrValue').textContent = `${value} dB`;
    updateSettings({ snr_db: parseInt(value) });
}

function updatePacketLoss(value) {
    appState.packetLoss = parseInt(value);
    document.getElementById('lossValue').textContent = `${value}%`;
    updateSettings({ packet_loss: parseInt(value) / 100 });
}

function updateSpeed(value) {
    const speed = parseInt(value) / 10;
    appState.speed = speed;
    const labels = { 0.3: 'Sangat Cepat', 0.5: 'Cepat', 0.8: 'Normal', 1.0: 'Lambat', 1.5: 'Sangat Lambat' };
    let label = `${speed.toFixed(1)}s`;
    if (speed <= 0.4) label = 'Sangat Cepat';
    else if (speed <= 0.6) label = 'Cepat';
    else if (speed <= 0.9) label = 'Normal';
    else if (speed <= 1.2) label = 'Lambat';
    else label = 'Sangat Lambat';
    document.getElementById('speedValue').textContent = label;
    updateSettings({ speed: speed });
}

// --- API CALLS ---
async function updateSettings(settings) {
    try {
        await fetch('/api/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
    } catch (e) {
        console.error('Failed to update settings:', e);
    }
}

// --- INIT ---
window.addEventListener('DOMContentLoaded', () => {
    connectSSE();
});
