// ============================================================
// ECG Real-Time Monitor — Frontend Logic (Enhanced v2)
// ============================================================

// --- STATE ---
const appState = {
    isStreaming: true,
    model: 'CNN',
    snrDb: 30,
    packetLoss: 5,
    speed: 0.8,
    source: 'demo',
    packetLossCount: 0,
    alertCount: 0,
    lastPrediction: -1,   // -1 = unknown, 0 = normal, 1 = abnormal
};

let pendingWarning = false;
let ecgAlertTimeout = null;

// --- DATA RECORDING FOR CSV EXPORT ---
let recordedData = [];

// --- BENCHMARK TRACKING ---
let benchmarkStats = {
    cnn: { correct: 0, total: 0, totalMs: 0, tp: 0, fp: 0, fn: 0, tn: 0 },
    svm: { correct: 0, total: 0, totalMs: 0, tp: 0, fp: 0, fn: 0, tn: 0 }
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

// --- DRAW ECG WAVEFORM (medical-grade EKG grid) ---
function drawECG() {
    const W = canvas.width;
    const H = canvas.height;
    const midY = H / 2;

    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fillRect(0, 0, W, H);

    // --- Medical-grade EKG grid (red/pink like real ECG paper) ---
    const smallGrid = 10; // Small squares (1mm equivalent)
    const bigGrid = 50;   // Large squares (5mm equivalent)

    // Small grid lines
    ctx.strokeStyle = 'rgba(220, 60, 60, 0.08)';
    ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += smallGrid) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += smallGrid) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Large grid lines
    ctx.strokeStyle = 'rgba(220, 60, 60, 0.18)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += bigGrid) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += bigGrid) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // Center baseline
    ctx.strokeStyle = 'rgba(220, 60, 60, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(W, midY);
    ctx.stroke();

    // Scale label
    ctx.fillStyle = 'rgba(220, 60, 60, 0.4)';
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillText('25 mm/s | 10 mm/mV', 8, H - 6);

    if (beatBuffer.length < 2) {
        requestAnimationFrame(drawECG);
        return;
    }

    const scale = H * 0.35;

    // Draw glow effect
    if (appState.lastPrediction === 1) {
        // ABNORMAL — red waveform
        ctx.shadowColor = 'rgba(239, 68, 68, 0.6)';
        ctx.shadowBlur = 10;
        ctx.strokeStyle = '#ef4444';
    } else {
        // NORMAL — cyan waveform
        ctx.shadowColor = 'rgba(34, 211, 238, 0.5)';
        ctx.shadowBlur = 8;
        ctx.strokeStyle = '#22d3ee';
    }
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

    // --- P-QRS-T Annotation (simplified peak detection) ---
    drawPQRSTAnnotations(W, H, midY, scale);

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

// --- P-QRS-T ANNOTATION ---
function drawPQRSTAnnotations(W, H, midY, scale) {
    if (beatBuffer.length < 187) return;

    // Only annotate the last full beat (187 samples)
    const beatLen = 187;
    const startIdx = Math.max(0, beatBuffer.length - beatLen);
    const beat = beatBuffer.slice(startIdx, startIdx + beatLen);

    // Find QRS peak (max absolute value, typically around index 80-120)
    let maxVal = -Infinity;
    let qrsIdx = 0;
    for (let i = 50; i < 140 && i < beat.length; i++) {
        if (Math.abs(beat[i]) > maxVal) {
            maxVal = Math.abs(beat[i]);
            qrsIdx = i;
        }
    }

    // Only annotate if we have a clear peak
    if (maxVal < 0.15) return;

    const drawIdx = startIdx + qrsIdx;
    const xPos = (drawIdx / MAX_POINTS) * W;

    // P wave region (before QRS, ~20-40 samples before peak)
    const pIdx = startIdx + Math.max(0, qrsIdx - 35);
    const pX = (pIdx / MAX_POINTS) * W;

    // T wave region (after QRS, ~40-80 samples after peak)
    const tIdx = startIdx + Math.min(beat.length - 1, qrsIdx + 55);
    const tX = (tIdx / MAX_POINTS) * W;

    ctx.font = 'bold 10px Inter, sans-serif';

    // P label
    ctx.fillStyle = 'rgba(96, 165, 250, 0.7)';
    ctx.fillText('P', pX, 14);

    // QRS label
    ctx.fillStyle = 'rgba(239, 68, 68, 0.8)';
    ctx.fillText('QRS', xPos - 8, 14);

    // T label
    ctx.fillStyle = 'rgba(52, 211, 153, 0.7)';
    ctx.fillText('T', tX, 14);
}

// Start render loop
requestAnimationFrame(drawECG);

// --- ADD BEAT TO BUFFER ---
function addBeatToBuffer(beatData) {
    for (let i = 0; i < beatData.length; i++) {
        beatBuffer.push(beatData[i]);
    }
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
        setTimeout(connectSSE, 3000);
    };
}

// --- HANDLE INCOMING BEAT DATA ---
const CONFIDENCE_THRESHOLD = 0.70;

function handleBeatData(data) {
    updateConnectionStatus(true);

    if (data.type === 'packet_loss') {
        appState.packetLossCount++;
        document.getElementById('statLoss').textContent = appState.packetLossCount;

        const flatBeat = new Array(30).fill(0);
        addBeatToBuffer(flatBeat);

        addAlert({
            time: new Date().toLocaleTimeString('id-ID'),
            msg: `Packet lost (Beat #${data.beat_index})`,
            type: 'packet_loss'
        });

        // Update connection quality color
        updateConnectionQuality();
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
    updateWarningIndicator(hr);

    // --- CONFIDENCE THRESHOLD CHECK ---
    const classCard = document.getElementById('classCard');
    const classLabel = document.getElementById('classLabel');
    const classConfidence = document.getElementById('classConfidence');
    const classModel = document.getElementById('classModel');

    if (confidence < CONFIDENCE_THRESHOLD) {
        // Signal too noisy — refuse to classify
        classCard.className = 'card card-metric card-status noisy';
        classLabel.innerHTML = '⚠️ Sinyal Terlalu Noisy';
        classConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}% (< 70%)`;
        classModel.textContent = `Model: ${data.model} — Unable to Classify`;
    } else {
        classConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
        classModel.textContent = `Model: ${data.model}`;

        if (pred === 0) {
            classCard.className = 'card card-metric card-status normal';
            classLabel.innerHTML = '✅ ' + label;
        } else {
            classCard.className = 'card card-metric card-status abnormal';
            classLabel.innerHTML = '⚠️ ' + label;

            playAlarmSound();
            simulateDeviceFeedback(hr);

            addAlert({
                time: new Date().toLocaleTimeString('id-ID'),
                msg: `Aritmia terdeteksi (Beat #${data.beat_index})`,
                confidence: confidence,
                hr: hr,
                type: 'abnormal'
            });
        }
    }

    // Update stats
    document.getElementById('statTotal').textContent = stats.total;
    document.getElementById('statNormal').textContent = stats.normal;
    document.getElementById('statAbnormal').textContent = stats.abnormal;

    // Update meta tags
    document.getElementById('metaModel').textContent = `Model: ${data.model}`;
    document.getElementById('metaSNR').textContent = `SNR: ${data.snr_db} dB`;
    document.getElementById('metaBeat').textContent = `Beat: ${data.beat_index}`;

    // --- BENCHMARK UPDATE ---
    if (data.benchmark) {
        updateBenchmark(data.benchmark, data.true_label);
        updateSideBySide(data.benchmark, data.beat_index);

        // Update meta inference time (for selected model)
        const chosenMs = data.model === 'CNN' ? data.benchmark.cnn_ms : data.benchmark.svm_ms;
        document.getElementById('metaInferenceTime').textContent = `\u23f1 Inferensi: ${chosenMs} ms`;
    }

    // --- ECG ALERT OVERLAY ---
    if (pred === 1 && confidence >= CONFIDENCE_THRESHOLD) {
        appState.lastPrediction = 1;
        showEcgAlert();
    } else {
        appState.lastPrediction = 0;
        hideEcgAlert();
    }

    // --- RECORD DATA FOR CSV ---
    recordedData.push({
        beat: data.beat_index,
        time: new Date().toLocaleTimeString('id-ID'),
        model: data.model,
        prediction: label,
        confidence: confidence,
        hr: hr,
        snr: data.snr_db,
        true_label: data.true_label,
        cnn_pred: data.benchmark ? data.benchmark.cnn_pred : '',
        svm_pred: data.benchmark ? data.benchmark.svm_pred : '',
        cnn_ms: data.benchmark ? data.benchmark.cnn_ms : '',
        svm_ms: data.benchmark ? data.benchmark.svm_ms : ''
    });
    if (recordedData.length > 5000) recordedData = recordedData.slice(-5000);
}

// --- BENCHMARK TABLE ---
function updateBenchmark(bm, trueLabel) {
    // Update running stats
    ['cnn', 'svm'].forEach(m => {
        benchmarkStats[m].total++;
        benchmarkStats[m].totalMs += bm[`${m}_ms`];

        const pred = bm[`${m}_pred`];
        if (trueLabel >= 0) {
            if (pred === trueLabel) benchmarkStats[m].correct++;
            if (pred === 1 && trueLabel === 1) benchmarkStats[m].tp++;
            if (pred === 1 && trueLabel === 0) benchmarkStats[m].fp++;
            if (pred === 0 && trueLabel === 1) benchmarkStats[m].fn++;
            if (pred === 0 && trueLabel === 0) benchmarkStats[m].tn++;
        }
    });

    // Update DOM
    ['cnn', 'svm'].forEach(m => {
        const s = benchmarkStats[m];
        const acc = s.total > 0 ? ((s.correct / s.total) * 100).toFixed(1) : '--';
        const precision = (s.tp + s.fp) > 0 ? s.tp / (s.tp + s.fp) : 0;
        const recall = (s.tp + s.fn) > 0 ? s.tp / (s.tp + s.fn) : 0;
        const f1 = (precision + recall) > 0 ? ((2 * precision * recall) / (precision + recall) * 100).toFixed(1) : '--';
        const avgMs = s.total > 0 ? (s.totalMs / s.total).toFixed(1) : '--';

        const accEl = document.getElementById(`bm-${m}-acc`);
        const f1El = document.getElementById(`bm-${m}-f1`);
        const msEl = document.getElementById(`bm-${m}-ms`);
        const countEl = document.getElementById(`bm-${m}-count`);

        if (accEl) accEl.textContent = acc + '%';
        if (f1El) f1El.textContent = f1 + '%';
        if (msEl) msEl.textContent = avgMs + ' ms';
        if (countEl) countEl.textContent = s.total;
    });

    // Highlight winner
    ['acc', 'f1', 'ms'].forEach(metric => {
        const cnnEl = document.getElementById(`bm-cnn-${metric}`);
        const svmEl = document.getElementById(`bm-svm-${metric}`);
        if (!cnnEl || !svmEl) return;

        cnnEl.classList.remove('benchmark-winner');
        svmEl.classList.remove('benchmark-winner');

        const cnnVal = parseFloat(cnnEl.textContent);
        const svmVal = parseFloat(svmEl.textContent);
        if (isNaN(cnnVal) || isNaN(svmVal)) return;

        if (metric === 'ms') {
            // Lower is better for latency
            if (cnnVal < svmVal) cnnEl.classList.add('benchmark-winner');
            else if (svmVal < cnnVal) svmEl.classList.add('benchmark-winner');
        } else {
            if (cnnVal > svmVal) cnnEl.classList.add('benchmark-winner');
            else if (svmVal > cnnVal) svmEl.classList.add('benchmark-winner');
        }
    });
}

// --- SIDE-BY-SIDE CNN vs SVM COMPARISON (PER-BEAT) ---
function updateSideBySide(bm, beatIndex) {
    // Beat number
    const beatNumEl = document.getElementById('compBeatNum');
    if (beatNumEl) beatNumEl.textContent = `Beat: ${beatIndex}`;

    // CNN
    const cnnPred = bm.cnn_pred;
    const cnnConf = bm.cnn_conf;
    const cnnMs = bm.cnn_ms;

    const cnnPredEl = document.getElementById('compCnnPred');
    if (cnnPred === 0) {
        cnnPredEl.className = 'comp-prediction pred-normal';
        cnnPredEl.innerHTML = '<span class="comp-pred-label">\u2705 Normal</span>';
    } else {
        cnnPredEl.className = 'comp-prediction pred-abnormal';
        cnnPredEl.innerHTML = '<span class="comp-pred-label">\u26a0\ufe0f Abnormal</span>';
    }

    document.getElementById('compCnnConf').textContent = `${(cnnConf * 100).toFixed(1)}%`;
    document.getElementById('compCnnBar').style.width = `${(cnnConf * 100).toFixed(0)}%`;
    document.getElementById('compCnnTime').textContent = `${cnnMs} ms`;

    // SVM
    const svmPred = bm.svm_pred;
    const svmConf = bm.svm_conf;
    const svmMs = bm.svm_ms;

    const svmPredEl = document.getElementById('compSvmPred');
    if (svmPred === 0) {
        svmPredEl.className = 'comp-prediction pred-normal';
        svmPredEl.innerHTML = '<span class="comp-pred-label">\u2705 Normal</span>';
    } else {
        svmPredEl.className = 'comp-prediction pred-abnormal';
        svmPredEl.innerHTML = '<span class="comp-pred-label">\u26a0\ufe0f Abnormal</span>';
    }

    document.getElementById('compSvmConf').textContent = `${(svmConf * 100).toFixed(1)}%`;
    document.getElementById('compSvmBar').style.width = `${(svmConf * 100).toFixed(0)}%`;
    document.getElementById('compSvmTime').textContent = `${svmMs} ms`;

    // Winner badge
    const winnerEl = document.getElementById('compWinner');
    const cnnCol = document.querySelector('.comp-cnn');
    const svmCol = document.querySelector('.comp-svm');
    cnnCol.classList.remove('comp-winner-highlight');
    svmCol.classList.remove('comp-winner-highlight');

    if (cnnConf > svmConf + 0.01) {
        winnerEl.textContent = '\ud83e\udde0 CNN Unggul';
        winnerEl.className = 'comp-winner winner-cnn';
        cnnCol.classList.add('comp-winner-highlight');
    } else if (svmConf > cnnConf + 0.01) {
        winnerEl.textContent = '\ud83d\udcd0 SVM Unggul';
        winnerEl.className = 'comp-winner winner-svm';
        svmCol.classList.add('comp-winner-highlight');
    } else {
        winnerEl.textContent = '\u2696\ufe0f Seri';
        winnerEl.className = 'comp-winner winner-tie';
    }

    // Speed comparison (smaller = faster = better)
    const cnnTimeEl = document.getElementById('compCnnTime');
    const svmTimeEl = document.getElementById('compSvmTime');
    cnnTimeEl.style.color = cnnMs <= svmMs ? '#34d399' : '#f87171';
    svmTimeEl.style.color = svmMs <= cnnMs ? '#34d399' : '#f87171';
}

// --- ECG ALERT OVERLAY ---
function showEcgAlert() {
    const overlay = document.getElementById('ecgAlertOverlay');
    const ecgCard = document.querySelector('.card-ecg');
    if (overlay) overlay.style.display = 'flex';
    if (ecgCard) ecgCard.classList.add('ecg-alerting');

    if (ecgAlertTimeout) clearTimeout(ecgAlertTimeout);
    ecgAlertTimeout = setTimeout(hideEcgAlert, 3000);
}

function hideEcgAlert() {
    const overlay = document.getElementById('ecgAlertOverlay');
    const ecgCard = document.querySelector('.card-ecg');
    if (overlay) overlay.style.display = 'none';
    if (ecgCard) ecgCard.classList.remove('ecg-alerting');
}

// --- CSV EXPORT ---
function exportCSV() {
    if (recordedData.length === 0) {
        alert('Belum ada data untuk diekspor. Jalankan monitoring terlebih dahulu.');
        return;
    }

    const headers = ['Beat#', 'Waktu', 'Model', 'Prediksi', 'Confidence', 'HR(BPM)', 'SNR(dB)', 'True_Label', 'CNN_Pred', 'SVM_Pred', 'CNN_ms', 'SVM_ms'];
    const rows = recordedData.map(d =>
        [d.beat, d.time, d.model, d.prediction, d.confidence, d.hr, d.snr, d.true_label, d.cnn_pred, d.svm_pred, d.cnn_ms, d.svm_ms].join(',')
    );

    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `ecg_recording_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
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
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);

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

function updateWarningIndicator(hr) {
    const indicator = document.getElementById('warningIndicator');
    if (!indicator) return;

    if (hr === '--' || isNaN(hr) || hr === 0) {
        indicator.innerHTML = '💡 <b>Status:</b> Menunggu data HR...';
        indicator.style.color = '#a5b4fc';
        indicator.style.borderColor = 'rgba(79, 70, 229, 0.3)';
        indicator.style.background = 'rgba(79, 70, 229, 0.1)';
        return;
    }

    if (hr > 100) {
        indicator.innerHTML = '⚠️ <b>Tachycardia (HR Tinggi)</b><br>Rekomendasi: <b>Tarik Napas & Istirahat</b>';
        indicator.style.color = '#fca5a5'; 
        indicator.style.borderColor = 'rgba(248, 113, 113, 0.4)';
        indicator.style.background = 'rgba(220, 38, 38, 0.15)';
    } else if (hr < 60) {
        indicator.innerHTML = '⚠️ <b>Bradycardia (HR Rendah)</b><br>Rekomendasi: <b>Hubungi RS / Minum Obat</b>';
        indicator.style.color = '#fcd34d'; 
        indicator.style.borderColor = 'rgba(251, 191, 36, 0.4)';
        indicator.style.background = 'rgba(245, 158, 11, 0.15)';
    } else {
        indicator.innerHTML = '✅ <b>HR Normal</b><br>Tidak butuh intervensi khusus.';
        indicator.style.color = '#6ee7b7'; 
        indicator.style.borderColor = 'rgba(52, 211, 153, 0.4)';
        indicator.style.background = 'rgba(16, 185, 129, 0.1)';
    }
}

function triggerManualWarning() {
    const action = document.getElementById('warningType').value;
    const screen = document.getElementById('deviceScreen');
    const status = document.getElementById('feedbackStatus');
    const actionPanel = document.getElementById('deviceAction');
    const actionText = document.getElementById('deviceActionText');
    
    if (feedbackTimeout) clearTimeout(feedbackTimeout);
    
    screen.className = 'device-screen alert-mode';
    screen.querySelector('.device-status').innerHTML = '⚠️ PERINTAH DOKTER';
    
    actionText.textContent = `Pesan: "${action}"`;
    actionPanel.style.display = 'block';
    
    status.className = 'badge-status badge-alerting';
    status.innerHTML = 'Peringatan terkirim! Menunggu konfirmasi pasien...';
    pendingWarning = true;
    
    if (navigator.vibrate) navigator.vibrate([300, 100, 300, 100, 300]);
    
    addAlert({
        time: new Date().toLocaleTimeString('id-ID'),
        msg: `Intervensi Medis: ${action}`,
        type: 'manual_action',
        confidence: 1,
        hr: '--'
    });
}

function acknowledgeWarning() {
    if (!pendingWarning) return;
    
    const screen = document.getElementById('deviceScreen');
    const status = document.getElementById('feedbackStatus');
    const actionPanel = document.getElementById('deviceAction');
    
    screen.className = 'device-screen';
    screen.querySelector('.device-status').innerHTML = 'Memantau...';
    actionPanel.style.display = 'none';
    
    status.className = 'badge-status badge-connected';
    status.innerHTML = '✅ Pasien telah menerima peringatan.';
    
    pendingWarning = false;
    
    addAlert({
        time: new Date().toLocaleTimeString('id-ID'),
        msg: `Pasien mengkonfirmasi instruksi medis`,
        type: 'acknowledge',
        confidence: 1,
        hr: '--'
    });
    
    setTimeout(() => {
        if (!pendingWarning) {
            status.innerHTML = 'Menunggu Alert / Sinkronisasi...';
        }
    }, 4000);
}

function simulateDeviceFeedback(hr) {
    const screen = document.getElementById('deviceScreen');
    const status = document.getElementById('feedbackStatus');
    const hrVal = document.getElementById('deviceHrValue');

    if (pendingWarning) return;

    hrVal.textContent = hr;

    screen.className = 'device-screen alert-mode';
    screen.querySelector('.device-status').innerHTML = '⚠️ PERINGATAN<br>ARITMIA!';

    status.className = 'badge-status badge-alerting';
    status.innerHTML = 'Menerima Sinyal Bahaya dari AI...';

    if (navigator.vibrate) {
        navigator.vibrate([200, 100, 200, 100, 500]);
    }

    if (feedbackTimeout) clearTimeout(feedbackTimeout);

    feedbackTimeout = setTimeout(() => {
        if (pendingWarning) return;
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

// --- ALERTS (with LocalStorage persistence) ---
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
    } else if (alert.type === 'manual_action') {
        alertEl.innerHTML = `
            <span class="alert-time">${alert.time}</span>
            <span class="alert-msg" style="color:#60a5fa;">👨‍⚕️ ${alert.msg}</span>
            <span class="alert-badge" style="color:#60a5fa; border-color:#60a5fa;">SENT</span>
        `;
    } else if (alert.type === 'acknowledge') {
        alertEl.innerHTML = `
            <span class="alert-time">${alert.time}</span>
            <span class="alert-msg" style="color:#34d399;">✅ ${alert.msg}</span>
            <span class="alert-badge" style="color:#34d399; background:rgba(16, 185, 129, 0.15)">OK</span>
        `;
    } else {
        alertEl.innerHTML = `
            <span class="alert-time">${alert.time}</span>
            <span class="alert-msg">🚨 ${alert.msg}</span>
            <span class="alert-badge">${(alert.confidence * 100).toFixed(0)}% | ${alert.hr} BPM</span>
        `;
    }

    alertsBody.insertBefore(alertEl, alertsBody.firstChild);

    appState.alertCount++;
    document.getElementById('alertCount').textContent = appState.alertCount;

    while (alertsBody.children.length > 30) {
        alertsBody.removeChild(alertsBody.lastChild);
    }

    // Save to localStorage
    saveAlertsToStorage(alert);
}

function saveAlertsToStorage(alert) {
    try {
        let stored = JSON.parse(localStorage.getItem('ecg_alerts') || '[]');
        stored.unshift(alert);
        if (stored.length > 50) stored = stored.slice(0, 50);
        localStorage.setItem('ecg_alerts', JSON.stringify(stored));
    } catch (e) { /* ignore storage errors */ }
}

function loadAlertsFromStorage() {
    try {
        const stored = JSON.parse(localStorage.getItem('ecg_alerts') || '[]');
        if (stored.length === 0) return;

        const alertsBody = document.getElementById('alertsBody');
        const emptyMsg = alertsBody.querySelector('.alert-empty');
        if (emptyMsg) emptyMsg.innerHTML = `📂 ${stored.length} alert tersimpan dari sesi sebelumnya.`;
    } catch (e) { /* ignore */ }
}

// --- CONNECTION STATUS (3-COLOR: Green / Yellow / Red) ---
let recentLossCount = 0;
let recentBeatCount = 0;

function updateConnectionQuality() {
    recentLossCount++;
}

function updateConnectionStatus(connected) {
    const el = document.getElementById('connectionStatus');
    recentBeatCount++;

    if (!connected) {
        el.className = 'connection-status disconnected';
        el.querySelector('.status-text').textContent = 'Disconnected';
        return;
    }

    // Calculate recent loss rate
    const lossRate = recentBeatCount > 10 ? (recentLossCount / recentBeatCount) : 0;

    if (lossRate > 0.15) {
        el.className = 'connection-status unstable';
        el.querySelector('.status-text').textContent = 'Unstable Connection';
    } else {
        el.className = 'connection-status connected';
        el.querySelector('.status-text').textContent = 'Connected';
    }

    // Reset counters periodically
    if (recentBeatCount > 50) {
        recentLossCount = 0;
        recentBeatCount = 0;
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

    // Update SNR quality indicator
    const fill = document.getElementById('snrQualityFill');
    const desc = document.getElementById('snrQualityDesc');
    const pct = ((parseInt(value) - 5) / 35) * 100; // 5-40 dB → 0-100%
    fill.style.width = `${pct}%`;

    fill.classList.remove('quality-good', 'quality-medium', 'quality-poor');
    if (value >= 25) {
        fill.classList.add('quality-good');
        desc.textContent = '\ud83d\udcf6 Koneksi Wi-Fi Indoor \u2014 Sinyal Baik';
    } else if (value >= 15) {
        fill.classList.add('quality-medium');
        desc.textContent = '\u26a0\ufe0f Koneksi Outdoor / Seluler \u2014 Gangguan Sedang';
    } else {
        fill.classList.add('quality-poor');
        desc.textContent = '\ud83d\udea8 Lingkungan Penuh Interferensi \u2014 Sinyal Sangat Buruk';
    }
}

function updatePacketLoss(value) {
    appState.packetLoss = parseInt(value);
    document.getElementById('lossValue').textContent = `${value}%`;
    updateSettings({ packet_loss: parseInt(value) / 100 });
}

function setSource(source) {
    appState.source = source;
    document.getElementById('btnSourceDemo').className = source === 'demo' ? 'toggle-btn active' : 'toggle-btn';
    document.getElementById('btnSourceLive').className = source === 'live' ? 'toggle-btn active' : 'toggle-btn';

    document.getElementById('liveInstructions').style.display = source === 'live' ? 'block' : 'none';

    beatBuffer = [];

    updateSettings({ source: source });
}

function updateSpeed(value) {
    const speed = parseInt(value) / 10;
    appState.speed = speed;
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
    loadAlertsFromStorage();
    connectSSE();
});
