// ============================================================
// ECG Professional Dashboard — Frontend Logic
// ============================================================

const appState = {
    isStreaming: true, model: 'CNN', snrDb: 30, packetLoss: 5,
    speed: 0.8, source: 'demo', packetLossCount: 0, alertCount: 0,
    lastPrediction: -1, selectedPatient: null, patientsList: [],
    consecutiveAbnormal: 0, ecgViewMode: 'single', selectedLead: 'II',
    telegramRegistrations: {},
    currentTab: 'home'  // Track active tab for conditional notifications
};

let ecgAlertTimeout = null;
let recordedData = [];
let benchmarkStats = {
    cnn: { correct:0, total:0, totalMs:0, tp:0, fp:0, fn:0, tn:0, history:[] },
    svm: { correct:0, total:0, totalMs:0, tp:0, fp:0, fn:0, tn:0, history:[] }
};
let comparisonChart = null;
let radarChartInstance = null;
let snrChartInstance = null;

// ===== CIRCULAR BUFFER =====
class CircularBuffer {
    constructor(max) { this.max=max; this.buf=new Float32Array(max); this.head=0; this.count=0; }
    push(v) { this.buf[this.head]=v; this.head=(this.head+1)%this.max; if(this.count<this.max)this.count++; }
    get(i) { if(i>=this.count)return 0; const s=(this.head-this.count+this.max)%this.max; return this.buf[(s+i)%this.max]; }
    get length() { return this.count; }
    clear() { this.head=0; this.count=0; }
}

const MAX_POINTS = 1800;
const ecgBuffer = new CircularBuffer(MAX_POINTS);

// ===== CANVAS =====
const canvas = document.getElementById('ecgCanvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
    const w = canvas.parentElement;
    if (w && w.clientWidth > 0 && w.clientHeight > 0) {
        canvas.width = Math.max(100, w.clientWidth - 48);
        canvas.height = Math.max(100, w.clientHeight - 48);
    } else {
        canvas.width = 300;
        canvas.height = 150;
    }
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

let mouseX = -1;
let mouseY = -1;
let isMouseOver = false;

canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    mouseX = (e.clientX - rect.left) * (canvas.width / rect.width);
    mouseY = (e.clientY - rect.top) * (canvas.height / rect.height);
    isMouseOver = true;
});

canvas.addEventListener('mouseleave', () => {
    isMouseOver = false;
    mouseX = -1;
    mouseY = -1;
});

canvas.addEventListener('mouseenter', () => {
    isMouseOver = true;
});


function drawECG() {
    const W=canvas.width, H=canvas.height, midY=H/2;
    ctx.fillStyle='#ffffff';
    ctx.fillRect(0,0,W,H);

    // Medical grid
    ctx.strokeStyle='rgba(220,60,60,0.1)'; ctx.lineWidth=0.5;
    for(let x=0;x<W;x+=10){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=10){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
    ctx.strokeStyle='rgba(220,60,60,0.22)'; ctx.lineWidth=1;
    for(let x=0;x<W;x+=50){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
    for(let y=0;y<H;y+=50){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}

    // Center line
    ctx.strokeStyle='rgba(220,60,60,0.35)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(0,midY); ctx.lineTo(W,midY); ctx.stroke();

    // Scale label
    ctx.fillStyle='rgba(220,60,60,0.4)'; ctx.font='9px JetBrains Mono,monospace';
    ctx.fillText('25 mm/s | 10 mm/mV',8,H-6);

    if(ecgBuffer.length<2){return;}
    const scale=H*0.35;

    if(appState.lastPrediction===1){
        ctx.shadowColor='rgba(220,38,38,0.4)';ctx.shadowBlur=6;ctx.strokeStyle='#dc2626';
    } else {
        ctx.shadowColor='rgba(37,99,235,0.3)';ctx.shadowBlur=4;ctx.strokeStyle='#2563eb';
    }
    ctx.lineWidth=2; ctx.beginPath();
    for(let i=0;i<ecgBuffer.length&&i<MAX_POINTS;i++){
        const x=(i/MAX_POINTS)*W, y=midY-ecgBuffer.get(i)*scale;
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.stroke(); ctx.shadowBlur=0;

    // Sweep line
    if(ecgBuffer.length<MAX_POINTS){
        const sx=(ecgBuffer.length/MAX_POINTS)*W;
        const g=ctx.createLinearGradient(sx-30,0,sx,0);
        g.addColorStop(0,'transparent');g.addColorStop(1,'rgba(37,99,235,0.5)');
        ctx.strokeStyle=g;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(sx,0);ctx.lineTo(sx,H);ctx.stroke();
    }
    // P-QRS-T annotation
    if(ecgBuffer.length>=187){
        const bLen=187, si=Math.max(0,ecgBuffer.length-bLen);
        let mv=-Infinity,qi=0;
        for(let i=50;i<140;i++){const v=Math.abs(ecgBuffer.get(si+i));if(v>mv){mv=v;qi=i;}}
        if(mv>=0.15){
            ctx.font='bold 10px Inter,sans-serif';
            ctx.fillStyle='rgba(96,165,250,0.7)';ctx.fillText('P',((si+Math.max(0,qi-35))/MAX_POINTS)*W,14);
            ctx.fillStyle='rgba(220,38,38,0.8)';ctx.fillText('QRS',((si+qi)/MAX_POINTS)*W-8,14);
            ctx.fillStyle='rgba(34,197,94,0.7)';ctx.fillText('T',((si+Math.min(bLen-1,qi+55))/MAX_POINTS)*W,14);
        }
    }

    // Interactive mouse hover coordinate/amplitude detector
    if (isMouseOver && mouseX >= 0 && mouseX <= W && ecgBuffer.length > 0) {
        // Calculate matching sample index
        const idx = Math.min(ecgBuffer.length - 1, Math.floor((mouseX / W) * ecgBuffer.length));
        const val = ecgBuffer.get(idx);
        const signalY = midY - val * scale;

        // Draw vertical coordinate guide line
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(mouseX, 0);
        ctx.lineTo(mouseX, H);
        ctx.stroke();
        ctx.setLineDash([]); // Reset line dash

        // Draw horizontal coordinate guide line
        ctx.strokeStyle = 'rgba(100, 116, 139, 0.25)';
        ctx.beginPath();
        ctx.moveTo(0, signalY);
        ctx.lineTo(W, signalY);
        ctx.stroke();

        // Draw high-visibility coordinate anchor point
        ctx.fillStyle = '#dc2626';
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(mouseX, signalY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Draw tooltip card next to cursor
        const tooltipW = 140;
        const tooltipH = 50;
        let tooltipX = mouseX + 15;
        let tooltipY = signalY - 25;

        // Keep tooltip inside canvas boundaries
        if (tooltipX + tooltipW > W) tooltipX = mouseX - tooltipW - 15;
        if (tooltipY + tooltipH > H) tooltipY = H - tooltipH - 10;
        if (tooltipY < 10) tooltipY = 10;

        // Render card shadow/background
        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)'; // Dark premium theme matching design
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY, tooltipW, tooltipH, 8);
        ctx.fill();

        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.fillText(`Beat Sample: #${idx}`, tooltipX + 10, tooltipY + 18);

        ctx.fillStyle = '#93c5fd';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.fillText(`Amp: ${val.toFixed(3)} mV`, tooltipX + 10, tooltipY + 36);
    }
}

// ===== MAIN DRAW LOOP =====
function mainDrawLoop() {
    drawECG();
    requestAnimationFrame(mainDrawLoop);
}

// Initialize main loop
mainDrawLoop();

// ===== SSE =====
let eventSource = null;
function connectSSE() {
    if(eventSource) eventSource.close();
    eventSource = new EventSource('/api/stream');
    eventSource.onopen = () => updateConnectionStatus(true);
    eventSource.onmessage = (e) => {
        const d=JSON.parse(e.data);
        if(d.type==='patient_change') handlePatientChange(d);
        else handleBeatData(d);
    };
    eventSource.onerror = () => { updateConnectionStatus(false); setTimeout(connectSSE,3000); };
}

function updateConnectionStatus(connected) {
    const badge = document.querySelector('.live-badge');
    if(!badge) return;
    if(connected) {
        badge.innerHTML = '<div class="live-dot"></div><span>Live Stream</span>';
        badge.style.borderColor = 'rgba(16,185,129,0.2)';
    } else {
        badge.innerHTML = '<span style="color:var(--red);font-weight:900;">⚠ DISCONNECTED</span>';
        badge.style.borderColor = 'rgba(220,38,38,0.3)';
    }
}

// ===== HANDLE BEAT DATA =====
const CONF_THRESHOLD = 0.60;

function handleBeatData(data) {
    updateConnectionStatus(true);

    if(data.type==='packet_loss'){
        appState.packetLossCount++;
        if(data.interpolated_data&&data.interpolated_data.length>0){
            data.interpolated_data.forEach(v=>ecgBuffer.push(v*0.6));
        } else { for(let i=0;i<30;i++)ecgBuffer.push(0); }
        addAlert({time:new Date().toLocaleTimeString('id-ID'),msg:`Packet lost (Beat #${data.beat_index})`,type:'loss'});
        return;
    }
    if(data.type==='heartbeat'||data.type==='end'||!data.beat_data) return;

    // Push waveform
    data.beat_data.forEach(v=>ecgBuffer.push(v));

    const pred=data.prediction, label=data.label, conf=data.confidence, hr=data.heart_rate, stats=data.stats;

    // Heart Rate
    document.getElementById('heartRate').textContent=hr;
    const hrCard=document.getElementById('hrCard');
    if(hr>100||hr<50){hrCard.classList.add('critical');}else{hrCard.classList.remove('critical');}

    // Simulated vitals
    document.getElementById('spo2Value').textContent=(95+Math.floor(Math.random()*4));
    const sys=115+Math.floor(Math.random()*15), dia=75+Math.floor(Math.random()*10);
    document.getElementById('bpValue').textContent=sys+'/'+dia;

    // AI Status Cards (CNN & SVM)
    const cnnSt = document.getElementById('aiStatusCNN');
    const cnnConfEl = document.getElementById('aiConfidenceCNN');
    const cnnCard = document.getElementById('aiCardCNN');

    const svmSt = document.getElementById('aiStatusSVM');
    const svmConfEl = document.getElementById('aiConfidenceSVM');
    const svmCard = document.getElementById('aiCardSVM');

    const cnnPred = data.benchmark ? data.benchmark.cnn_pred : (data.model === 'CNN' ? pred : null);
    const cnnConf = data.benchmark ? data.benchmark.cnn_conf : (data.model === 'CNN' ? conf : null);
    const svmPred = data.benchmark ? data.benchmark.svm_pred : (data.model === 'SVM' ? pred : null);
    const svmConf = data.benchmark ? data.benchmark.svm_conf : (data.model === 'SVM' ? conf : null);

    // Update CNN Card
    if (cnnPred !== null && cnnSt && cnnConfEl && cnnCard) {
        if (cnnPred === 1) {
            cnnSt.textContent = '🔴 ABNORMAL'; cnnSt.style.color = 'var(--red)';
            cnnConfEl.textContent = `Confidence: ${(cnnConf * 100).toFixed(1)}%`;
            cnnCard.classList.add('critical');
        } else {
            cnnSt.textContent = '✅ NORMAL'; cnnSt.style.color = 'var(--emerald)';
            cnnConfEl.textContent = `Confidence: ${(cnnConf * 100).toFixed(1)}%`;
            cnnCard.classList.remove('critical');
        }
    }

    // Update SVM Card
    if (svmPred !== null && svmSt && svmConfEl && svmCard) {
        if (svmPred === 1) {
            svmSt.textContent = '🔴 ABNORMAL'; svmSt.style.color = 'var(--red)';
            svmConfEl.textContent = `Confidence: ${(svmConf * 100).toFixed(1)}%`;
            svmCard.classList.add('critical');
        } else {
            svmSt.textContent = '✅ NORMAL'; svmSt.style.color = 'var(--emerald)';
            svmConfEl.textContent = `Confidence: ${(svmConf * 100).toFixed(1)}%`;
            svmCard.classList.remove('critical');
        }
    }

    // Handle Active Model Actions (Alarms, overlays, history alerts)
    const isTachycardia = hr > 100;
    const isBradycardia = hr < 60;
    const isIrregular = pred === 1;
    const isOnDashboard = appState.currentTab === 'dashboard';

    if (conf < CONF_THRESHOLD) {
        if (cnnSt) { cnnSt.textContent = '⚠️ Noisy Signal'; cnnSt.style.color = '#d97706'; }
        if (svmSt) { svmSt.textContent = '⚠️ Noisy Signal'; svmSt.style.color = '#d97706'; }
        appState.lastPrediction = -1;
    } else if (isIrregular || isTachycardia || isBradycardia) {
        appState.lastPrediction = 1;

        // Visual pop-up alerts ONLY when monitoring a patient on dashboard tab
        if (isOnDashboard) {
            if (isAlarmEnabled) {
                document.getElementById('btnAlarm').classList.add('active');
                playAlarmSound();
            }
            showEcgAlert();
        }

        let alarmMsg = `Aritmia (Beat #${data.beat_index})`;
        if (isTachycardia) alarmMsg = `Takikardia (Beat #${data.beat_index} - ${hr} BPM)`;
        else if (isBradycardia) alarmMsg = `Bradikardia (Beat #${data.beat_index} - ${hr} BPM)`;
        
        // Alert Center (sidebar) ALWAYS receives notifications
        addAlert({time:new Date().toLocaleTimeString('id-ID'),msg:alarmMsg,conf:conf,hr:hr,type:'abnormal'});
        appState.consecutiveAbnormal++;

        // Top toast alert disabled as requested by user (too annoying)
        if(appState.consecutiveAbnormal>=3){
            // if (isOnDashboard) showTopAlert(data);
            appState.consecutiveAbnormal=0;
        }
    } else {
        appState.lastPrediction = 0;
        document.getElementById('btnAlarm').classList.remove('active');
        hideEcgAlert();
        appState.consecutiveAbnormal=0;
    }

    // Meta chips
    const mp=document.getElementById('metaPatient');if(mp)mp.textContent='Patient: '+data.record_id;
    const mm=document.getElementById('metaModel');if(mm)mm.textContent=data.model;
    const mb=document.getElementById('metaBeat');if(mb)mb.textContent='Beat: '+data.beat_index;

    // Benchmark
    if(data.benchmark){
        const chosen=data.model==='CNN'?data.benchmark.cnn_ms:data.benchmark.svm_ms;
        document.getElementById('metaInferenceTime').textContent='⏱ '+chosen+' ms';
        updateBenchmark(data.benchmark, data.true_label);
    }

    // Record
    recordedData.push({beat:data.beat_index,time:new Date().toLocaleTimeString('id-ID'),model:data.model,prediction:label,confidence:conf,hr:hr,snr:data.snr_db});
    if(recordedData.length>5000)recordedData=recordedData.slice(-5000);
}

// ===== ALERTS =====
function addAlert(a) {
    appState.alertCount++;
    document.getElementById('alertCount').textContent=appState.alertCount;
    const body=document.getElementById('alertsBody');
    const empty=body.querySelector('.alert-empty');
    if(empty)empty.remove();

    const row=document.createElement('div');
    row.className='alert-item'+(a.type==='loss'?' loss':'');
    row.innerHTML=`<span class="alert-item-time">${a.time}</span><span class="alert-item-msg">${a.msg}</span>${a.conf?`<span class="alert-item-badge">${(a.conf*100).toFixed(0)}%</span>`:''}`;
    body.insertBefore(row, body.firstChild);
    if(body.children.length>50)body.removeChild(body.lastChild);

    // Right Sidebar Alert Center is now updated via global polling (pollGlobalAlerts)
}

// ===== ECG OVERLAY =====
function showEcgAlert(){
    const o=document.getElementById('ecgAlertOverlay');if(o)o.style.display='block';
    const hb=document.getElementById('headerAlertBadge');if(hb)hb.style.display='block';
    if(ecgAlertTimeout)clearTimeout(ecgAlertTimeout);
    ecgAlertTimeout=setTimeout(hideEcgAlert,3000);
}
function hideEcgAlert(){
    const o=document.getElementById('ecgAlertOverlay');if(o)o.style.display='none';
    const hb=document.getElementById('headerAlertBadge');if(hb)hb.style.display='none';
}

// ===== TOP ALERT TOAST =====
function showTopAlert(data){
    const t=document.getElementById('topAlertToast');
    let cond = "Detak jantung tidak beraturan";
    if(data.heart_rate > 100) cond = "Takikardia (Terlalu Cepat)";
    else if(data.heart_rate < 60) cond = "Bradikardia (Terlalu Lambat)";
    
    document.getElementById('topAlertMsg').innerHTML=`
        <strong>Pasien: ${data.record_id}</strong><br>
        Kondisi: ${cond} (${data.heart_rate} BPM)<br>
        <em>Tindakan: Peringatan Telegram otomatis telah dikirim ke pasien.</em>
    `;
    document.getElementById('topAlertTime').textContent=new Date().toLocaleTimeString('id-ID');
    t.classList.add('show');
    setTimeout(()=>t.classList.remove('show'),8000);
}

// ===== AUDIO =====
let audioCtx=null, isAlarmEnabled=true;
function toggleAlarm(){
    isAlarmEnabled=!isAlarmEnabled;
    const b=document.getElementById('btnAlarm');
    const span = b.querySelector('span');
    if(isAlarmEnabled){
        b.classList.remove('muted');
        if(span)span.textContent='Pulse Alarm Active';
    }else{
        b.classList.add('muted');
        b.classList.remove('active');
        if(span)span.textContent='Pulse Alarm Muted';
    }
}
function playAlarmSound(){
    if(!isAlarmEnabled)return;
    if(!audioCtx)audioCtx=new(window.AudioContext||window.webkitAudioContext)();
    const o=audioCtx.createOscillator(),g=audioCtx.createGain();
    o.type='sine';o.frequency.setValueAtTime(880,audioCtx.currentTime);
    g.gain.setValueAtTime(0,audioCtx.currentTime);
    g.gain.linearRampToValueAtTime(1.2,audioCtx.currentTime+0.05);
    g.gain.linearRampToValueAtTime(0,audioCtx.currentTime+0.15);
    o.connect(g);g.connect(audioCtx.destination);o.start();o.stop(audioCtx.currentTime+0.2);
}

// ===== NAVIGATION =====
function switchTab(tab){
    appState.currentTab = tab;  // Track current tab for notification routing
    document.querySelectorAll('.tab-panel').forEach(s=>s.classList.remove('active'));
    document.querySelectorAll('.sb-nav-item').forEach(n=>n.classList.remove('active'));
    const panel = document.getElementById('tab-'+tab);
    const nav = document.getElementById('nav-'+tab);
    if(panel) panel.classList.add('active');
    if(nav) nav.classList.add('active');
    if(tab==='dashboard'){
        setTimeout(resizeCanvas, 50);
    }
    if(tab==='analytics'){
        if(!comparisonChart) initComparisonChart();
        if(comparisonChart) {
            comparisonChart.reset(); comparisonChart.update({duration: 1000, easing: 'easeOutQuart'});
        }
        if(radarChartInstance) {
            radarChartInstance.reset(); radarChartInstance.update({duration: 1000, easing: 'easeOutQuart'});
        }
        if(snrChartInstance) {
            snrChartInstance.reset(); snrChartInstance.update({duration: 1000, easing: 'easeOutQuart'});
        }
        // CSS popup animation
        document.querySelectorAll('#tab-analytics .an-chart-card').forEach((c, i) => {
            c.style.animation = 'none';
            void c.offsetWidth; // trigger reflow
            c.style.animation = `modalIn 0.5s ease-out ${i*0.1}s both`;
        });
    }
    if(tab==='settings'){
        loadTelegramConfig();
        loadTelegramRegistrations();
    }
}

// ===== CONTROLS =====
function toggleStream(){
    appState.isStreaming=!appState.isStreaming;
    fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({is_streaming:appState.isStreaming})});
}
function updateSNR(v){document.getElementById('snrValue').textContent=v+' dB';fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({snr_db:parseInt(v)})});}
function updatePacketLoss(v){document.getElementById('lossValue').textContent=v+'%';fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({packet_loss:parseInt(v)/100})});}
function updateSpeed(v){
    const speeds={3:'Ultra Fast',5:'Fast',8:'Normal',10:'Slow',15:'Very Slow'};
    const sp=parseFloat(v)/10;
    document.getElementById('speedValue').textContent=speeds[parseInt(v)]||sp.toFixed(1)+'s';
    fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({speed:sp})});
}
function setModel(m){
    appState.model=m;
    document.getElementById('btnCNN').className='mt-btn'+(m==='CNN'?' active':'');
    document.getElementById('btnSVM').className='mt-btn'+(m==='SVM'?' active':'');
    fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_choice:m})});
}

// ===== PATIENTS =====
function loadPatients(){
    fetch('/api/patients').then(r=>r.json()).then(data=>{
        appState.patientsList=data.patients;
        appState.selectedPatient=data.selected;
        document.getElementById('patientTotal').textContent='Total: '+data.patients.length;
        const grid=document.getElementById('patientsGrid');
        grid.innerHTML='';
        data.patients.forEach(p=>{
            const card=document.createElement('div');
            card.className='patient-card'+(p.record_id===data.selected?' selected':'');
            const isCritical=p.rate>20;
            const hr = isCritical ? 115 : 72;
            
            const hasTg = appState.telegramRegistrations && appState.telegramRegistrations[p.record_id];
            card.innerHTML=`
                <div class="pc-top">
                    <div class="pc-info">
                        <div class="pc-avatar"><img src="https://api.dicebear.com/7.x/avataaars/svg?seed=${p.name}" alt="${p.name}"></div>
                        <div class="pc-name"><h3>${p.name}</h3><p>${p.record_id}</p></div>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        ${hasTg ? '<div style="display:flex;align-items:center;gap:4px;font-size:11px;font-weight:700;color:#0ea5e9;background:#e0f2fe;padding:5px 10px;border-radius:20px;" title="Telegram terdaftar">📲 TG</div>' : ''}
                        <div class="pc-status-badge" style="display:flex; align-items:center; gap:6px; font-size:12px; font-weight:700; color:${isCritical?'var(--red)':'var(--text-muted)'}; background:${isCritical?'#fee2e2':'#f1f5f9'}; padding:6px 12px; border-radius:20px;">
                            ${isCritical ? '⚠️ ARITMIA' : '✓ STABIL'}
                        </div>
                    </div>
                </div>
                <div class="pc-vitals">
                    <div class="pc-vital-row"><span class="vl">Dataset Beats</span><span class="vv">${p.total_beats}</span></div>
                    <div class="pc-vital-row"><span class="vl">Arrhythmia Rate</span><span class="vv">${parseFloat(p.rate).toFixed(1)}%</span></div>
                    <div class="pc-vital-row"><span class="vl">Est. Heart Rate</span><span class="vv" style="${isCritical?'color:var(--red)':''}">${hr} BPM</span></div>
                    <div class="pc-vital-row"><span class="vl">Notifikasi</span><span class="vv" style="${hasTg?'color:#0ea5e9':'color:var(--text-muted)'}">${hasTg ? '📲 Telegram Aktif' : '— Belum Terdaftar'}</span></div>
                </div>
                <div class="pc-action">
                    <button class="btn-open" onclick="selectPatient('${p.record_id}'); event.stopPropagation();" style="width:100%; background:var(--blue); border:none; color:#fff; font-weight:600; padding:10px; border-radius:12px; transition:all 0.2s; cursor:pointer;">Lihat Monitor EKG</button>
                </div>
            `;
            card.onclick=()=>selectPatient(p.record_id);
            grid.appendChild(card);
        });
    });
}

// Patient search filtering
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('patientSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const query = this.value.trim().toLowerCase();
            document.querySelectorAll('.patient-card').forEach(card => {
                const idEl = card.querySelector('.pc-name p');
                const nameEl = card.querySelector('.pc-name h3');
                const id = idEl ? idEl.textContent.toLowerCase() : '';
                const name = nameEl ? nameEl.textContent.toLowerCase() : '';
                if (!query || id.includes(query) || name.includes(query)) {
                    card.style.display = '';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }
});

function selectPatient(id){
    let targetBtn = null;
    document.querySelectorAll('.patient-card').forEach(c => {
        const pId = c.querySelector('.pc-name p');
        if (pId && pId.textContent.includes(id)) {
            targetBtn = c.querySelector('.btn-open');
        }
    });
    
    const originalText = targetBtn ? targetBtn.textContent : "Lihat Monitor EKG";
    if (targetBtn) {
        targetBtn.textContent = "Loading...";
        targetBtn.style.background = "#64748b";
        targetBtn.disabled = true;
    }

    fetch('/api/select_patient',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({record_id:id})
    })
    .then(r => {
        if (!r.ok) throw new Error("HTTP error " + r.status);
        return r.json();
    })
    .then(d => {
        if (targetBtn) {
            targetBtn.textContent = originalText;
            targetBtn.style.background = "var(--blue)";
            targetBtn.disabled = false;
        }
        if (d.status === 'ok') {
            appState.selectedPatient = id;
            ecgBuffer.clear();
            document.querySelectorAll('.patient-card').forEach(c => {
                c.classList.remove('selected');
                const pId = c.querySelector('.pc-name p');
                if (pId && pId.textContent.includes(id)) c.classList.add('selected');
            });
            switchTab('dashboard');
        } else {
            alert("Gagal memilih pasien: " + (d.message || "Error tidak dikenal."));
        }
    })
    .catch(err => {
        console.error(err);
        if (targetBtn) {
            targetBtn.textContent = originalText;
            targetBtn.style.background = "var(--blue)";
            targetBtn.disabled = false;
        }
        alert("Gagal menghubungi server. Mohon pastikan server aplikasi di Hugging Face sudah selesai loading dan online.");
    });
}

function handlePatientChange(data){
    appState.selectedPatient=data.record_id;
    ecgBuffer.clear();
    appState.alertCount=0;
    appState.packetLossCount=0;
    appState.consecutiveAbnormal=0;
    recordedData=[];
    document.getElementById('alertCount').textContent='0';
    document.getElementById('alertsBody').innerHTML='<div class="alert-empty">Pasien baru dipilih. Menunggu data...</div>';
    
    // Reset benchmark stats for both CNN and SVM
    benchmarkStats = {
        cnn: { correct:0, total:0, totalMs:0, tp:0, fp:0, fn:0, tn:0, history:[] },
        svm: { correct:0, total:0, totalMs:0, tp:0, fp:0, fn:0, tn:0, history:[] }
    };
    
    // Clear and update the live comparison chart
    if(comparisonChart){
        comparisonChart.data.labels = [];
        comparisonChart.data.datasets[0].data = [];
        comparisonChart.data.datasets[1].data = [];
        comparisonChart.update('none');
    }

    // Reset UI Score Cards and Confusion Matrix elements in the DOM
    ['cnn', 'svm'].forEach(m => {
        ['acc', 'f1', 'prec', 'rec', 'ms'].forEach(met => {
            const el = document.getElementById('bm-' + m + '-' + met);
            if (el) el.textContent = '--';
        });
        const cnt = document.getElementById('bm-' + m + '-count');
        if (cnt) cnt.textContent = '0';

        ['tn', 'fp', 'fn', 'tp'].forEach(cell => {
            const el = document.getElementById(m + '-' + cell);
            if (el) el.textContent = '0';
        });
    });
    
    // Update topbar patient info
    const pName = document.getElementById('currentPatientName');
    const pMeta = document.getElementById('currentPatientMeta');
    if(pName && data.meta) pName.textContent = data.meta.name || data.record_id;
    if(pMeta && data.meta) pMeta.textContent = `${data.meta.age} Yrs • ${data.meta.condition || 'Monitoring'}`;
}

// ===== BENCHMARK =====
function updateBenchmark(bm,trueLabel){
    ['cnn','svm'].forEach(m=>{
        benchmarkStats[m].total++;benchmarkStats[m].totalMs+=bm[m+'_ms'];
        const pred=bm[m+'_pred'];
        if(trueLabel>=0){
            if(pred===trueLabel)benchmarkStats[m].correct++;
            if(pred===1&&trueLabel===1)benchmarkStats[m].tp++;
            if(pred===1&&trueLabel===0)benchmarkStats[m].fp++;
            if(pred===0&&trueLabel===1)benchmarkStats[m].fn++;
            if(pred===0&&trueLabel===0)benchmarkStats[m].tn++;
        }
    });
    ['cnn','svm'].forEach(m=>{
        const s=benchmarkStats[m];
        const acc=s.total>0?((s.correct/s.total)*100).toFixed(1):'--';
        const prec=(s.tp+s.fp)>0?s.tp/(s.tp+s.fp):0;
        const rec=(s.tp+s.fn)>0?s.tp/(s.tp+s.fn):0;
        const f1=(prec+rec)>0?((2*prec*rec)/(prec+rec)*100).toFixed(1):'--';
        const avg=s.total>0?(s.totalMs/s.total).toFixed(1):'--';
        const ae=document.getElementById('bm-'+m+'-acc');if(ae)ae.textContent=acc+'%';
        const fe=document.getElementById('bm-'+m+'-f1');if(fe)fe.textContent=f1+'%';
        const me=document.getElementById('bm-'+m+'-ms');if(me)me.textContent=avg+' ms';
        const pe=document.getElementById('bm-'+m+'-prec');if(pe)pe.textContent=(prec>0?(prec*100).toFixed(1):'0.0')+'%';
        const re=document.getElementById('bm-'+m+'-rec');if(re)re.textContent=(rec>0?(rec*100).toFixed(1):'0.0')+'%';
        const ce=document.getElementById('bm-'+m+'-count');if(ce)ce.textContent=s.total;
        
        // Confusion Matrix
        const e_tn=document.getElementById(m+'-tn');if(e_tn)e_tn.textContent=s.tn;
        const e_tp=document.getElementById(m+'-tp');if(e_tp)e_tp.textContent=s.tp;
        const e_fn=document.getElementById(m+'-fn');if(e_fn)e_fn.textContent=s.fn;
        const e_fp=document.getElementById(m+'-fp');if(e_fp)e_fp.textContent=s.fp;
    });
    // Highlight winners
    // Highlight winners
    ['acc','f1','prec','rec','ms'].forEach(met=>{
        const c=document.getElementById('bm-cnn-'+met),s=document.getElementById('bm-svm-'+met);
        if(!c||!s)return;c.classList.remove('winner');s.classList.remove('winner');
        const cv=parseFloat(c.textContent),sv=parseFloat(s.textContent);
        if(isNaN(cv)||isNaN(sv))return;
        if(met==='ms'){if(cv<sv)c.classList.add('winner');else if(sv<cv)s.classList.add('winner');}
        else{if(cv>sv)c.classList.add('winner');else if(sv>cv)s.classList.add('winner');}
    });
    // Update chart
    if(comparisonChart){
        ['cnn','svm'].forEach((m,i)=>{
            const s=benchmarkStats[m];
            const v=s.total>0?((s.correct/s.total)*100):0;
            s.history.push(parseFloat(v.toFixed(1)));if(s.history.length>30)s.history.shift();
            comparisonChart.data.datasets[i].data=[...s.history];
        });
        const ml=Math.max(benchmarkStats.cnn.history.length,benchmarkStats.svm.history.length);
        comparisonChart.data.labels=Array.from({length:ml},(_,i)=>i+1);
        comparisonChart.update('none');
    }
    // Update radar chart with live data
    updateRadarFromBenchmark();
}

function initComparisonChart(){
    const el=document.getElementById('comparisonChart');
    if(el){
        comparisonChart=new Chart(el,{
            type:'line',
            data:{labels:[],datasets:[
                {label:'CNN Accuracy',data:[],borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.1)',fill:true,tension:0.4,pointRadius:2},
                {label:'SVM Accuracy',data:[],borderColor:'#f59e0b',backgroundColor:'rgba(245,158,11,0.1)',fill:true,tension:0.4,pointRadius:2}
            ]},
            options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'top'}},scales:{y:{min:0,max:100,title:{display:true,text:'Accuracy (%)'}},x:{title:{display:true,text:'Beat Sequence'}}}}
        });
    }

    // Radar Chart — initialized with zeros, updated dynamically from live benchmarkStats
    const radarEl=document.getElementById('radarChart');
    if(radarEl){
        radarChartInstance=new Chart(radarEl,{
            type:'radar',
            data:{
                labels: ['F1 Score', 'Precision', 'Recall', 'SNR Tolerance', 'Latency', 'Accuracy'],
                datasets: [
                    {label: 'CNN (Deep Learning)', data: [0, 0, 0, 0, 0, 0], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.4)', pointBackgroundColor: '#2563eb'},
                    {label: 'SVM (Machine Learning)', data: [0, 0, 0, 0, 0, 0], borderColor: '#94a3b8', backgroundColor: 'rgba(148,163,184,0.2)', pointBackgroundColor: '#94a3b8'}
                ]
            },
            options:{responsive:true,maintainAspectRatio:false,plugins:{tooltip:{enabled:true,mode:'index'}},scales:{r:{min:0,max:100,ticks:{display:false}}}}
        });
    }

    // SNR Chart — Real thesis Bab 4 research data (tested on 500 beats across 6 SNR levels)
    const snrEl=document.getElementById('snrChart');
    if(snrEl){
        snrChartInstance=new Chart(snrEl,{
            type:'line',
            data:{
                labels: ['5 dB\n(Sgt Buruk)', '10 dB\n(Buruk)', '15 dB\n(Cukup)', '20 dB\n(Sedang)', '30 dB\n(Baik)', '40 dB\n(Bersih)'],
                datasets: [
                    {label: 'CNN (Deep Learning)', data: [95.4, 97.8, 99.0, 99.4, 99.4, 99.4], borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,0.15)', fill: true, tension: 0.3, pointRadius: 5, pointBackgroundColor: '#2563eb', borderWidth: 2.5},
                    {label: 'SVM (Machine Learning)', data: [95.0, 99.6, 99.4, 99.4, 99.4, 99.4], borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,0.08)', fill: false, borderDash: [6,3], tension: 0.3, pointRadius: 5, pointBackgroundColor: '#f59e0b', borderWidth: 2.5}
                ]
            },
            options:{responsive:true,maintainAspectRatio:false,
                plugins:{tooltip:{enabled:true,mode:'index',intersect:false,callbacks:{label:function(ctx){return ctx.dataset.label+': '+ctx.parsed.y.toFixed(1)+'%';}}}},
                scales:{y:{min:92,max:101,title:{display:true,text:'Akurasi (%)'},ticks:{callback:function(v){return v+'%';}}},x:{title:{display:true,text:'Signal to Noise Ratio (SNR)'}}}
            }
        });
    }
}

// Update Radar Chart dynamically from live benchmarkStats
function updateRadarFromBenchmark(){
    if(!radarChartInstance) return;
    ['cnn','svm'].forEach((m,i)=>{
        const s=benchmarkStats[m];
        if(s.total===0) return;
        const prec=(s.tp+s.fp)>0? (s.tp/(s.tp+s.fp))*100 : 0;
        const rec=(s.tp+s.fn)>0? (s.tp/(s.tp+s.fn))*100 : 0;
        const f1=(prec+rec)>0? (2*prec*rec)/(prec+rec) : 0;
        const acc=(s.correct/s.total)*100;
        const avgMs=s.totalMs/s.total;
        // Latency score: invert so lower ms = higher score (cap at 100)
        const latencyScore = Math.min(100, Math.max(0, 100 - (avgMs / 5)));
        // SNR Tolerance from thesis research data
        const snrTolerance = m==='cnn' ? 97.8 : 99.6; // accuracy at SNR 10dB from Bab 4
        radarChartInstance.data.datasets[i].data = [
            parseFloat(f1.toFixed(1)),
            parseFloat(prec.toFixed(1)),
            parseFloat(rec.toFixed(1)),
            snrTolerance,
            parseFloat(latencyScore.toFixed(1)),
            parseFloat(acc.toFixed(1))
        ];
    });
    radarChartInstance.update('none');
}

function refreshComparisonChart(){if(comparisonChart)comparisonChart.update();}

// ===== TELEGRAM CONFIG =====
function saveTelegramConfig(){
    const token=document.getElementById('tgToken').value;
    const chatId=document.getElementById('tgChatId').value;
    const apiUrl=document.getElementById('tgApiUrl').value;
    fetch('/api/telegram_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({telegram_token:token,telegram_chat_id:chatId,telegram_api_url:apiUrl})}).then(()=>{
        const s=document.getElementById('tgSaveStatus');s.style.opacity='1';setTimeout(()=>s.style.opacity='0',2000);
        updateWebhookPreview();
    });
}
function loadTelegramConfig(){
    fetch('/api/telegram_config').then(r=>r.json()).then(d=>{
        document.getElementById('tgToken').value=d.telegram_token||'';
        document.getElementById('tgChatId').value=d.telegram_chat_id||'';
        if(document.getElementById('tgApiUrl')) document.getElementById('tgApiUrl').value=d.telegram_api_url||'';
        updateWebhookPreview();
    });
}
function updateWebhookPreview(){
    const previewEl = document.getElementById('tgWebhookUrlPreview');
    if (previewEl) {
        previewEl.textContent = "Webhook URL: " + window.location.origin + "/telegram_webhook";
    }
}
function openTelegramWebhookLink(action){
    const token = document.getElementById('tgToken').value.trim();
    if (!token) {
        alert("Masukkan Bot Token terlebih dahulu dan simpan!");
        return;
    }
    const webhookUrl = window.location.origin + "/telegram_webhook";
    let url = "";
    if (action === 'set') {
        url = `https://api.telegram.org/bot${token}/setWebhook?url=${encodeURIComponent(webhookUrl)}`;
    } else {
        url = `https://api.telegram.org/bot${token}/deleteWebhook`;
    }
    window.open(url, '_blank');
}

// ===== PER-PATIENT TELEGRAM REGISTRATION =====
function loadTelegramRegistrations(){
    fetch('/api/telegram/registrations').then(r=>r.json()).then(data=>{
        const list=document.getElementById('tgRegList');
        const badge=document.getElementById('tgRegCountBadge');
        if(!list) return;
        
        // Store for patient card rendering
        appState.telegramRegistrations={};
        data.registrations.forEach(r=>{ appState.telegramRegistrations[r.record_id]=r.chat_id; });
        
        if(badge) badge.textContent=data.registrations.length+' Terdaftar';
        
        if(data.registrations.length===0){
            list.innerHTML='<div style="text-align:center;padding:32px;color:var(--text-muted);font-size:0.875rem;background:var(--bg-main);border-radius:16px;border:1px dashed var(--border);">Belum ada pasien terdaftar. Gunakan form di atas untuk mendaftarkan.</div>';
            return;
        }
        list.innerHTML='';
        data.registrations.forEach(r=>{
            const sexLabel=r.sex==='Female'?'Wanita':r.sex==='Male'?'Pria':r.sex;
            const item=document.createElement('div');
            item.style.cssText='display:flex;align-items:center;justify-content:space-between;padding:14px 18px;background:var(--bg-main);border:1px solid var(--border);border-radius:16px;transition:all 0.2s;';
            item.innerHTML=`
                <div style="display:flex;align-items:center;gap:14px;">
                    <div style="width:40px;height:40px;border-radius:50%;background:#e0f2fe;display:flex;align-items:center;justify-content:center;font-size:18px;">📲</div>
                    <div>
                        <div style="font-size:0.875rem;font-weight:700;color:var(--text-main);">${r.name}</div>
                        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">${r.record_id} • ${r.age} thn • ${sexLabel} • Aritmia: ${parseFloat(r.rate).toFixed(1)}%</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:10px;">
                    <span style="font-size:0.7rem;color:#0ea5e9;font-weight:600;font-family:var(--font-mono);">ID: ${r.chat_id}</span>
                    <button onclick="unregisterTelegram('${r.record_id}')" style="width:28px;height:28px;border-radius:50%;border:1px solid #fecaca;background:#fef2f2;color:#dc2626;font-size:14px;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all 0.2s;" title="Hapus registrasi">✕</button>
                </div>
            `;
            list.appendChild(item);
        });
    });
}

function registerTelegram(){
    const recordId=document.getElementById('tgRegRecordId').value.trim();
    const chatId=document.getElementById('tgRegChatId').value.trim();
    const statusEl=document.getElementById('tgRegStatus');
    const btn=document.getElementById('tgRegBtn');
    
    if(!recordId||!chatId){
        statusEl.textContent='⚠️ Harap isi Record ID dan Chat ID';
        statusEl.style.color='#dc2626';
        statusEl.style.display='inline';
        return;
    }
    
    btn.disabled=true;
    btn.textContent='Memvalidasi...';
    statusEl.style.display='none';
    
    fetch('/api/telegram/register',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({record_id:recordId,chat_id:chatId})
    })
    .then(r=>r.json().then(data=>({status:r.status,data})))
    .then(({status,data})=>{
        if(status===200){
            statusEl.textContent='✅ '+data.message+' — Cek Telegram!';
            statusEl.style.color='#16a34a';
            document.getElementById('tgRegRecordId').value='';
            document.getElementById('tgRegChatId').value='';
            loadTelegramRegistrations();
            loadPatients();
        } else {
            statusEl.textContent='❌ '+data.message;
            statusEl.style.color='#dc2626';
        }
        statusEl.style.display='inline';
    })
    .catch(err=>{
        statusEl.textContent='❌ Gagal: '+err.message;
        statusEl.style.color='#dc2626';
        statusEl.style.display='inline';
    })
    .finally(()=>{
        btn.disabled=false;
        btn.textContent='Daftarkan Pasien';
    });
}

function unregisterTelegram(recordId){
    if(!confirm('Hapus registrasi Telegram untuk '+recordId+'?')) return;
    fetch('/api/telegram/unregister',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({record_id:recordId})
    })
    .then(r=>r.json())
    .then(data=>{
        loadTelegramRegistrations();
        loadPatients();
    });
}

// ===== GLOBAL ALERT CENTER POLLING =====
let lastGlobalAlertId = null;
let lastSelectedPatient = null;
function pollGlobalAlerts() {
    fetch('/api/global_alerts')
    .then(r => r.json())
    .then(data => {
        if (!data.alerts) return;
        const rsAlerts = document.getElementById('rsAlerts');
        if (!rsAlerts) return;

        const latestAlert = data.alerts.length > 0 ? data.alerts[data.alerts.length - 1] : null;
        const latestId = latestAlert ? latestAlert.id : null;

        // Rebuild if:
        // 1. The newest alert's unique ID has changed
        // 2. The selected patient has changed (so highlight border moves)
        // 3. The sidebar was empty but we now have alerts
        if (latestId === lastGlobalAlertId && appState.selectedPatient === lastSelectedPatient && rsAlerts.children.length > 0) {
            return;
        }

        lastGlobalAlertId = latestId;
        lastSelectedPatient = appState.selectedPatient;

        // If no alerts, show empty state message
        if (data.alerts.length === 0) {
            rsAlerts.innerHTML = `
                <div class="alert-empty" style="text-align: center; padding: 2rem; color: #94a3b8;">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 32px; height: 32px; margin: 0 auto 12px auto; opacity: 0.5;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
                    <p style="font-size: 0.85rem; font-weight: 500;">Belum ada deteksi aritmia</p>
                </div>
            `;
            return;
        }

        // Rebuild the sidebar with ALL patient alerts (newest first)
        const sorted = data.alerts.slice().reverse();
        rsAlerts.innerHTML = '';
        const maxShow = 30;
        sorted.slice(0, maxShow).forEach(a => {
            const isCurrentPatient = (a.patient_id === appState.selectedPatient);
            const rsRow = document.createElement('div');
            rsRow.className = 'rs-alert-item';
            if (isCurrentPatient) {
                rsRow.style.borderLeft = '3px solid #f87171';
                rsRow.style.background = 'rgba(239, 68, 68, 0.03)';
            }

            let condText = 'Aritmia';
            if (a.heart_rate > 100) condText = 'Takikardia (' + a.heart_rate + ' BPM)';
            else if (a.heart_rate < 60) condText = 'Bradikardia (' + a.heart_rate + ' BPM)';
            else condText = 'Aritmia (' + a.heart_rate + ' BPM)';

            rsRow.innerHTML = `
                <div class="rs-icon critical"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/></svg></div>
                <div class="rs-alert-body">
                    <div class="rs-alert-meta"><span style="font-weight:700;color:${isCurrentPatient ? '#ef4444' : '#94a3b8'}">${a.patient_id}</span><span>${a.time}</span></div>
                    <p>${condText} — ${(a.confidence * 100).toFixed(0)}% [${a.model}]</p>
                </div>
            `;
            // Click to switch to that patient
            rsRow.style.cursor = 'pointer';
            rsRow.addEventListener('click', () => {
                selectPatient(a.patient_id);
            });
            rsAlerts.appendChild(rsRow);
        });
    })
    .catch(() => {});
}

// Poll every 3 seconds
setInterval(pollGlobalAlerts, 3000);

// ===== BROWSER-SIDE TELEGRAM DISPATCH =====
// HF Spaces blocks outbound HTTP to *.workers.dev and api.telegram.org.
// The browser has NO such restriction, so we dispatch queued alerts from here.
async function dispatchTelegramOutbox() {
    try {
        const res = await fetch('/api/telegram_outbox');
        const items = await res.json();
        if (!items || items.length === 0) return;

        const clearedIds = [];

        for (const item of items) {
            const apiUrl = (item.api_url || 'https://api.telegram.org').replace(/\/+$/, '');
            const sendUrl = `${apiUrl}/bot${item.token}/sendMessage`;

            for (const chatId of item.targets) {
                try {
                    const tgRes = await fetch(sendUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            chat_id: chatId,
                            text: item.message,
                            parse_mode: 'Markdown'
                        })
                    });
                    const tgData = await tgRes.json();
                    if (tgData.ok) {
                        console.log(`[TG-DISPATCH] Alert sent to ${chatId} for ${item.patient_id}`);
                    } else {
                        console.warn(`[TG-DISPATCH] Telegram error for ${chatId}:`, tgData.description);
                    }
                } catch (e) {
                    console.error(`[TG-DISPATCH] Network error sending to ${chatId}:`, e);
                }
            }
            clearedIds.push(item.id);
        }

        // Clear dispatched items from server queue
        if (clearedIds.length > 0) {
            await fetch('/api/telegram_outbox/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: clearedIds })
            });
            console.log(`[TG-DISPATCH] Cleared ${clearedIds.length} dispatched alert(s)`);
        }
    } catch (e) {
        // Silent fail — network errors are expected during page load
    }
}

// Poll outbox every 3 seconds
setInterval(dispatchTelegramOutbox, 3000);

// ===== INIT =====
loadPatients();
loadTelegramConfig();
loadTelegramRegistrations();
connectSSE();
pollGlobalAlerts();
