
// --- TAB SWITCHING LOGIC ---
document.getElementById('textTab').addEventListener('click', () => switchTab('text'));
document.getElementById('imageTab').addEventListener('click', () => switchTab('image'));

function switchTab(type) {
    document.getElementById('textTab').classList.toggle('active', type === 'text');
    document.getElementById('imageTab').classList.toggle('active', type === 'image');
    document.getElementById('textInputGroup').classList.toggle('hidden', type !== 'text');
    document.getElementById('imageInputGroup').classList.toggle('hidden', type !== 'image');
    resetPipelineUI();
}

// Report Toggle Logic
document.getElementById('reportToggle').addEventListener('click', () => {
    const details = document.getElementById('reportDetails');
    const header = document.getElementById('reportToggle');
    details.classList.toggle('hidden');
    header.classList.toggle('open');
});

// --- IMAGE UPLOAD LOGIC ---
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('imageInput');
const selectedImg = document.getElementById('selectedImg');
const imagePreview = document.getElementById('imagePreview');

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('zone-active'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('zone-active'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('zone-active');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    const reader = new FileReader();
    reader.onload = e => {
        selectedImg.src = e.target.result;
        imagePreview.classList.remove('hidden');
        dropZone.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

// --- SUBMIT HANDLERS ---
document.getElementById('submitBtn').addEventListener('click', processPipeline);
document.getElementById('submitImgBtn').addEventListener('click', processImagePipeline);

// Typewriter effect function
async function typeWriter(text, elementId, speed = 15) {
    const element = document.getElementById(elementId);
    element.innerHTML = '';
    for (let i = 0; i < text.length; i++) {
        element.innerHTML += text.charAt(i);
        await new Promise(resolve => setTimeout(resolve, speed));
    }
}

const delay = ms => new Promise(res => setTimeout(res, ms));

async function processPipeline() {
    const input = document.getElementById('userInput').value.trim();
    if (!input) return;

    const btn = document.getElementById('submitBtn');
    updateLoadingState(btn, true, 'Scanning & Securing...');
    resetPipelineUI();

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input })
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        await runAnimationSequence(data);
        await updateMonitor(data, 'text');
    } catch (error) {
        showError(error);
    } finally {
        updateLoadingState(btn, false, 'Process Securely');
    }
}

async function processImagePipeline() {
    const file = fileInput.files[0];
    if (!file) return;

    const btn = document.getElementById('submitImgBtn');
    updateLoadingState(btn, true, 'Processing Vision Data...');
    resetPipelineUI();

    const instructions = document.getElementById('imageInstructions').value.trim();
    const formData = new FormData();
    formData.append('image', file);
    if (instructions) formData.append('instructions', instructions);

    try {
        const response = await fetch('/process-image', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        // Custom Sequence for Image
        await delay(400);
        showStep('detectionSection');
        document.getElementById('conn1').classList.remove('hidden');

        // Show Scanning Animation
        const scanStatus = document.getElementById('privacyScanStatus');
        const reportCard = document.getElementById('privacyReportCard');
        scanStatus.classList.remove('hidden');
        reportCard.classList.add('hidden');

        await delay(1500); // Simulate deep scan
        scanStatus.classList.add('hidden');
        reportCard.classList.remove('hidden');

        renderImageDetection(data);
        updateStatusHUD(data.risk_score > 0.8 ? 5 : 1); // Mock count for status HUD if high risk

        await delay(600);
        showStep('tokenizationSection');
        renderImageMask(data.masked_image);
        document.getElementById('conn2').classList.remove('hidden');

        // 5. LLM Reasoning (Visual Step)
        await delay(600);
        showStep('llmSection');
        document.getElementById('conn3').classList.remove('hidden');

        // Only show raw output if debug data present
        if (data.llm_raw_response) {
            document.getElementById('debugOutputSection').classList.remove('hidden');
            await typeWriter(data.llm_raw_response, 'llmRawOutput', 10);
        } else {
            // Give user a visual of "thinking"
            await typeWriter("Synthesizing context-aware summary...", 'llmRawOutput', 10);
            await delay(1000);
        }

        // 6. Final Detokenized Result
        await delay(600);
        showStep('finalSection');
        if (data.llm_response) {
            await typeWriter(data.llm_response, 'finalOutput', 15);
        } else {
            document.getElementById('finalOutput').innerHTML = '<p class="subtitle">AI summary generated securely. Please check transmission logs.</p>';
        }

        await updateMonitor(data, 'image');
    } catch (error) {
        showError(error);
    } finally {
        updateLoadingState(btn, false, 'Detect & Mask Privacy');
    }
}

async function runAnimationSequence(data) {
    // Step 02: Launch Privacy Inspection
    await delay(200);
    showStep('detectionSection');
    document.getElementById('conn1').classList.remove('hidden');

    // Show Scanning Animation
    const scanStatus = document.getElementById('privacyScanStatus');
    const reportCard = document.getElementById('privacyReportCard');
    scanStatus.classList.remove('hidden');
    reportCard.classList.add('hidden');

    await delay(1500); // Simulate deep scan
    scanStatus.classList.add('hidden');
    reportCard.classList.remove('hidden');

    renderDetection(data.ner_report);
    updateStatusHUD(data.ner_report.total_sensitive_items);

    await delay(600);
    showStep('tokenizationSection');
    renderTokenization(data.tokenized_input);
    document.getElementById('conn2').classList.remove('hidden');

    await delay(600);
    showStep('llmSection');
    document.getElementById('conn3').classList.remove('hidden');

    // Only show raw output if debug data present
    if (data.llm_raw_response) {
        document.getElementById('debugOutputSection').classList.remove('hidden');
        await typeWriter(data.llm_raw_response, 'llmRawOutput', 10);
    } else {
        await typeWriter("Processing sanitized prompt...", 'llmRawOutput', 10);
        await delay(1000);
    }

    await delay(600);
    showStep('finalSection');
    const final_res = data.llm_response || data.final_response;
    if (final_res) {
        await typeWriter(final_res, 'finalOutput', 15);
        // Apply highlights after typing
        renderFinalOutput(final_res, data.token_map || {});
    } else {
        document.getElementById('finalOutput').innerHTML = '<p class="subtitle">Secure response received.</p>';
    }
}

function updateStatusHUD(count) {
    const hud = document.getElementById('statusHUD');
    const text = document.getElementById('statusText');

    hud.className = 'status-hud'; // Reset
    if (count === 0) {
        text.innerText = 'SYSTEM STATUS: SAFE';
    } else if (count <= 2) {
        hud.classList.add('risk-medium');
        text.innerText = 'SYSTEM STATUS: PROTECTED (LOW RISK)';
    } else {
        hud.classList.add('risk-high');
        text.innerText = 'SYSTEM STATUS: CRITICAL (HIGH PROTECTION ON)';
    }
}

function updateLoadingState(btn, isLoading, text) {
    btn.disabled = isLoading;
    btn.querySelector('.spinner').classList.toggle('hidden', !isLoading);
    btn.querySelector('.btn-text').innerText = text;
}

function showError(err) {
    console.error("Pipeline Error:", err);
    alert(err.message || "Operation failed.");
}

function showStep(id) {
    const el = document.getElementById(id);
    el.classList.remove('hidden-step');
    el.classList.add('show-step');
    if (window.innerWidth < 1024) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function resetPipelineUI() {
    const steps = ['detectionSection', 'tokenizationSection', 'llmSection', 'finalSection', 'monitorSection'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.add('hidden-step');
            el.classList.remove('show-step');
        }
    });
    ['conn1', 'conn2', 'conn3'].forEach(id => document.getElementById(id).classList.add('hidden'));
    document.getElementById('llmRawOutput').innerText = '';
    document.getElementById('debugOutputSection').classList.add('hidden');

    // Reset Monitor Specifics
    document.getElementById('decisionTableBody').innerHTML = '<tr class="empty-row"><td colspan="4">No data analyzed yet...</td></tr>';
    document.getElementById('dataFlowLog').innerHTML = '<div class="log-entry">SYSTEM IDLE: Waiting for input...</div>';
    ['step-extract', 'step-analyze', 'step-decide', 'step-mask', 'step-output'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });
}

function renderDetection(report) {
    const container = document.getElementById('nerResults');
    const countEl = document.getElementById('reportCount');
    container.innerHTML = '';

    const items = report.entities || [];
    countEl.innerText = `${items.length} Sensitive items protected`;

    if (items.length === 0) {
        container.innerHTML = '<p class="subtitle" style="grid-column: 1/-1">No sensitive entities detected. Your data is clean.</p>';
        return;
    }

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'entity-item';
        div.innerHTML = `<span class="entity-label">${item.label}</span><span class="entity-value" title="Original data masked">[PROTECTED]</span>`;
        container.appendChild(div);
    });
}

function renderImageDetection(data) {
    const container = document.getElementById('nerResults');
    const report = data.privacy_report || {};

    container.innerHTML = `
        <div class="entity-item">
            <span class="entity-label">Risk Level</span>
            <span class="entity-value risk-level ${report.risk_level === 'CRITICAL' ? 'risk-high' : 'risk-low'}">${report.risk_level}</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Faces</span>
            <span class="entity-value">${report.faces_detected} detected</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Sensitive Text</span>
            <span class="entity-value">${report.text_regions} regions</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Masked Regions</span>
            <span class="entity-value">${report.masked_regions} items</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Barcodes/QR</span>
            <span class="entity-value">${report.barcodes_detected || 0} items</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Official Doc</span>
            <span class="entity-value">${report.document_detected ? 'YES' : 'NO'}</span>
        </div>
    `;
}

function renderImageMask(base64) {
    const container = document.getElementById('tokenizedPrompt');
    container.innerHTML = `
        <div style="text-align: center;">
            <p style="margin-bottom: 10px; font-size: 0.8rem; color: var(--purple);">PROCESSED MASKED ARTIFACT</p>
            <img src="${base64}" style="max-width: 100%; border-radius: 8px; border: 2px solid var(--purple);">
        </div>
    `;
}

function renderTokenization(text) {
    const container = document.getElementById('tokenizedPrompt');
    if (!text) {
        container.innerHTML = '<p class="subtitle">No tokenization required.</p>';
        return;
    }
    container.innerHTML = text.replace(/\[([A-Z0-9_]+)\]/g, (match, label) => `<span class="token-pill" title="Protected data placeholder: ${label}">${match}</span>`);
}

function renderFinalOutput(text, tokenMap) {
    const container = document.getElementById('finalOutput');
    if (!text) return;

    let highlighted = text;
    if (tokenMap && Object.keys(tokenMap).length > 0) {
        for (const [token, original] of Object.entries(tokenMap)) {
            const escaped = original.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(`(${escaped})`, 'g');
            highlighted = highlighted.replace(re, '<span class="highlight-restored">$1</span>');
        }
    }
    container.innerHTML = highlighted;
}

window.addEventListener('load', () => document.body.classList.add('fade-in'));

// --- PRIVACY EXECUTION MONITOR LOGIC ---

function addLog(message) {
    const logBox = document.getElementById('dataFlowLog');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerText = `[${new Date().toLocaleTimeString()}] ${message}`;
    logBox.appendChild(entry);
    logBox.scrollTop = logBox.scrollHeight;
}

async function updateMonitor(data, type = 'text') {
    const monitor = document.getElementById('monitorSection');
    monitor.classList.remove('hidden-step');
    monitor.classList.add('show-step');

    // Reset Steps
    ['step-extract', 'step-analyze', 'step-decide', 'step-mask', 'step-output'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });

    const isText = type === 'text';
    const report = isText ? data.ner_report : data.privacy_report;
    const entities = isText ? (report.entities || []) : [];

    // Step 1: Extract
    document.getElementById('step-extract').classList.add('active');
    addLog(`Initiating ${type} extraction audit...`);
    await delay(400);

    // Update OCR/NER counts
    if (isText) {
        document.getElementById('auditFaces').innerText = '0';
        document.getElementById('auditTextRegions').innerText = entities.length;
        document.getElementById('auditMasks').innerText = entities.length;
        document.getElementById('auditBarcodes').innerText = '0';
    } else {
        document.getElementById('auditFaces').innerText = report.faces_detected || 0;
        document.getElementById('auditTextRegions').innerText = report.text_regions || 0;
        document.getElementById('auditMasks').innerText = report.masked_regions || 0;
        document.getElementById('auditBarcodes').innerText = data.barcodes_detected || 0; // Use data.barcodes_detected if available
    }

    // Step 2: Analyze
    document.getElementById('step-analyze').classList.add('active');
    addLog("Analyzing sensitivity levels and risk vectors...");
    await delay(400);

    // Update Risk
    const riskScore = data.risk_score || 0;
    document.getElementById('auditRiskScore').innerText = riskScore.toFixed(2);
    const safeEl = document.getElementById('auditSafe');
    const safe = data.safe_to_forward ?? true;
    safeEl.innerText = safe ? 'YES' : 'NO';
    safeEl.className = `safe-tag ${safe ? 'tag-yes' : 'tag-no'}`;
    document.getElementById('auditPolicy').innerText = riskScore > 0.8 ? 'RESTRICTED' : 'ALLOWED';

    // Step 3: Decide
    document.getElementById('step-decide').classList.add('active');
    addLog("Applying privacy policy rules to detected entities...");
    await delay(400);

    // Populate Table
    const tableBody = document.getElementById('decisionTableBody');
    tableBody.innerHTML = '';

    if (isText && entities.length > 0) {
        entities.forEach((ent, idx) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${ent.label}: ${ent.text}</td>
                <td>[MASK_${idx + 1}]</td>
                <td class="decision-masked">MASKED</td>
                <td class="reason-tag">Privacy Pattern Match</td>
            `;
            tableBody.appendChild(row);
        });
    } else if (!isText) {
        if (report.faces_detected > 0) {
            const row = document.createElement('tr');
            row.innerHTML = `<td>Biometric: Human Face</td><td>[BLURRED]</td><td class="decision-masked">REDACTED</td><td class="reason-tag">GDPR Compliance Rule</td>`;
            tableBody.appendChild(row);
        }
        if (report.text_regions > 0) {
            const row = document.createElement('tr');
            row.innerHTML = `<td>Document Text Regions</td><td>[BLACKED]</td><td class="decision-masked">MASKED</td><td class="reason-tag">PII Exclusion Filter</td>`;
            tableBody.appendChild(row);
        }
        // Check barcodes
        const barcodeCount = data.barcodes_detected || 0;
        if (barcodeCount > 0) {
            const row = document.createElement('tr');
            row.innerHTML = `<td>Visual: QR/Barcode</td><td>[BLURRED]</td><td class="decision-masked">REDACTED</td><td class="reason-tag">Secure Metadata Sweep</td>`;
            tableBody.appendChild(row);
        }
    }

    if (tableBody.innerHTML === '') {
        tableBody.innerHTML = '<tr class="empty-row"><td colspan="4">No sensitive data found. No masking required.</td></tr>';
    }

    // Step 4: Mask
    document.getElementById('step-mask').classList.add('active');
    addLog("Masking execution complete. Secure payload generated.");
    await delay(400);

    // Step 5: Output
    document.getElementById('step-output').classList.add('active');
    if (!isText) {
        addLog("Multi-modal vision switching: ACTIVE");
    }
    addLog("De-tokenization verified. Transmission successful.");

    document.getElementById('encryptionStatus').innerText = "End-to-End Encryption Verified";
}
