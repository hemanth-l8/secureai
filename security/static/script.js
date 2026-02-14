
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

    const formData = new FormData();
    formData.append('image', file);

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
        renderImageDetection(data);
        document.getElementById('conn1').classList.remove('hidden');

        await delay(600);
        showStep('tokenizationSection');
        renderImageMask(data.masked_image);
        document.getElementById('conn2').classList.remove('hidden');

        await delay(600);
        showStep('llmSection');
        document.getElementById('conn3').classList.remove('hidden');
        await typeWriter(data.llm_response, 'llmRawOutput', 20);

        await delay(600);
        showStep('finalSection');
        renderFinalOutput(`Privacy Risk Assessment Completed. Score: ${data.risk_score}`, {});
    } catch (error) {
        showError(error);
    } finally {
        updateLoadingState(btn, false, 'Detect & Mask Privacy');
    }
}

async function runAnimationSequence(data) {
    await delay(400);
    showStep('detectionSection');
    renderDetection(data.ner_report);
    document.getElementById('conn1').classList.remove('hidden');

    await delay(600);
    showStep('tokenizationSection');
    renderTokenization(data.tokenized_input);
    document.getElementById('conn2').classList.remove('hidden');

    await delay(600);
    showStep('llmSection');
    document.getElementById('conn3').classList.remove('hidden');
    await typeWriter(data.llm_raw_response, 'llmRawOutput', 20);

    await delay(600);
    showStep('finalSection');
    renderFinalOutput(data.final_response, data.token_map);
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
    const steps = ['detectionSection', 'tokenizationSection', 'llmSection', 'finalSection'];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.classList.add('hidden-step');
        el.classList.remove('show-step');
    });
    ['conn1', 'conn2', 'conn3'].forEach(id => document.getElementById(id).classList.add('hidden'));
    document.getElementById('llmRawOutput').innerText = '';
}

function renderDetection(report) {
    const container = document.getElementById('nerResults');
    container.innerHTML = '';
    let items = [];
    if (report.structured_data) Object.entries(report.structured_data).forEach(([l, list]) => list.forEach(v => items.push({ label: l, value: v })));
    if (report.contextual_regex_data) Object.entries(report.contextual_regex_data).forEach(([l, list]) => list.forEach(v => items.push({ label: l, value: v })));
    if (report.contextual_data) Object.entries(report.contextual_data).forEach(([l, list]) => list.forEach(v => items.push({ label: l, value: v })));

    if (items.length === 0) {
        container.innerHTML = '<p class="subtitle" style="grid-column: 1/-1">No sensitive entities detected.</p>'; return;
    }
    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'entity-item';
        div.innerHTML = `<span class="entity-label">${item.label}</span><span class="entity-value" title="${item.value}">${item.value}</span>`;
        container.appendChild(div);
    });
}

function renderImageDetection(data) {
    const container = document.getElementById('nerResults');
    container.innerHTML = `
        <div class="entity-item">
            <span class="entity-label">Risk Level</span>
            <span class="entity-value risk-level ${data.risk_score > 0.8 ? 'risk-high' : 'risk-low'}">${data.risk_score > 0.8 ? 'CRITICAL' : 'LOW'}</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Faces</span>
            <span class="entity-value">${data.faces_detected} detected</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Sensitive Text</span>
            <span class="entity-value">${data.sensitive_text_regions} regions</span>
        </div>
        <div class="entity-item">
            <span class="entity-label">Official Doc</span>
            <span class="entity-value">${data.document_detected ? 'YES' : 'NO'}</span>
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
    container.innerHTML = text.replace(/\[([A-Z0-9_]+)\]/g, (match, label) => `<span class="token-pill" title="Protected data placeholder: ${label}">${match}</span>`);
}

function renderFinalOutput(text, tokenMap) {
    const container = document.getElementById('finalOutput');
    let highlighted = text;
    for (const [token, original] of Object.entries(tokenMap)) {
        const escaped = original.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp(`(${escaped})`, 'g');
        highlighted = highlighted.replace(re, '<span class="highlight-restored">$1</span>');
    }
    container.innerHTML = highlighted;
}

window.addEventListener('load', () => document.body.classList.add('fade-in'));
