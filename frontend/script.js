// API endpoints
const API_URL = '/runs';

// DOM Elements
const apiKeyInput = document.getElementById('apiKeyInput');
const openaiKeyInput = document.getElementById('openaiKeyInput');
const saveApiKeyBtn = document.getElementById('saveApiKeyBtn');
const evalForm = document.getElementById('evalForm');
const triggerBtn = document.getElementById('triggerBtn');
const triggerFeedback = document.getElementById('triggerFeedback');
const runsTableBody = document.getElementById('runsTableBody');
const generatorProvider = document.getElementById('generatorProvider');
const modelNameInput = document.getElementById('modelName');

// State
let pollingInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    const savedKey = localStorage.getItem('eval_api_key');
    const savedOpenaiKey = localStorage.getItem('eval_openai_api_key');
    
    if (savedOpenaiKey) {
        openaiKeyInput.value = savedOpenaiKey;
    }
    
    if (savedKey) {
        apiKeyInput.value = savedKey;
        fetchRuns();
        startPolling();
    }
});

// Handle Provider Change
generatorProvider.addEventListener('change', () => {
    if (generatorProvider.value === 'openai') {
        modelNameInput.value = 'gpt-4o-mini';
    } else {
        modelNameInput.value = 'llama3';
    }
});

// Save API Keys
saveApiKeyBtn.addEventListener('click', () => {
    const key = apiKeyInput.value.trim();
    const openaiKey = openaiKeyInput.value.trim();
    
    if (openaiKey) {
        localStorage.setItem('eval_openai_api_key', openaiKey);
    } else {
        localStorage.removeItem('eval_openai_api_key');
    }

    if (key) {
        localStorage.setItem('eval_api_key', key);
        alert('API Keys saved locally.');
        fetchRuns();
        startPolling();
    } else {
        alert('Please enter a valid App API Key.');
    }
});

// Trigger New Evaluation
evalForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const apiKey = localStorage.getItem('eval_api_key');
    if (!apiKey) {
        showFeedback('Please save an App API Key first.', 'error');
        return;
    }

    const judgeProvider = document.getElementById('judgeProvider').value;
    const genProvider = generatorProvider.value;
    const openaiKey = localStorage.getItem('eval_openai_api_key');

    if ((judgeProvider === 'openai' || genProvider === 'openai') && !openaiKey) {
        showFeedback('OpenAI API Key is required for cloud generation/judging.', 'error');
        return;
    }

    const payload = {
        model_name: modelNameInput.value.trim(),
        generator_provider: genProvider,
        judge_provider: judgeProvider,
        openai_api_key: openaiKey || null,
        prompt_variant: document.getElementById('promptVariant').value.trim(),
        n_samples: parseInt(document.getElementById('nSamples').value, 10)
    };

    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Starting...';

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': apiKey
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            let msg = 'Failed to start evaluation';
            if (err.detail) {
                if (Array.isArray(err.detail)) {
                    // Extract readable messages from Pydantic validation errors
                    msg = err.detail.map(d => d.msg).join(', ');
                } else if (typeof err.detail === 'string') {
                    msg = err.detail;
                }
            }
            throw new Error(msg);
        }

        showFeedback('Evaluation started successfully!', 'success');
        fetchRuns(); // Update table immediately
    } catch (error) {
        showFeedback(error.message, 'error');
    } finally {
        triggerBtn.disabled = false;
        triggerBtn.textContent = 'Start Evaluation';
        setTimeout(() => showFeedback('', ''), 3000); // Clear feedback after 3s
    }
});

// Fetch and display runs
async function fetchRuns() {
    const apiKey = localStorage.getItem('eval_api_key');
    if (!apiKey) return;

    try {
        // Use cache-busting to prevent the browser from caching the GET request
        const response = await fetch(`${API_URL}?t=${Date.now()}`, {
            method: 'GET',
            headers: {
                'X-API-Key': apiKey,
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });

        if (response.status === 403) {
            stopPolling();
            runsTableBody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: red;">Invalid API Key. Please update and save.</td></tr>';
            return;
        }

        if (!response.ok) throw new Error('Failed to fetch runs');

        const runs = await response.json();
        renderRuns(runs);
    } catch (error) {
        console.error('Error fetching runs:', error);
    }
}

function renderRuns(runs) {
    if (!runs || runs.length === 0) {
        runsTableBody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #666;">No runs found.</td></tr>';
        return;
    }

    runsTableBody.innerHTML = runs.map(run => {
        const shortId = run.run_id.substring(0, 8);
        
        // Format metrics
        let cp = '-', ar = '-', f = '-';
        if (run.metrics) {
            cp = run.metrics.context_precision !== null ? run.metrics.context_precision.toFixed(4) : 'NaN';
            ar = run.metrics.answer_relevancy !== null ? run.metrics.answer_relevancy.toFixed(4) : 'NaN';
            f = run.metrics.faithfulness !== null ? run.metrics.faithfulness.toFixed(4) : 'NaN';
        }

        // Format duration
        const duration = run.duration_seconds ? `${Math.round(run.duration_seconds)}s` : '-';

        return `
            <tr>
                <td class="run-id" title="${run.run_id}" style="cursor: help; border-bottom: 1px dotted #ccc;">${shortId}...</td>
                <td><span class="status-badge status-${run.status}">${run.status}</span></td>
                <td>${run.model_name} <br> <small style="color:#666;">(${run.prompt_variant})</small></td>
                <td>${run.n_samples}</td>
                <td>${cp}</td>
                <td>${ar}</td>
                <td>${f}</td>
                <td>${duration}</td>
            </tr>
        `;
    }).join('');
}

// Polling mechanism
function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    pollingInterval = setInterval(fetchRuns, 5000); // Poll every 5 seconds
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// UI Helpers
function showFeedback(message, type) {
    triggerFeedback.textContent = message;
    triggerFeedback.className = `feedback-msg ${type}`;
}
