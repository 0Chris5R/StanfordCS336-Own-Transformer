/**
 * Browser LM Demo - Pure browser inference with ONNX Runtime Web
 */

// Global state
let session = null;
let tokenizer = null;
let modelConfig = null;
let isGenerating = false;

// Special token IDs (set after tokenizer loads)
let THINK_TOKEN = null;
let ANSWER_TOKEN = null;
let REFUSE_TOKEN = null;
let END_TOKEN = null;

// DOM elements
const statusEl = document.getElementById('status');
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('send');
const statsEl = document.getElementById('stats');
const tempSlider = document.getElementById('temperature');
const topPSlider = document.getElementById('top-p');
const maxTSlider = document.getElementById('max-tokens');

// Update slider displays
tempSlider.addEventListener('input', () => {
    document.getElementById('temp-value').textContent = tempSlider.value;
});
topPSlider.addEventListener('input', () => {
    document.getElementById('topp-value').textContent = topPSlider.value;
});
maxTSlider.addEventListener('input', () => {
    document.getElementById('maxt-value').textContent = maxTSlider.value;
});

// Initialize
async function init() {
    try {
        updateStatus('Loading tokenizer...', 'loading');

        // Load tokenizer
        tokenizer = new BPETokenizer();
        await tokenizer.load('tokenizer.json');

        // Get special token IDs
        const specialTokens = tokenizer.specialTokens || [];
        const baseOffset = 256; // Special tokens start after byte tokens
        THINK_TOKEN = baseOffset + specialTokens.indexOf('<|think|>');
        ANSWER_TOKEN = baseOffset + specialTokens.indexOf('<|answer|>');
        REFUSE_TOKEN = baseOffset + specialTokens.indexOf('<|refuse|>');
        END_TOKEN = baseOffset + specialTokens.indexOf('<|endoftext|>');

        // Load model config
        const configResponse = await fetch('model_config.json');
        modelConfig = await configResponse.json();

        updateStatus('Loading model (270MB, please wait)...', 'loading');

        // Configure ONNX Runtime (single-threaded for file:// protocol compatibility)
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;

        // Load ONNX model with memory-optimized settings
        const startLoad = performance.now();
        session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
            enableCpuMemArena: false,
            enableMemPattern: false,
        });
        const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);

        updateStatus(`Model loaded in ${loadTime}s. Ready to generate!`, 'ready');
        inputEl.disabled = false;
        sendBtn.disabled = false;
        inputEl.focus({ preventScroll: true });

    } catch (error) {
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

function updateStatus(message, type) {
    statusEl.textContent = message;
    statusEl.className = `status ${type}`;
}

function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    chatEl.appendChild(div);
    chatEl.scrollTop = chatEl.scrollHeight;
    return div;
}

function createAssistantMessage() {
    const container = document.createElement('div');
    container.className = 'message assistant';

    // Thinking indicator (shown during thinking phase)
    const thinkingIndicator = document.createElement('div');
    thinkingIndicator.className = 'thinking-indicator';
    thinkingIndicator.innerHTML = '<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span> Thinking...';
    container.appendChild(thinkingIndicator);

    // Main response content
    const responseContent = document.createElement('div');
    responseContent.className = 'response-content';
    container.appendChild(responseContent);

    // CoT toggle (hidden by default, shown after generation if there's thinking)
    const cotToggle = document.createElement('div');
    cotToggle.className = 'cot-toggle';
    cotToggle.style.display = 'none';
    cotToggle.innerHTML = '<button class="cot-btn">Show reasoning</button>';
    container.appendChild(cotToggle);

    // CoT content (hidden by default)
    const cotContent = document.createElement('div');
    cotContent.className = 'cot-content';
    cotContent.style.display = 'none';
    container.appendChild(cotContent);

    // Toggle functionality
    cotToggle.querySelector('.cot-btn').addEventListener('click', () => {
        const isHidden = cotContent.style.display === 'none';
        cotContent.style.display = isHidden ? 'block' : 'none';
        cotToggle.querySelector('.cot-btn').textContent = isHidden ? 'Hide reasoning' : 'Show reasoning';
    });

    chatEl.appendChild(container);
    chatEl.scrollTop = chatEl.scrollHeight;

    return {
        container,
        thinkingIndicator,
        responseContent,
        cotToggle,
        cotContent
    };
}

// Softmax with temperature
function softmax(logits, temperature = 1.0) {
    const scaled = logits.map(x => x / temperature);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

// Top-p (nucleus) sampling
function sampleTopP(probs, topP) {
    const indexed = probs.map((p, i) => [p, i]);
    indexed.sort((a, b) => b[0] - a[0]);

    let cumSum = 0;
    let cutoff = indexed.length;
    for (let i = 0; i < indexed.length; i++) {
        cumSum += indexed[i][0];
        if (cumSum >= topP) {
            cutoff = i + 1;
            break;
        }
    }

    const topTokens = indexed.slice(0, cutoff);
    const totalProb = topTokens.reduce((sum, [p, _]) => sum + p, 0);

    let r = Math.random() * totalProb;
    for (const [prob, idx] of topTokens) {
        r -= prob;
        if (r <= 0) return idx;
    }
    return topTokens[0][1];
}

async function generate(prompt) {
    if (isGenerating || !session || !tokenizer) return;

    isGenerating = true;
    sendBtn.disabled = true;
    inputEl.disabled = true;

    const temperature = parseFloat(tempSlider.value);
    const topP = parseFloat(topPSlider.value);
    const maxTokens = parseInt(maxTSlider.value);

    // Add user message (strip format tokens for display)
    const displayText = prompt.replace('<|user|>', '').replace('<|assistant|>', '');
    addMessage(displayText, 'user');

    // Create structured assistant message
    const msgElements = createAssistantMessage();

    // Generation state
    let phase = 'initial'; // 'initial' -> 'thinking' -> 'answering'
    let thinkingText = '';
    let answerText = '';
    let responseType = null; // 'answer' or 'refuse'

    try {
        // Encode prompt
        let inputIds = tokenizer.encode(prompt);

        let totalTokens = 0;
        const startTime = performance.now();

        for (let i = 0; i < maxTokens; i++) {
            // Truncate if too long for context
            if (inputIds.length > modelConfig.context_length - 1) {
                inputIds = inputIds.slice(-modelConfig.context_length + 1);
            }

            // Create tensor from input ids
            const inputTensor = new ort.Tensor(
                'int64',
                BigInt64Array.from(inputIds.map(x => BigInt(x))),
                [1, inputIds.length]
            );

            // Run inference
            const outputs = await session.run({ input_ids: inputTensor });
            const logits = outputs.logits.data;

            // Get logits for last position
            const vocabSize = modelConfig.vocab_size;
            const seqLen = inputIds.length;
            const startIdx = (seqLen - 1) * vocabSize;
            const lastLogits = Array.from(logits.slice(startIdx, startIdx + vocabSize));

            // Dispose tensors to prevent memory leaks
            if (inputTensor.dispose) inputTensor.dispose();
            if (outputs.logits.dispose) outputs.logits.dispose();

            // Sample next token
            const probs = softmax(lastLogits, temperature);
            const nextToken = sampleTopP(probs, topP);

            // Check for end of text token
            if (nextToken === END_TOKEN) {
                break;
            }

            // Handle special tokens for phase transitions
            if (nextToken === THINK_TOKEN) {
                phase = 'thinking';
                inputIds.push(nextToken);
                totalTokens++;
                continue;
            } else if (nextToken === ANSWER_TOKEN) {
                phase = 'answering';
                responseType = 'answer';
                // Hide thinking indicator, prepare for answer
                msgElements.thinkingIndicator.style.display = 'none';
                inputIds.push(nextToken);
                totalTokens++;
                continue;
            } else if (nextToken === REFUSE_TOKEN) {
                phase = 'answering';
                responseType = 'refuse';
                // Hide thinking indicator, prepare for refusal
                msgElements.thinkingIndicator.style.display = 'none';
                inputIds.push(nextToken);
                totalTokens++;
                continue;
            }

            // Decode token
            const tokenText = tokenizer.decode([nextToken]);

            // Update appropriate text based on phase
            if (phase === 'thinking') {
                thinkingText += tokenText;
                // Keep showing thinking indicator
            } else if (phase === 'answering') {
                answerText += tokenText;
                msgElements.responseContent.textContent = answerText;
            } else {
                // Initial phase - might be raw output without thinking
                answerText += tokenText;
                msgElements.thinkingIndicator.style.display = 'none';
                msgElements.responseContent.textContent = answerText;
            }

            chatEl.scrollTop = chatEl.scrollHeight;

            // Append to input for next iteration
            inputIds.push(nextToken);
            totalTokens++;

            // Yield to UI to keep it responsive
            await new Promise(r => setTimeout(r, 0));
        }

        // Finalize display
        msgElements.thinkingIndicator.style.display = 'none';

        // Show CoT toggle if there was thinking
        if (thinkingText.trim() || responseType) {
            // Build CoT display with badge and classification
            msgElements.cotContent.innerHTML = '';

            // Add response type badge
            if (responseType) {
                const badge = document.createElement('span');
                badge.className = `response-badge ${responseType === 'refuse' ? 'refused' : 'answered'}`;
                badge.textContent = responseType === 'refuse' ? 'REFUSED' : 'ANSWERED';
                msgElements.cotContent.appendChild(badge);
            }

            // Add classification text
            if (thinkingText.trim()) {
                const classificationDiv = document.createElement('div');
                classificationDiv.style.marginTop = '0.5rem';
                classificationDiv.innerHTML = `<strong>Classification:</strong> ${thinkingText.trim()}`;
                msgElements.cotContent.appendChild(classificationDiv);
            }

            msgElements.cotToggle.style.display = 'block';
        }

        const endTime = performance.now();
        const elapsed = (endTime - startTime) / 1000;
        const tokensPerSec = totalTokens / elapsed;
        statsEl.textContent = `Generated ${totalTokens} tokens in ${elapsed.toFixed(2)}s (${tokensPerSec.toFixed(1)} tok/s)`;

    } catch (error) {
        msgElements.thinkingIndicator.style.display = 'none';
        msgElements.responseContent.textContent = `Error: ${error.message}`;
    }

    isGenerating = false;
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus({ preventScroll: true });
}

// Format user input for Q&A model
function formatPrompt(userText) {
    return `<|user|>${userText}<|assistant|>`;
}

// Event listeners
sendBtn.addEventListener('click', () => {
    const text = inputEl.value.trim();
    if (text) {
        inputEl.value = '';
        generate(formatPrompt(text));
    }
});

inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const text = inputEl.value.trim();
        if (text) {
            inputEl.value = '';
            generate(formatPrompt(text));
        }
    }
});

// Start initialization
init();
