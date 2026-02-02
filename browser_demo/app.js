/**
 * Browser LM Demo - Pure browser inference with ONNX Runtime Web
 */

// Global state
let session = null;
let tokenizer = null;
let modelConfig = null;
let isGenerating = false;

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
        console.log('Tokenizer loaded');

        // Load model config
        const configResponse = await fetch('model_config.json');
        modelConfig = await configResponse.json();
        console.log('Model config:', modelConfig);

        updateStatus('Loading model (270MB, please wait)...', 'loading');

        // Configure ONNX Runtime for best performance
        ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
        ort.env.wasm.simd = true;

        // Load ONNX model
        const startLoad = performance.now();
        session = await ort.InferenceSession.create('model.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all',
        });
        const loadTime = ((performance.now() - startLoad) / 1000).toFixed(1);

        updateStatus(`Model loaded in ${loadTime}s. Ready to generate!`, 'ready');
        inputEl.disabled = false;
        sendBtn.disabled = false;
        inputEl.focus();

        console.log('ONNX session created');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);

    } catch (error) {
        console.error('Initialization error:', error);
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

    // Add user message
    addMessage(prompt, 'user');

    // Create assistant message element for streaming
    const assistantDiv = addMessage('', 'assistant');
    assistantDiv.classList.add('generating');

    try {
        // Encode prompt
        let inputIds = tokenizer.encode(prompt);
        console.log('Encoded prompt:', inputIds.slice(0, 10), '...');

        let generatedText = '';
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

            // Sample next token
            const probs = softmax(lastLogits, temperature);
            const nextToken = sampleTopP(probs, topP);

            // Check for end of text token (256 = <|endoftext|>)
            if (nextToken === 256) {
                console.log('End of text token reached');
                break;
            }

            // Decode and display
            const tokenText = tokenizer.decode([nextToken]);
            generatedText += tokenText;
            assistantDiv.textContent = generatedText;
            chatEl.scrollTop = chatEl.scrollHeight;

            // Append to input for next iteration
            inputIds.push(nextToken);
            totalTokens++;

            // Yield to UI to keep it responsive
            await new Promise(r => setTimeout(r, 0));
        }

        const endTime = performance.now();
        const elapsed = (endTime - startTime) / 1000;
        const tokensPerSec = totalTokens / elapsed;
        statsEl.textContent = `Generated ${totalTokens} tokens in ${elapsed.toFixed(2)}s (${tokensPerSec.toFixed(1)} tok/s)`;

    } catch (error) {
        console.error('Generation error:', error);
        assistantDiv.textContent = `Error: ${error.message}`;
    }

    assistantDiv.classList.remove('generating');
    isGenerating = false;
    sendBtn.disabled = false;
    inputEl.disabled = false;
    inputEl.focus();
}

// Event listeners
sendBtn.addEventListener('click', () => {
    const text = inputEl.value.trim();
    if (text) {
        inputEl.value = '';
        generate(text);
    }
});

inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        const text = inputEl.value.trim();
        if (text) {
            inputEl.value = '';
            generate(text);
        }
    }
});

// Start initialization
init();
