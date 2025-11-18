// Main Application Logic

// Global state
let currentNetwork = null;
let currentDataset = null;
let isTraining = false;
let trainingChart = null;

// Initialize on load
document.addEventListener('DOMContentLoaded', function() {
    initializeTabs();
    generateTruthTables();
});

// Tab Management
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            // Update active states
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// Logic Expression Evaluator
function evaluateLogic() {
    const expr = document.getElementById('logic-expr').value;
    const a = parseInt(document.getElementById('var-a').value);
    const b = parseInt(document.getElementById('var-b').value);
    const c = parseInt(document.getElementById('var-c').value);

    try {
        // Parse and evaluate expression
        const result = evalTernaryExpression(expr, { a, b, c });

        const resultDiv = document.getElementById('logic-result');
        resultDiv.innerHTML = `
            <strong>Result:</strong> ${formatTrit(result)}<br>
            <strong>Expression:</strong> ${expr}<br>
            <strong>Variables:</strong> a=${formatTrit(a)}, b=${formatTrit(b)}, c=${formatTrit(c)}
        `;
    } catch (error) {
        document.getElementById('logic-result').innerHTML =
            `<span style="color: red;">Error: ${error.message}</span>`;
    }
}

function evalTernaryExpression(expr, vars) {
    // Replace variable names with values
    let processed = expr.toUpperCase();

    for (const [name, value] of Object.entries(vars)) {
        const regex = new RegExp(name.toUpperCase(), 'g');
        processed = processed.replace(regex, value.toString());
    }

    // Evaluate operations
    processed = processed.replace(/NOT\s+(-?\d+)/g, (_, val) => {
        return (-parseInt(val)).toString();
    });

    processed = processed.replace(/(-?\d+)\s+AND\s+(-?\d+)/g, (_, a, b) => {
        return Math.min(parseInt(a), parseInt(b)).toString();
    });

    processed = processed.replace(/(-?\d+)\s+OR\s+(-?\d+)/g, (_, a, b) => {
        return Math.max(parseInt(a), parseInt(b)).toString();
    });

    processed = processed.replace(/(-?\d+)\s+XOR\s+(-?\d+)/g, (_, a, b) => {
        const sum = parseInt(a) + parseInt(b);
        if (sum > 1) return '-1';
        if (sum < -1) return '1';
        return sum.toString();
    });

    // Handle parentheses (simple version)
    while (processed.includes('(')) {
        processed = processed.replace(/\((-?\d+)\)/g, '$1');
    }

    return parseInt(processed);
}

function formatTrit(value) {
    const colors = { '-1': 'red', '0': 'gray', '1': 'green' };
    const symbols = { '-1': '-', '0': '0', '1': '+' };
    return `<span style="color: ${colors[value]}; font-weight: bold;">
        ${symbols[value]} (${value})
    </span>`;
}

// Generate Truth Tables
function generateTruthTables() {
    const values = [-1, 0, 1];

    // AND table
    let andHTML = '<tr><th>A</th><th>B</th><th>A AND B</th></tr>';
    for (const a of values) {
        for (const b of values) {
            const result = Math.min(a, b);
            andHTML += `<tr>
                <td class="trit-${a === -1 ? 'minus' : a === 0 ? 'zero' : 'plus'}">${a}</td>
                <td class="trit-${b === -1 ? 'minus' : b === 0 ? 'zero' : 'plus'}">${b}</td>
                <td class="trit-${result === -1 ? 'minus' : result === 0 ? 'zero' : 'plus'}">${result}</td>
            </tr>`;
        }
    }
    document.getElementById('and-table').innerHTML = andHTML;

    // OR table
    let orHTML = '<tr><th>A</th><th>B</th><th>A OR B</th></tr>';
    for (const a of values) {
        for (const b of values) {
            const result = Math.max(a, b);
            orHTML += `<tr>
                <td class="trit-${a === -1 ? 'minus' : a === 0 ? 'zero' : 'plus'}">${a}</td>
                <td class="trit-${b === -1 ? 'minus' : b === 0 ? 'zero' : 'plus'}">${b}</td>
                <td class="trit-${result === -1 ? 'minus' : result === 0 ? 'zero' : 'plus'}">${result}</td>
            </tr>`;
        }
    }
    document.getElementById('or-table').innerHTML = orHTML;

    // NOT table
    let notHTML = '<tr><th>A</th><th>NOT A</th></tr>';
    for (const a of values) {
        const result = -a;
        notHTML += `<tr>
            <td class="trit-${a === -1 ? 'minus' : a === 0 ? 'zero' : 'plus'}">${a}</td>
            <td class="trit-${result === -1 ? 'minus' : result === 0 ? 'zero' : 'plus'}">${result}</td>
        </tr>`;
    }
    document.getElementById('not-table').innerHTML = notHTML;
}

// Neural Network Builder
function addLayer() {
    const container = document.getElementById('hidden-layers');
    const div = document.createElement('div');
    div.className = 'layer-input';
    div.innerHTML = `
        <input type="number" class="layer-neurons" value="64" min="1" placeholder="Neurons" />
        <button onclick="removeLayer(this)">Remove</button>
    `;
    container.appendChild(div);
}

function removeLayer(btn) {
    btn.parentElement.remove();
}

function buildNetwork() {
    const inputSize = parseInt(document.getElementById('input-size').value);
    const outputSize = parseInt(document.getElementById('output-size').value);
    const learningRate = parseFloat(document.getElementById('learning-rate').value);
    const threshold = parseFloat(document.getElementById('threshold').value);

    // Get hidden layers
    const hiddenLayers = Array.from(document.querySelectorAll('.layer-neurons'))
        .map(input => parseInt(input.value));

    // Build architecture
    const architecture = [inputSize, ...hiddenLayers, outputSize];

    // Create network
    currentNetwork = new TernaryNeuralNetwork(architecture, learningRate, threshold);

    // Display info
    const modelSize = currentNetwork.getModelSize();
    document.getElementById('network-info').innerHTML = `
        <div class="result">
            <strong>Network Created!</strong><br>
            Architecture: ${architecture.join(' → ')}<br>
            Total Parameters: ${modelSize.params.toLocaleString()}<br>
            Memory (Float32): ${(modelSize.float32Bytes / 1024).toFixed(2)} KB<br>
            Memory (Ternary): ${(modelSize.ternaryBytes / 1024).toFixed(2)} KB<br>
            Compression: ${modelSize.compression.toFixed(1)}x
        </div>
    `;
}

// Dataset Loading
function loadMNIST() {
    // Generate synthetic MNIST-like data
    const numSamples = 1000;
    const inputSize = 784;
    const numClasses = 10;

    const X = [];
    const y = [];

    // Generate class prototypes
    const prototypes = [];
    for (let c = 0; c < numClasses; c++) {
        const proto = Array(inputSize).fill(0).map(() => Math.random() - 0.5);
        prototypes.push(proto);
    }

    // Generate samples
    for (let i = 0; i < numSamples; i++) {
        const classIdx = i % numClasses;
        const sample = prototypes[classIdx].map(v => v + (Math.random() - 0.5) * 0.3);
        X.push(sample);
        y.push(classIdx);
    }

    currentDataset = { X, y, inputSize, outputSize: numClasses };

    document.getElementById('dataset-info').innerHTML = `
        <div class="result">
            <strong>MNIST Sample Loaded!</strong><br>
            Samples: ${numSamples}<br>
            Input size: ${inputSize}<br>
            Classes: ${numClasses}<br>
            Shape: (${numSamples}, ${inputSize})
        </div>
    `;

    // Update network inputs if not built yet
    document.getElementById('input-size').value = inputSize;
    document.getElementById('output-size').value = numClasses;
}

function loadCSV(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        const text = e.target.result;
        parseCSV(text);
    };

    reader.readAsText(file);
}

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const headers = lines[0].split(',');

    // Show configuration UI
    const configDiv = document.getElementById('dataset-config');
    configDiv.style.display = 'block';

    // Populate target column dropdown
    const targetSelect = document.getElementById('target-column');
    targetSelect.innerHTML = headers.map(h =>
        `<option value="${h}">${h}</option>`
    ).join('');

    // Parse data
    const data = lines.slice(1).map(line => {
        const values = line.split(',');
        const row = {};
        headers.forEach((h, i) => {
            row[h] = isNaN(values[i]) ? values[i] : parseFloat(values[i]);
        });
        return row;
    });

    // Store for later
    window.csvData = { headers, data };

    document.getElementById('dataset-info').innerHTML = `
        <div class="result">
            <strong>CSV Loaded!</strong><br>
            Rows: ${data.length}<br>
            Columns: ${headers.length}<br>
            Headers: ${headers.join(', ')}
        </div>
    `;
}

function loadImages(event) {
    alert('Image loading coming soon! For now, use MNIST sample or CSV data.');
}

// Training
function trainNetwork() {
    if (!currentNetwork) {
        alert('Please build a network first!');
        return;
    }

    if (!currentDataset) {
        alert('Please load a dataset first!');
        return;
    }

    const epochs = parseInt(document.getElementById('epochs').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);

    document.getElementById('train-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';

    isTraining = true;
    currentNetwork.trainingHistory = { loss: [], accuracy: [] };

    trainLoop(epochs, batchSize, 0);
}

function trainLoop(epochs, batchSize, currentEpoch) {
    if (!isTraining || currentEpoch >= epochs) {
        // Training complete
        document.getElementById('train-btn').style.display = 'inline-block';
        document.getElementById('stop-btn').style.display = 'none';
        isTraining = false;

        // Update results
        updateResults();
        return;
    }

    // Create mini-batches
    const batches = createMiniBatches(currentDataset.X, currentDataset.y, batchSize);

    let epochLoss = 0;
    let epochAcc = 0;

    for (const batch of batches) {
        const { loss, accuracy } = currentNetwork.trainStep(batch.x, batch.y);
        epochLoss += loss;
        epochAcc += accuracy;
    }

    epochLoss /= batches.length;
    epochAcc /= batches.length;

    // Update history
    currentNetwork.trainingHistory.loss.push(epochLoss);
    currentNetwork.trainingHistory.accuracy.push(epochAcc);

    // Update display
    const logDiv = document.getElementById('training-log');
    logDiv.innerHTML = `
        <div class="log-entry">
            Epoch ${currentEpoch + 1}/${epochs}: Loss=${epochLoss.toFixed(4)}, Accuracy=${(epochAcc * 100).toFixed(2)}%
        </div>
    ` + logDiv.innerHTML;

    // Update chart
    updateChart();

    // Continue training
    setTimeout(() => trainLoop(epochs, batchSize, currentEpoch + 1), 10);
}

function stopTraining() {
    isTraining = false;
}

function updateChart() {
    const canvas = document.getElementById('loss-chart');
    const ctx = canvas.getContext('2d');
    const history = currentNetwork.trainingHistory;

    const width = canvas.width;
    const height = canvas.height;
    const padding = 40;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (history.loss.length === 0) return;

    // Find max values
    const maxLoss = Math.max(...history.loss);
    const maxAcc = 1.0;

    // Draw grid
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
        const y = padding + (height - 2 * padding) * i / 5;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(width - padding, y);
        ctx.stroke();
    }

    // Draw loss curve
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < history.loss.length; i++) {
        const x = padding + (width - 2 * padding) * i / (history.loss.length - 1 || 1);
        const y = height - padding - (height - 2 * padding) * history.loss[i] / maxLoss;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw accuracy curve
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 2;
    ctx.beginPath();

    for (let i = 0; i < history.accuracy.length; i++) {
        const x = padding + (width - 2 * padding) * i / (history.accuracy.length - 1 || 1);
        const y = height - padding - (height - 2 * padding) * history.accuracy[i] / maxAcc;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '12px sans-serif';
    ctx.fillText('Loss (red) & Accuracy (green)', padding, 20);
    ctx.fillText('Epoch', width / 2, height - 10);
}

function updateResults() {
    const sparsity = currentNetwork.getSparsity();
    const modelSize = currentNetwork.getModelSize();

    // Get final accuracy
    const finalAcc = currentNetwork.trainingHistory.accuracy.slice(-1)[0] || 0;

    document.getElementById('model-size').textContent =
        `${(modelSize.ternaryBytes / 1024).toFixed(2)} KB`;
    document.getElementById('sparsity').textContent =
        `${(sparsity * 100).toFixed(1)}%`;
    document.getElementById('accuracy').textContent =
        `${(finalAcc * 100).toFixed(2)}%`;
    document.getElementById('compression').textContent =
        `${modelSize.compression.toFixed(1)}x`;
}

// Encoding
function encodeValue() {
    const value = parseInt(document.getElementById('encode-value').value);
    const tryte = new Tryte(value);

    // Compact encoding
    const compact = CompactEncoder.encode(tryte);
    document.getElementById('compact-result').innerHTML = `
        <strong>Balanced Ternary:</strong> ${tryte.toBalancedTernary()}<br>
        <strong>Bits:</strong> ${compact.bits}<br>
        <strong>Bytes:</strong> ${compact.bytes}<br>
        <strong>Efficiency:</strong> ${compact.efficiency}
    `;

    // Perfect encoding
    const perfect = PerfectEncoder.encode(tryte);
    document.getElementById('perfect-result').innerHTML = `
        <strong>Balanced Ternary:</strong> ${tryte.toBalancedTernary()}<br>
        <strong>Bits:</strong> ${perfect.bits}<br>
        <strong>Bytes:</strong> ${perfect.bytes}<br>
        <strong>Efficiency:</strong> ${perfect.efficiency}
    `;

    // Optimal block encoding
    const optimal = OptimalBlockEncoder.encode(tryte);
    document.getElementById('optimal-result').innerHTML = `
        <strong>Balanced Ternary:</strong> ${tryte.toBalancedTernary()}<br>
        <strong>Bits:</strong> ${optimal.bits}<br>
        <strong>Bytes:</strong> ${optimal.bytes}<br>
        <strong>Efficiency:</strong> ${optimal.efficiency}
    `;
}
