// Ternary Neural Network - JavaScript Implementation

class TernaryLinearLayer {
    constructor(inFeatures, outFeatures, threshold = 0.3) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.threshold = threshold;

        // Initialize weights (full precision for training)
        this.weightsFp = Array(outFeatures).fill(0).map(() =>
            Array(inFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.2)
        );

        this.biasFp = Array(outFeatures).fill(0);

        // Ternary weights
        this.weightsTernary = null;
        this.biasTernary = null;

        // Cache for backprop
        this.cacheInput = null;
    }

    quantize() {
        this.weightsTernary = this.weightsFp.map(row =>
            row.map(w => ternaryQuantize(w, this.threshold))
        );
        this.biasTernary = this.biasFp.map(b => ternaryQuantize(b, this.threshold));
    }

    forward(input, training = true) {
        // Quantize weights
        this.quantize();

        const batchSize = input.length;
        const output = Array(batchSize).fill(0).map(() => Array(this.outFeatures).fill(0));

        // Matrix multiplication: output = input @ W^T + b
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < this.outFeatures; j++) {
                let sum = 0;
                for (let k = 0; k < this.inFeatures; k++) {
                    sum += input[i][k] * this.weightsTernary[j][k];
                }
                output[i][j] = sum + this.biasTernary[j];
            }
        }

        if (training) {
            this.cacheInput = input;
        }

        return output;
    }

    backward(gradOutput) {
        const batchSize = gradOutput.length;

        // Gradient w.r.t. input
        const gradInput = Array(batchSize).fill(0).map(() => Array(this.inFeatures).fill(0));

        for (let i = 0; i < batchSize; i++) {
            for (let k = 0; k < this.inFeatures; k++) {
                let sum = 0;
                for (let j = 0; j < this.outFeatures; j++) {
                    sum += gradOutput[i][j] * this.weightsTernary[j][k];
                }
                gradInput[i][k] = sum;
            }
        }

        // Gradient w.r.t. weights (straight-through estimator)
        const gradWeights = Array(this.outFeatures).fill(0).map(() => Array(this.inFeatures).fill(0));

        for (let j = 0; j < this.outFeatures; j++) {
            for (let k = 0; k < this.inFeatures; k++) {
                let sum = 0;
                for (let i = 0; i < batchSize; i++) {
                    sum += gradOutput[i][j] * this.cacheInput[i][k];
                }
                gradWeights[j][k] = sum / batchSize;
            }
        }

        // Gradient w.r.t. bias
        const gradBias = Array(this.outFeatures).fill(0);
        for (let j = 0; j < this.outFeatures; j++) {
            let sum = 0;
            for (let i = 0; i < batchSize; i++) {
                sum += gradOutput[i][j];
            }
            gradBias[j] = sum / batchSize;
        }

        return { gradInput, gradWeights, gradBias };
    }

    update(gradWeights, gradBias, learningRate) {
        // Update full-precision weights
        for (let i = 0; i < this.outFeatures; i++) {
            for (let j = 0; j < this.inFeatures; j++) {
                this.weightsFp[i][j] -= learningRate * gradWeights[i][j];
            }
            this.biasFp[i] -= learningRate * gradBias[i];
        }

        // Re-quantize
        this.quantize();
    }

    getSparsity() {
        let total = 0;
        let zeros = 0;

        for (let i = 0; i < this.outFeatures; i++) {
            for (let j = 0; j < this.inFeatures; j++) {
                total++;
                if (this.weightsTernary[i][j] === 0) zeros++;
            }
        }

        return zeros / total;
    }
}

class TernaryNeuralNetwork {
    constructor(layerSizes, learningRate = 0.01, threshold = 0.3) {
        this.layerSizes = layerSizes;
        this.learningRate = learningRate;
        this.threshold = threshold;
        this.layers = [];

        // Create layers
        for (let i = 0; i < layerSizes.length - 1; i++) {
            const layer = new TernaryLinearLayer(
                layerSizes[i],
                layerSizes[i + 1],
                threshold
            );
            this.layers.push(layer);
        }

        this.trainingHistory = {
            loss: [],
            accuracy: []
        };
    }

    relu(x) {
        return x.map(row => row.map(val => Math.max(0, val)));
    }

    reluGradient(x) {
        return x.map(row => row.map(val => val > 0 ? 1 : 0));
    }

    forward(input, training = true) {
        let x = input;
        const activations = [x];

        // Forward through all layers except last
        for (let i = 0; i < this.layers.length - 1; i++) {
            x = this.layers[i].forward(x, training);
            x = this.relu(x);
            activations.push(x);
        }

        // Last layer (no activation)
        x = this.layers[this.layers.length - 1].forward(x, training);
        activations.push(x);

        this.activations = activations;
        return x;
    }

    softmax(logits) {
        return logits.map(row => {
            const maxVal = Math.max(...row);
            const exps = row.map(x => Math.exp(x - maxVal));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map(x => x / sum);
        });
    }

    crossEntropyLoss(logits, labels) {
        const probs = this.softmax(logits);
        const batchSize = logits.length;
        let loss = 0;

        const gradLogits = probs.map(row => [...row]);

        for (let i = 0; i < batchSize; i++) {
            const trueLabel = labels[i];
            loss -= Math.log(probs[i][trueLabel] + 1e-8);
            gradLogits[i][trueLabel] -= 1;
        }

        loss /= batchSize;

        // Normalize gradient
        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < gradLogits[i].length; j++) {
                gradLogits[i][j] /= batchSize;
            }
        }

        return { loss, gradLogits };
    }

    backward(gradOutput) {
        let grad = gradOutput;
        const gradients = [];

        // Backward through last layer
        let result = this.layers[this.layers.length - 1].backward(grad);
        gradients.unshift({
            weights: result.gradWeights,
            bias: result.gradBias
        });
        grad = result.gradInput;

        // Backward through remaining layers
        for (let i = this.layers.length - 2; i >= 0; i--) {
            // Backward through ReLU
            const reluGrad = this.reluGradient(this.activations[i + 1]);
            grad = grad.map((row, idx) =>
                row.map((val, jdx) => val * reluGrad[idx][jdx])
            );

            // Backward through layer
            result = this.layers[i].backward(grad);
            gradients.unshift({
                weights: result.gradWeights,
                bias: result.gradBias
            });
            grad = result.gradInput;
        }

        return gradients;
    }

    update(gradients) {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].update(
                gradients[i].weights,
                gradients[i].bias,
                this.learningRate
            );
        }
    }

    trainStep(xBatch, yBatch) {
        // Forward pass
        const logits = this.forward(xBatch, true);

        // Compute loss and gradients
        const { loss, gradLogits } = this.crossEntropyLoss(logits, yBatch);

        // Backward pass
        const gradients = this.backward(gradLogits);

        // Update weights
        this.update(gradients);

        // Compute accuracy
        const predictions = logits.map(row =>
            row.indexOf(Math.max(...row))
        );
        const accuracy = predictions.filter((pred, idx) =>
            pred === yBatch[idx]
        ).length / yBatch.length;

        return { loss, accuracy };
    }

    predict(x) {
        const logits = this.forward(x, false);
        return logits.map(row => row.indexOf(Math.max(...row)));
    }

    getSparsity() {
        let totalSparsity = 0;
        for (const layer of this.layers) {
            totalSparsity += layer.getSparsity();
        }
        return totalSparsity / this.layers.length;
    }

    getTotalParameters() {
        let total = 0;
        for (const layer of this.layers) {
            total += layer.inFeatures * layer.outFeatures + layer.outFeatures;
        }
        return total;
    }

    getModelSize() {
        const params = this.getTotalParameters();
        const float32Size = params * 4; // 4 bytes per float32
        const ternarySize = params * 0.27 / 8; // ~0.27 bits per ternary weight

        return {
            params,
            float32Bytes: float32Size,
            ternaryBytes: Math.ceil(ternarySize),
            compression: float32Size / ternarySize
        };
    }
}

// Helper function to create mini-batches
function createMiniBatches(X, y, batchSize) {
    const batches = [];
    const numSamples = X.length;

    // Shuffle indices
    const indices = Array.from({ length: numSamples }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Create batches
    for (let i = 0; i < numSamples; i += batchSize) {
        const batchIndices = indices.slice(i, Math.min(i + batchSize, numSamples));
        const xBatch = batchIndices.map(idx => X[idx]);
        const yBatch = batchIndices.map(idx => y[idx]);
        batches.push({ x: xBatch, y: yBatch });
    }

    return batches;
}
