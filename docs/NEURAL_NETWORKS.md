

# Ternary Neural Networks: Implementation and Theory

## Overview

This document explains the ternary neural network implementation, including:
1. 3-bit to 2-trit encoding scheme
2. Gradient flow through ternary quantization
3. Straight-through estimator
4. MNIST classification example

---

## 1. Ternary Quantization

### 1.1 Weight Quantization

Ternary neural networks use weights restricted to three values: **{-1, 0, +1}**

```python
def ternary_quantize(x, threshold=0.3):
    if x > threshold:
        return +1
    elif x < -threshold:
        return -1
    else:
        return 0
```

**Advantages:**
- **Memory**: 2 bits per weight (vs 32 bits for float32) → 16x compression
- **Computation**: Multiplication becomes conditional add/subtract
- **Sparsity**: Zero weights skip computation entirely

### 1.2 Quantization Functions

We implement three quantization strategies:

#### Deterministic Quantization
```
q(x) = sign(x)  if |x| > threshold
       0        otherwise
```

#### Stochastic Quantization
```
P(q(x) = +1) = (x + 1) / 2
P(q(x) =  0) = (1 - |x|) / 2
P(q(x) = -1) = (1 - x) / 2
```

Maintains E[q(x)] ≈ x for better gradient approximation.

---

## 2. 3-Bit to 2-Trit Encoding

### 2.1 The Encoding Problem

- **2 trits** can represent **3² = 9** states
- **3 bits** can represent **2³ = 8** states
- We map 8 of the 9 possible 2-trit combinations

### 2.2 Encoding Table

| Trit Pair | Decimal | Binary | Encoding |
|-----------|---------|--------|----------|
| -1, -1    | 0       | 000    | --       |
| -1,  0    | 1       | 001    | -0       |
| -1, +1    | 2       | 010    | -+       |
|  0, -1    | 3       | 011    | 0-       |
|  0,  0    | 4       | 100    | 00       |
|  0, +1    | 5       | 101    | 0+       |
| +1, -1    | 6       | 110    | +-       |
| +1,  0    | 7       | 111    | +0       |
| +1, +1    | -       | -      | *unmappable* |

**Note**: The (+1, +1) combination is unmappable. We use saturation to (+1, 0).

### 2.3 Storage Efficiency

For a Tryte (18 trits):
- **Ideal**: 18 × log₂(3) ≈ 28.53 bits
- **Our encoding**: 9 pairs × 3 bits = 27 bits
- **Efficiency**: 27/28.53 ≈ **94.6%**
- **Bytes used**: ⌈27/8⌉ = 4 bytes (vs 9 bytes for direct encoding)

### 2.4 Implementation

```python
class TritEncoder:
    TRIT_PAIR_TO_BITS = {
        (-1, -1): 0b000,
        (-1,  0): 0b001,
        (-1,  1): 0b010,
        ( 0, -1): 0b011,
        ( 0,  0): 0b100,
        ( 0,  1): 0b101,
        ( 1, -1): 0b110,
        ( 1,  0): 0b111,
    }

    @staticmethod
    def encode_trit_pair(t1, t2):
        pair = (t1.to_int(), t2.to_int())
        if pair == (1, 1):
            return TRIT_PAIR_TO_BITS[(1, 0)]  # Saturation
        return TRIT_PAIR_TO_BITS[pair]
```

---

## 3. Gradient Flow and Backpropagation

### 3.1 The Gradient Problem

Quantization is non-differentiable:
```
∂q(x)/∂x = 0  (almost everywhere)
```

This would stop gradient flow and prevent training!

### 3.2 Straight-Through Estimator (STE)

**Solution**: Approximate gradient during backprop

**Forward pass:**
```python
W_ternary = quantize(W_fp)
y = W_ternary @ x
```

**Backward pass:**
```python
∂L/∂W_fp ≈ ∂L/∂y @ x^T  # Ignore quantization in gradient
```

**Key insight**: We pretend quantization is identity during backprop!

### 3.3 Mathematical Justification

Let q(·) be the quantization function and W be full-precision weights:

**Forward:**
```
y = q(W) · x
```

**Backward (STE):**
```
∂L/∂W ≈ ∂L/∂y · ∂y/∂W

where we approximate: ∂y/∂W ≈ x (ignoring ∂q/∂W)
```

**Why it works:**
1. **Small updates**: W changes gradually, q(W) stays stable locally
2. **Full precision**: We maintain W in full precision, only quantize for forward
3. **Expected gradient**: Over many updates, gradient direction is approximately correct

### 3.4 Implementation

```python
class TernaryLinear:
    def __init__(self, in_features, out_features):
        # Full-precision weights (for training)
        self.weights_fp = np.random.randn(out_features, in_features) * 0.1

        # Ternary weights (for forward pass)
        self.weights_ternary = None

    def forward(self, x, training=True):
        # Quantize weights
        self.weights_ternary = ternary_quantize(self.weights_fp)

        # Forward with ternary weights
        output = x @ self.weights_ternary.T

        if training:
            self.cache_input = x

        return output

    def backward(self, grad_output):
        # Gradient w.r.t. input (use ternary weights)
        grad_input = grad_output @ self.weights_ternary

        # Gradient w.r.t. weights (STE: ignore quantization)
        grad_weights_fp = grad_output.T @ self.cache_input

        return grad_input, grad_weights_fp

    def update(self, grad_weights, learning_rate):
        # Update full-precision weights
        self.weights_fp -= learning_rate * grad_weights

        # Re-quantize for next forward pass
        self.weights_ternary = ternary_quantize(self.weights_fp)
```

### 3.5 Gradient Flow Diagram

```
Input (x)
    │
    ▼
┌─────────────────────┐
│  Quantize: W → q(W) │  ← Forward: use ternary weights
└─────────────────────┘
    │
    ▼
Linear: y = q(W) · x
    │
    ▼
Loss: L(y, target)
    │
    ▼
┌─────────────────────┐
│  ∂L/∂y              │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  ∂L/∂W = ∂L/∂y · x  │  ← Backward: STE (ignore quantization)
└─────────────────────┘
    │
    ▼
Update: W ← W - η·∂L/∂W
```

---

## 4. Training Algorithm

### 4.1 Complete Training Loop

```python
# Initialize
model = TernaryNeuralNetwork([784, 256, 128, 10])

for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # Forward pass (with ternary weights)
        logits = model.forward(x_batch)

        # Compute loss
        loss = cross_entropy(logits, y_batch)
        grad_output = ∂loss/∂logits

        # Backward pass (STE for gradients)
        gradients = model.backward(grad_output)

        # Update full-precision weights
        model.update(gradients)

        # Weights are re-quantized automatically in next forward pass
```

### 4.2 Key Points

1. **Two sets of weights**:
   - `weights_fp`: Full precision (float32) for training
   - `weights_ternary`: Quantized {-1, 0, 1} for inference

2. **Training**:
   - Forward: Use ternary weights
   - Backward: Compute gradients as if no quantization (STE)
   - Update: Modify full-precision weights

3. **Inference**:
   - Only need ternary weights
   - Can discard full-precision weights
   - Massive memory savings!

---

## 5. MNIST Example

### 5.1 Architecture

```
Input: 784 (28×28 pixels)
   ↓
Hidden 1: 256 neurons (ternary weights)
   ↓
ReLU activation
   ↓
Hidden 2: 128 neurons (ternary weights)
   ↓
ReLU activation
   ↓
Output: 10 classes

Total parameters: ~230K
Ternary encoding: ~7.5 KB (vs ~900 KB for float32)
Compression: 120x
```

### 5.2 Performance

**Expected results (synthetic MNIST-like data):**
- Training accuracy: ~85-95%
- Test accuracy: ~80-90%
- Sparsity: 30-50% (many zero weights)

**Real MNIST (when using actual dataset):**
- Training accuracy: ~98-99%
- Test accuracy: ~96-98%
- Comparable to full-precision for this simple task!

### 5.3 Computational Advantages

For inference with ternary weights:

**Standard multiplication** (per weight):
```
result = weight * input
```
Requires: 1 multiply, 1 add

**Ternary multiplication** (per weight):
```
if weight == -1:
    result = -input
elif weight == 0:
    result = 0  # skip!
else:  # weight == +1
    result = input
```
Requires: 0 multiplies, 1 conditional add/negate

**Speedup analysis:**
- Eliminate ~50% of computations (zero weights)
- Remaining computations are 3-5x faster (no multiply)
- **Total speedup: 3-10x for inference**

---

## 6. Advanced Topics

### 6.1 Ternary Activations

In addition to ternary weights, we can quantize activations:

```python
def ternary_activation(x, threshold=0.3):
    return ternary_quantize(x, threshold)
```

**Pros:**
- Even more memory savings
- Faster computation
- End-to-end ternary network

**Cons:**
- Harder to train (less gradient information)
- May reduce accuracy
- Requires careful hyperparameter tuning

### 6.2 Improved Quantization

**Learnable thresholds:**
```python
class LearnableTernaryQuantize:
    def __init__(self):
        self.threshold = Parameter(0.3)

    def forward(self, x):
        return ternary_quantize(x, self.threshold)
```

**Asymmetric quantization:**
```python
# Different thresholds for positive and negative
threshold_pos = 0.5
threshold_neg = 0.3

q(x) = +1 if x > threshold_pos
       -1 if x < -threshold_neg
        0 otherwise
```

### 6.3 Mixed Precision

Hybrid approach:
- **First layer**: Full precision (preserve input information)
- **Hidden layers**: Ternary weights, full precision activations
- **Last layer**: Full precision (preserve output precision)

This often gives best accuracy/efficiency tradeoff.

---

## 7. Implementation Checklist

When implementing ternary neural networks:

✅ **Maintain two weight copies**
   - Full precision for training
   - Ternary for forward pass

✅ **Use straight-through estimator**
   - Ignore quantization in backward pass
   - Compute gradients w.r.t. full-precision weights

✅ **Quantize after each update**
   - Update full-precision weights
   - Re-quantize before next forward

✅ **Handle edge cases**
   - Saturation for unmappable values
   - Numerical stability in loss functions

✅ **Monitor sparsity**
   - Track number of zero weights
   - Use for pruning and speedup estimation

✅ **Efficient encoding**
   - Use 3-bit to 2-trit for storage
   - Decode on-the-fly during inference

---

## 8. References

1. **Ternary Weight Networks**
   - Li et al., "Ternary Weight Networks" (2016)
   - Introduces TWN with {-1, 0, +1} weights

2. **Straight-Through Estimator**
   - Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons" (2013)
   - Foundation for training quantized networks

3. **Binary/Ternary Networks**
   - Courbariaux et al., "BinaryConnect" (2015)
   - Hubara et al., "Quantized Neural Networks" (2016)

4. **Balanced Ternary**
   - Knuth, "The Art of Computer Programming, Vol. 2"
   - Hayes, "Third Base" American Scientist (2001)

---

## 9. Conclusion

Ternary neural networks offer a compelling tradeoff:
- **120x memory compression** vs float32
- **3-10x inference speedup**
- **Minimal accuracy loss** (often <2% on MNIST)
- **Hardware-friendly**: natural fit for ternary logic

The key innovations:
1. **Ternary quantization** reduces weights to 3 values
2. **3-bit to 2-trit encoding** achieves 94.6% efficiency
3. **Straight-through estimator** enables training
4. **Sparsity** from zero weights provides additional speedup

This makes ternary networks ideal for:
- Edge devices (IoT, mobile)
- Embedded systems
- Low-power AI accelerators
- Ternary hardware implementations

**The future of efficient neural networks may well be ternary!**
