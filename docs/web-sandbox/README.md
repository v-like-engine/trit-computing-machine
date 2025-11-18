# Ternary Computing Interactive Sandbox

An interactive web-based environment for exploring balanced ternary computing, ternary neural networks, and encoding methods.

## 🌐 Live Demo

**Visit:** [https://v-like-engine.github.io/trit-computing-machine/web-sandbox/](https://v-like-engine.github.io/trit-computing-machine/web-sandbox/)

(Once deployed - see deployment instructions below)

## ✨ Features

### 1. Logic Expression Evaluator
- Evaluate ternary logic expressions
- Interactive truth tables for AND, OR, NOT
- Real-time variable updates
- Support for complex expressions with parentheses

### 2. Neural Network Builder
- Build custom ternary neural network architectures
- Configure layers and neurons interactively
- Train on datasets with real-time visualization
- View sparsity, compression, and accuracy metrics

### 3. Dataset Support
- **MNIST Sample:** Built-in synthetic MNIST-like data
- **CSV Upload:** Upload your own CSV datasets
- **Image Upload:** Support for image datasets (coming soon)
- Configure feature and target columns

### 4. Encoding Comparison
- **Compact Encoding:** 3 bits → 2 trits (94.6% efficient, minimal loss)
- **Perfect Encoding:** Radix conversion (98.4% efficient, lossless)
- **Optimal Block:** 5 trits → 8 bits (94.9% efficient, lossless)
- Side-by-side comparison of all methods

### 5. Training Visualization
- Real-time loss and accuracy charts
- Training logs
- Model statistics
- Performance metrics

## 🚀 Quick Start

### Online (No Installation)

Just visit the live demo URL above!

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/v-like-engine/trit-computing-machine.git
cd trit-computing-machine/docs/web-sandbox
```

2. **Start a local server:**

Using Python:
```bash
python3 -m http.server 8000
```

Or using Node.js:
```bash
npx http-server -p 8000
```

3. **Open in browser:**
```
http://localhost:8000
```

## 📖 Usage Examples

### Example 1: Evaluate Ternary Logic

1. Go to **Logic Expressions** tab
2. Enter expression: `(a AND b) OR (NOT c)`
3. Set variables: `a=1`, `b=0`, `c=-1`
4. Click **Evaluate**
5. See result and explanation

### Example 2: Build and Train Neural Network

1. Go to **Neural Network** tab
2. Click **Load MNIST Sample**
3. Click **Build Network** (uses default 784→128→10)
4. Set **Epochs: 10**, **Batch Size: 32**
5. Click **Train**
6. Watch real-time training progress
7. View final accuracy and model statistics

### Example 3: Compare Encodings

1. Go to **Encoding** tab
2. Enter a number (e.g., `12345`)
3. Click **Encode**
4. Compare three encoding methods:
   - Compact (fastest, slight information loss)
   - Perfect (lossless, good efficiency)
   - Optimal Block (lossless, block-based)

## 🎯 Use Cases

### Education
- Learn balanced ternary concepts interactively
- Understand ternary logic operations
- Visualize neural network training
- Compare encoding efficiency

### Research
- Prototype ternary neural network architectures
- Test encoding strategies
- Experiment with different thresholds
- Validate ternary computing concepts

### Development
- Quick testing of ternary algorithms
- Dataset preprocessing
- Model architecture exploration
- Performance benchmarking

## 🛠️ Technical Details

### Architecture

```
docs/web-sandbox/
├── index.html          # Main HTML structure
├── styles.css          # CSS styling
├── ternary-core.js     # Ternary logic implementation
├── ternary-nn.js       # Neural network implementation
└── app.js              # UI logic and interactions
```

### Technologies

- **Pure JavaScript:** No frameworks required
- **HTML5 Canvas:** For training visualization
- **CSS3:** Modern responsive design
- **Client-side only:** No server needed

### Performance

- Runs entirely in browser
- Typical limits:
  - Dataset: ~10,000 samples
  - Network: up to 1M parameters
  - Training: depends on device speed

## 📊 Features in Detail

### Ternary Logic Operations

| Operation | Formula | Example |
|-----------|---------|---------|
| AND | min(a, b) | 1 AND 0 = 0 |
| OR | max(a, b) | 1 OR 0 = 1 |
| NOT | -a | NOT 1 = -1 |
| XOR | Balanced sum | 1 XOR 1 = -1 |

### Neural Network Training

- **Algorithm:** Mini-batch SGD
- **Loss:** Cross-entropy
- **Activation:** ReLU (configurable)
- **Quantization:** Straight-through estimator
- **Weights:** {-1, 0, +1} ternary

### Encoding Efficiency

| Method | Bits/Tryte | Lossless | Efficiency |
|--------|-----------|----------|------------|
| Compact | 27 | No* | 94.6% |
| Perfect | 29 | Yes | 98.4% |
| Optimal | 32+header | Yes | 94.9% |

*Compact has saturation for (++,) case

## 🎨 Customization

### Modify Network Defaults

Edit `app.js`:
```javascript
// Change default architecture
function buildNetwork() {
    const architecture = [784, 256, 128, 10]; // Modify here
    // ...
}
```

### Add New Logic Operations

Edit `ternary-core.js`:
```javascript
class Trit {
    // Add custom operation
    customOp(other) {
        return new Trit(/* your logic */);
    }
}
```

### Adjust Styles

Edit `styles.css`:
```css
:root {
    --primary-color: #2563eb; /* Change theme color */
    /* ... */
}
```

## 📦 Deployment

See [GITHUB_PAGES_DEPLOYMENT.md](../GITHUB_PAGES_DEPLOYMENT.md) for detailed instructions.

Quick deploy:
```bash
git add docs/web-sandbox/
git commit -m "Add ternary computing sandbox"
git push origin main

# Enable GitHub Pages in repository settings
# Settings → Pages → Source: main → Folder: /docs
```

## 🐛 Known Limitations

1. **Large datasets:** May slow down browser (limit ~10K samples)
2. **Image uploads:** Not yet implemented (coming soon)
3. **Model export:** Cannot save/load trained models (planned)
4. **Browser compatibility:** Requires modern browser (Chrome, Firefox, Safari, Edge)

## 🔮 Future Enhancements

- [ ] Model save/load functionality
- [ ] More dataset formats (images, JSON)
- [ ] Additional neural network layers (convolution, dropout)
- [ ] More encoding methods
- [ ] Ternary arithmetic calculator
- [ ] Assembly language editor
- [ ] WebGL acceleration
- [ ] Offline PWA support

## 🤝 Contributing

Contributions welcome!

1. Fork the repository
2. Make changes in `docs/web-sandbox/`
3. Test locally
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the Setun ternary computer (1958)
- Built on balanced ternary concepts
- Educational tool for ternary computing research

## 📞 Support

- [GitHub Issues](https://github.com/v-like-engine/trit-computing-machine/issues)
- [Documentation](../README.md)
- [Neural Networks Guide](../NEURAL_NETWORKS.md)

---

**Enjoy exploring ternary computing!** 🔺
