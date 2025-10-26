# AI Surveillance Camera - High Precision Face Detection

Real-time face detection and analysis with high-precision embeddings optimized for face recognition. Features float64 precision, multiple model support, and modern PyQt5 interface. Built with InsightFace, OpenCV, and ONNX Runtime.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![InsightFace](https://img.shields.io/badge/InsightFace-0.7.3+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### High-Precision Embeddings
- **Float64 Precision**: Double precision for all embedding calculations
- **FP32 TensorRT Inference**: FP16 disabled to prevent quantization errors
- **Welford's Algorithm**: Numerically stable averaging
- **NaN/Inf Validation**: Automatic detection and filtering
- **Epsilon Safeguards**: Prevents division by zero (ε = 1e-12)

### Face Recognition
- **Cosine Similarity Matching**: Built-in face comparison
- **Stable Embeddings**: 1-second interval averaging with high precision
- **Quality Validation**: Embedding quality checks and metrics
- **Ready for Production**: Optimized for real-world systems

### Core Functionality
- **Real-time Face Detection**: High-accuracy detection using InsightFace models
- **Age & Gender Prediction**: Automatic demographic analysis
- **Face Embedding Extraction**: 512-dimensional embeddings with float64 precision
- **Multiple Model Support**: Speed, Balance, and Accuracy modes
- **Model Switching**: Dynamic model selection via GUI slider

### Performance
- **TensorRT Acceleration**: GPU-accelerated FP32 inference
- **Multi-Provider Support**: ONNX Runtime with CPU, CUDA, and TensorRT
- **Real-time FPS Monitoring**: Live performance metrics
- **Memory Management**: Automatic cleanup of old data
- **Error Recovery**: Robust error handling

## Requirements

### Tested Configuration
- **Python**: 3.8
- **CUDA**: 12.6
- **cuDNN**: v9.13
- **TensorRT**: 10.3 GA
- **OS**: Windows

⚠️ **Note**: For best results, use similar versions.

### Hardware
- **CPU**: Modern multi-core processor
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA GPU with CUDA support (recommended)
  - Tested: GTX 1650 Ti with CUDA 12.6, cuDNN v9.13, TensorRT 10.3 GA
- **Webcam**: USB or built-in camera

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ai-surveillance-camera.git
cd ai-surveillance-camera
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3b. Install PyTorch for GPU (CUDA 12.6)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
*Note: Adjust CUDA version as needed. Skip if using CPU only.*

### 4. Download Models
```bash
python setup.py
```

Downloads 2 buffalo models (~120MB):
- **buffalo_s**: Speed mode (fast processing, 20MB)
- **buffalo_l**: Accuracy mode (best accuracy, 100MB)

### 5. Run Application

**PyQt5 GUI (Recommended)**
```bash
python gui_app_qt.py
```

**Terminal Version**
```bash
python test.py
```

## PyQt5 GUI Features

### Modern Interface
- Beautiful gradient UI with smooth animations
- Professional dark theme
- Rich HTML-formatted embeddings display
- Anti-aliased rendering

### Controls
- **Model Button**: Toggle between Speed/Accuracy
- **Precision Toggle**: Switch between float32/float64
- **Camera Switch**: Cycle through available cameras
- **Real-time Stats**: FPS, face count, model status

### Model Selection

Click button to toggle between:
- **Speed**: buffalo_s, float32, ~32 FPS (Orange)
- **Accuracy**: buffalo_l, float64, ~22 FPS (Green)

### Display Elements
- **Green Bounding Boxes**: Detected faces
- **Cyan Text**: Age and gender
- **Top Stats Bar**: FPS, face count, model, precision
- **Right Panel**: Face embeddings (raw, normalized, averaged)

## Terminal Version

### Controls
- Press **q**: Quit application
- Press **h**: Toggle precision mode (float64 ↔ float32)

### Display
- Red bounding boxes around faces
- Cyan text: Age and gender
- Yellow text: 1s averaged embedding norm
- Gray text: Instant embedding norm
- White text: FPS and face count

## Configuration

### Model Selection
| Model | Speed | Accuracy | Size | FPS (GTX 1650 Ti) |
|-------|-------|----------|------|-------------------|
| buffalo_s | Fast | Good | 20MB | 35+ |
| buffalo_l | Slow | Best | 100MB | 24+ |

### Detection Parameters
```python
DETECTION_SIZE = (640, 640)  # Input resolution
det_thresh = 0.5             # Detection threshold
avg_window_seconds = 1.0     # Embedding averaging window
```

### Precision Settings
```python
EMBEDDING_DTYPE = np.float64  # Double precision
EPSILON = 1e-12               # Numerical stability
```

## Troubleshooting

**"No ONNX models found"**
- Run `python setup.py` to download models
- Check models directory exists

**"Could not open webcam"**
- Check webcam permissions
- Try different camera indices
- Close other apps using camera

**Low FPS**
- Switch to Speed model via slider
- Reduce DETECTION_SIZE to (320, 320)
- Install onnxruntime-gpu for GPU acceleration

**"TensorRT Provider not available"**
- Install NVIDIA TensorRT separately
- Use CUDA provider instead
- Fallback to CPU provider

## Model Comparison

| Feature | Speed | Accuracy |
|---------|-------|----------|
| Model | buffalo_s | buffalo_l |
| Precision | float32 | float64 |
| FPS | ~32 | ~22 |
| Size | 20MB | 100MB |
| Use Case | Real-time | Recognition |

## Face Recognition Example

```python
from gui_app_qt import Face

# Compare two faces
similarity = face1.compute_similarity(face2)
if similarity > 0.6:
    print("Same person")

# Access embeddings
instant = face.embedding       # Current frame
averaged = face.avg_embedding  # 1s average
normed = face.normed_embedding # Normalized
```

## Performance Tips

1. **Model Selection**: Use slider to find optimal model
   - Speed: Real-time monitoring
   - Balance: General use
   - Accuracy: Face recognition

2. **Precision Toggle**: 
   - float64: Best for face matching
   - float32: Faster processing

3. **GPU Acceleration**: Install onnxruntime-gpu

4. **Resolution**: Lower DETECTION_SIZE for speed

## Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Submit pull request

## License

MIT License - see LICENSE file

## Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis models
- [OpenCV](https://opencv.org/) - Computer vision library
- [ONNX Runtime](https://onnxruntime.ai/) - Inference engine
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - GPU acceleration

## Version History

### Current
- Float64 precision embeddings
- Three model support (Speed/Balance/Accuracy)
- PyQt5 modern GUI with model selector slider
- Runtime precision toggle
- Welford's algorithm for stable averaging
- NaN/Inf validation
- Cosine similarity matching
- 1-second embedding averaging

---

**Built for computer vision and AI surveillance applications**

For questions or issues, open an issue on GitHub.
