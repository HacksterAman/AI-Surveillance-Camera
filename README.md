# AI Surveillance Camera v1.0

A real-time face detection and analysis system with 2-second embedding averaging for stable measurements. Built with InsightFace, OpenCV, and ONNX Runtime with optional TensorRT acceleration.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![InsightFace](https://img.shields.io/badge/InsightFace-0.7.3+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Features (v1.0)

### Core Functionality
- **Real-time Face Detection**: High-accuracy face detection using InsightFace models
- **Age & Gender Prediction**: Automatic demographic analysis for detected faces
- **Face Embedding Extraction**: 512-dimensional face embeddings for identity analysis
- **2-Second Embedding Averaging**: Stable measurements using time-windowed averaging
- **Multiple Model Support**: Detection, recognition, and demographic analysis models

### Performance & Optimization  
- **TensorRT Acceleration**: GPU-accelerated inference for NVIDIA hardware
- **Multi-Provider Support**: ONNX Runtime with CPU, CUDA, and TensorRT providers
- **Real-time FPS Monitoring**: Live performance metrics display
- **Memory Management**: Automatic cleanup of old embedding data
- **Error Recovery**: Robust error handling with graceful degradation

### User Interface
- **Live Video Feed**: Real-time webcam processing with overlay information
- **Visual Overlays**: Bounding boxes, age/gender labels, embedding metrics
- **Performance Metrics**: FPS counter and active face count
- **Comparison Display**: Shows both instant and averaged embedding values

### Technical Features
- **Float32 Enforcement**: All embeddings forced to float32 for consistency and performance
- **Cached Properties**: Optimized embedding norm calculations  
- **Time-based Buffers**: Efficient sliding window for embedding averaging
- **Model Validation**: Automatic model loading with error handling
- **Configurable Parameters**: Adjustable detection thresholds and window sizes

## 📋 Requirements

### Tested Configuration
This project has been tested and verified on the following configuration:
- **Python**: 3.8
- **CUDA**: 12.6
- **cuDNN**: v9.13
- **TensorRT**: 10.3 GA
- **Operating System**: Windows

⚠️ **Version Compatibility**: No guarantee of working in case of version mismatch. For best results, use similar versions.

### General Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Hardware**: 
  - CPU: Modern multi-core processor
  - RAM: 4GB minimum, 8GB recommended
  - GPU: NVIDIA GPU with CUDA support (optional, for acceleration)
- **Webcam**: USB or built-in camera

## 🚀 Quick Start

### 1. Clone the Repository
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

### 3b. Install PyTorch for GPU Acceleration (Optional)
For CUDA 12.6 support (tested configuration):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

⚠️ **Note**: This command is for CUDA 12.6. Adjust the URL for different CUDA versions or use CPU-only version if no GPU available.

### 4. Download InsightFace Models

#### Automatic Setup (Recommended)
Use the included setup script for easy model installation:

```bash
python setup.py
```

This interactive script will:
- Download your chosen model (buffalo_s, buffalo_m, or buffalo_l)
- Configure the system automatically  
- Verify everything is working

#### Model Files and Their Purposes
When downloaded, the models directory will contain these ONNX files:

```
models/
└── buffalo_l/  (example)
    ├── det_10g.onnx          # Face detection (primary detector)
    ├── genderage.onnx        # Age (0-100) and gender prediction (0=female, 1=male)
    ├── w600k_r50.onnx        # Face recognition embeddings (512-dim vectors)
    ├── 1k3d68.onnx           # 3D face alignment (68 landmarks)
    └── 2d106det.onnx         # Detailed facial landmark detection (106 points)
```

**Why `.gitkeep`?** The `models/.gitkeep` file ensures the models directory is tracked in git, even when empty, so the setup script knows where to place downloaded models.

### 5. Run the Application
```bash
python test.py
```

The system will use the model configured by `setup.py`. To change models, simply run `python setup.py` again and select a different option.

## 🎮 Usage

### Basic Controls
- **Start**: Run `python test.py`
- **Quit**: Press `q` key in the video window
- **Camera**: The system will try webcam index 1, then 0 if unavailable

### Display Elements
- **Red Bounding Boxes**: Detected face regions
- **Cyan Text**: Age and gender information
- **Yellow Text**: Averaged embedding norm (2-second window)
- **Gray Text**: Instant embedding norm (for comparison)
- **White Text**: FPS and face count (top-left corner)

### Understanding the Output
```
FPS: 25.3 | Faces: 2          # Performance metrics
Female, 28                     # Age and gender prediction
Avg Emb: 12.543               # 2-second averaged embedding norm
Inst: 12.891                  # Instant embedding norm
```

## ⚙️ Configuration

### Model Selection
Use the setup script to change models:
```bash
python setup.py
```
Then choose from:
- `buffalo_s`: Faster, good accuracy (35+ FPS on GTX 1650 Ti)
- `buffalo_m`: Balanced performance (30+ FPS on GTX 1650 Ti)  
- `buffalo_l`: Best accuracy (24+ FPS on GTX 1650 Ti)

### Detection Parameters
```python
DETECTION_SIZE = (640, 640)  # Input resolution
det_thresh = 0.5            # Detection confidence threshold
```

### Averaging Window
```python
avg_window_seconds = 2.0    # Embedding averaging window
cleanup_interval = 5.0      # Memory cleanup interval
```

### Hardware Acceleration
For GPU acceleration, install CUDA-compatible ONNX Runtime:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

**Tested GPU Configuration**:
- NVIDIA GPU with CUDA 12.6, cuDNN v9.13, TensorRT 10.3 GA
- For optimal performance, ensure your system matches these versions

## 🔧 Troubleshooting

### Common Issues

**"No ONNX models found"**
- Ensure models are in `./models/buffalo_m/` directory
- Check file names match expected ONNX files
- Try re-downloading the model pack

**"Could not open webcam"**
- Check webcam permissions
- Try different camera indices (0, 1, 2)
- Ensure no other applications are using the camera

**Low FPS Performance**
- **Switch to smaller model**: Run `python setup.py` and choose buffalo_m or buffalo_s
- Reduce `DETECTION_SIZE` to (320, 320)
- Install GPU-accelerated ONNX Runtime
- Close unnecessary applications
- **Reference**: GTX 1650 Ti achieves 24+ FPS with buffalo_l

**"TensorRT Provider not available"**
- Install NVIDIA TensorRT separately
- Use CUDA provider instead: modify `PROVIDERS` in code
- Fallback to CPU provider for compatibility

**Version Compatibility Issues**
- This project was tested on Python 3.8, CUDA 12.6, cuDNN v9.13, TensorRT 10.3 GA
- For different versions: try CPU-only mode first
- Check ONNX Runtime compatibility with your CUDA version
- Consider using virtual environment to avoid conflicts

### Performance Optimization

1. **Model Selection**: Run `python setup.py` to choose the right model for your hardware
   - GTX 1650 Ti: buffalo_l (24+ FPS), buffalo_m (30+ FPS), buffalo_s (35+ FPS)
   - Lower-end GPUs: buffalo_s recommended
   - CPU-only: buffalo_s strongly recommended

2. **GPU Acceleration**: Install `onnxruntime-gpu` and ensure CUDA is available
3. **Resolution**: Lower `DETECTION_SIZE` for better performance  
4. **TensorRT**: Enable TensorRT provider for NVIDIA GPUs

### Debug Mode
Enable verbose logging by uncommenting debug prints in the code:
```python
print(f"DEBUG: Processing frame {frame_count}")
print(f"DEBUG: Detected {len(faces)} faces")
```

## 📊 Model Comparison

| Model | Speed | Accuracy | Model Size | FPS (GTX 1650 Ti) | Use Case |
|-------|--------|----------|------------|-------------------|----------|
| buffalo_s | Fast | Good | ~20MB | 35+ FPS | Real-time applications, lower-end hardware |
| buffalo_m | Medium | Better | ~50MB | 30+ FPS | Balanced performance, recommended default |
| buffalo_l | Slow | Best | ~100MB | **24+ FPS** | High-accuracy, tested configuration |

**Performance Note**: Testing on GTX 1650 Ti shows buffalo_l achieves 24+ FPS. For better performance on lower-end hardware, switch to buffalo_m or buffalo_s models.

**Hardware Scaling**: 
- Modern GPUs (RTX 30/40 series): All models run smoothly
- Mid-range GPUs (GTX 1650-1660): buffalo_l works well, buffalo_m recommended
- Older/Lower-end GPUs: Use buffalo_s for best performance
- CPU-only: buffalo_s strongly recommended

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis models and framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [ONNX Runtime](https://onnxruntime.ai/) - High-performance inference engine
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) - GPU acceleration

## 📈 Version History

### v1.0.0 (Current)
- ✅ Real-time face detection and analysis
- ✅ Age and gender prediction
- ✅ 2-second embedding averaging system
- ✅ TensorRT acceleration support
- ✅ FPS monitoring and performance metrics
- ✅ Robust error handling and recovery
- ✅ Memory management and cleanup
- ✅ Multiple model support (buffalo series)
- ✅ Float32 embedding enforcement (prevents ONNX float64 issues)

---

**Built with ❤️ for computer vision and AI surveillance applications**

For questions, issues, or feature requests, please open an issue on GitHub.
