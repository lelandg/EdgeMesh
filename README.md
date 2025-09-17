# 🎨 EdgeMesh - Advanced 3D Mesh Generation from Images

<div align="center">

[![GitHub](https://img.shields.io/github/license/lelandg/EdgeMesh)](https://github.com/lelandg/EdgeMesh/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-Latest-green)](https://www.riverbankcomputing.com/software/pyqt/)
[![Open3D](https://img.shields.io/badge/Open3D-0.19.0-orange)](http://www.open3d.org/)

**Transform 2D images into stunning 3D meshes using AI-powered depth estimation and advanced edge detection** ✨

[**🚀 Quick Start**](#-quick-start) | [**📸 Features**](#-key-features) | [**🛠️ Installation**](#-installation) | [**📖 Documentation**](Docs/CodeMap.md) | [**🎯 Examples**](#-examples)

<img src="https://github.com/lelandg/EdgeMesh/assets/YOUR_IMAGE.png" alt="EdgeMesh Demo" width="600"/>

</div>

---

## 🌟 Overview

**EdgeMesh** is a powerful PyQt6-based desktop application that converts 2D images into 3D meshes using state-of-the-art depth estimation models and sophisticated edge detection algorithms. Whether you're a 3D artist, game developer, researcher, or hobbyist, EdgeMesh provides an intuitive interface for creating detailed 3D models from ordinary photographs.

### 🎯 What Can EdgeMesh Do?

- 🖼️ **Convert any image to a 3D mesh** in seconds
- 🧠 **AI-powered depth estimation** using multiple models (MiDaS, DPT, ZoeDepth, Depth-Anything)
- 🎨 **Advanced edge detection** with customizable parameters
- 🔧 **Real-time 3D preview** with interactive viewport
- 📁 **Export to standard formats** (.obj, .stl) for 3D printing or modeling software
- 🎮 **SpaceMouse support** for professional 3D navigation

---

## ✨ Key Features

### 🖼️ Image Processing
- **Smart Background Removal** - Automatically detects and removes backgrounds based on corner colors
- **Edge Detection** - Advanced Canny edge detection with adjustable thresholds
- **Multiple Smoothing Methods** - Gaussian, Bilateral, Median, and Anisotropic Diffusion
- **Grayscale Conversion** - Process images in grayscale for enhanced control
- **Dynamic Depth Adjustment** - Create front-half meshes perfect for 3D printing

### 🧠 Depth Estimation Models
Choose from multiple state-of-the-art models:
- **MiDaS** (Small & Large variants) - Fast and reliable
- **DPT** (Large & Hybrid) - High accuracy
- **ZoeDepth** (K, N, NK, N-indoor) - Specialized for different scenarios
- **Depth-Anything** - Latest cutting-edge models

### 🎨 3D Mesh Generation
- **Depth-to-3D Conversion** - Direct conversion from depth maps
- **Edge-based Mesh Creation** - Generate meshes from detected edges
- **Surface Partitioning** - Intelligent surface segmentation
- **Extrusion Projection** - Complex 3D reconstructions from contours
- **Real-time Preview** - See your mesh as it's created

### 🛠️ Professional Tools
- **3D Text Generator** - Create 3D text meshes using the included MeshTools
- **Mesh Manipulation** - Scale, rotate, and transform meshes
- **Custom Coloring** - Apply gradients and color transitions
- **Measurement Grid** - Visualize dimensions and proportions
- **Batch Processing** - Process multiple images efficiently

---

## 🚀 Quick Start

### 📋 Prerequisites

- **Python 3.12** (⚠️ Not compatible with Python 3.13+)
- **Windows** (primary support) or **Linux/WSL**
- **4GB+ RAM** recommended
- **NVIDIA GPU** (optional, for faster processing)

### 🛠️ Installation

#### Option 1: Automated Installation (Recommended) 🎯

```bash
# Clone the repository
git clone https://github.com/lelandg/EdgeMesh.git
cd EdgeMesh

# Run the automated installer
python install_requirements.py
```

#### Option 2: Manual Installation 📦

```bash
# Clone the repository
git clone https://github.com/lelandg/EdgeMesh.git
cd EdgeMesh

# Update pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For Open3D compatibility issues
python install_open3d.py
```

### ▶️ Running EdgeMesh

#### Windows 🪟
```bash
python edge_mesh.py
# or simply:
run.bat
```

#### Linux/WSL 🐧
```bash
python3 edge_mesh.py
```

---

## 📖 How to Use

### Step 1: Load an Image 📸
Click **"Load Image"** and select any image file (PNG, JPG, etc.)

### Step 2: Configure Settings ⚙️
- **Depth Model**: Choose your preferred AI model
- **Smoothing**: Select smoothing method and intensity
- **Edge Detection**: Adjust thresholds for edge sensitivity
- **Resolution**: Set mesh resolution (higher = more detail)

### Step 3: Process the Image 🔄
Click **"Generate 3D Mesh"** to create your 3D model

### Step 4: Preview and Export 💾
- Use the 3D viewport to inspect your mesh
- **Mouse controls**: Rotate, zoom, pan
- Export as `.obj` or `.stl` for 3D printing or further editing

### Pro Tips 💡
- Enable **"Dynamic Depth"** for 3D printing-friendly meshes
- Use **"Remove Background"** for cleaner results
- Try different depth models for various image types
- Indoor scenes work best with ZoeDepth N-indoor model

---

## 🏗️ Architecture

EdgeMesh follows a modular pipeline architecture:

```
📷 Image Input
    ↓
🎨 Preprocessing (Edge Detection, Smoothing)
    ↓
🧠 Depth Estimation (AI Models)
    ↓
🔨 Mesh Generation (Depth-to-3D or Edge-based)
    ↓
✨ 3D Viewport (Interactive Preview)
    ↓
💾 Export (.obj, .stl)
```

For detailed architecture information, see the [Code Map](Docs/CodeMap.md).

---

## 🎯 Examples

### Creating a 3D Portrait
1. Load a portrait photo
2. Select **MiDaS Large** model
3. Enable **Dynamic Depth**
4. Generate and export for 3D printing

### Architectural Visualization
1. Load building photograph
2. Use **DPT Hybrid** for accuracy
3. Enable edge detection for sharp details
4. Export to modeling software

### 3D Text Creation
```bash
python MeshTools/text_3d.py --text "Hello 3D" --height 100 --depth 20
```

---

## 🔧 Troubleshooting

### Python Version Compatibility ⚠️
**Important**: Open3D 0.19.0 is not compatible with Python 3.13+

**Solutions**:
1. Use Python 3.12:
   ```bash
   # Check your version
   python --version

   # If 3.13+, install Python 3.12
   ```

2. Create a virtual environment:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Run compatibility checker:
   ```bash
   python install_open3d.py
   ```

### Common Issues

**Issue**: ImportError with Open3D
**Solution**: Reinstall with specific version
```bash
pip install open3d==0.19.0 --force-reinstall --no-cache-dir
```

**Issue**: CUDA/GPU not detected
**Solution**: Install PyTorch with CUDA support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Documentation

- 📖 [**Code Map**](Docs/CodeMap.md) - Complete project structure and navigation
- 🔄 [**3D Mesh Creation Flow**](docs/3D_Mesh_Creation_Flow.md) - Detailed processing pipeline
- 📝 [**Version History**](docs/Versions.md) - Release notes and changes
- 🛠️ [**API Reference**](Docs/CodeMap.md#core-application) - Detailed class and method documentation

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💻 Make your changes
4. ✅ Test thoroughly
5. 📝 Commit (`git commit -m 'Add AmazingFeature'`)
6. 📤 Push (`git push origin feature/AmazingFeature`)
7. 🎯 Open a Pull Request

### Development Setup

```bash
# Clone with submodules
git clone --recursive https://github.com/lelandg/EdgeMesh.git

# Install in development mode
pip install -e .

# Run tests (when available)
python -m pytest
```

---

## 🌐 Related Projects

### 🎨 ImageAI by Leland Green
For AI-powered image generation that can serve as input for 3D mesh creation, check out [**ImageAI**](https://github.com/lelandg/ImageAI) - a comprehensive tool supporting multiple AI providers including:
- Google Gemini
- OpenAI DALL-E
- Stability AI
- Local Stable Diffusion

Perfect for creating unique images to convert into 3D meshes! 🚀

---

## 👨‍💻 Author

**Leland Green**
- 🌐 Website: [LelandGreen.com](https://www.lelandgreen.com)
- 📧 Email: contact@lelandgreen.com
- 💬 Discord: [The Intersection of Art and AI](https://discord.gg/a64xRg9w)
- 🐙 GitHub: [@lelandg](https://github.com/lelandg)

*Also creator of [ImageAI](https://github.com/lelandg/ImageAI) - Advanced AI Image & Video Generation Tool*

---

## 📄 License

This project is proprietary software. Please see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Open3D Team** - For the excellent 3D processing library
- **PyTorch Team** - For deep learning framework
- **MiDaS Contributors** - For depth estimation models
- **Qt/PyQt6** - For the robust GUI framework
- **Community Contributors** - For feedback and improvements

---

## 🚀 Roadmap

### Coming Soon 🔜
- [ ] 🤖 More AI models (SAM integration)
- [ ] 📱 Mobile companion app
- [ ] ⚡ GPU acceleration optimizations
- [ ] 🔧 Plugin system for custom processors

### Future Vision 🔮
- Real-time 3D reconstruction from webcam
- VR/AR export capabilities
- Cloud processing for large batches
- AI-assisted mesh refinement

---

## 💖 Support

If you find EdgeMesh useful, please:
- ⭐ Star the repository
- 🐛 Report bugs via [Issues](https://github.com/lelandg/EdgeMesh/issues)
- 💡 Suggest features
- 📣 Share with others
- 🤝 Join our [Discord Community](https://discord.gg/a64xRg9w)

---

<div align="center">

**Made with ❤️ by [Leland Green](https://www.lelandgreen.com)**

[🔝 Back to Top](#-edgemesh---advanced-3d-mesh-generation-from-images)

</div>