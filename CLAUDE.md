# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EdgeMesh is a PyQt6-based application for advanced image processing, edge detection, and 3D mesh generation from depth maps. It provides an interactive GUI for professionals working in computer vision and 3D modeling.

## Commands

### Running the Application
```bash
# Windows
python edge_mesh.py  # or run.bat

# Linux/WSL
python3 edge_mesh.py
```

### Installing Dependencies
```bash
# Automated installation (recommended)
python install_requirements.py

# Manual installation
pip install -r requirements.txt

# For Open3D compatibility issues with Python 3.13+
python install_open3d.py
```

### Building Standalone Executable
```bash
# Windows (using Nuitka)
build.bat
```

## Architecture

### Core Processing Pipeline
The application follows a multi-stage pipeline for 3D mesh generation:
1. **Image Loading & Preprocessing** → `image_processor.py`
2. **Depth Estimation** → `depth_to_3d.py` (uses various torch models: MiDaS, DPT, ZoeDepth)
3. **Edge Detection** → `edge_detection.py` (Canny edge detection with customizable parameters)
4. **Mesh Generation** → `mesh_generator.py` or direct depth-to-mesh conversion
5. **3D Visualization** → `MeshTools/viewport_3d.py` (Open3D-based rendering)

### Key Components

**Main Entry Point**: `edge_mesh.py`
- Contains `MainWindowImageProcessing` class
- Manages PyQt6 GUI and user interactions
- Coordinates between processing modules

**Depth Processing Models** (`depth_to_3d.py`):
- MiDaS variants (small, large)
- DPT models (large, hybrid)
- ZoeDepth models (K, N, NK, N-indoor)
- Depth-Anything models

**Processing Features**:
- Multiple smoothing methods (Gaussian, Bilateral, Median, Anisotropic Diffusion)
- Background removal based on corner color detection
- Dynamic depth adjustment for front-half mesh generation
- Edge clustering and shape analysis (not fully exposed in GUI)
- Surface partitioning and extrusion projection capabilities

### MeshTools Submodule
Located in `MeshTools/` directory, provides:
- `viewport_3d.py`: Standalone 3D viewport with export capabilities (.obj, .stl)
- `mesh_tools.py`: Core mesh manipulation utilities
- `text_3d.py`: 3D text mesh generation using Open3D and PyVista
- SpaceMouse controller support for 3D navigation

## Key Technical Details

### Python Version Compatibility
- **Critical**: Open3D 0.19.0 is NOT compatible with Python 3.13+
- Use Python 3.12 or earlier for full functionality
- The project includes `install_open3d.py` to detect version and provide guidance

### Image Processing Flow
1. Images are loaded in BGR format (OpenCV standard)
2. Internal processing maintains BGR for consistency
3. Preview images are converted to RGB for Qt display
4. Depth maps are normalized to 0-255 range
5. 3D vertices are generated from depth values with configurable scaling

### GUI State Management
- Uses Qt checkboxes and sliders for real-time parameter adjustment
- Configuration persistence through ConfigParser
- Supports both processed and original image as input source
- "Use Processed Image" checkbox determines processing pipeline source

### Mesh Generation Approaches
1. **Depth-based**: Converts depth map directly to 3D mesh
2. **Edge-based**: Uses detected edges to generate mesh structure
3. Both approaches support visualization of intermediate steps

## Development Notes

### File Organization
- Core processing modules are at project root
- GUI extensions and utilities in `qt_extensions.py`
- Torch/model utilities in `torch_utils.py`, `torch_test.py`
- File operations helpers in `file_tools.py`
- Logging utilities in `log_utils.py` (debug mode configurable)

### Important Flags
- `debug = False` in `edge_mesh.py` - Controls debug output
- `visualize_images = False` - Enables OpenCV visualization windows
- Various visualization flags for mesh generation stages

### External Dependencies
Key libraries required:
- PyQt6 for GUI
- OpenCV (cv2) for image processing
- Open3D for 3D operations and visualization
- PyTorch and torchvision for depth estimation models
- Trimesh for mesh manipulation
- NumPy, SciPy, scikit-learn for computational operations
- Transformers for model loading