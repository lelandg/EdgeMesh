# EdgeMesh Code Map

*Last Updated: 2025-09-17 11:48:26*

## Table of Contents

| Section | Line Number |
|---------|-------------|
| [Quick Navigation](#quick-navigation) | 25 |
| [Visual Architecture Overview](#visual-architecture-overview) | 40 |
| [Project Structure](#project-structure) | 75 |
| [Core Configuration Files](#core-configuration-files) | 142 |
| [Core Application](#core-application) | 158 |
| [Image Processing Modules](#image-processing-modules) | 208 |
| [Depth Processing Modules](#depth-processing-modules) | 232 |
| [3D Mesh Generation](#3d-mesh-generation) | 280 |
| [MeshTools Library](#meshtools-library) | 292 |
| [UI Components and Extensions](#ui-components-and-extensions) | 345 |
| [Utility Modules](#utility-modules) | 360 |
| [Cross-File Dependencies](#cross-file-dependencies) | 382 |
| [Architecture Patterns](#architecture-patterns) | 432 |
| [Development Guidelines](#development-guidelines) | 467 |
| [Performance Considerations](#performance-considerations) | 501 |

## Quick Navigation

### Primary Entry Points
- **Main Application**: `edge_mesh.py:1544` - Main PyQt6 application entry
- **Depth Processing**: `depth_to_3d.py:25` - DepthTo3D class for depth map generation
- **Edge Detection**: `edge_detection.py:5` - Edge detection algorithms
- **Mesh Generator**: `mesh_generator.py:1` - 3D mesh generation from images
- **3D Viewport**: `MeshTools/viewport_3d.py:1` - 3D visualization component

### Key Components
- **Image Processing**: `image_processor.py:11` - ImageProcessor class
- **Qt Extensions**: `qt_extensions.py:1` - Custom Qt widgets and utilities
- **MeshTools**: `MeshTools/mesh_tools.py:66` - MeshTools class for 3D operations
- **Configuration**: `config.ini` - Application settings persistence

## Visual Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    EdgeMesh Application                       │
│                    edge_mesh.py (PyQt6)                      │
└──────────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│   Image Processing       │          │   UI Components          │
│   - edge_detection       │◄─────────►   - MainWindow          │
│   - image_processor      │          │   - Qt Extensions        │
│   - smoothing_utils      │          │   - Flow Layout         │
└──────────────────────────┘          └──────────────────────────┘
          ▲                                           ▲
          └─────────────────────┬─────────────────────┘
                                ▼
          ┌──────────────────────────────────────────┐
          │         Depth Processing                 │
          │   - depth_to_3d                         │
          │   - depth_cue_estimator                 │
          │   - smoothing_depth_map_utils           │
          └──────────────────────────────────────────┘
                                │
                                ▼
          ┌──────────────────────────────────────────┐
          │         3D Mesh Generation               │
          │   - mesh_generator                       │
          │   - MeshTools/mesh_tools                │
          │   - MeshTools/viewport_3d               │
          └──────────────────────────────────────────┘
```

## Project Structure

```
EdgeMesh/
├── edge_mesh.py                    # Main application (1546 lines)
├── depth_to_3d.py                  # Depth map to 3D conversion
├── edge_detection.py               # Edge detection algorithms
├── mesh_generator.py               # 2D to 3D mesh generation
├── image_processor.py              # Image processing utilities
├── qt_extensions.py                # Custom Qt widgets
├── flow_layout.py                  # Flow layout for Qt
│
├── Depth Processing/
│   ├── depth_cue_estimator_util.py    # Depth estimation utilities
│   ├── depth_based3d_reconstruction.py # 3D reconstruction
│   ├── smoothing_depth_map_utils.py   # Depth map smoothing
│   └── depth_anything.py              # Depth Anything model integration
│
├── MeshTools/                      # 3D mesh manipulation library
│   ├── mesh_tools.py              # Core mesh operations
│   ├── viewport_3d.py             # 3D visualization viewport
│   ├── mesh_manipulation.py       # Mesh manipulation utilities
│   ├── mesh_gradient_colorizer.py # Mesh coloring utilities
│   ├── measurement_grid_visualizer.py # Grid visualization
│   ├── text_3d.py                # 3D text rendering
│   ├── space_mouse_controller.py  # SpaceMouse integration
│   ├── space_mouse_event_handler.py # SpaceMouse events
│   └── file_tools.py             # File management utilities
│
├── Depth Models/                  # Depth estimation models
│   ├── depth_estimation_model.py
│   └── estimate_depth.py
│
├── Utils/
│   ├── spinner.py                 # Progress spinner
│   ├── file_tools.py             # File utilities
│   ├── torch_utils.py            # PyTorch utilities
│   └── log_utils.py              # Logging utilities
│
├── Analysis/
│   ├── edge_clustering_analyzer.py # Edge clustering analysis
│   ├── shape_analyzer.py          # Shape analysis
│   └── surface_partitioning.py    # Surface partitioning
│
├── Configuration/
│   ├── config.ini                 # User settings
│   ├── _version.py               # Version information
│   └── python_requirements.txt   # Python dependencies
│
├── Build/
│   ├── setup.py                  # Setup script
│   ├── install_requirements.py   # Requirements installer
│   ├── install_open3d.py        # Open3D installer
│   └── build.bat                 # Windows build script
│
├── Documentation/
│   ├── README.md                 # Main documentation
│   ├── Docs/                     # Additional documentation
│   │   └── CodeMap.md           # This file
│   └── docs/
│       ├── 3D_Mesh_Creation_Flow.md
│       └── Versions.md
│
└── Images/                       # Sample images
    └── example.png              # Example image for testing
```

## Core Configuration Files

### Build Configuration
- `setup.py` - Python package setup configuration
- `python_requirements.txt` - Python package dependencies
- `build.bat` - Windows build automation script
- `run.bat` - Windows run script

### Application Configuration
- `config.ini` - User preferences and UI settings persistence
- `_version.py` - Version tracking (current: defined in file)

### Development Configuration
- `.gitignore` - Git ignore patterns
- `LICENSE` - License information

## Core Application

### MainWindowImageProcessing
**Path**: `edge_mesh.py` - 1546 lines
**Purpose**: Main application window managing UI and workflow orchestration
**Language**: Python (PyQt6)

#### Table of Contents
| Section | Line Number |
|---------|-------------|
| Imports and Configuration | 1 |
| Helper Functions | 73 |
| Class Definition | 101 |
| UI Initialization | 172 |
| Event Handlers | 688 |
| Image Processing | 935 |
| 3D Viewport Management | 1140 |
| Configuration Management | 1331 |
| Main Entry Point | 1531 |

#### Properties
| Name | Line | Type | Access | Description |
|------|------|------|--------|-------------|
| verbose | 104 | bool | public | Enable verbose logging |
| depth_to_3d | 105 | DepthTo3D | public | Depth processing instance |
| mesh_3d | 108 | Trimesh | public | Generated 3D mesh |
| mesh_from_2d | 109 | Trimesh | public | 2D edge-based mesh |
| image | 117 | np.ndarray | public | Original loaded image |
| processed_image | 119 | np.ndarray | public | Processed image result |
| image_path | 118 | str | public | Path to current image |
| three_d_viewport | 115 | ThreeDViewport | public | 3D visualization viewport |

#### Key Methods
| Method | Line | Access | Returns | Async | Description |
|--------|------|--------|---------|-------|-------------|
| __init__() | 102 | public | None | No | Initialize main window |
| _init_ui() | 172 | private | None | No | Setup UI components |
| load_image() | 1253 | public | None | No | Load image from file |
| process_image() | 935 | public | None | No | Generate 3D mesh from depth |
| generate_mesh() | 1090 | public | None | No | Generate mesh from edges |
| update_preview() | 1189 | public | None | No | Update processed image preview |
| update_3d_viewport() | 1140 | public | None | No | Refresh 3D visualization |
| export_mesh() | 1165 | public | None | No | Export mesh to file |
| save_image() | 1318 | public | None | No | Save processed image |
| toggle_grayscale() | 850 | public | None | No | Toggle grayscale mode |
| toggle_edge_detection() | 857 | public | None | No | Toggle edge detection |
| toggle_invert_colors() | 1116 | public | None | No | Toggle color inversion |
| enable_color_picker_mode() | 694 | public | None | No | Enable color picker |
| reset_defaults() | 1390 | public | None | No | Reset all settings |

## Image Processing Modules

### ImageProcessor
**Path**: `image_processor.py` - ~100 lines
**Purpose**: Core image processing operations including blending and filters
**Language**: Python

#### Methods
| Method | Line | Access | Returns | Description |
|--------|------|--------|---------|-------------|
| blend_images() | 17 | static | np.ndarray | Blend two images with percentage |
| process_with_edge_detection() | - | public | np.ndarray | Apply edge detection |
| apply_smoothing() | - | public | np.ndarray | Apply smoothing filters |

### Edge Detection Module
**Path**: `edge_detection.py` - 82 lines
**Purpose**: Edge detection algorithms using OpenCV Canny

#### Functions
| Function | Line | Returns | Description |
|----------|------|---------|-------------|
| detect_and_project_edges() | 5 | np.ndarray | Detect and project edges |
| detect_edges() | 70 | np.ndarray | Simplified edge detection wrapper |

## Depth Processing Modules

### DepthTo3D
**Path**: `depth_to_3d.py` - ~500+ lines
**Purpose**: Convert images to depth maps and generate 3D meshes
**Language**: Python

#### Table of Contents
| Section | Line Number |
|---------|-------------|
| Imports and Configuration | 1 |
| Model Names Dictionary | 21 |
| Class Definition | 25 |
| Model Loading | 43 |
| Depth Estimation | - |
| Mesh Generation | - |

#### Properties
| Name | Line | Type | Access | Description |
|------|------|------|--------|-------------|
| verbose | 31 | bool | public | Enable verbose output |
| mesh_tools | 32 | MeshTools | public | Mesh manipulation tools |
| depth_map | 33 | np.ndarray | public | Generated depth map |
| model_type | 37 | str | public | Depth model type |
| device | 38 | torch.device | public | Compute device (CPU/GPU) |
| model | 39 | torch.nn.Module | public | Loaded depth model |

#### Methods
| Method | Line | Access | Returns | Description |
|--------|------|--------|---------|-------------|
| load_model() | 43 | public | tuple | Load depth estimation model |
| process_image() | - | public | tuple | Process image to 3D mesh |
| estimate_depth() | - | private | np.ndarray | Generate depth map |

### Depth Cue Estimator
**Path**: `depth_cue_estimator_util.py`
**Purpose**: Estimate depth cues from 2D images

### Smoothing Depth Map Utils
**Path**: `smoothing_depth_map_utils.py`
**Purpose**: Apply smoothing algorithms to depth maps

#### Smoothing Methods
- Anisotropic diffusion (edge-preserving)
- Gaussian smoothing
- Bilateral filtering
- Median filtering

## 3D Mesh Generation

### MeshGenerator
**Path**: `mesh_generator.py`
**Purpose**: Generate 3D meshes from 2D edge-detected images

#### Key Methods
- `generate()` - Main mesh generation pipeline
- Edge clustering analysis
- Surface partitioning
- Mesh triangulation

## MeshTools Library

### MeshTools Class
**Path**: `MeshTools/mesh_tools.py` - ~500+ lines
**Purpose**: Core 3D mesh manipulation operations
**Language**: Python

#### Table of Contents
| Section | Line Number |
|---------|-------------|
| Documentation Header | 1 |
| Imports | 49 |
| Class Definition | 66 |
| Mesh Operations | 100+ |

#### Properties
| Name | Line | Type | Access | Description |
|------|------|------|--------|-------------|
| mesh | 81 | Trimesh | public | Current mesh object |
| verbose | 82 | bool | public | Verbose output flag |
| input_mesh | 85 | str | public | Input mesh filename |

#### Methods
| Method | Line | Access | Returns | Description |
|--------|------|--------|---------|-------------|
| __init__() | 73 | public | None | Initialize with mesh |
| rotate_mesh() | - | public | Trimesh | Rotate mesh by angles |
| mirror_mesh() | - | public | Trimesh | Mirror across axis |
| fix_mesh() | - | public | Trimesh | Fix mesh issues |
| solidify_mesh() | - | public | Trimesh | Make mesh solid |

### ThreeDViewport
**Path**: `MeshTools/viewport_3d.py`
**Purpose**: 3D visualization viewport using Open3D
**Language**: Python

#### Key Features
- Real-time 3D mesh visualization
- Camera controls and navigation
- Mesh coloring and texturing
- Export functionality
- SpaceMouse support (optional)

#### Methods
| Method | Line | Access | Returns | Description |
|--------|------|--------|---------|-------------|
| __init__() | - | public | None | Initialize viewport |
| load_mesh() | - | public | None | Load mesh for display |
| clear_geometries() | - | public | None | Clear displayed meshes |
| run() | - | public | None | Start visualization loop |
| export_mesh_as_obj() | - | public | None | Export as OBJ file |
| export_mesh_as_stl() | - | public | None | Export as STL file |

## UI Components and Extensions

### Qt Extensions Module
**Path**: `qt_extensions.py`
**Purpose**: Custom Qt widgets and utility functions

#### Components
- `FlowLayout` - Custom flow layout widget
- `state_to_bool()` - Convert checkbox state to boolean
- Custom validators and widgets

### Flow Layout
**Path**: `flow_layout.py`
**Purpose**: Implement flow layout for dynamic UI arrangement

## Utility Modules

### File Tools
**Path**: `file_tools.py` & `MeshTools/file_tools.py`
**Purpose**: File management utilities

#### Functions
- `find_newest_file_in_directory()` - Find most recent file
- `get_matching_files()` - Get files matching pattern

### Spinner
**Path**: `spinner.py` & `MeshTools/spinner.py`
**Purpose**: Console progress spinner for long operations

### Torch Utils
**Path**: `torch_utils.py`
**Purpose**: PyTorch utility functions and helpers

### Log Utils
**Path**: `log_utils.py`
**Purpose**: Logging configuration and utilities

## Cross-File Dependencies

### Core Dependency Flows

#### Image Processing Pipeline
**Flow**: User Input → Image Loading → Processing → Preview
- `edge_mesh.py:load_image()` → loads image
- `edge_mesh.py:update_preview()` → calls processing
- `edge_detection.py:detect_edges()` → edge detection
- `image_processor.py:blend_images()` → blending
- `edge_mesh.py:display_processed_image()` → display result

#### Depth Estimation Pipeline
**Flow**: Image → Depth Model → Depth Map → 3D Mesh
- `edge_mesh.py:process_image()` → initiates depth processing
- `depth_to_3d.py:DepthTo3D()` → manages depth pipeline
- `depth_to_3d.py:load_model()` → loads AI model
- `depth_to_3d.py:process_image()` → generates depth map
- `MeshTools/mesh_tools.py` → mesh manipulation
- `MeshTools/viewport_3d.py` → 3D visualization

#### Configuration Management
**Managed by**: `edge_mesh.py` configuration methods
**Consumed by**:
- `edge_mesh.py:load_ui_settings()` (line 1419) - Load saved preferences
- `edge_mesh.py:save_ui_settings()` (line 1355) - Save preferences
- All UI components for state persistence

### Module Import Dependencies

#### edge_mesh.py imports:
- `_version` - Version information
- `image_processor` - Image processing
- `depth_to_3d` - Depth processing
- `edge_detection` - Edge detection
- `mesh_generator` - Mesh generation
- `MeshTools.viewport_3d` - 3D visualization
- `qt_extensions` - Custom Qt widgets
- PyQt6 modules - GUI framework
- OpenCV (cv2) - Image operations
- NumPy - Array operations
- Open3D - 3D operations

#### depth_to_3d.py imports:
- PyTorch - Deep learning models
- Transformers - AI models
- `MeshTools.mesh_tools` - Mesh operations
- `smoothing_depth_map_utils` - Depth smoothing
- `spinner` - Progress indication

## Architecture Patterns

### Design Patterns Used

#### Model-View Pattern
- **Model**: Image data, depth maps, 3D meshes
- **View**: Qt UI components, 3D viewport
- **Implementation**: Separation between data processing and UI

#### Factory Pattern
- **Implementation**: `depth_to_3d.py:load_model()`
- **Purpose**: Dynamic model loading based on type

#### Observer Pattern
- **Implementation**: Qt signals and slots
- **Purpose**: Event-driven UI updates

#### Strategy Pattern
- **Implementation**: Multiple depth models and smoothing methods
- **Purpose**: Interchangeable algorithms

### Architectural Style
- **Style**: Modular Pipeline Architecture
- **Rationale**: Clear separation of processing stages
- **Benefits**: Easy to extend and maintain

### Processing Pipeline Stages
1. **Input Stage**: Image loading and validation
2. **Preprocessing**: Grayscale, edge detection, color operations
3. **Depth Estimation**: AI model inference
4. **Smoothing**: Depth map refinement
5. **Mesh Generation**: 3D reconstruction
6. **Visualization**: Real-time 3D display
7. **Export**: File output in various formats

## Development Guidelines

### Adding New Features

#### Adding a New Depth Model
1. Add model name to `model_names` dict in `depth_to_3d.py`
2. Implement loading logic in `load_model()` method
3. Add UI dropdown option in `edge_mesh.py`
4. Test with sample images

#### Adding Image Filters
1. Implement filter in `image_processor.py`
2. Add UI controls in `edge_mesh.py:_init_ui()`
3. Connect to `update_preview()` pipeline
4. Update configuration save/load methods

### Code Standards

#### Naming Conventions
- Classes: PascalCase (e.g., `MainWindowImageProcessing`)
- Functions/Methods: snake_case (e.g., `process_image`)
- Constants: UPPER_SNAKE_CASE
- Private methods: Leading underscore (e.g., `_init_ui`)

#### File Organization
- One main class per file
- Related utilities grouped in modules
- Separate UI from processing logic

#### Documentation Standards
- Docstrings for all public methods
- Type hints where applicable
- Comments for complex algorithms

## Performance Considerations

### Optimization Strategies

#### Image Processing
- Resolution limiting (default: 700px)
- Cached processed images
- Efficient NumPy operations
- GPU acceleration when available

#### Depth Estimation
- Model caching after first load
- Batch processing support
- Configurable resolution
- Device selection (CPU/GPU)

#### 3D Rendering
- Level-of-detail for large meshes
- Viewport culling
- Efficient mesh formats
- Texture optimization

### Known Performance Considerations
- Large images (>4K): Use resolution limiting
- Complex meshes: Enable mesh simplification
- Memory usage: Clear unused meshes
- GPU memory: Monitor VRAM usage

### Recommended Settings
- Default resolution: 700px for real-time preview
- Depth models: DepthAnythingV2 for quality/speed balance
- Smoothing: Anisotropic for best quality
- Export format: PLY for compatibility, STL for 3D printing