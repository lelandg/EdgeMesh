# User Documentation for the EdgeMesh Application
# 3D Text Generator

This tool creates 3D text meshes using Open3D and PyVista.

## Installation

There are two ways to install the required dependencies:

### Option 1: Automated Installation

Run the provided installation script:

```bash
python install_requirements.py
```

This will install all dependencies, including Open3D 0.19.0 which is compatible with Python 3.12.

### Option 2: Manual Installation

If the automated installation doesn't work, try these steps:

1. Update pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Install core dependencies:
   ```bash
   python -m pip install numpy matplotlib
   ```

3. Install Open3D specifically:
   ```bash
   python -m pip install open3d==0.19.0 --no-cache-dir
   ```

4. Install remaining requirements:
   ```bash
   python -m pip install -r MeshTools/requirements.txt
   ```

## Troubleshooting

### Python Version Compatibility

**Important**: Open3D 0.19.0 is **not compatible** with Python 3.13 or newer versions.

If you're using Python 3.13+, you'll encounter an error importing Open3D. This is because Open3D 0.19.0 was developed before Python 3.13 was released and hasn't been updated to support it yet.

### Resolution Options:

1. **Downgrade to Python 3.12 (Recommended)**:
   - Install Python 3.12 from [python.org](https://www.python.org/downloads/)
   - Reinstall dependencies with `python install_requirements.py`

2. **Use a Python 3.12 Virtual Environment**:
   ```bash
   # Install Python 3.12 first
   # Then create a virtual environment
   python3.12 -m venv venv

   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate

   # Install requirements
   python install_requirements.py
   ```

3. **Wait for Open3D Update**: Future versions of Open3D may add support for Python 3.13+.

### General Troubleshooting:

1. Try running the `install_open3d.py` script which will detect your Python version and provide guidance:
   ```bash
   python install_open3d.py
   ```

2. If you're using Python 3.12 or earlier and still have issues, try:
   ```bash
   python -m pip install open3d==0.19.0 --force-reinstall --no-cache-dir
   ```

## Usage

To create a 3D text mesh:

```bash
python MeshTools/text_3d.py --text "Your Text" --height 100 --depth 20
```

This will create a 3D mesh of your text, save it as a PLY file, and display it in a visualization window.
## Introduction
This application provides advanced functionality for processing images, performing edge detection, generating 3D meshes from depth maps, and rendering them in a 3D viewport. Primarily designed for professionals working in computer vision, 3D modeling, and similar fields, the application offers an interactive GUI to manipulate features like edge detection, smoothing methods, dynamic depth adjustments, and more.

To install, you need pip. Use:
`pip install -r requirements.txt`

---

## Features
Note that some of these features are not exposed in the GUI, but are available in the code. Like Edge Clustering, Shape Analysis, Surface Partitioning, and Extrusion Projection. These may be implemented later.

### 1. **Image Processing**
- **Image Loading**: Users can load images into the application for processing.
- **Edge Detection**: Tools for detecting edges in grayscale or color images. Adjustable parameters include:
  - Low and high thresholds.
  - Edge thickness.
  - Option to project on the original image or a clean canvas.
- **Grayscale Conversion**: Option to process images in grayscale for better control over data visualization.

### 2. **Depth Map Processing**
- **Depth-to-3D Conversion**:
  - Converts an image with depth information into a 3D mesh.
  - Model options to suit various use cases (e.g., `DepthTo3D` class provides customizable depth translation).
- **Depth Smoothing Techniques**:
  - Gaussian Smoothing.
  - Bilateral Smoothing.
  - Median Smoothing.
  - Anisotropic Diffusion.
- **Dynamic Depth Adjustments**:
  - Users can dynamically toggle depth adjustments for better representation of 3D objects.

### 3. **3D Mesh Generation**
- **Processing Mesh**: The application processes image data to generate 3D models using a variety of techniques. Features include:
  - **Edge Clustering**: Analyze and cluster detected edges.
  - **Shape Analysis**: Extract geometrical primitives such as polygons and ellipses.
  - **Surface Partitioning**: Segment surfaces into regions for reconstruction.
  - **Extrusion Projection**: Handle complex 3D reconstructions from contours.
- **Visualization**:
  - Options to visualize edge clustering, depth map, partitioning, and extruded edges.

### 4. **Image Manipulation and Visualization**
- Resize and sharpen processed images for best results in edge and mesh processing.
- If you select "Remove Background" and all four corners of the image are the same color, the background is removed automatically.
- Toggle between processed and unprocessed outputs for real-time comparisons.

### 5. **3D Viewport**
- **Interactive Viewport**: Display generated 3D meshes in an open3d-based environment.
- **Mesh Manipulation**:
  - Zooming and panning to adjust the view.
  - Center meshes within the viewport.
  - Export meshes as `.obj` or `.stl` files for external use.
- **Background Color Customization**: Change the background color of the viewport automatically when background is detected.

### 6. **Edge Detection**
- Utilize the `detect_and_project_edges` function for fine-tuned edge detection.
- Adjustable parameters include:
  - Thresholds for edge sensitivity.
  - Edge thickness.
  - Projection options for detected edges.

### 7. **Configuration-Based Customization**
- Configuration file support for maintaining user preferences and settings.
- Save and load custom configurations for workflow consistency.

### 8. **Exporting and Saving**
- Export processed images or meshes in multiple formats for further use in modeling software or analysis.
- Save states within the application to retain project-specific information.

---

## How to Use the Application

### Step 1: Load an Image
1. Start by loading the image using the **Load Image** button in the GUI.
2. Select the image path, and it will appear in the workspace for further operations.

### Step 2: Adjust Image Settings (Optional)
- Set parameters for smoothness, edge detection sensitivity, resolution, and grayscale conversion.
- It's important to realize that you'll be processing the image on the left, **Unless** you select **Use Processed Image**. Then you'll be processing the image on the right.

### Step 3: Perform Edge Detection
1. Enable the **Edge Detection Checkbox** to perform edge detection on the image.
2. Adjust thresholds, thickness, and select options to project the edges on the original image or a plain background.

### Step 4: Depth Map Adjustments (Optional)
1. Utilize depth adjustment inputs to modify how the 3D mesh should be shaped from the depth map.
2. Toggle the **Dynamic Depth** to create only the "front half" of the mesh. This is useful for 3D printing. Normally, the front half (that we can see) is mirrored to the back half.


### Step 5: Generate 3D Mesh
1. Click the **Generate 3D Mesh** button to process the loaded image into a 3D mesh.
2. Visualize intermediate steps like depth maps, edge detection, and surface partitioning if visualization options are enabled.

### Step 6: Render Mesh in 3D Viewport
1. The 3D mesh is automatically loaded into the viewport. Use pan, zoom, and background customization to adjust the view.
2. Export the mesh in `.obj` or `.stl` formats using the corresponding buttons.

### Step 7: Save and Export
- Save your project or export the resulting mesh/image for use in other tools or workflows.

### Caveats
- When **Use Processed Image** is enabled, and **Grayscale Mode** or **Edge Detection** are also selected, this will affect the edge detection and depth map generation process. 

---

## Key User Interface Components

### Buttons
- **Load Image**: Select and load images for processing.
- **Generate Mesh**: Process the currently loaded image into a 3D mesh.
- **Export Mesh**: Save the final mesh in the desired format.

### Toggles and Checkboxes
- **Grayscale**: Toggle grayscale conversion for the image.
- **Edge Detection**: Enable edge detection and configuration.
- **Dynamic Depth**: Dynamically adjust depth values.

### Inputs
- **Thresholds and Sensitivity Sliders**: Adjust edge detection parameters such as sensitivity and thresholds.
- **Resolution**: Set the resolution of the 3D mesh.
- **Depth Input**: Modify the amount of depth applied.

### Viewport
- **Mesh Viewer**: A real-time 3D viewer that allows for mesh inspection.
- **Pan/Zoom Controls**: Interactive manipulation of the 3D viewport.

---

## Notes on Additional Features
- The application is extensible, allowing developers to enhance existing methods or add new features by modifying/integrating additional utility files.
- Debugging tools and logging are implemented for troubleshooting or refining operations.

---

## Advanced Features for Developers

The application comprises multiple modular scripts. Key components include:

1. **Image Preprocessing**:
   - Smoothing techniques provided in `smoothing_depth_map_utils.py`.
   - Versatile edge detection using `edge_detection.py`.

2. **3D Processing**:
   - Depth processing and 3D mesh reconstruction are implemented within the `depth_to_3d.py`.

3. **Custom Rendering**:
   - Open3D-based viewport handling is defined in `viewport_3d.py`.

4. **Mesh Generation**:
   - High-level processing and mesh generation logic are encapsulated in `mesh_generator.py`.

These scripts employ libraries like OpenCV and NumPy for computational efficiency.

---

## Conclusion

This application combines cutting-edge computer vision and 3D rendering techniques to provide a robust tool for mesh generation from depth maps. With its extensive customization options and intuitive GUI, it is suitable for both professionals and enthusiasts in the fields of 3D modeling and image analysis.
- *A bonus utility is **viewport_3d.py**.* It can be used standalone or imported into your code.
- It now has export_mesh_as_obj() and export_mesh_as_stl() methods, and a SUPPORTED_EXTENSIONS list, so you can be sure of your file type(s) before import/export!
- It takes a list of mesh files as arguments and opens a viewport for each valid file. (One at a time.)
- Be careful when loading a lot of files! *Especially* if they're large. It will work just fine, but you must wait for each window to close before the next one opens. For this reason, running from a Python IDE like PyCharm is very recommended. From there, you can press <Ctrl>+<F2> to stop the program. (I'm sure other IDEs have similar features.)
- Large files take a few seconds to minutes to load, depending on your system specs.
- *Or* pass it file names with full path and/or liberal wildcards. (I tested multiple asterisks, for example.)
- *Or* use it from your code to display a mesh.

Notes: "Invert Colors" is intended for edge detection. (Should it just be automatic?) IDK, because it's fun
          to have the option. It's like a filter. If you enable it, colors become their opposite, or complementary
          color. This may be useful sometimes for a grayscale image. Or, if you have a picture of a negative! 
          So I think it's fun _and_ useful.
        **Warning** Use your new inverted colors with care.  
