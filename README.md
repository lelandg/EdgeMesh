# User Documentation for the EdgeMesh Application

## Introduction
This application provides advanced functionality for processing images, performing edge detection, generating 3D meshes from depth maps, and rendering them in a 3D viewport. Primarily designed for professionals working in computer vision, 3D modeling, and similar fields, the application offers an interactive GUI to manipulate features like edge detection, smoothing methods, dynamic depth adjustments, and more.

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
- **Interactive Viewport**: Display generated 3D meshes in an OpenGL-based environment.
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
   - OpenGL-based viewport handling is defined in `viewport_3d.py`.

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


Version history:
* 0.5.13:
    * Big UI update. Everything has tooltips now. Labels stick to associated widgets.
    * Fixed image scaling issues at startup.
* 0.5.12:
    * Add hotkeys to buttons.
    * Fix for direction of faces for flat back mesh.
    * Add Images/example.png for testing. Set as default image if config.ini is missing.
    * Add run.bat and run.sh to easily run the program from a command line. The .sh assumes python3 is in your path.
* 0.5.11:
    * Much improved "flat back" functionality.
* 0.5.10:
    * Added "pick color" to let you pick the background color from the image. (Or from anything on your desktop!)
    * After you pick it, the background will not be detected until you click "Clear Color"
    * Improved UI layout. Saves a few settings I missed before.
    * Fixes issues with windows size, position and state saving/loading.
* 0.5.9:
    * Started reworking UI. Window now can be sized smaller and still access all widgets.
    * Saves/restores window size and position in config.ini.
    * Fixed window name of 3D viewport to include the file name if given.
* 0.5.8:
    * Added _version to store the version so the main file is not always updated.
    * Removed __last_updated__ from the main file since I didn't update it.
* 0.5.7:
    * Added rotate left and right with arrow keys in 3D viewport. It's slow, but it works. Mouse is responsive.
    * Note the new MeshManipulation.move() function is untested, but it's not used, yet. :)
* 0.5.6:
    * Fix bug I just created in depth_to_3d.py
* 0.5.5:
    * Added save/load configuration for all user inputs.
    * Added "Reset Defaults" button to reset all user inputs to default values.
    * Yet more color fixes. (I hope this is the last time I have to say that.) <-- Suggested by CoPilot! (The part in parentheses was! Haha!)
* 0.5.4:
    * Added a 'D' toggle to the 'G' grid toggle. It toggles between percentages and depth values. 
      Sides of mesh are coming out colored based on the front/back portion. Need to look into that, but sides are flat.
* 0.5.3:
    * Move history to ReadMe.md
    * Add mesh_gradient_colorizer to shade the mesh based on depth, given a range of colors. Press 'C' for "Colorize".
    * Window can now be sized much smaller. (Still need to rework UI to organize widgets better!)
    * Added the FlowLayout class to implement this. (It's a start.)
    * Disabled tensor OneDNN option. It may be slower, but it's more accurate. And it's not much slower.
* 0.5.2:
    * Updates: Grid lines are now rainbow colored and numbered at each end. 
    * Added text_3d.py for 3D text generation in color, at position.
* 0.5.1:
    * Adds: G key now toggles a "measuring grid" on and off. It's just color lines on the edge of the mesh, right now.
      This should come in handy to figure out what percentage depth drop to use!
* 0.5.0:
    * Fixes:
        * Grayscale mode works without edge detection.
        * Only flip the image before processing when DepthAnythingV2 is used. (Not sure why it reverses depth map?)
        * ThreeDViewport class no longer binds some keys, mostly so W for wireframe works.
    * Adds:
        * Tweaks to depth map to allow modifications.
* 0.4.3:
    * UI improvements. Added tooltips to all input fields and buttons.
    * Fixed the depth removal by percentage. 
* 0.4.2:
    * Corrected input validation for depth to allow float.
* 0.4.1:
    * Implements depth amount. This is just a simple multiplier for the depth map. (Still does not use "max depth".)
    * Changed default depth amount to 1.0.
    * Fixes a bug where the processed image was not displayed correctly in the preview at startup.
    * Fix problems when using grayscale mode.
    * Tweak output file name to include more information.
* 0.4.0
    * Fixes several issues listed as "Known Issues" in the previous commit. Good checkpoint.
* 0.3.7 
    * Fixes: 
        * Color issues (again). 
    * Adds: 
        * "Project on Original" option. Works with and without grayscale enabled.
    * Known Issues:
        * Preview window may not update correctly. Change options a time or two to "fix it". 
        * Preview image scaling is not perfect at startup. Resize the window to fix it. (For now.)
* 0.3.6 Adds:
    * Add method to mirror and stitch the generated mesh. Not perfect, but it's a start. 
    * You can still use the old method (no mirrored back) by checking "Dynamic Depth". That's better for 3D printing.
    * Adds spinner.py for fun.
* 0.3.5 Adds: 
    * Includes "date_time" at the end output file name, so you'll always get a new file! 
                    *** Clean your <output folder> as needed. ***
    " Use Processed Image" option to create 3D mesh from current results, i.e., the image on the current right in GUI.
    " Dynamic Depth" option to make the mesh dynamically shaped on the back, approximating the front.
    " Edge Detection" option to enable or disable edge detection.
    " Grayscale" option to enable or disable grayscale mode.
    
    *Notes: "Invert Colors" is intended for edge detection. Should it just be automatic? IDK, because it's fun
              to have the option. It's like a filter. If you enable it, colors become their opposite, or complementary
              color. This may be useful sometimes for a grayscale image. Or, if you have a picture of a negative! 
              So I think it's fun _and_ useful.
            ***Warning*** Use your new inverted colors with care.  
    Fixes:
        * Background color for 3D viewport.
        * Supports 0 for resolution input. When you use this, the maximum of the image's width or height is used.
    * Updated:
        * ReadMe.md    
* 0.3.4 Adds background color as background color for 3D viewport. For more expected user experience.
              Note this background color is not saved in the exported mesh. It's simply carried over from
              the original image to the Open3D viewport.
* 0.3.3 Adds automatic background removal. This only happens when all four corners are a solid color.
* 0.3.2 Reopens the 3D viewport when closed, and clears existing geometry before loading a new mesh. 
* 0.3.1 Adds full color .PLY export support and a new depth map smoothing method.
              viewport_3d.py now has export_mesh_as_obj and export_mesh_as_stl methods, and a SUPPORTED_EXTENSIONS list.
                             It takes a list of mesh files as arguments and opens a viewport for each valid file. 
                             (One at a time.) Be careful with many files! *Especially* if they're large. 
                             Large files take a few seconds to minutes to load, depending on your system specs.
              Fixes 3D viewport update issue. (Now controlled by mouse.)
* 0.3.0 adds depth map smoothing options and anisotropic diffusion.
* 0.2.0 adds depth map generation from shading and light cues.
* 0.1.0 adds edge detection and 3D mesh generation from edges.


    Notes: "Invert Colors" is intended for edge detection. Should it just be automatic? IDK, because it's fun
              to have the option. It's like a filter. If you enable it, colors become their opposite, or complementary
              color. This may be useful sometimes for a grayscale image. Or, if you have a picture of a negative! 
              So I think it's fun _and_ useful.
            **Warning** Use your new inverted colors with care.  

*   **WIP 20250213** *Leland Green, Springfield, MO, USA*   
*   *This is a work in progress.*
*   **Please report any issues via GitHub/lelandg/EdgeMesh/issues.**