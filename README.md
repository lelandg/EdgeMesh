# EdgeMesh
Image utilities created using opencv, numpy, scikit, open3d and more. 
Features:
* Edge detection in a GUI with major tweaking. 
* Depth map generation via select (implemented) methods. 
* Then 3D mesh generation from the resulting depth map.

* *A bonus utility is **viewport_3d.py**.* It can be used standalone or imported into your code. 
* It now has export_mesh_as_obj() and export_mesh_as_stl() methods, and a SUPPORTED_EXTENSIONS list, so you can be sure 
of your file type(s) before import/export!
* It takes a list of mesh files as arguments and opens a viewport for each valid file. (One at a time.) 
* Be careful when loading a lot of files! *Especially* if they're large. It will work just fine, but you must wait for each window 
to close before the next one opens. For this reason, running from a Python IDE like PyCharm is very recommended. From
there, you can press <Ctrl>+<F2> to stop the program. (I'm sure other IDEs have similar features.)
* Large files take a few seconds to minutes to load, depending on your system specs.
* *Or* pass it file names with full path and/or liberal wildcards. (I tested multiple asterisks, for example.)
* *Or* use it from your code to display a mesh.


History:
* Version 0.3.6 
  * Fixes:
      * Background color for 3D viewport.
      * Supports 0 for resolution input. When you use this, the maximum of the image's width or height is used.
  * Updated:
    * ReadMe.md
* Version 0.3.5 Adds: 
    *Includes "date_time" at the end output file name, so you'll always get a new file! 
                    ** Clean your <output folder> as needed. **
    * "Use Processed Image" option to create 3D mesh from current results, i.e., the image on the current right in GUI.
    * "Dynamic Depth" option to make the mesh dynamically shaped on the back, approximating the front.
    * "Edge Detection" option to enable or disable edge detection.
    * "Grayscale" option to enable or disable grayscale mode.
        
* Version 0.3.4 Adds background color as background color for 3D viewport. For more expected user experience.
              Note this background color is not saved in the exported mesh. It's simply carried over from
              the original image to the Open3D viewport.
* Version 0.3.3 Adds automatic background removal. This only happens when all four corners are a solid color.
* Version 0.3.2 Reopens the 3D viewport when closed and/or clears existing geometry before loading a new mesh. 
* Version 0.3.1 Adds full color .PLY export support and a new depth map smoothing method.
              viewport_3d.py now has export_mesh_as_obj and export_mesh_as_stl methods, and a SUPPORTED_EXTENSIONS list.
                             It takes a list of mesh files as arguments and opens a viewport for each valid file. 
                             (One at a time.) Be careful with many files! *Especially* if they're large. 
                             Large files take a few seconds to minutes to load, depending on your system specs.
              Fixes 3D viewport update issue. (Now controlled by mouse.)
* Version 0.3.0 adds depth map smoothing options and anisotropic diffusion.
* Version 0.2.0 adds depth map generation from shading and light cues.
                                  Fixes 3D viewport update issue. (Now controlled by mouse.)
* Version 0.1.0 adds edge detection and 3D mesh generation from edges.

    Notes: "Invert Colors" is intended for edge detection. Should it just be automatic? IDK, because it's fun
              to have the option. It's like a filter. If you enable it, colors become their opposite, or complementary
              color. This may be useful sometimes for a grayscale image. Or, if you have a picture of a negative! 
              So I think it's fun _and_ useful.
            **Warning** Use your new inverted colors with care.  

*   **WIP 20250213** *Leland Green, Springfield, MO, USA*   
*   *This is a work in progress.*
*   **Please report any issues via github/lelandg/EdgeMesh/issues.**