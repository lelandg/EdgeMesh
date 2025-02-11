# EdgeMesh
Features: 
* Depth map generation via select (implemented) methods via torch, etc. 
* Then 3D mesh generation from the depth map.
* Additional utilities using opencv for edge detection in a GUI. 

Version 0.3.2 Reopens the 3D viewport when closed, and clears existing geometry before loading a new mesh. 

Version 0.3.1 Adds full color .PLY export support and a new depth map smoothing method.

              viewport_3d.py now has export_mesh_as_obj and export_mesh_as_stl methods, and a SUPPORTED_EXTENSIONS list.
              
                              It takes a list of mesh files as arguments and opens a viewport for each valid file. 
                              
                              (One at a time.) Be careful with many files! *Especially* if they're large. 
                              
                              Large files take a few seconds to minutes to load, depending on your system specs.
              
              Fixes 3D viewport update issue. (Now controlled by mouse.)

Version 0.3.0 adds depth map smoothing options and anisotropic diffusion.

Version 0.2.0 adds depth map generation from shading and light cues.

Version 0.1.0 adds edge detection and 3D mesh generation from edges.
