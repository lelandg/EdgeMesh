import glob
import sys
import os
import open3d as o3d

class ThreeDViewport:
    def __init__(self, initial_mesh_file=None, background_color=None):
        """
        Initialize the 3D viewport using Open3D with default parameters
        for camera movement, rotation, and mesh rendering.
        :param initial_mesh_file: Optional initial mesh file to load and display.
        :param background_color: Background color for the viewport as a list [R, G, B].
                                 Uses dark gray [0.2, 0.2, 0.2] if None.
        """
        self.viewer = o3d.visualization.VisualizerWithKeyCallback()
        title = f"3D Viewport - Open3D v{o3d.__version__} - Press 'H' for help - {initial_mesh_file}"
        self.viewer.create_window(title, width=800, height=600, left=800, top=50)

        self.mesh = None  # Placeholder for the loaded 3D mesh
        self.zoom_factor = 1.0  # Default zoom factor
        self.pan_x = 0.0  # Pan translation on x-axis
        self.pan_y = 0.0  # Pan translation on y-axis

        # Set the background color
        if background_color is None:
            background_color = [0.2, 0.2, 0.2]  # Dark gray
        self.background_color = [x / 255.0 for x in background_color]
        self.viewer.get_render_option().background_color = self.background_color

        # Register interaction callbacks
        self._setup_key_callbacks()
        if initial_mesh_file:
            self.load_mesh(initial_mesh_file)

    # def __init__(self, initial_mesh_file=None):
    #     """
    #     Initialize the 3D viewport using Open3D with default parameters
    #     for camera movement, rotation, and mesh rendering.
    #     """
    #     self.viewer = o3d.visualization.VisualizerWithKeyCallback()
    #     title = f"3D Viewport - Open3D v{o3d.__version__} - Press 'H' for help - {initial_mesh_file}"
    #     self.viewer.create_window(title, width=800, height=600, left=800, top=50)
    #
    #     self.mesh = None  # Placeholder for the loaded 3D mesh
    #     self.zoom_factor = 1.0  # Default zoom factor
    #     self.pan_x = 0.0  # Pan translation on x-axis
    #     self.pan_y = 0.0  # Pan translation on y-axis
    #
    #     # Register interaction callbacks
    #     self._setup_key_callbacks()
    #     if initial_mesh_file:
    #         self.load_mesh(initial_mesh_file)

    def _setup_key_callbacks(self):
        """
        Set up key callbacks for panning, zooming, and navigation in the viewport.
        """
        # Panning with arrow keys (WASD for directions)
        self.viewer.register_key_callback(ord("W"), lambda _: self.pan(0, 10))  # Pan up
        self.viewer.register_key_callback(ord("S"), lambda _: self.pan(0, -10))  # Pan down
        self.viewer.register_key_callback(ord("A"), lambda _: self.pan(-10, 0))  # Pan left
        self.viewer.register_key_callback(ord("D"), lambda _: self.pan(10, 0))  # Pan right

        # Zooming with '+' and '-' keys
        self.viewer.register_key_callback(ord("+"), lambda _: self.zoom(10))  # Zoom in
        self.viewer.register_key_callback(ord("-"), lambda _: self.zoom(-10))  # Zoom out

        # Support for Shift + Arrow Keys for zoom
        self.viewer.register_key_callback(ord("U"), lambda _: self.zoom(0.1))  # Zoom in
        self.viewer.register_key_callback(ord("J"), lambda _: self.zoom(-0.1))  # Zoom out

    def clear_geometries(self):
        """
        Clear all geometries currently loaded in the 3D viewer.
        This allows loading a new mesh without accumulating old geometries.
        """
        self.viewer.clear_geometries()
        print("Existing geometries cleared from the viewport.")

    def load_mesh(self, mesh_file):
        """
        Load a new 3D triangular mesh file into the viewport.
        Clears any existing geometry to ensure no duplicates.

        :param mesh_file: Path to the new mesh file (.obj, .stl, .ply, etc.).
        """
        try:
            # Clear existing geometry before loading a new mesh
            self.clear_geometries()

            self.mesh_file = mesh_file
            self.mesh = o3d.io.read_triangle_mesh(mesh_file)
            if self.mesh.is_empty():
                raise ValueError(f"Could not load mesh from {mesh_file}.")

            self.mesh.compute_vertex_normals()

            # Add the new mesh for rendering
            self.viewer.add_geometry(self.mesh)
            self._center_mesh_in_view()
            self.viewer.get_render_option().background_color = self.background_color

            print(f"Mesh loaded: {mesh_file}")
        except Exception as e:
            print(f"Error loading mesh: {e}")

    def _center_mesh_in_view(self):
        """
        Automatically center the mesh in the viewport by adjusting
        the camera's viewpoint and zoom level based on the mesh's bounding box.
        """
        if self.mesh is None:
            return

        bbox = self.mesh.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        # extent = bbox.get_extent()

        ctr = self.viewer.get_view_control()
        ctr.set_lookat(center.tolist())  # Set the camera to look at the mesh's center
        ctr.set_zoom(1.0 / (self.zoom_factor))  # Adjust zoom level based on the mesh size
        ctr.set_front([0.0, 0.0, 1.0])  # Set camera front direction
        ctr.set_up([0.0, 1.0, 0.0])  # Set the camera's up direction

    def pan(self, dx, dy):
        """
        Pan the camera by translating it in the x and y directions.

        :param dx: Translation along the x-axis.
        :param dy: Translation along the y-axis.
        """
        self.pan_x += dx
        self.pan_y += dy
        ctr = self.viewer.get_view_control()
        ctr.translate(dx, dy, 0.0)  # Translation in the x, y plane

    def zoom(self, delta):
        """
        Zoom the camera in or out based on a delta factor.

        :param delta: Positive for zoom in, negative for zoom out.
        """
        self.zoom_factor += delta
        self.zoom_factor = max(0.1, min(self.zoom_factor, 10.0))  # Clamp zoom factor between 0.1 and 10.0
        ctr = self.viewer.get_view_control()
        ctr.set_zoom(1.0 / self.zoom_factor)  # Adjust zoom level

    def run(self):
        """
        Start the Open3D visualization window.
        """
        print(f"3D viewport is running for {self.mesh_file}")
        self.viewer.run()
        self.viewer.destroy_window()

    def export_mesh_as_obj(self, output_file):
        """
        Export the current mesh to a Wavefront OBJ file.

        :param output_file: The path to save the OBJ file.
        """
        if self.mesh is None:
            print("No mesh loaded to export.")
            return
        try:
            o3d.io.write_triangle_mesh(output_file, self.mesh)
            print(f"Successfully exported the mesh to {output_file}")
        except Exception as e:
            print(f"Error exporting mesh to OBJ: {e}")

    def export_mesh_as_stl(self, output_file):
        """
        Export the current mesh to an STL file.

        :param output_file: The path to save the STL file.
        """
        if self.mesh is None:
            print("No mesh loaded to export.")
            return
        try:
            o3d.io.write_triangle_mesh(output_file, self.mesh, write_ascii=True)
            print(f"Successfully exported the mesh to {output_file}")
        except Exception as e:
            print(f"Error exporting mesh to STL: {e}")


if __name__ == "__main__":
    # Supported mesh formats
    SUPPORTED_EXTENSIONS = [".obj", ".ply", ".stl", ".off", ".gltf", ".glb"]

    def get_matching_files(patterns, supported_extensions):
        """Expand wildcard patterns and return a list of matching files with supported extensions."""
        matched_files = []
        for pattern in patterns:
            # Resolve wildcard patterns
            for file in glob.glob(pattern, recursive=True):
                if os.path.isfile(file) and os.path.splitext(file)[1].lower() in supported_extensions:
                    matched_files.append(file)
        return matched_files


    # Check if there are any arguments passed to the script
    if len(sys.argv) > 1:
        # Collect files using wildcards and filter by extensions
        input_patterns = sys.argv[1:]  # Exclude the script name
        valid_files = get_matching_files(input_patterns, SUPPORTED_EXTENSIONS)

        if not valid_files:
            print("No valid mesh files were provided.")
        else:
            # Open a separate viewport for each valid file
            for mesh_file in valid_files:
                print(f"Opening viewport for: {mesh_file}")
                try:
                    viewport = ThreeDViewport(initial_mesh_file=mesh_file)
                    viewport.run()
                except Exception as e:
                    print(f"Error while loading or visualizing {mesh_file}: {e}")
    else:
        fname = "g:/Downloads/UseMe.obj"
        print(f"Usage: python {os.path.basename(__file__)} [path_to_mesh1] [path_to_mesh2] ...")
        print(f"Wildcards are supported for matching multiple files.")
        print("`*.obj`: Matches all `.obj` files in the current directory.")
        print("`models/**/*.stl`: Matches all `.stl` files recursively in the `models` directory.\r\n"
                "Supported mesh formats: ", ", ".join(SUPPORTED_EXTENSIONS))
        print(f"Example: python {os.path.basename(__file__)} *.obj *.ply sample.stl")
        print(f"Using hard-coded mesh file for demonstration: {fname}")
        viewport = ThreeDViewport(initial_mesh_file=fname)
        viewport.run()
