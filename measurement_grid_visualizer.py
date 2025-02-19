import numpy as np
import open3d as o3d

from color_transition_gradient_generator import ColorTransition


class MeasurementGrid:
    def __init__(self, trimesh):
        """
        Initialize the MeasurementGrid class with a TriMesh instance.

        :param trimesh: TriMesh instance representing the target mesh.
        """
        self.mesh = trimesh
        self.colors = ColorTransition("red", "orange", "yellow").generate_gradient(21)

    def create_measurement_grid(self):
        """
        Create a measurement grid using Open3D's LineSet to overlay on the viewport.

        :return: An Open3D LineSet object representing the measurement grid.
        """
        if self.mesh is None:
            print("No mesh loaded to create a measurement grid.")
            return None

        # Get the bounding box of the mesh
        bounding_box = self.mesh.get_axis_aligned_bounding_box()
        min_bound = bounding_box.get_min_bound()
        max_bound = bounding_box.get_max_bound()

        # Dimensions of the mesh
        width = max_bound[0] - min_bound[0]  # x-axis
        height = max_bound[1] - min_bound[1]  # y-axis
        depth = max_bound[2] - min_bound[2]  # z-axis

        # Define grid spacing (step size)
        spacing = depth * 0.05  # 5% of the depth

        # Grid vertices and edges
        vertices = []
        edges = []
        line_colors = []  # To store colors for each line

        # Generate grid lines
        num_intervals = 21  # 21 intervals for 5% steps (0 to 100%)
        for i in range(num_intervals):
            z = min_bound[2] + i * spacing  # Calculate z-level

            # Horizontal line along the x-axis, at fixed y and z
            vertices.append([min_bound[0], min_bound[1], z])  # Start point
            vertices.append([max_bound[0], min_bound[1], z])  # End point
            edges.append([len(vertices) - 2, len(vertices) - 1])  # Connect start and end
            line_colors.append(self.colors[i])  # Assign corresponding color

            # Vertical line along the y-axis, at fixed x and z
            vertices.append([min_bound[0], min_bound[1], z])  # Start point
            vertices.append([min_bound[0], max_bound[1], z])  # End point
            edges.append([len(vertices) - 2, len(vertices) - 1])  # Connect start and end
            line_colors.append(self.colors[i])  # Assign corresponding color

        # Convert vertices and edges to numpy arrays
        vertices = np.array(vertices, dtype=np.float64)
        edges = np.array(edges, dtype=np.int32)

        # Create and configure LineSet for the grid
        grid_lines = o3d.geometry.LineSet()
        grid_lines.points = o3d.utility.Vector3dVector(vertices)
        grid_lines.lines = o3d.utility.Vector2iVector(edges)

        # Assign colors to lines
        grid_lines.colors = o3d.utility.Vector3dVector(line_colors)

        return grid_lines

    def overlay_on_mesh(self, viewer=None):
        """
        Adds the measurement grid to the mesh and visualizes it.

        :param viewer: Open3D visualization environment to render the mesh and grid. If none, a new window is opened.
        """
        if self.grid_lines is None:
            self.create_measurement_grid()

        if viewer is None:
            # Create a new visualization environment
            viewer = o3d.visualization.Visualizer()
            viewer.create_window()

            # Add the mesh and grid to the viewer
            viewer.add_geometry(self.mesh)
            viewer.add_geometry(self.grid_lines)

            # Run the viewer
            viewer.run()
            viewer.destroy_window()
        else:
            # If a viewer is provided, add the grid overlay
            viewer.add_geometry(self.grid_lines)
