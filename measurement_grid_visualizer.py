import traceback

import numpy as np
import open3d as o3d

import text_3d

from color_transition_gradient_generator import ColorTransition

rainbow_colors = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]

class MeasurementGrid:
    def __init__(self, trimesh, colors=None):
        """
        Initialize the MeasurementGrid class with a TriMesh instance.

        :param trimesh: TriMesh instance representing the target mesh.
        """
        self.mesh = trimesh
        if not colors:
            colors = rainbow_colors

        self.colors = ColorTransition(*colors).generate_gradient(21)

    def _create_grid_with_labels(self, custom_labels=None):
        """
        Create the grid lines with associated labels positioned at the endpoints.

        :param custom_labels: Optional list of 21 labels to use instead of the default percentage labels.
        :return: List of grid line vertices, edges, colors, and text labels as Open3D geometries.
        """
        if self.mesh is None:
            print("No mesh loaded to create a measurement grid.")
            return None

        try:
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

            # Grid vertices, edges, line colors, and labels container
            vertices = []
            edges = []
            line_colors = []  # To store colors for each line
            labels = []  # Store label geometries (text objects)

            # Use custom labels if provided, otherwise generate default percentage labels
            if custom_labels:
                if len(custom_labels) != 21:
                    raise ValueError("custom_labels must be a list of exactly 21 elements.")
                if not isinstance(custom_labels[0], str):
                    if type(custom_labels[0]) in [int, float]:
                        custom_labels = [f"{int(label)}" for label in custom_labels]
                    else:
                        raise ValueError("custom_labels must be a list of strings or numbers. Got list of: ", type(custom_labels[0]))
                label_texts = custom_labels
            else:
                label_texts = [f"{i * 5}%" for i in range(21)]  # Default labels (0, 5, 10, ..., 100)

            # Generate grid lines and labels
            num_intervals = 21  # 21 intervals for 5% steps (0 to 100%)
            for i, label_text in enumerate(label_texts):
                z = min_bound[2] + i * spacing  # Calculate z-level

                # Horizontal line along the x-axis, at fixed y and z
                start_x = [min_bound[0], min_bound[1], z]
                end_x = [max_bound[0], min_bound[1], z]
                vertices.extend([start_x, end_x])
                edges.append([len(vertices) - 2, len(vertices) - 1])  # Connect start and end
                line_colors.append(self.colors[i])  # Assign corresponding color

                # Add text label at the end of the horizontal line
                text_label_x = text_3d.create_text_3d(label_text, position=end_x, color=self.colors[i], height=20, depth=2)
                labels.append(text_label_x)

                # Vertical line along the y-axis, at fixed x and z
                start_y = [min_bound[0], min_bound[1], z]
                end_y = [min_bound[0], max_bound[1], z]
                vertices.extend([start_y, end_y])
                edges.append([len(vertices) - 2, len(vertices) - 1])  # Connect start and end
                line_colors.append(self.colors[i])  # Assign corresponding color

                # Add text label at the end of the vertical line
                text_label_y = text_3d.create_text_3d(label_text, position=end_y, color=self.colors[i], height=20, depth=2)
                labels.append(text_label_y)

            # Convert vertices and edges to numpy arrays
            vertices = np.array(vertices, dtype=np.float64)
            edges = np.array(edges, dtype=np.int32)

            return vertices, edges, line_colors, labels
        except Exception as e:
            print(f"Error in creating the measurement grid: {traceback.format_exc()}")
            return None

    def create_measurement_grid(self, labels=None):
        """
        Create a list of Open3D geometries (LineSet and text labels)
        to be used for the measurement grid.

        :return: A list of Open3D geometries, including the grid lines and text labels.
        """
        if self.mesh is None:
            print("No mesh loaded to create a measurement grid.")
            return []

        # Generate grid components
        vertices, edges, line_colors, labels = self._create_grid_with_labels(labels)

        # Create and configure LineSet for the grid
        grid_lines = o3d.geometry.LineSet()
        grid_lines.points = o3d.utility.Vector3dVector(vertices)
        grid_lines.lines = o3d.utility.Vector2iVector(edges)
        grid_lines.colors = o3d.utility.Vector3dVector(line_colors)

        # Collect all geometries (LineSet + text labels) into a list
        geometries = [grid_lines]  # Start with the grid lines
        geometries.extend(labels)  # Add all the text labels

        return geometries
