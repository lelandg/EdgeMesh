import numpy as np
import open3d as o3d


class MeshColorizer:
        @staticmethod
        def apply_gradient_to_mesh(mesh, gradient_colors):
                """
                Apply a gradient of colors to a TriangleMesh from back to front.

                :param mesh: The TriangleMesh object containing vertices and other properties.
                :param gradient_colors: A list of either RGB tuples (0-1 range) or color names (strings) to color the mesh.
                :return: A new TriangleMesh identical to the input, but colored with the gradient.
                """
                # Make a deep copy of the mesh to avoid modifying the original
                colored_mesh = o3d.geometry.TriangleMesh(mesh)

                # Extract Z-coordinates of vertices and determine the range
                z_coords = np.array([vertex[2] for vertex in np.asarray(colored_mesh.vertices)])
                z_min, z_max = z_coords.min(), z_coords.max()

                if z_min == z_max:
                        raise ValueError("All vertices have the same Z-coordinate; gradient cannot be applied.")

                # Normalize Z-coordinates to range [0, 1] for mapping to gradient
                normalized_z = (z_coords - z_min) / (z_max - z_min)

                # Map normalized Z-coordinates to gradient colors
                num_colors = len(gradient_colors)

                # Convert gradient colors to RGB tuples, only calling _color_to_rgb if the value is a string
                gradient_colors_rgb = [
                        MeshColorizer._color_to_rgb(color) if isinstance(color, str) else color
                        for color in gradient_colors
                ]

                vertex_colors = [
                        gradient_colors_rgb[int(np.clip(z * (num_colors - 1), 0, num_colors - 1))]
                        for z in normalized_z
                ]

                # Assign the colors to the mesh
                colored_mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vertex_colors))

                return colored_mesh


        @staticmethod
        def _color_to_rgb(color_name):
                """
                Convert a color name into an RGB tuple (0-1 range).
                Example: 'red' -> (1.0, 0.0, 0.0).

                :param color_name: Name of the color (e.g., 'red', 'blue').
                :return: RGB tuple in the range of 0-1.
                """
                color_map = {
                    "red": (1.0, 0.0, 0.0),
                    "orange": (1.0, 0.5, 0.0),
                    "yellow": (1.0, 1.0, 0.0),
                    "green": (0.0, 1.0, 0.0),
                    "blue": (0.0, 0.0, 1.0),
                    "indigo": (0.3, 0.0, 0.5),
                    "violet": (0.5, 0.0, 1.0),
                }
                return color_map.get(color_name.lower(), (0.0, 0.0, 0.0))  # Default to black if not found
