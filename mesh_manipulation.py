import numpy as np
import open3d as o3d


class MeshManipulation:
    def __init__(self, viewport, mesh):
        """
        Initialize the MeshManipulation with a given 3D viewport.

        :param viewport: A 3D viewport controlling the display of the mesh.
        :param mesh: The mesh geometry to manipulate.
        """
        self.viewport = viewport
        self.mesh = mesh
        self.mesh_center = self.mesh.get_center() if self.mesh else None  # Cache the center for performance

    def move_object(self, dx, dy, dz=0.0, zoom_factor=1.0):
        """
        Move the mesh within the 3D viewport using Open3D utilities.

        :param dx: Amount to move along the x-axis (in world units).
        :param dy: Amount to move along the y-axis (in world units).
        :param dz: Amount to move along the z-axis (in world units), default is 0.
        :param zoom_factor: Scaling factor to zoom the viewport.
        """
        if self.mesh is None:
            print("Error: No mesh is loaded!")
            return

        # Combine translation and scaling into a single transformation matrix
        translation_vector = np.array([dx, dy, dz])

        # Compute scaling only if zoom_factor is not 1.0
        if zoom_factor != 1.0:
            # Update cached center if required
            if self.mesh_center is None:
                self.mesh_center = self.mesh.get_center()

            translation_matrix = np.eye(4)
            translation_matrix[:3, 3] = translation_vector

            scaling_matrix = np.eye(4)
            scaling_matrix[:3, :3] *= zoom_factor
            scaling_matrix[:3, 3] = self.mesh_center * (1 - zoom_factor)

            # Combine transformations
            transformation = scaling_matrix @ translation_matrix
            self.mesh.transform(transformation)
        else:
            # Only apply translation
            self.mesh.translate(translation_vector, relative=True)

        # Update the viewport once
        self.update_viewport()
        print(f"Moved object by dx: {dx}, dy: {dy}, dz: {dz} with zoom factor: {zoom_factor}.")

    def rotate_object(self, angle_degrees, counter_clockwise=False):
        """
        Rotate the mesh within the 3D viewport.

        :param angle_degrees: Angle to rotate the mesh by, in degrees.
        :param counter_clockwise: Direction of rotation (default: True).
        """
        if self.mesh is None:
            print("Error: No mesh is loaded!")
            return

        # Convert the angle to radians and set the correct direction
        angle_radians = np.radians(angle_degrees)
        if not counter_clockwise:
            angle_radians *= -1

        # Cache center if not already cached
        if self.mesh_center is None:
            self.mesh_center = self.mesh.get_center()

        # Create a rotation matrix for the Y axis
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_radians, 0])
        # rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_degrees, 0])

        # Apply the rotation to the mesh
        self.mesh.rotate(rotation_matrix, center=self.mesh_center)

        # Update the viewport once
        self.update_viewport()
        print(f"Rotated object by {angle_degrees} degrees {'counter-clockwise' if counter_clockwise else 'clockwise'}.")

    def update_viewport(self):
        """
        Refreshes the viewport display with the current state of the mesh.
        """
        self.viewport.clear_geometries()
        self.viewport.add_geometry(self.mesh)
