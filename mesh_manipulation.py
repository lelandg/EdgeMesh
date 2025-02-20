import numpy as np
import open3d as o3d

class MeshManipulation:

    def __init__(self, viewport, mesh):
        """
        Initialize the MeshManipulation with a given 3D viewport.

        :param viewport: A 3D viewport controlling the display of the mesh
        """
        self.viewport = viewport
        self.mesh = mesh

    def move_object(self, dx, dy, dz=0.0, zoom_factor=1.0):
        """
        Move the mesh within the 3D viewport using Open3D utilities.

        :param dx: Amount to move along the x-axis (in world units)
        :param dy: Amount to move along the y-axis (in world units)
        :param dz: Amount to move along the z-axis (in world units), default is 0
        :param zoom_factor: Scaling factor to zoom the viewport
        """
        if self.mesh is None:
            print("Error: No mesh is loaded!")
            return

        # Translation vector
        translation_vector = np.array([dx, dy, dz])

        # Apply translation to the mesh
        self.mesh.translate(translation_vector, relative=True)

        # If a zoom factor is applied, apply scaling centered at the mesh's center
        if zoom_factor != 1.0:
            center = self.mesh.get_center()
            scaling_matrix = np.eye(4)
            scaling_matrix[:3, :3] *= zoom_factor
            scaling_matrix[:3, 3] = center * (1 - zoom_factor)  # Adjust to scale about the center
            self.mesh.transform(scaling_matrix)

        # Update the viewport
        self.viewport.clear_geometries()
        self.viewport.add_geometry(self.mesh)

        print(f"Moved object by dx: {dx}, dy: {dy}, dz: {dz} with zoom factor: {zoom_factor}.")

    def rotate_object(self, angle_degrees, counter_clockwise=True):
        if self.mesh is None:
            print("Error: No mesh is loaded!")
            return

        # Convert the angle to radians
        angle_radians = np.radians(angle_degrees)
        if not counter_clockwise:
            angle_radians *= -1

        # Create a rotation matrix for the Z axis
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle_radians, 0])

        # Apply the rotation to the mesh
        self.mesh.rotate(rotation_matrix, center=self.mesh.get_center())

        # Clear the viewer and re-add the rotated mesh
        self.viewport.clear_geometries()
        self.viewport.add_geometry(self.mesh)
        print(f"Rotated object by {angle_degrees} degrees {'counter-clockwise' if counter_clockwise else 'clockwise'}.")
