import argparse
import os
import random

# Function to create 3D text mesh
import numpy as np
import open3d as o3d
import pyvista as pv


def create_text_3d(text, position=[0, 0, 0], depth=10, height=100, color=[0.5, 0.5, 0.5]):
    """
    Function to create a 3D text mesh and move it such that its bottom-left corner is at the specified position.

    Parameters:
        text (str): The text string to generate in 3D.
        position (list): The desired position of the bottom-left corner [x, y, z].
        depth (float): Depth of the 3D text extrusion.
        height (float): Height of the 3D text.
        color (list): RGB color as a list (default is gray, values must be between 0 and 1).

    Returns:
        o3d.geometry.TriangleMesh: A transformed 3D text mesh with the specified color and position.
    """
    # Create the PyVista 3D text mesh
    pv_mesh = pv.Text3D(text, depth=depth, height=height)

    # Extract vertices and faces
    vertices = np.asarray(pv_mesh.points)  # PyVista stores points as a NumPy array
    faces = np.asarray(pv_mesh.faces).reshape(-1, 4)[:, 1:]  # Ignore the first element in each face (n_verts)

    # Create Open3D TriangleMesh object
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Ensure color values are clipped to the valid range [0, 1]
    color = tuple(max(0.0, min(1.0, c)) for c in color)
    o3d_mesh.paint_uniform_color(color)

    # Get bounding box to calculate the offset for transformation
    bounding_box = o3d_mesh.get_axis_aligned_bounding_box()
    min_bound = np.array(bounding_box.get_min_bound())  # Bottom-left corner of the bounding box
    offset = np.array(position) - min_bound  # Vector to move the bottom-left corner to the specified position

    # Apply translation to move the mesh
    o3d_mesh.translate(offset)

    return o3d_mesh


# Main function to use command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Create 3D text using command-line inputs.")

    # Positional argument for the 3D text to be created
    parser.add_argument("--text", type=str, help="The text string to generate in 3D.")

    # Optional arguments for customization
    parser.add_argument("--depth", type=float, default=20,
                        help="Depth of the 3D text extrusion. Default is 20.")
    parser.add_argument("--height", type=float, default=100.0,
                        help="Height of the 3D text. Default is 100.0.")

    args = parser.parse_args()
    text = "3D Text!"
    if args.text:
        text = args.text

    # Create the 3D text mesh
    print(f"Creating 3D text: '{text}' with height {args.height} and depth {args.depth}")
    r = random.random()
    g = random.random()
    b = random.random()
    text_mesh = create_text_3d(text, height=args.height, depth=args.depth, color=[r, g, b])

    # Save the 3D text mesh to a file named "{args.text}.ply"
    output_filename = f"{text}.ply"
    # print(f"Saving 3D text mesh to file: {output_filename}")
    o3d.io.write_triangle_mesh(output_filename, text_mesh)
    if os.path.exists(output_filename):
        print(f"File saved successfully: {output_filename}")

    # Display the 3D text in an Open3D visualization window
    o3d.visualization.draw_geometries([text_mesh])


if __name__ == "__main__":
    main()
