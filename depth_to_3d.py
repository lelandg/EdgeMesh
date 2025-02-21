import argparse
import os
import cv2
from datetime import datetime
import numpy as np
import torch
import trimesh
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import smoothing_depth_map_utils
from spinner import Spinner

import PyQt5.QtGui as QtGui

class DepthTo3D:
    def __init__(self, model_type="dpt"):
        """
        Initialize the depth estimation and mesh generation pipeline.
        :param model_type: "midas" (default) or "dense_depth". Specifies the depth estimation model.
        """
        self.depth_map = None
        self.depth_values = None
        self.depth_labels = None
        self.solid_mesh = None
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self.load_model()
        self.spinner = Spinner(f"{{time}} ")

    model_names = {"DenseDepth": "dense_depth", "MiDaS": "midas", "DPT": "dpt", "LeReS": "leres",
                   "DepthAnythingV2": "depth_anything_v2"}

    # model_lookup = {k: v for k, v in model_names.items()}
    # print (f"model_lookup = {self.model_lookup}")

    def load_model(self):
        """
        Load the depth estimation model.
        :return: model and corresponding transform pipeline
        """
        if self.model_type == "midas":
            # MiDaS model (Microsoft)
            from torch.hub import load
            model = load("intel-isl/MiDaS", "MiDaS_small", pretrained=True).to(self.device).eval()
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.model_type == "dense_depth":
            # DenseDepth model
            from dense_depth_model import DenseDepth
            model = DenseDepth().to(self.device).eval()
            transform = None  # Replace with DenseDepth-specific preprocessing logic if required

        elif self.model_type == "dpt":
            # Dense Prediction Transformer (DPT)
            from torch.hub import load
            model = load("intel-isl/MiDaS", "DPT_Large", pretrained=True).to(self.device).eval()
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        elif self.model_type == "leres":
            # LeReS (Lightweight Estimation)
            from leres_model import LeReS  # Custom import
            model = LeReS().to(self.device).eval()
            transform = None  # Replace with LeReS-specific preprocessing logic if necessary

        elif self.model_type == "depth_anything_v2":
            # Depth Anything V2 implementation
            # from depth_anything_v2.dpt import DepthAnythingV2
            # model = DepthAnythingV2().to(self.device).eval()
            # transform = Compose([
            #     ToTensor(),
            #     Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # ])  # Replace this line with actual DepthAnythingV2-specific preprocessing logic if necessary

            # import torch
            # import numpy as np
            # import requests

            # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            # image = Image.open(requests.get(url, stream=True).raw)
            #
            image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        else:
            raise ValueError(
                f"Unsupported model type. Use one of:\n{', '.join(self.model_names.keys())}.")
        return model, transform

    def estimate_depth(self, image, target_size=(500, 500), flip=False):
        """
        Estimate depth from a single image.
        :param image: Input image (numpy array).
        :param target_size: Tuple defining the target size (width, height) for the output depth map.
        :return: Depth map (numpy array) of the limited size.
        """
        print(f"Estimating depth with target size: {target_size}...")
        img_h, img_w, _ = image.shape  # Use NumPy shape to get height and width
        depth = None

        # Resize to ensure divisibility by 32
        new_h = (img_h + 31) // 32 * 32
        new_w = (img_w + 31) // 32 * 32
        resized_image = cv2.resize(image, (new_w, new_h)) # Resize
        if flip:
            resized_image = cv2.flip(resized_image, 1) # Flip horizontally


        # Convert to tensor
        img_input = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_input).unsqueeze(0).to(self.device) if self.transform else None

        # Predict depth
        with torch.no_grad():
            if self.model_type == "dense_depth" or self.model_type == "leres":
                depth = self.model(img_input)[0]  # Replace with respective model's output logic
            else:
                if self.model_type == "depth_anything_v2":
                    # Depth Anything V2 (local weights)
                    # Prepare image for the model
                    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                    inputs = image_processor(images=image, return_tensors="pt")

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    # Interpolate to original size
                    depth = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(img_h, img_w),  # Match original image dimensions here
                        mode="bicubic",
                        align_corners=False,
                    )

                else:
                    depth = self.model(img_tensor)

        # Resize depth back to original image dimensions
        depth = depth.squeeze().cpu().numpy()
        depth = cv2.resize(depth, (img_w, img_h))  # Match original input image size

        if target_size and target_size != (0,0) and target_size != (img_w, img_h):
            # Limit the depth map size
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize for visualization (optional)
        depth = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        return depth

    def solidify_mesh(self, mesh, depth_offset=-1.0):
        """
        Extend the hollow mesh backwards along the depth axis to make it solid.
        :param mesh: Existing 3D trimesh object.
        :param depth_offset: The offset applied to create the "back" face of the solid model (negative for backward extension).
        :return: A new solidified trimesh object.
        """
        # Extract the original vertices and faces
        original_vertices = mesh.vertices
        original_faces = mesh.faces

        # Create the "back" face vertices by shifting along the z-axis
        back_vertices = original_vertices.copy()
        back_vertices[:, 2] += depth_offset  # Adjust depth axis (z-axis)

        # Combine original vertices and back vertices
        combined_vertices = np.vstack([original_vertices, back_vertices])

        # Create faces for the back surface
        num_vertices = len(original_vertices)
        back_faces = original_faces + num_vertices  # Shift indices for the back faces

        # Create side faces to connect the front and back vertices
        side_faces = []
        for face in original_faces:
            for i in range(3):
                # Get the current edge (start, end)
                start = face[i]
                end = face[(i + 1) % 3]

                # Create two faces to cover the side
                side_faces.append([start, end, end + num_vertices])
                side_faces.append([start, end + num_vertices, start + num_vertices])

        side_faces = np.array(side_faces)

        # Combine all faces: front, back, and side
        combined_faces = np.vstack([original_faces, back_faces, side_faces])

        # Create a new mesh with the combined vertices and faces
        solid_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)

        return solid_mesh

    def solidify_mesh_with_flat_back(self, mesh, flat_back_depth=-1.0):
        """
        Solidify the mesh by making the back side flat while preserving vertex colors.
        :param mesh: Existing 3D trimesh object with vertex colors.
        :param flat_back_depth: The depth value for the flat back surface.
        :return: A new solidified trimesh object with vertex colors preserved.
        """
        # Extract the original vertices, faces, and vertex colors
        original_vertices = mesh.vertices
        original_faces = mesh.faces
        original_colors = mesh.visual.vertex_colors if hasattr(mesh.visual, 'vertex_colors') else None

        # Assign default colors if none exist
        if original_colors is None:
            original_colors = np.ones((len(original_vertices), 3))  # Default white color

        # Create the "flat back" vertices by setting all z values to flat_back_depth
        flat_back_vertices = original_vertices.copy()
        flat_back_vertices[:, 2] = flat_back_depth

        # Combine original vertices and flat back vertices
        combined_vertices = np.vstack([original_vertices, flat_back_vertices])

        # Duplicate vertex colors for the flat back vertices
        combined_colors = np.vstack([original_colors, original_colors])

        # Create faces for the flat back surface
        num_vertices = len(original_vertices)
        flat_back_faces = original_faces + num_vertices  # Shift indices for the back faces

        # Create side faces to connect the front and flat back vertices
        side_faces = []
        for face in original_faces:
            for i in range(3):
                # Get the current edge (start, end)
                start = face[i]
                end = face[(i + 1) % 3]

                # Create two faces to cover each side
                side_faces.append([start, end, end + num_vertices])
                side_faces.append([start, end + num_vertices, start + num_vertices])

        side_faces = np.array(side_faces)

        # Combine all faces: front, flat back, and side
        combined_faces = np.vstack([original_faces, flat_back_faces, side_faces])

        # Create a new mesh with the combined vertices, faces, and preserved colors
        solid_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            vertex_colors=combined_colors
        )

        return solid_mesh

    # Assuming 'mesh' is your created Trimesh object
    def flip_mesh(self, mesh):
        """
        Flip the mesh geometry horizontally (flipping the y-axis).
        Adjust the vertex colors accordingly if vertex colors are present.

        :param mesh: Trimesh object to be flipped.
        :return: Transformed Trimesh object with flipped geometry and adjusted colors.
        """
        # Define the transformation matrix for flipping along the y-axis
        flip_matrix = np.array([
            [1, 0, 0, 0],  # No change to x-axis
            [0, -1, 0, 0],  # Flip y-axis
            [0, 0, 1, 0],  # No change to z-axis
            [0, 0, 0, 1],  # Homogeneous coordinate
        ])

        # Apply the transformation to the mesh
        mesh.apply_transform(flip_matrix)

        # # Check if the mesh has vertex colors
        # if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        #     # Optionally adjust the vertex colors during flipping
        #     # Example: If flipping affects orientation-dependent effects, handle it here
        #     # For instance, flipping colors (if mirrored) can be implemented, but often the color remains unchanged
        #     flipped_colors = mesh.visual.vertex_colors.copy()  # Currently colors are unchanged
        #
        #     # Update flipped colors (if needed) - placeholder for any operation on colors.
        #     mesh.visual.vertex_colors = flipped_colors

        return mesh

    def add_mirror_mesh(self, mesh):
        """
        Add a mirrored backside to the provided 3D mesh and stitch the halves together.
        This creates a water-tight mesh by connecting the edges of the original
        mesh and the mirrored counterpart.

        :param mesh: A Trimesh object representing the front-facing mesh.
        :return: A new Trimesh object with a water-tight mirrored back side.
        """
        print("Adding mirrored backside to the mesh...")

        # Original mesh vertices and faces
        original_vertices = mesh.vertices
        original_faces = mesh.faces

        # Create mirrored vertices by negating the z-axis
        mirrored_vertices = original_vertices.copy()
        mirrored_vertices[:, 2] = -mirrored_vertices[:, 2]

        # Adjust face indices for mirrored vertices
        num_original_vertices = len(original_vertices)
        mirrored_faces = original_faces.copy() + num_original_vertices

        # Reverse the face winding for the mirrored side
        mirrored_faces = mirrored_faces[:, ::-1]

        # Combine original and mirrored vertices and faces
        combined_vertices = np.vstack([original_vertices, mirrored_vertices])
        combined_faces = np.vstack([original_faces, mirrored_faces])

        print("Finding boundary edges...")
        original_edges = mesh.edges_sorted
        mirrored_edges = np.roll(original_edges, shift=1, axis=1) + num_original_vertices

        original_edges_set = set(map(tuple, original_edges))
        mirrored_edges_set = set(map(tuple, mirrored_edges))

        # shared_edges = original_edges_set.intersection(mirrored_edges_set)
        # boundary_edges = original_edges_set.symmetric_difference(shared_edges)
        boundary_edges = original_edges_set - mirrored_edges_set

        if len(boundary_edges) == 0:
            print("Warning: No boundary edges detected! Mesh may already be watertight.")

        print("Stitching boundary edges...")
        stitching_faces = []
        for edge in boundary_edges:
            v1, v2 = edge
            mv1, mv2 = v1 + num_original_vertices, v2 + num_original_vertices
            stitching_faces.append([v1, v2, mv1])
            stitching_faces.append([v2, mv2, mv1])
            stitching_faces.append([mv2, v2, v1])
            stitching_faces.append([mv1, mv2, v1])

        stitching_faces = np.array(stitching_faces)

        print("Combining faces, applying colors, creating watertight mesh...")
        # Combine stitching faces with others
        watertight_faces = np.vstack([combined_faces, stitching_faces])

        # Optionally combine vertex colors if provided
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            original_colors = mesh.visual.vertex_colors
            combined_colors = np.vstack([original_colors, original_colors])
        else:
            combined_colors = None

        # Create the watertight Trimesh object
        watertight_mesh = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=watertight_faces,
            vertex_colors=combined_colors
        )

        # print("Removing duplicate faces...")
        # watertight_mesh.remove_duplicate_faces()
        # print("Removing degenerate faces...")
        # watertight_mesh.remove_degenerate_faces()
        # print("Fixing normals...")
        # watertight_mesh.fix_normals()  # Ensure outward normals
        # print("Filling holes...")
        # watertight_mesh.fill_holes()  # Fix any remaining holes

        print("Finished creating mesh.")
        return watertight_mesh

    def create_3d_mesh(self, image, depth, filename, smoothing_method, target_size, dynamic_depth, grayscale_enabled,
                       edge_detection_enabled, invert_colors_enabled=False, depth_amount=1.0, project_on_original=False,
                       background_removal=False, background_tolerance=10, color_to_remove=None):
        """
        Args:
            image: Input image data.
            depth: Depth data corresponding to the image.
            filename: The path to save the generated 3D mesh.
            smoothing_method: (Optional) Method used for smoothing depth data.
            target_size: (Optional) Desired output size.
            dynamic_depth: (Optional) Adjust depth dynamically based on the data.
            grayscale_enabled: (Optional) Whether to enable grayscale processing.
            edge_detection_enabled: (Optional) Whether to enable edge detection.
            invert_colors_enabled: (Optional) Whether to invert colors for depth data.
            depth_amount: (Optional) Factor to control the depth scaling (1.0 = current behavior, 0.5 = half, 2.0 = double).
                          Maximum allowed value is 100.0.
            project_on_original: (Optional) Whether to project the mesh onto the original image.
            background_removal: (Optional) Whether to remove the background based on the average color.
            background_tolerance: (Optional) Tolerance for background color removal.
            color_to_remove: (Optional) RGB color to explicitly remove as the background. Overrides automatic detection.
        Returns:
            Generated mesh (or related object).
        """
        print(f"Creating 3D mesh with depth amount: {depth_amount}...")
        depth_amount = max(0.0, min(depth_amount, 100.0))

        # Adjust the depth data based on the depth_amount
        depth = depth * depth_amount

        # Step 1: Background processing
        h, w, _ = image.shape

        # If color_to_remove is provided, use it as the background color.
        if color_to_remove is not None:
            # Handle QColor if it is passed
            if isinstance(color_to_remove, QtGui.QColor):
                # Convert QColor to a list of RGB values
                color_to_remove = list(color_to_remove.getRgb()[:3])  # Discard the alpha channel
            # Convert to uint8 numpy array
            background_color = np.array(color_to_remove, dtype=np.uint8)
        else:
            # Otherwise, calculate the background color from image corners
            corners = [image[0, 0], image[0, w - 1], image[h - 1, 0], image[h - 1, w - 1]]
            avg_color = np.mean(corners, axis=0)

            if background_removal and all(
                    np.all(np.abs(corner - avg_color) < background_tolerance) for corner in corners):
                background_color = avg_color.astype(np.uint8)
            else:
                background_color = None

        if background_color is not None:
            print(f"Masking background color = {background_color.tolist()}")
            mask = cv2.inRange(image, background_color - background_tolerance, background_color + background_tolerance)
            if target_size and target_size != (0, 0) and target_size != (w, h):
                mask_resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            mask_resized = cv2.bitwise_not(mask_resized)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_resized = cv2.dilate(mask_resized, kernel, iterations=2)
        else:
            mask_resized = np.ones(target_size, dtype=np.uint8) * 255

        depth[mask_resized == 0] = 0
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        # Ensure consistent RGB color conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image[mask_resized == 0] = 0

        # Step 2: Create 3D vertices
        h, w = target_size
        y, x = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w), indexing="ij")
        z = depth
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid_mask = z.reshape(-1) != 0
        valid_vertices = vertices[valid_mask]

        # Step 3: Re-map faces
        index_map = -np.ones(vertices.shape[0], dtype=int)
        index_map[valid_mask] = np.arange(len(valid_vertices))

        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                if depth[i, j] != 0 and depth[i, j + 1] != 0 and depth[i + 1, j] != 0:
                    remapped = [index_map[idx], index_map[idx + 1], index_map[idx + w]]
                    if all(idx >= 0 for idx in remapped):
                        faces.append(remapped)
                if depth[i + 1, j] != 0 and depth[i, j + 1] != 0 and depth[i + 1, j + 1] != 0:
                    remapped = [index_map[idx + 1], index_map[idx + w + 1], index_map[idx + w]]
                    if all(idx >= 0 for idx in remapped):
                        faces.append(remapped)
        faces = np.array(faces)

        # Flatten the image to apply vertex_colors
        colors = image.reshape(-1, 3)
        valid_colors = colors[valid_mask]

        # Create the mesh
        mesh = self.flip_mesh(trimesh.Trimesh(vertices=valid_vertices, faces=faces, vertex_colors=valid_colors))

        if dynamic_depth:
            flat_back_depth = np.median(z[z != 0]) - np.std(z[z != 0])
            print(f"Dynamic flat back depth = {flat_back_depth}")
            solid_mesh = self.solidify_mesh_with_flat_back(mesh, flat_back_depth=flat_back_depth)
        else:
            solid_mesh = self.add_mirror_mesh(mesh)

        # Construct file name suffix based on enabled options
        file_suffix = f"_D{depth_amount}".replace(".", "_")
        if smoothing_method:
            file_suffix += f"_{smoothing_method}"
        if project_on_original:
            file_suffix += "_projected"
        if background_removal:
            file_suffix += "_noBG"
        file_suffix += f"-{self.model_type}"
        if background_color is not None:
            file_suffix += f"_B" + "".join(str(c) for c in background_color)
        file_suffix += f"_R{target_size[0]}x{target_size[1]}"
        if grayscale_enabled:
            file_suffix += "_gray"
        if edge_detection_enabled:
            file_suffix += "_edge"
        if invert_colors_enabled:
            file_suffix += "_invert"
        if dynamic_depth:
            file_suffix += "_dynamic"

        # For part of file name, format current date and time in format: YYYYMMDD_HHmmss
        now = datetime.now()
        file_suffix += now.strftime("%Y%m%d_%H%M%S")

        output_ply_filename = (f"{os.path.splitext(filename)[0]}{file_suffix}.ply")
        solid_mesh.export(output_ply_filename)
        print(f"3D mesh saved to {output_ply_filename}")

        if background_color is None:
            background_color = [-1, -1, -1]
        return output_ply_filename, background_color

    def modify_depth(self, depth_array, percentage):
        """
        Modify depth by removing the lowest values based on a given percentage.

        Args:
            depth_array (ndarray): 2D or 3D numpy array of depth values.
            percentage (float): Percentage of lowest depth values to remove (0-100). Usually < 10 is good.

        Returns:
            ndarray: Modified depth array with the lowest values removed.
        """
        if not 0 < percentage <= 100:
            return depth_array

        # Flatten the depth array for percentile calculation
        flattened = depth_array.flatten()

        # Calculate the threshold value based on the given percentage
        threshold = np.percentile(flattened, percentage)

        # Apply a mask to set values below the threshold to zero
        modified_depth = np.where(depth_array > threshold, depth_array, 0)

        return modified_depth

    def create_text_values_from_depth(self, depth_map, num_segments=21):
        """
            Generate 21 text values based on a depth map divided evenly.

            Args:
                depth_map (numpy.ndarray): Depth map containing depth values.

            Returns:
                list: List of 21 string values representing evenly divided intervals.

            Raises:
                ValueError: If depth_map is not a valid NumPy array.
            """
        # Ensure the depth_map is a numpy array
        if not isinstance(depth_map, np.ndarray):
            try:
                depth_map = np.array(depth_map)
            except Exception as e:
                raise ValueError(f"Invalid depth map format. Expected ndarray, got {type(depth_map)}. Error: {e}")

        # Flatten depth_map to 1D and get min, max values
        depth_values = depth_map.flatten()
        min_depth, max_depth = depth_values.min(), depth_values.max()

        # Create 21 evenly spaced intervals
        intervals = np.linspace(min_depth, max_depth, num_segments)

        # Format the intervals as text
        text_values = [f"{value:.2f}" for value in intervals]

        return depth_values, text_values

    def process_image(self, image_path, smoothing_method="anisotropic", target_size=(500, 500), dynamic_depth=False,
                      grayscale_enabled=False, edge_detection_enabled=False, invert_colors_enabled=False,
                      depth_amount=1.0, depth_drop_percentage=0, project_on_original=False, background_removal=False,
                      background_tolerance=0, background_color=None):
        """
        Process the input image to estimate depth, project into 3D space, and save as a PLY file.
        :param image_path: Path to the input image.
        :param smoothing_method: Depth map smoothing method.
        :param target_size: Target resolution of the depth map.
        :param dynamic_depth: If True, adjusts the back of the mesh dynamically.
        :param grayscale_enabled: If True, indicates grayscale input was used.
        :param edge_detection_enabled: If True, indicates edge detection was used.
        """
        # Load the image
        print(f"Processing image: {image_path} \r\n\t\t\tWith: Depth map resolution = {target_size}, "
              f"Grayscale = {grayscale_enabled} Edge detection = {edge_detection_enabled}, "
              f"Invert colors = {invert_colors_enabled}, Dynamic depth = {dynamic_depth}, Depth amount = {depth_amount}, "
              f"Removing {depth_drop_percentage}% of the lowest depth values, Smoothing method = {smoothing_method}, "
              f"Project on original = {project_on_original}, Background removal = {background_removal}, "
              f"Background tolerance = {background_tolerance}")
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        if target_size == (0,0):
            size = max(img_h, img_w)
            target_size = (size, size)

        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # Estimate depth
        flip = False
        if self.model_type == "depth_anything_v2":
            flip = True
        depth = self.estimate_depth(image, target_size, flip)
        # Remove the lowest values. 0.05 = remove 5% of the lowest depth values.
        depth = self.modify_depth(depth, depth_drop_percentage)
        self.depth_values, self.depth_labels = self.create_text_values_from_depth(depth)

        # Apply depth smoothing
        try:
            depth = smoothing_depth_map_utils.SmoothingDepthMapUtils().apply_smoothing(depth, method=smoothing_method)
        except Exception as e:
            print(f"Warning: Could not smooth depth. Proceeding with raw depth. {e}")

        # Generate and save 3D mesh with updated file suffix
        return self.create_3d_mesh(image, depth, f"{image_path}", smoothing_method, target_size,
                                   dynamic_depth, grayscale_enabled, edge_detection_enabled,
                                   invert_colors_enabled, depth_amount, project_on_original,
                                   background_removal, background_tolerance=background_tolerance,
                                   color_to_remove=background_color)

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="3D Depth Estimation and Mesh Generation")
    parser.add_argument(
        "image_path", type=str, help="Path to input image. Generates a 3D OBJ file in the same directory."
    )
    parser.add_argument(
        "--model_type", type=str, default="midas", choices=["midas", "dense_depth"],
        help="Type of pre-trained depth model to use (midas or dense_depth). Default: midas"
    )
    args = parser.parse_args()

    # Run the depth-to-3D mesh pipeline
    depth_to_3d = DepthTo3D(model_type=args.model_type)
    depth_to_3d.process_image(args.image_path)
