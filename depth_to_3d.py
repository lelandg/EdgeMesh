import argparse
import os
from pathlib import Path
from data_contracts import as_bgr, output_shape, normalized_depth, normalized_inverse_depth, foreground_mask as validate_mask
from log_utils import get_logger
import cv2
from datetime import datetime
import numpy as np
import torch
import trimesh
from trimesh import Trimesh

from PIL import Image
from transformers import AutoImageProcessor as AutoImageProcessor  # Backwards-compatible public import.
from model_store import ModelStore
from da3_backend import DA3_ALIASES
from MeshTools.mesh_tools import MeshTools

import smoothing_depth_map_utils
from spinner import Spinner

import PySide6.QtGui as QtGui

"""!@brief DepthTo3D modelnames supported by the DepthTo3D class."""
model_names = {"MiDaS": "midas", "DPT": "dpt",
               "DepthAnythingV2": "depth_anything_v2", "Depth Pro": "depth_pro", **DA3_ALIASES}


class DepthTo3D:
    def __init__(self, model_type="dpt", verbose=True, model_store=None, allow_download=False, cancelled=None,
                 device=None):
        """
        Initialize the depth estimation and mesh generation pipeline.
        :param model_type: Supported UI name or internal model identifier.
        :param allow_download: Explicit opt-in to fetch missing pinned HF files.
        :param cancelled: Optional callable checked between preparation stages.
        :param device: Optional torch device. Explicit CPU avoids probing CUDA.
        """
        self.verbose = verbose
        self.mesh_tools = None
        self.depth_map = None
        self.depth_values = None
        self.depth_labels = None
        self.solid_mesh = None
        self.model_type = model_names.get(model_type, model_type)
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_info = {}
        self.model_store = model_store if model_store is not None else ModelStore()
        self.allow_download = allow_download
        self.cancelled = cancelled
        self.model, self.transform = self.load_model()
        self.spinner = Spinner(f"{{time}} ")


    def load_model(self):
        """Use the same offline/pinned loader for GUI and direct Python callers."""
        model, transform, self.model_info = self.model_store.get_depth(
            self.model_type, self.device, allow_download=self.allow_download,
            cancelled=self.cancelled)
        return model, transform

    def process_depth_map(depth_map, percentage_attenuate, percent_reduce):
        """
        Processes a depth map by attenuating values below a computed median_value and smoothing them.

        Args:
            depth_map (numpy array): A 2D array representing the depth map.
            percentage_attenuate (float): The percentile (0-100) used to compute the median_value.
            percent_reduce (float): The percentage (0-100) that determines the lower bound for smoothing.

        Returns:
            numpy array: The processed depth map.
        """
        # Flatten the depth map to find percentile
        depth_values = depth_map.flatten()

        # Calculate the value at percentage_attenuate percentile
        median_value = np.percentile(depth_values, percentage_attenuate)
        print(f"Median value at {percentage_attenuate}%: {median_value}")

        # Calculate the reduced value based on percent_reduce
        reduce_value = (percent_reduce / 100) * median_value

        # Create a mask for values below the median_value
        mask = depth_map < median_value

        # Spread the values evenly between reduce_value and median_value
        depth_map[mask] = np.interp(
            depth_map[mask],  # Original values below the median
            (depth_map[mask].min(), median_value),  # Original scale
            (reduce_value, median_value)  # New scale
        )

        return depth_map

    def estimate_depth(self, image, target_size=(500, 500), flip=False):
        """Estimate normalized depth from BGR input; target_size is (height, width).

        (0, 0) or None preserves the original dimensions. A processing flip is
        undone on the prediction so geometry remains aligned with image colors.
        """
        image = as_bgr(image)
        img_h, img_w = image.shape[:2]
        output_size = output_shape(image.shape, target_size)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if flip:
            rgb_image = cv2.flip(rgb_image, 1)

        if getattr(self, "model_info", {}).get("backend") == "depth_anything_3":
            depth = self.model.predict_depth(rgb_image, cancelled=self.cancelled)
            if flip:
                depth = np.fliplr(depth).copy()
            # DA3 returns distance; the relief surface requires nearer-is-higher
            # inverse depth, as returned by the existing HF/MiDaS model paths.
            return normalized_inverse_depth(depth, output_size)

        with torch.no_grad():
            if getattr(self, "model_info", {}).get("backend") == "huggingface" or self.model_type in ("depth_anything_v2", "depth_pro"):
                inputs = self.transform(images=rgb_image, return_tensors="pt")
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
                depth = self.model(**inputs).predicted_depth
            else:
                resized = cv2.resize(rgb_image, ((img_w + 31) // 32 * 32, (img_h + 31) // 32 * 32))
                if self.model_type in ("midas", "dpt"):
                    img_tensor = self.transform(resized).to(self.device)
                    depth = self.model(img_tensor)
                else:
                    img_tensor = self.transform(Image.fromarray(resized)).unsqueeze(0).to(self.device)
                    depth = self.model(img_tensor)[0]

        # Preserve singleton spatial dimensions when stripping batch/channel axes.
        depth = depth.detach().cpu().numpy()
        depth = depth.reshape(depth.shape[-2:])
        if flip:
            depth = np.fliplr(depth).copy()
        return normalized_depth(depth, output_size)

    # def solidify_mesh(self, mesh, depth_offset=-1.0):
    #     """
    #     Extend the hollow mesh backwards along the depth axis to make it solid.
    #     :param mesh: Existing 3D trimesh object.
    #     :param depth_offset: The offset applied to create the "back" face of the solid model (negative for backward extension).
    #     :return: A new solidified trimesh object.
    #     """
    #     # Extract the original vertices and faces
    #     original_vertices = mesh.vertices
    #     original_faces = mesh.faces
    #
    #     # Create the "back" face vertices by shifting along the z-axis
    #     back_vertices = original_vertices.copy()
    #     back_vertices[:, 2] += depth_offset  # Adjust depth axis (z-axis)
    #
    #     # Combine original vertices and back vertices
    #     combined_vertices = np.vstack([original_vertices, back_vertices])
    #
    #     # Create faces for the back surface
    #     num_vertices = len(original_vertices)
    #     back_faces = original_faces + num_vertices  # Shift indices for the back faces
    #
    #     # Create side faces to connect the front and back vertices
    #     side_faces = []
    #     for face in original_faces:
    #         for i in range(3):
    #             # Get the current edge (start, end)
    #             start = face[i]
    #             end = face[(i + 1) % 3]
    #
    #             # Create two faces to cover the side
    #             side_faces.append([start, end, end + num_vertices])
    #             side_faces.append([start, end + num_vertices, start + num_vertices])
    #
    #     side_faces = np.array(side_faces)
    #
    #     # Combine all faces: front, back, and side
    #     combined_faces = np.vstack([original_faces, back_faces, side_faces])
    #
    #     # Create a new mesh with the combined vertices and faces
    #     solid_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    #
    #     return solid_mesh

    def remove_masked_islands(self, mask):
        """Keep only background pixels connected to an image border.

        Label pixels rather than filling contours, which erases foreground holes.
        """
        _, labels = cv2.connectedComponents((mask != 0).astype(np.uint8), connectivity=8)
        border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        border_labels = border_labels[border_labels != 0]
        return np.where(np.isin(labels, border_labels), 255, 0).astype(np.uint8)

    @staticmethod
    def _background_color(image, color_to_remove, background_removal, background_tolerance):
        """Return BGR for processing; explicitly selected colors arrive as RGB."""
        if color_to_remove is not None:
            if isinstance(color_to_remove, QtGui.QColor):
                color_to_remove = color_to_remove.getRgb()[:3]
            return np.asarray(color_to_remove, dtype=np.int32)[::-1]
        corners = np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]], dtype=np.float32)
        average = corners.mean(axis=0)
        if background_removal and np.all(np.abs(corners - average) <= background_tolerance):
            return average.astype(np.int32)
        return None

    def create_background_mask(self, image, color_to_remove=None,
                               background_removal=False, background_tolerance=0):
        """Mask border-connected BGR background; selected colors are RGB/QColor."""
        background_color = self._background_color(image, color_to_remove, background_removal, background_tolerance)
        if background_color is None:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        # Signed arithmetic prevents black/white tolerance bounds wrapping at 255.
        lower = np.clip(background_color - background_tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(background_color + background_tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower, upper)
        return self.remove_masked_islands(mask)

    def create_3d_mesh(self, image, depth, filename, smoothing_method, target_size, flat_back, grayscale_enabled,
                       edge_detection_enabled, invert_colors_enabled=False, depth_amount=1.0, project_on_original=False,
                       background_removal=False, background_tolerance=10, color_to_remove=None,
                       subject_mask=None, cancel_check=None):
        """
        Args:
            image: Input image data.
            depth: Depth data corresponding to the image.
            filename: The path to save the generated 3D mesh.
            smoothing_method: (Optional) Method used for smoothing depth data.
            target_size: (Optional) Desired output size.
            flat_back: (Optional) Adjust depth dynamically based on the data.
            grayscale_enabled: (Optional) Whether to enable grayscale processing.
            edge_detection_enabled: (Optional) Whether to enable edge detection.
            invert_colors_enabled: (Optional) Whether to invert colors for depth data.
            depth_amount: (Optional) Relative relief scale. At 1.0, a normalized
                          depth of 255 reaches half the longest image side;
                          0.5 is half that relief and 2.0 is double. Independent
                          of output resolution. Maximum allowed value is 100.0.
            project_on_original: (Optional) Whether to project the mesh onto the original image.
            background_removal: (Optional) Whether to remove the background based on the average color.
            background_tolerance: (Optional) Tolerance for background color removal.
            color_to_remove: (Optional) RGB color to explicitly remove as the background. Overrides automatic detection.
        Returns:
            Generated mesh (or related object).
        """
        print(f"Creating 3D mesh with depth amount: {depth_amount}...")
        # Assume depth has been normalized so 0 is the minimum value

        check = cancel_check or (lambda: None)
        check()
        image = as_bgr(image)
        target_size = output_shape(image.shape, target_size)
        depth = np.asarray(depth, dtype=np.float32).copy()
        if depth.shape != target_size or not np.isfinite(depth).all():
            raise ValueError("Depth dimensions must match the mesh and all values must be finite.")
        depth_amount = max(0.0, min(depth_amount, 100.0))

        # X/Y are pixel coordinates, but model depth is normalized to 0..255.
        # Scale Z with the same image extent so resolution changes only detail,
        # never the object's proportions. This is relative relief, not metres.
        longest_side = max(target_size) - 1
        depth = depth * (depth_amount * longest_side / (2.0 * 255.0))

        # Step 1: Background processing
        h, w, _ = image.shape
        print(f"Image size: {h}x{w}, target_size: {target_size}")
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        h, w, _ = image.shape
        background_color = self._background_color(image, color_to_remove, background_removal, background_tolerance)

        if background_color is not None:
            # print(f"Masking background color = {background_color.tolist()}")
            # mask = cv2.inRange(image, background_color - background_tolerance, background_color + background_tolerance)
            mask = self.create_background_mask(image,
                                               color_to_remove=color_to_remove,
                                               background_removal=background_removal,
                                               background_tolerance=background_tolerance)
            mask = cv2.bitwise_not(mask)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # mask = cv2.dilate(mask, kernel, iterations=2)
        else:
            mask = np.ones(target_size, dtype=np.uint8) * 255

        print(f"Image size: {image.shape}, depth size: {depth.shape}, mask size: {mask.shape}")
        if subject_mask is not None:
            mask[~validate_mask(subject_mask, target_size)] = 0
        depth[mask == 0] = 0
        surface = (mask != 0) & ((depth != 0) if depth_amount > 0 else True)
        # Ensure consistent RGB color conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image[mask == 0] = 0

        # Step 2: Create 3D vertices
        h, w = target_size
        y, x = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w), indexing="ij")
        z = depth
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid_mask = surface.reshape(-1)
        valid_vertices = vertices[valid_mask]

        # Step 3: Re-map faces
        index_map = -np.ones(vertices.shape[0], dtype=int)
        index_map[valid_mask] = np.arange(len(valid_vertices))

        faces = []
        for i in range(h - 1):
            if i % 32 == 0:
                check()
            for j in range(w - 1):
                idx = i * w + j
                if surface[i, j] and surface[i, j + 1] and surface[i + 1, j]:
                    remapped = [index_map[idx], index_map[idx + 1], index_map[idx + w]]
                    if all(idx >= 0 for idx in remapped):
                        faces.append(remapped)
                if surface[i + 1, j] and surface[i, j + 1] and surface[i + 1, j + 1]:
                    remapped = [index_map[idx + 1], index_map[idx + w + 1], index_map[idx + w]]
                    if all(idx >= 0 for idx in remapped):
                        faces.append(remapped)
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if not len(faces):
            raise ValueError("No surface remains. Increase depth or keep more foreground pixels.")
        check()

        # Flatten the image to apply vertex_colors
        colors = image.reshape(-1, 3)
        valid_colors = colors[valid_mask]

        # Create the mesh
        mesh = trimesh.Trimesh(vertices=valid_vertices, faces=faces, vertex_colors=valid_colors)
        self.mesh_tools = MeshTools(mesh, verbose=self.verbose)
        mesh = self.mesh_tools.flip_mesh(mesh)

        if depth_amount == 0:
            solid_mesh = mesh
        elif flat_back:
            solid_mesh = self.mesh_tools.solidify_mesh_with_flat_back(mesh, flat_back_depth=0.0)
        else:
            solid_mesh = self.mesh_tools.add_mirror_mesh(mesh)

        # Construct file name suffix based on enabled options
        file_suffix = f"_D{depth_amount}".replace(".", "_")
        if smoothing_method:
            file_suffix += f"_{smoothing_method}"
        if project_on_original:
            file_suffix += "_proj"
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
            file_suffix += "_inv"
        if flat_back:
            file_suffix += "_dyn"

        # For part of file name, format current date and time in format: YYYYMMDD_HHmmss
        now = datetime.now()
        file_suffix += now.strftime("%Y%m%d_%H%M%S")

        output_ply_filename = (f"{os.path.splitext(filename)[0]}{file_suffix}.ply")
        check()
        solid_mesh.export(output_ply_filename)
        check()
        print(f"3D mesh saved to {output_ply_filename}")

        if background_color is None:
            background_color = [-1, -1, -1]
        else:
            background_color = background_color[::-1].tolist()  # Viewport expects RGB.
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

        original_length = len(depth_array)
        # Flatten the depth array for percentile calculation
        flattened = depth_array.flatten()

        # Calculate the threshold value based on the given percentage
        # threshold = np.percentile(flattened, percentage)
        percentage = percentage / 100
        threshold = percentage * flattened.max()

        if self.verbose:
            print(f"Removed {percentage}% of the lowest depth values. Original had {original_length}. Threshold: {threshold} "
                  f"Has {len(flattened)} values. Min depth value: {flattened.min()}. Max depth value: {flattened.max()}.")
        # Apply a mask to set values below the threshold to zero
        modified_depth = np.where(depth_array > threshold, depth_array, threshold)
        min = modified_depth.min()
        max = modified_depth.max()
        if self.verbose: print(f"Modified depth is {modified_depth.shape}.\nMin depth value: {min} Max depth value: {max}")
        if min > 0:
            modified_depth = modified_depth - min # Normalize the depth values
            min = modified_depth.min()
            max = modified_depth.max()
            if self.verbose: print(f"Normalized depth {modified_depth.shape}.\nMin depth value: {min} Max depth value: {max}")

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

    def pad_to_square(self, image):
        """!
        Pads a given cv2 image to make it square with padding evenly distributed.
        Uses the average color of the image corners for padding to create a more
        natural transition.

        @param image (numpy.ndarray) The input cv2 image.

        @Return (numpy.ndarray) The square-padded image.
        """
        if image is None or len(image.shape) < 2:
            raise ValueError("Input image is invalid or None")

        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        # If already square, return the original image
        if height == width:
            return image

        # Get all four corner colors
        top_left_color = image[0, 0].tolist() if channels == 3 else int(image[0, 0])
        top_right_color = image[0, -1].tolist() if channels == 3 else int(image[0, -1])
        bottom_left_color = image[-1, 0].tolist() if channels == 3 else int(image[-1, 0])
        bottom_right_color = image[-1, -1].tolist() if channels == 3 else int(image[-1, -1])

        # Determine padding color by averaging the corners
        if channels == 3:
            padding_color = [
                int(sum(c) / 4) for c in zip(top_left_color, top_right_color, bottom_left_color, bottom_right_color)
            ]
        else:
            padding_color = int((top_left_color + top_right_color + bottom_left_color + bottom_right_color) / 4)

        # Calculate padding dimensions to center the image
        if height > width:
            diff = height - width
            pad_left = diff // 2
            pad_right = diff - pad_left
            pad_top, pad_bottom = 0, 0
        else:  # width > height
            diff = width - height
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            pad_left, pad_right = 0, 0

        # Add padding using cv2.copyMakeBorder
        square_image = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_color
        )

        return square_image


    def process_image(self, image_path, smoothing_method="anisotropic", target_size=(500, 500), flat_back=False,
                      grayscale_enabled=False, edge_detection_enabled=False, invert_colors_enabled=False,
                      depth_amount=1.0, depth_drop_percentage=0, project_on_original=False, background_removal=False,
                      background_tolerance=0, background_color=None, *, image_data=None, subject_mask=None,
                      output_dir=None, progress=None, cancel_check=None):
        """Process an immutable BGR snapshot; optional output_dir stages a worker job."""
        check = cancel_check or (lambda: None)
        progress = progress or (lambda message: None)
        check()
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED) if image_data is None else image_data
        if image is None:
            raise ValueError(f"Image not found: {image_path}")
        image = as_bgr(image)
        target_size = output_shape(image.shape, target_size)
        progress("Estimating depth")
        depth = self.estimate_depth(image, target_size, flip=False)
        check()
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            image_path = str(Path(output_dir) / Path(image_path).name)
        fname, _ = os.path.splitext(image_path)
        if not cv2.imwrite(f"{fname}{self.model_type}_depth_map.png", np.rint(depth).astype(np.uint8)):
            raise OSError("Could not save the generated depth preview.")
        depth = self.modify_depth(depth, depth_drop_percentage)
        self.depth_values, self.depth_labels = self.create_text_values_from_depth(depth)
        if depth.min() > 0:
            depth = depth - depth.min()
        progress("Smoothing depth")
        check()
        depth = smoothing_depth_map_utils.SmoothingDepthMapUtils().apply_smoothing(depth, method=smoothing_method)
        self.depth_map = depth.copy()
        check()
        progress("Building mesh")
        result = self.create_3d_mesh(image, depth, str(image_path), smoothing_method, target_size,
                                   flat_back, grayscale_enabled, edge_detection_enabled,
                                   invert_colors_enabled, depth_amount, project_on_original,
                                   background_removal, background_tolerance=background_tolerance,
                                   color_to_remove=background_color, subject_mask=subject_mask,
                                   cancel_check=check)
        check()
        return result

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="3D Depth Estimation and Mesh Generation")
    parser.add_argument(
        "image_path", type=str, help="Path to input image. Generates a PLY mesh in the same directory."
    )
    parser.add_argument(
        "--model_type", type=str, default="midas", choices=sorted(set(model_names.values())),
        help="Registered or cached depth model. Default: midas"
    )
    parser.add_argument("--allow-download", action="store_true", help="Allow a requested Hugging Face model download; otherwise offline.")
    args = parser.parse_args()

    # Run the depth-to-3D mesh pipeline
    depth_to_3d = DepthTo3D(model_type=args.model_type, allow_download=args.allow_download)
    depth_to_3d.process_image(args.image_path)
