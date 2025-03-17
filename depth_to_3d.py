import argparse
import os
import cv2
from datetime import datetime
import numpy as np
import torch
import trimesh
from trimesh import Trimesh

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from MeshTools.mesh_tools import MeshTools

import smoothing_depth_map_utils
from spinner import Spinner

import PyQt6.QtGui as QtGui

"""!@brief DepthTo3D modelnames supported by the DepthTo3D class."""
model_names = {"DenseDepth": "dense_depth", "MiDaS": "midas", "DPT": "dpt", "LeReS": "leres",
               "DepthAnythingV2": "depth_anything_v2", "Depth Pro": "depth_pro"}


class DepthTo3D:
    def __init__(self, model_type="dpt", verbose = True):
        """
        Initialize the depth estimation and mesh generation pipeline.
        :param model_type: "midas" (default) or "dense_depth". Specifies the depth estimation model.
        """
        self.verbose = verbose
        self.mesh_tools = None
        self.depth_map = None
        self.depth_values = None
        self.depth_labels = None
        self.solid_mesh = None
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = self.load_model()
        self.spinner = Spinner(f"{{time}} ")


    def load_model(self):
        """
        Load the depth estimation model.
        :return: model and corresponding transform pipeline
        """
        print(f"Loading depth estimation model: {self.model_type}...")
        from torch.hub import load
        if self.model_type == "midas":
            # MiDaS model (Microsoft)
            model = load("intel-isl/MiDaS", "MiDaS", pretrained=True).to(self.device).eval()
            transforms = load("intel-isl/MiDaS", "transforms")
            transform = transforms.dpt_transform

        elif self.model_type == "dense_depth":
            # DenseDepth model
            from dense_depth_model import DenseDepth
            model = DenseDepth().to(self.device).eval()
            transform = DenseDepth.transform

        elif self.model_type == "dpt":
            # Dense Prediction Transformer (DPT)
            from torch.hub import load
            model = load("intel-isl/MiDaS", "DPT_Large", pretrained=True).to(self.device).eval()
            transforms = load("intel-isl/MiDaS", "transforms")
            transform = transforms.dpt_transform

        elif self.model_type == "leres":
            # LeReS (Lightweight Estimation)
            from leres_model import LeReS  # Custom import
            model = LeReS().to(self.device).eval()
            transform = LeReS.transform  # Replace with LeReS-specific preprocessing logic if necessary
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
            model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif self.model_type == "depth_pro":
            # Depth Pro (local weights)
            model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf")
            transform = Compose([
                ToTensor(),
                Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            raise ValueError(
                f"Unsupported model type. Use one of:\n{', '.join(model_names.keys())}.")
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
        """
        Estimate depth from a single image.
        :param image: Input image (numpy array).
        :param target_size: Tuple defining the target size (width, height) for the output depth map.
        :param flip: Whether to horizontally flip the image for processing.
        :return: Depth map (numpy array) of the limited size.
        """
        print(f"Estimating depth with target size: {target_size}...")
        img_h, img_w, _ = image.shape  # Use NumPy shape to get height and width
        print(f"Input image size: {img_h}x{img_w}")

        # Convert the image to grayscale for cv2.minMaxLoc
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Ensure grayscale image is passed to minMaxLoc
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray_image)
        print(f"Max value in image has ranges from {min_val} - {max_val}, at {min_loc} and {max_loc}, respectively.")

        # Resize to ensure divisibility by 32
        new_h = (img_h + 31) // 32 * 32
        new_w = (img_w + 31) // 32 * 32
        resized_image = cv2.resize(image, (new_w, new_h))  # Resize to divisible by 32

        if flip:
            resized_image = cv2.flip(resized_image, 1)  # Flip horizontally
            print(f"Flipped image size = {resized_image.shape}")
        # Convert to tensor
        
        # Predict depth
        with torch.no_grad():
            if self.model_type == "dense_depth" or self.model_type == "leres":
                img_input = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                img_tensor = self.transform(img_input).unsqueeze(0).to(self.device) if self.transform else None
                # depth = self.model(img_input)[0]  # Replace with respective model's output logic
                depth = self.model(img_tensor)[0]  # Replace with respective model's output logic
            else:
                if self.model_type == "depth_anything_v2":
                    img_input = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                    img_tensor = self.transform(img_input).unsqueeze(0).to(self.device) if self.transform else None
                    # Depth Anything V2 (local weights)
                    # Prepare image for the model
                    image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
                    inputs = image_processor(images=image, return_tensors="pt")

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    if target_size and target_size != (0, 0) and target_size != (img_h, img_w):
                        print("Target size specified, resizing to match.")
                        # Interpolate to original size
                        depth = torch.nn.functional.interpolate(
                            predicted_depth.unsqueeze(1),
                            size=target_size,  # Match original image dimensions here
                            mode="bicubic",
                            align_corners=False,
                        )
                    else:
                        print ("No target size specified, using original size.")
                        # Interpolate to original size
                        depth = torch.nn.functional.interpolate(
                            predicted_depth.unsqueeze(1),
                            # scale_factor=1.0,
                            size=(img_h, img_w),  # Match original image dimensions here
                            mode="bicubic",
                            align_corners=False,
                        )
                elif self.model_type == "depth_pro":
                    img_input = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
                    img_tensor = self.transform(img_input).unsqueeze(0).to(self.device) if self.transform else None
                    # Depth Pro (local weights)
                    # Prepare image for the model
                    image_processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
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
                elif self.model_type == "midas":
                    img_input = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                    img_tensor = self.transform(img_input).to(self.device)
                    depth = self.model(img_tensor)
                else:
                    depth = self.model(img_tensor)
        print(f"Initial depth map shape: {depth.shape}, values: {depth.min()} - {depth.max()}.")
        # Resize depth back to original image dimensions
        depth = depth.squeeze().cpu().numpy()
        print(f"Squeezed depth map shape: {depth.shape}, values: {depth.min()} - {depth.max()} {np.sum(depth < 0)} negative values.")
        depth[depth < 0] = 0  # replace negative values with 0
        if target_size and target_size != (0, 0) and target_size != (img_h, img_w):
            print(f"Target size specified: {target_size} Resizing to match.")
            # Limit the depth map size
            depth = cv2.resize(depth, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST_EXACT)  # Match original input image size
            print(f"Resized depth map shape: {depth.shape}, values: {depth.min()} - {depth.max()} {np.sum(depth < 0)} negative values.")

        # Normalize for visualization (optional)
        depth = cv2.normalize(depth, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        print(f"Final depth map shape: {depth.shape}, values: {depth.min()} - {depth.max()}")
        return depth

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
        """
        Remove all masked regions (white areas) from the mask that are not contiguous with the edge of the image.

        :param mask: Input binary mask (numpy array, 0 for unmasked and 255 for masked areas).
        :return: Cleaned mask where only regions connected to the image edges remain.
        """
        # Step 1: Create a blank mask to store the cleaned output
        cleaned_mask = np.zeros_like(mask, dtype=np.uint8)

        # Step 2: Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Step 3: Check each contour to see if it touches the edges of the image
        h, w = mask.shape
        for contour in contours:
            # Check if the contour connects with the edge of the image
            is_connected_to_edge = False
            for point in contour:
                x, y = point[0]  # Contour points are stored as [[x, y]]
                if x == 0 or y == 0 or x == w - 1 or y == h - 1:
                    is_connected_to_edge = True
                    break

            # If the region is connected to the edge, keep it
            if is_connected_to_edge:
                cv2.drawContours(cleaned_mask, [contour], -1, 255, thickness=cv2.FILLED)

        return cleaned_mask

    def create_background_mask(self, image, color_to_remove=None,
                               background_removal=False, background_tolerance=0):
        """
        Create a background mask for an image by removing the specified background color.
        Smooth the edges by eroding and blending edges. After that, remove masks surrounded by unmasked areas.

        :param image: Input image (numpy array).
        :param color_to_remove: Specific color to remove from the background.
        :param background_removal: Whether background removal is enabled.
        :param background_tolerance: Tolerance for background color matching.
        :return: Final processed mask.
        """
        # Step 1: Background processing
        h, w, _ = image.shape

        if color_to_remove is not None:
            if isinstance(color_to_remove, QtGui.QColor):
                # Convert QColor to a list of RGB values
                color_to_remove = list(color_to_remove.getRgb()[:3])  # Discard the alpha channel
            background_color = np.array(color_to_remove, dtype=np.uint8)
        else:
            corners = [image[0, 0], image[0, w - 1], image[h - 1, 0], image[h - 1, w - 1]]
            avg_color = np.mean(corners, axis=0)

            if background_removal and all(
                    np.all(np.abs(corner - avg_color) < background_tolerance) for corner in corners):
                background_color = avg_color.astype(np.uint8)
            else:
                background_color = None

        # Step 2: Create the background mask
        if background_color is not None:
            print(f"Masking background color = {background_color.tolist()}")
            mask = cv2.inRange(image, background_color - background_tolerance, background_color + background_tolerance)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)  # Default to all zeros (no mask)

        # # Step 3: Smooth the edges of the mask
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Larger kernel for smoothing
        # mask_dilate1 = cv2.dilate(mask, kernel, iterations=3)
        # # mask_dilate2 = cv2.addWeighted(mask, 0.7, mask_dilate1, 0.3, 0)
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # Smaller kernel for edge adjustment
        # mask_eroded = cv2.erode(mask, kernel, iterations=1)
        #
        # mask_smoothed = cv2.dilate(mask_eroded, kernel, iterations=1)
        # smoothed_mask = cv2.addWeighted(mask_smoothed, 0.8, mask_smoothed, 0.2, 0)
        # contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cleaned_mask = np.zeros_like(smoothed_mask)  # Start with a blank mask

        # Step 4: Remove masked regions surrounded by unmasked areas
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_mask = np.zeros_like(mask)  # Start with a blank mask

        for contour in contours:
            # Filter contours based on their area
            if cv2.contourArea(contour) > 0:  # Keep significant contours
                cv2.drawContours(cleaned_mask, [contour], -1, (255), thickness=cv2.FILLED)

        cleaned_mask = self.remove_masked_islands(cleaned_mask)
        # Return the finalized mask (foreground is marked as white)
        return cleaned_mask

    def create_3d_mesh(self, image, depth, filename, smoothing_method, target_size, flat_back, grayscale_enabled,
                       edge_detection_enabled, invert_colors_enabled=False, depth_amount=1.0, project_on_original=False,
                       background_removal=False, background_tolerance=10, color_to_remove=None):
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
        # Assume depth has been normalized so 0 is the minimum value

        depth_amount = max(0.0, min(depth_amount, 100.0))

        # Adjust the depth data based on the depth_amount
        depth = depth * depth_amount

        # Step 1: Background processing
        h, w, _ = image.shape
        print(f"Image size: {h}x{w}, target_size: {target_size}")
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
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
        depth[mask == 0] = 0
        # Ensure consistent RGB color conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image[mask == 0] = 0

        # Step 2: Create 3D vertices
        h, w = target_size
        y, x = np.meshgrid(np.linspace(0, h - 1, h), np.linspace(0, w - 1, w), indexing="ij")
        z = depth
        vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        valid_mask = z.reshape(-1) >= 0
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
        mesh = trimesh.Trimesh(vertices=valid_vertices, faces=faces, vertex_colors=valid_colors)
        self.mesh_tools = MeshTools(mesh, verbose=self.verbose)
        mesh = self.mesh_tools.flip_mesh(mesh)

        if flat_back:
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
        Pads a given cv2 image to make it square.
        - If the two corners being padded have the same color, uses that color for padding.
        - Otherwise, uses black for padding.

        @param image (numpy.ndarray) The input cv2 image.

        @Return (numpy.ndarray) The square-padded image.
        """
        if image is None or len(image.shape) < 2:
            raise ValueError("Input image is invalid or None")

        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]

        # Default padding color is black
        padding_color = [0, 0, 0] if channels == 3 else 0

        # Check corner colors to determine padding color
        top_right_color = image[0, -1].tolist() if channels == 3 else image[0, -1]
        bottom_right_color = image[-1, -1].tolist() if channels == 3 else image[-1, -1]
        bottom_left_color = image[-1, 0].tolist() if channels == 3 else image[-1, 0]

        # Determine padding dimensions
        if height > width:
            diff = height - width
            pad_left = 0 # diff // 2
            pad_right = diff # - pad_left
            pad_top, pad_bottom = 0, 0
            if top_right_color == bottom_right_color:
                padding_color = top_right_color
        elif width > height:
            diff = width - height
            pad_top = 0 # diff // 2
            pad_bottom = diff # - pad_top
            pad_left, pad_right = 0, 0
            if bottom_right_color == bottom_left_color:
                padding_color = bottom_right_color
        else:
            # Already square
            return image

        # # Ensure padding_color format matches cv2 requirements
        # if channels == 3 and isinstance(padding_color, list):
        #     padding_color = [int(c) for c in padding_color]
        # elif channels == 1 and isinstance(padding_color, (list, np.ndarray)):
        #     padding_color = int(padding_color[0])

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
                      background_tolerance=0, background_color=None):
        """
        Process the input image to estimate depth, project into 3D space, and save as a PLY file.
        :param image_path: Path to the input image.
        :param smoothing_method: Depth map smoothing method.
        :param target_size: Target resolution of the depth map.
        :param flat_back: If True, flattens the back of the mesh at 0 depth.
        :param grayscale_enabled: If True, indicates grayscale input was used.
        :param edge_detection_enabled: If True, indicates edge detection was used.
        """
        # Load the image
        print(f"Processing image: {image_path} \r\n\t\t\tWith: Depth map resolution = {target_size}, "
              f"Grayscale = {grayscale_enabled} Edge detection = {edge_detection_enabled}, "
              f"Invert colors = {invert_colors_enabled}, Dynamic depth = {flat_back}, Depth amount = {depth_amount}, "
              f"Removing {depth_drop_percentage}% of the lowest depth values, Smoothing method = {smoothing_method}, "
              f"Project on original = {project_on_original}, Background removal = {background_removal}, "
              f"Background tolerance = {background_tolerance}")
        image = cv2.imread(image_path)
        # image = self.pad_to_square(image)
        # dname = os.path.dirname(image_path)
        # fname = os.path.join(dname, f"{os.path.basename(image_path)}_padded.png")
        # cv2.imwrite(fname, image)
        img_h, img_w, _ = image.shape
        if target_size == (0,0):
            target_size = (img_h, img_w)

        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        # Estimate depth
        flip = False
        if self.model_type == "depth_anything_v2":
            flip = True
        depth = self.estimate_depth(image, target_size, flip)
        fname, ext = os.path.splitext(image_path)
        cv2.imwrite(f"{fname}{self.model_type}_depth_map.png", depth)
        # Remove the lowest values. 0.05 = remove 5% of the lowest depth values.
        depth = self.modify_depth(depth, depth_drop_percentage)
        self.depth_values, self.depth_labels = self.create_text_values_from_depth(depth)

        min = depth.min()
        if min > 0:
            depth = depth - min # Normalize the depth values

        # Apply depth smoothing
        try:
            depth = smoothing_depth_map_utils.SmoothingDepthMapUtils().apply_smoothing(depth, method=smoothing_method)
        except Exception as e:
            print(f"Warning: Could not smooth depth. Proceeding with raw depth. {e}")

        # Generate and save 3D mesh with updated file suffix
        return self.create_3d_mesh(image, depth, f"{image_path}", smoothing_method, target_size,
                                   flat_back, grayscale_enabled, edge_detection_enabled,
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
