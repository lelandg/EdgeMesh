import os
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2

class DepthAnythingMapGenerator:
    def __init__(self, model_path: str, encoder: str = 'vitl', device: str = None):
        """
        Initialize the DepthMapGenerator with model configurations.

        :param model_path: Path to the .pth (model weights) file.
        :param encoder: Model encoder type. Options: 'vits', 'vitb', 'vitl', 'vitg'.
                        Default is 'vitl'.
        :param device: Device to run the model on ('cuda', 'cpu', or 'mps').
                       If not provided, it will auto-detect the best available.
        """
        # Auto-detect device if not provided
        self.device = device or ('cuda' if torch.cuda.is_available() else
                                 'mps' if torch.backends.mps.is_available() else 'cpu')

        # Supported encoder configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in self.model_configs:
            raise ValueError(f"Invalid encoder type: {encoder}. Supported encoders: {list(self.model_configs.keys())}")

        # Initialize the model
        self.model = DepthAnythingV2(**self.model_configs[encoder])

        # Load pretrained weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")

        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()

    def generate_depth_map(self, image_path: str, output_path: str, resolution: tuple = None):
        """
        Generate a 16-bit depth map for the given input image and save it to a file.

        :param image_path: Path to the input image file.
        :param output_path: Path to save the resulting depth map.
        :param resolution: Tuple (width, height) for resizing the output depth map. Default: No resizing.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image file not found at path: {image_path}")

        # Load the image
        raw_img = cv2.imread(image_path)
        if raw_img is None:
            raise ValueError(f"Failed to read the image file: {image_path}")

        # Optional resizing of the raw image to target resolution before inference
        original_size = raw_img.shape[:2]  # (height, width)
        if resolution:
            raw_img = cv2.resize(raw_img, resolution, interpolation=cv2.INTER_AREA)

        # Perform inference to generate the depth map
        depth = self.model.infer_image(raw_img)  # HxW raw depth map as a numpy array

        # Resize depth map back to original or specified resolution, if required
        if resolution:
            depth = cv2.resize(depth, original_size[::-1], interpolation=cv2.INTER_AREA)  # Resize to original size

        # Normalize depth map to 16-bit values (range 0-65535)
        depth_16bit = cv2.normalize(depth, None, 0, 65535, norm_type=cv2.NORM_MINMAX).astype(np.uint16)

        # Save the depth map as a 16-bit PNG image
        cv2.imwrite(output_path, depth_16bit)
        print(f"16-bit depth map saved to: {output_path}")


# Example Usage:
if __name__ == '__main__':
    # Specify paths to the model and input/output files
    model_file = 'checkpoints/depth_anything_v2_vitl.pth'
    input_image_1 = 'g:/Downloads/GettyImages-1731443210.jpg'
    output_image_1 = 'g:/Downloads/GettyImages-1731443210-depth-16bit.png'

    input_image_2 = 'g:/Downloads/lelandgreen_3D_relief_of_a_lions_head_roaring_carved_on_grey__544dd61a-20de-411a-9df3-f5e877c0a618_0.png'
    output_image_2 = 'g:/Downloads/lelandgreen_depth-16bit.png'

    # Initialize the DepthMapGenerator
    generator = DepthAnythingMapGenerator(model_path=model_file, encoder='vitl')

    # Generate depth map with default resolution
    print(f"Processing image: {input_image_1}")
    generator.generate_depth_map(image_path=input_image_1, output_path=output_image_1)

    # Generate depth map with a specific resolution (e.g., 1024x768)
    print(f"Processing image: {input_image_2} with target resolution (1024x768)")
    generator.generate_depth_map(image_path=input_image_2, output_path=output_image_2, resolution=(1024, 768))
