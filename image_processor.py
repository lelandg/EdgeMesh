# This is a bonus, standalone script that demonstrates how to process images using OpenCV and custom utility classes.
# The ImageProcessor class provides methods to apply various image processing techniques, such as edge detection, depth estimation, and smoothing.
# However, the real work of those are done by the SmoothingDepthMapUtils and DepthCueEstimator classes.
import cv2
import os
from edge_detection import detect_edges
from depth_cue_estimator_util import DepthCueEstimator
from smoothing_depth_map_utils import SmoothingDepthMapUtils


class ImageProcessor:
    def __init__(self):
        self.depth_estimator = DepthCueEstimator()
        self.smoothing_utils = SmoothingDepthMapUtils()

    def process_image(self, image_path, methods, output_directory=None):
        """
        Process an image using specified methods.

        Args:
            image_path (str): Full path to the input image.
            methods (list): List of processing methods to apply. 
                            Supported: ["edges", "depth_light", "depth_shading", "depth_combined", "smooth_gaussian", "smooth_bilateral", "smooth_median", "smooth_anisotropic"].
            output_directory (str): Directory to save output images. Defaults to none (no saving).

        Returns:
            dict: Dictionary of processed images with method names as keys.
        """
        # Read the input image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image from the provided path.")

        basename = os.path.splitext(os.path.basename(image_path))[0]
        processed_images = {}

        for method in methods:
            if method == "edges":
                # Detect edges with Canny
                processed_images[method] = detect_edges(image_path, 50, 150)

            elif method == "depth_light":
                grayscale = self._convert_to_grayscale(image)
                processed_images[method] = self.depth_estimator.estimate_depth_from_light(grayscale)

            elif method == "depth_shading":
                grayscale = self._convert_to_grayscale(image)
                processed_images[method] = self.depth_estimator.estimate_depth_from_shading(grayscale)

            elif method == "depth_combined":
                grayscale = self._convert_to_grayscale(image)
                light_depth = self.depth_estimator.estimate_depth_from_light(grayscale)
                shading_depth = self.depth_estimator.estimate_depth_from_shading(grayscale)
                processed_images[method] = self.depth_estimator.combine_depth_cues(light_depth, shading_depth)

            elif method.startswith("smooth_"):
                if "depth_combined" not in processed_images:
                    raise ValueError("Depth map needed for smoothing. Ensure you include 'depth_combined' in methods.")
                depth_map = processed_images["depth_combined"]
                if method == "smooth_gaussian":
                    processed_images[method] = self.smoothing_utils.gaussian_smoothing(depth_map)
                elif method == "smooth_bilateral":
                    processed_images[method] = self.smoothing_utils.bilateral_smoothing(depth_map)
                elif method == "smooth_median":
                    processed_images[method] = self.smoothing_utils.median_smoothing(depth_map)
                elif method == "smooth_anisotropic":
                    processed_images[method] = self.smoothing_utils.anisotropic_diffusion(depth_map)

            else:
                raise ValueError(f"Unsupported method: {method}")

            # Save output if output directory is specified
            if output_directory:
                output_path = os.path.join(output_directory, f"{basename}-{method}.png")
                cv2.imwrite(output_path, processed_images[method])

        return processed_images

    def _convert_to_grayscale(self, image):
        """Convert an image to grayscale if it is not already."""
        if len(image.shape) > 2:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


# Example usage
if __name__ == "__main__":
    image_processor = ImageProcessor()
    input_image_path = "./Images/example.png"
    output_dir = "./Images"

    # List of processing methods
    methods = ["edges", "depth_light", "depth_shading", "depth_combined", "smooth_gaussian"]

    results = image_processor.process_image(input_image_path, methods, output_directory=output_dir)

    # Display the processed images
    for method, output in results.items():
        cv2.imshow(method, output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()