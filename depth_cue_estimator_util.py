import cv2
import numpy as np


class DepthCueEstimator:
    def __init__(self):
        """
        Initialize the DepthCueEstimator class.
        This class uses light, shading, and depth maps (if available) to estimate relative depth layers.
        """
        pass

    def estimate_depth_from_light(self, image, threshold=50):
        """
        Estimate depth using light intensity in the grayscale image.

        Args:
            image (numpy.ndarray): Input grayscale image (single channel).
            threshold (int): Threshold to detect areas of intense light. Default is 50.

        Returns:
            numpy.ndarray: An estimated depth map where light intense areas are closer.
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Normalize the image intensity
        normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Inverse the intensity to treat brighter areas as closer regions
        inverted = 255 - normalized

        # Blur the image to smooth out noise
        depth_map = cv2.GaussianBlur(inverted, (5, 5), 0)

        _, binary_mask = cv2.threshold(normalized, threshold, 255, cv2.THRESH_BINARY)

        return depth_map

    def estimate_depth_from_shading(self, image, kernel_size=5):
        """
        Estimate depth using shading by detecting gradients.

        Args:
            image (numpy.ndarray): Input grayscale or BGR image.
            kernel_size (int): Size of Sobel operator kernel. Default is 5.

        Returns:
            numpy.ndarray: An estimated depth map based on shading gradients.
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute gradients in X and Y directions
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)

        # Compute the gradient magnitude
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)

        # Normalize gradient to scale from 0 to 255
        depth_map = cv2.normalize(gradient_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return depth_map.astype(np.uint8)

    def process_input_depth_data(self, depth_map, normalize=True):
        """
        Process input depth data if it is available.

        Args:
            depth_map (numpy.ndarray): Input depth data (single channel).
            normalize (bool): Whether to normalize the depth map to scale 0 to 255. Default is True.

        Returns:
            numpy.ndarray: Processed depth data.
        """
        if normalize:
            depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return depth_map.astype(np.uint8)

    def combine_depth_cues(self, light_depth, shading_depth, input_depth=None, weights=(0.5, 0.5, 0.0)):
        """
        Combine depth maps from different cues to compute a unified depth estimation.

        Args:
            light_depth (numpy.ndarray): Depth map estimated based on light intensity.
            shading_depth (numpy.ndarray): Depth map estimated based on shading/gradient.
            input_depth (numpy.ndarray): Input depth map if available (can be None).
            weights (tuple): Weights for combining light, shading, and input depth maps.

        Returns:
            numpy.ndarray: Combined and normalized depth map.
        """
        combined = weights[0] * light_depth + weights[1] * shading_depth

        if input_depth is not None:
            combined += weights[2] * input_depth

        # Normalize the combined depth map
        combined_depth_map = cv2.normalize(combined, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        return combined_depth_map.astype(np.uint8)


# Example usage
if __name__ == "__main__":
    # Read an input image
    image = cv2.imread("example_image.jpg")

    # Initialize the DepthCueEstimator
    depth_estimator = DepthCueEstimator()

    # Estimate depth from light
    light_depth = depth_estimator.estimate_depth_from_light(image)

    # Estimate depth from shading
    shading_depth = depth_estimator.estimate_depth_from_shading(image)

    # Combine light and shading depth cues
    combined_depth = depth_estimator.combine_depth_cues(light_depth, shading_depth)

    # Show the results
    cv2.imshow("Light-based Depth", light_depth)
    cv2.imshow("Shading-based Depth", shading_depth)
    cv2.imshow("Combined Depth", combined_depth)
    cv2.waitKey(0)
    cv2.destroyAllWindows()