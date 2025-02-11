import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SmoothingDepthMapUtils:
    def __init__(self):
        """
        A class for smoothing depth maps with various methods.
        """
        pass

    @staticmethod
    def gaussian_smoothing(depth_map, kernel_size=5, sigma=1.0):
        """
        Apply Gaussian smoothing to the depth map to reduce noise and jagged edges.
        :param depth_map: The input depth map as a numpy array.
        :param kernel_size: Size of the Gaussian kernel (odd integer).
        :param sigma: Standard deviation of the Gaussian kernel.
        :return: Smoothed depth map.
        """
        smoothed_map = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), sigma)
        return smoothed_map

    @staticmethod
    def bilateral_smoothing(depth_map, diameter=15, sigma_color=75, sigma_space=75):
        """
        Apply bilateral filtering to smooth the depth map while preserving edges.
        :param depth_map: The input depth map as a numpy array.
        :param diameter: Diameter of each pixel neighborhood used in filtering.
        :param sigma_color: Filter sigma in the color space.
        :param sigma_space: Filter sigma in the coordinate/space domain.
        :return: Smoothed depth map.
        """
        smoothed_map = cv2.bilateralFilter(depth_map, diameter, sigma_color, sigma_space)
        return smoothed_map

    @staticmethod
    def median_smoothing(depth_map, kernel_size=5):
        """
        Apply median filtering to the depth map to remove salt-and-pepper noise.
        :param depth_map: The input depth map as a numpy array.
        :param kernel_size: Size of the filter kernel (odd integer).
        :return: Smoothed depth map.
        """
        smoothed_map = cv2.medianBlur(depth_map, kernel_size)
        return smoothed_map

    @staticmethod
    def anisotropic_diffusion(depth_map, iterations=10, kappa=50, gamma=0.1):
        """
        Perform anisotropic diffusion (edge-aware smoothing) on the depth map.
        :param depth_map: The input depth map as a numpy array.
        :param iterations: Number of diffusion iterations.
        :param kappa: Conduction coefficient, controls sensitivity to edges.
        :param gamma: Integration constant (usually <= 0.25 for stability).
        :return: Smoothed depth map.
        """
        smoothed_map = depth_map.copy()
        for _ in range(iterations):
            nabla_n = np.roll(smoothed_map, -1, axis=0) - smoothed_map
            nabla_s = np.roll(smoothed_map, 1, axis=0) - smoothed_map
            nabla_e = np.roll(smoothed_map, -1, axis=1) - smoothed_map
            nabla_w = np.roll(smoothed_map, 1, axis=1) - smoothed_map

            diffusion = (
                    np.exp(-(nabla_n / kappa) ** 2) * nabla_n +
                    np.exp(-(nabla_s / kappa) ** 2) * nabla_s +
                    np.exp(-(nabla_e / kappa) ** 2) * nabla_e +
                    np.exp(-(nabla_w / kappa) ** 2) * nabla_w
            )
            smoothed_map += gamma * diffusion
        return smoothed_map

    @staticmethod
    def normalize_depth_map(depth_map, output_range=(0, 255)):
        """
        Normalize the depth map to a specified range (e.g., 0 to 255 for visualization).
        :param depth_map: The input depth map as a numpy array.
        :param output_range: Tuple specifying the output range (min, max).
        :return: Normalized depth map.
        """
        min_val, max_val = np.min(depth_map), np.max(depth_map)
        normalized_map = (depth_map - min_val) / (max_val - min_val) * (output_range[1] - output_range[0]) + \
                         output_range[0]
        return normalized_map.astype(np.uint8)

    def apply_smoothing(self, depth_map, method="gaussian", **kwargs):
        """
        Apply a specified smoothing method to the depth map.
        :param depth_map: The input depth map as a numpy array.
        :param method: Smoothing method to use ("gaussian", "bilateral", "median", "anisotropic").
        :param kwargs: Additional parameters for the chosen smoothing method.
        :return: Smoothed depth map.
        """
        if method == "gaussian":
            return self.gaussian_smoothing(depth_map, **kwargs)
        elif method == "bilateral":
            return self.bilateral_smoothing(depth_map, **kwargs)
        elif method == "median":
            return self.median_smoothing(depth_map, **kwargs)
        elif method == "anisotropic":
            return self.anisotropic_diffusion(depth_map, **kwargs)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")