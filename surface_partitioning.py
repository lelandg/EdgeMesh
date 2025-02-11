import cv2
import numpy as np
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point

from edge_clustering_analyzer import EdgeClustering


class SurfacePartitioning:
    """
    Class for partitioning image edges into planar or volumetric regions
    using Convex Hulls, Alpha-Shapes, and Region-Growing techniques.
    """

    def __init__(self, sensitivity=0.5, max_polygons=10):
        """
        Initialize the SurfacePartitioning class.
        
        Args:
            sensitivity (float): Threshold for edge detection and alpha shape.
                                 Value should be between 0.0 to 1.0.
            max_polygons (int): Maximum number of polygons or regions to generate.
        """
        self.sensitivity = sensitivity
        self.max_polygons = max_polygons

    def _detect_edges(self, image):
        """
        Detect edges in the image using Canny edge detection.
        
        Args:
            image (numpy.ndarray): Input grayscale image.
        
        Returns:
            numpy.ndarray: Binary edge map of the image.
        """
        edge_clustering_analyzer = EdgeClustering(image)
        self.edges = edge_clustering_analyzer.detect_edges()
        return self.edges

    def _find_contours(self, edges):
        """
        Find contours from the edge map.
        
        Args:
            edges (numpy.ndarray): Binary edge map.
        
        Returns:
            list: List of detected contours.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _compute_convex_hulls(self, contours):
        """
        Compute convex hulls from the detected contours.
        
        Args:
            contours (list): List of detected contours.
        
        Returns:
            list: List of convex hulls.
        """
        hulls = [cv2.convexHull(c) for c in contours]
        return hulls

    def _alpha_shapes(self, points, alpha):
        """
        Compute an alpha shape (concave hull) for a given set of points.
        
        Args:
            points (numpy.ndarray): Array of 2D points.
            alpha (float): Alpha parameter to control detail in the shape.
        
        Returns:
            list: List of polygons representing the alpha shape.
        """
        if len(points) < 4:  # Alpha shape cannot be computed with less than 4 points
            return Polygon(points)

        tri = Delaunay(points)
        edges = set()
        edge_points = []

        # Loop through the triangles
        for ia, ib, ic in tri.simplices:
            pa, pb, pc = points[ia], points[ib], points[ic]

            # Compute the circumradius
            a = np.linalg.norm(pa - pb)
            b = np.linalg.norm(pb - pc)
            c = np.linalg.norm(pc - pa)
            s = (a + b + c) / 2.0
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_radius = a * b * c / (4.0 * area)

            if circum_radius < 1.0 / alpha:
                edges.add((ia, ib))
                edges.add((ib, ic))
                edges.add((ic, ia))

        for i, j in edges:
            edge_points.append(points[[i, j]])

        return edge_points

    def _region_growing(self, image, seed_point, threshold=10):
        """
        Perform region-growing segmentation starting from a seed point.
        
        Args:
            image (numpy.ndarray): Input grayscale image.
            seed_point (tuple): Starting point for region growing.
            threshold (int): Intensity difference threshold for acceptance.
        
        Returns:
            numpy.ndarray: Binary mask of the grown region.
        """
        height, width = image.shape
        visited = np.zeros_like(image, dtype=bool)
        region = np.zeros_like(image, dtype=np.uint8)

        stack = [seed_point]
        seed_value = image[seed_point]

        while stack:
            x, y = stack.pop()

            if visited[y, x]:
                continue

            visited[y, x] = True
            intensity_diff = abs(int(image[y, x]) - int(seed_value))

            if intensity_diff <= threshold:
                region[y, x] = 255

                # Check neighbors
                for nx, ny in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        stack.append((nx, ny))

        return region

    def apply(self, image):
        """
        Apply the surface partitioning process to an image.

        Args:
            image (numpy.ndarray): Input image (grayscale).

        Returns:
            dict: Dictionary containing regions, polygons, and debug information.
        """
        # Step 1: Detect edges
        edges = self._detect_edges(image)

        # Step 2: Detect contours
        contours = self._find_contours(edges)

        # Step 3: Compute regions (Convex Hulls and/or Alpha Shapes)
        regions = []
        convex_hulls = []
        for contour in contours:
            if len(regions) >= self.max_polygons:
                break

            # Ensure the contour is valid and has enough points
            if len(contour) < 3:
                continue

            points = contour.squeeze()
            if points.ndim != 2 or points.shape[1] != 2:
                continue  # Skip if points are not 2D

            # Ensure the points have the correct datatype
            points = points.astype(np.float32)

            # Alpha shape or Convex Hull
            if len(points) >= 4:
                alpha_shape = self._alpha_shapes(points, alpha=self.sensitivity)
                regions.append(alpha_shape)
            else:
                convex_hull = cv2.convexHull(points)
                convex_hulls.append(convex_hull)

        # Step 4: Region Growing (optional)
        grown_region = self._region_growing(image, seed_point=(image.shape[1] // 2, image.shape[0] // 2))

        return {
            "edges": edges,
            "convex_hulls": convex_hulls,
            "alpha_shapes": regions,
            "region_grown": grown_region,
        }
