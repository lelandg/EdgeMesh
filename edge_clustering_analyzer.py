import cv2
import numpy as np
from sklearn.cluster import DBSCAN


class EdgeClustering:
    def __init__(self, image, edge_thresholds=(50, 150), cluster_eps=10, min_samples=10):
        """
        Initialize the EdgeClustering class.

        Args:
            image: Input image as a numpy array.
            edge_thresholds: Canny edge detection thresholds.
            cluster_eps: Maximum distance between points for DBSCAN clustering.
            min_samples: Minimum number of points to form a cluster in DBSCAN.
        """
        self.image = image
        self.edge_thresholds = edge_thresholds
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples

    def detect_edges(self):
        """
        Perform edge detection using the Canny edge detector.

        Returns:
            edges: Binary image with edges detected.
        """
        if len(self.image.shape) == 2:  # Image is already grayscale
            gray = self.image
        elif len(self.image.shape) == 3 and self.image.shape[2] == 3:  # Color image (BGR)
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image format. Ensure the input image has 1 or 3 channels.")

        # Ensure the image is of type uint8
        if gray.dtype != np.uint8:
            gray = (gray / gray.max() * 255).astype(np.uint8) if gray.max() > 1 else gray.astype(np.uint8)

        edges = cv2.Canny(gray, *self.edge_thresholds)
        return edges

    def detect_lines(self, edges):
        """
        Detect straight lines in the edge-detected image using the Hough Transform.

        Args:
            edges: Binary edge-detected image.

        Returns:
            lines: List of lines represented as (rho, theta).
        """
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        return lines if lines is not None else []

    def find_contours(self, edges):
        """
        Detect contours in the edge-detected image.

        Args:
            edges: Binary edge-detected image.

        Returns:
            contours: List of detected contours.
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def cluster_shapes(self, points):
        """
        Cluster edge points or shapes using DBSCAN.

        Args:
            points: Array of 2D points representing edges or shapes.

        Returns:
            labels: Cluster labels for each point (-1 for noise).
        """
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.min_samples).fit(points)
        return clustering.labels_

    def analyze_edges(self):
        """
        Perform edge detection, clustering, and grouping into potential object boundaries.

        Returns:
            clusters: Dictionary mapping cluster IDs to points.
            contours: List of contours detected.
            lines: List of lines detected.
        """
        # Step 1: Detect edges using Canny
        edges = self.detect_edges()

        # Step 2: Detect lines with Hough Transform
        lines = self.detect_lines(edges)

        # Step 3: Detect contours
        contours = self.find_contours(edges)

        # Step 4: Prepare points for clustering
        edge_points = np.column_stack(np.where(edges > 0))  # Extract non-zero edge points

        # Step 5: Cluster edge points or contours
        labels = self.cluster_shapes(edge_points)

        # Group points by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:  # Noise points
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(edge_points[i])

        return clusters, contours, lines

    def visualize_clusters(self, clusters):
        """
        Visualize clusters on the original image.

        Args:
            clusters: Dictionary mapping cluster IDs to points.
        """
        clustered_image = self.image.copy()
        colors = np.random.randint(0, 255, (len(clusters), 3))

        for cluster_id, points in clusters.items():
            for point in points:
                cv2.circle(clustered_image, tuple(point), 1, colors[cluster_id].tolist(), -1)

        cv2.imshow("Clustered Edges", clustered_image)

    def visualize_results(self, edges, contours, lines):
        """
        Visualize the edges, contours, and lines detected.

        Args:
            edges: Binary edge-detected image.
            contours: List of contours.
            lines: List of lines detected using Hough Transform.
        """
        # Create copies of the original image to draw on
        contour_image = self.image.copy()
        line_image = self.image.copy()

        # Draw contours
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Draw lines
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display results
        cv2.imshow("Edges", edges)
        cv2.imshow("Contours", contour_image)
        cv2.imshow("Lines", line_image)


# Example usage:
if __name__ == "__main__":
    image_path = "path_to_image.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    edge_clustering = EdgeClustering(image)
    clusters, contours, lines = edge_clustering.analyze_edges()

    # Visualize the result
    edge_clustering.visualize_clusters(clusters)
    edges = edge_clustering.detect_edges()
    edge_clustering.visualize_results(edges, contours, lines)