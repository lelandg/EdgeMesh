# New File: MeshGenerator.py

import cv2
import numpy as np

from depth_to_3d import DepthTo3D
from edge_clustering_analyzer import EdgeClustering
from shape_analyzer import ShapeAnalysis
from depth_cue_estimator_util import DepthCueEstimator
from surface_partitioning import SurfacePartitioning
from depth_based3_d_reconstruction import ExtrusionProjectionReconstruction


class MeshGenerator:
    def __init__(self, options=None):
        self.visualize_clustering = options.get("visualize_clustering", False)
        self.visualize_depth = options.get("visualize_depth", False)
        self.visualize_partitioning = options.get("visualize_partitioning", True)
        self.visualize_edges = options.get("visualize_edges", False)

    def sharpen_image(self, image):
        """Sharpen the processed image for more prominent edges."""
        image = image.astype(np.float32)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        laplacian = cv2.Laplacian(sharpened, cv2.CV_32F)
        sharpened = sharpened - 0.2 * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def generate(self, processed_image):
        """Handle the mesh generation process."""
        if processed_image is None:
            raise ValueError("No processed image provided for mesh generation.")

        # Resize image if necessary
        image_height, image_width = processed_image.shape[:2]
        if image_width > 1000:
            scale_factor = 1000 / image_width
            new_height = int(image_height * scale_factor)
            processed_image = cv2.resize(processed_image, (1000, new_height), interpolation=cv2.INTER_AREA)

        # Sharpen the image
        processed_image = self.sharpen_image(processed_image)

        # Edge clustering
        edge_clustering = EdgeClustering(processed_image)
        clusters, contours, lines = edge_clustering.analyze_edges()

        if self.visualize_clustering:
            edge_clustering.visualize_clusters(clusters)
            edge_clustering.visualize_results(edge_clustering.detect_edges(), contours, lines)

        # Shape analysis
        shape_analyzer = ShapeAnalysis(clustered_contours=contours)
        geometric_primitives = shape_analyzer.extract_geometric_primitives()
        polygons, ellipses = geometric_primitives.get("polygons"), geometric_primitives.get("ellipses")

        # Depth estimation
        depth_estimator = DepthCueEstimator()
        light_depth = depth_estimator.estimate_depth_from_light(processed_image)
        shading_depth = depth_estimator.estimate_depth_from_shading(processed_image)
        combined_depth = depth_estimator.combine_depth_cues(light_depth, shading_depth)

        if self.visualize_depth:
            cv2.imshow("Light-based Depth", light_depth)
            cv2.imshow("Shading-based Depth", shading_depth)
            cv2.imshow("Combined Depth", combined_depth)

        # Surface partitioning
        partitioner = SurfacePartitioning(sensitivity=0.4, max_polygons=5)
        result = partitioner.apply(processed_image)

        if self.visualize_partitioning:
            cv2.imshow("Region-Grown Output", result["region_grown"])

        if self.visualize_edges:
            cv2.imshow("Edges", result["edges"])
            for idx, hull in enumerate(result["convex_hulls"]):
                hull_image = np.zeros_like(processed_image)
                cv2.drawContours(hull_image, [hull], -1, 255, 2)
                cv2.imshow(f"Convex Hull {idx + 1}", hull_image)

        # Extrusion projection
        extrusion_projector = ExtrusionProjectionReconstruction(contours)
        extruded_edges = extrusion_projector.extrude()

        depth_to_3d = DepthTo3D(model_type="depth_anything_v2")
        depth_to_3d.process_image(processed_image)

        return extruded_edges