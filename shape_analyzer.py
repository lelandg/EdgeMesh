import cv2
import numpy as np


class ShapeAnalysis:
    def __init__(self, clustered_contours):
        """
        Initialize the ShapeAnalysis class with clustered contours.
        
        Parameters:
        - clustered_contours: List of contours obtained from edge clustering.
        """
        self.clustered_contours = clustered_contours

    def approximate_contours(self, epsilon_factor=0.02):
        """
        Approximate contours to polygons using cv2.approxPolyDP.
        
        Parameters:
        - epsilon_factor: A factor to adjust the approximation accuracy, higher values simplify more.
        
        Returns:
        - List of approximated polygons as contours.
        """
        approximated_shapes = []
        for contour in self.clustered_contours:
            # Epsilon determines the degree of approximation
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approximated_shapes.append(approx)
        return approximated_shapes

    def fit_ellipses(self):
        """
        Fit ellipses to contours using cv2.fitEllipse.
        
        Returns:
        - List of ellipse parameters (center, axes, angle) for applicable contours.
        """
        ellipses = []
        for contour in self.clustered_contours:
            # Only fit ellipses for contours with enough points
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ellipses.append(ellipse)
        return ellipses

    def extract_geometric_primitives(self):
        """
        Analyze contours and extract both polygons and ellipses.
        
        Returns:
        - Dictionary containing:
          - 'polygons': List of approximated polygons.
          - 'ellipses': List of fitted ellipse parameters.
        """
        polygons = self.approximate_contours()
        ellipses = self.fit_ellipses()
        return {
            "polygons": polygons,
            "ellipses": ellipses
        }