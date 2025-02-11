# Python | edge_detection.py
import cv2
import numpy as np


def detect_edges(image_path, low_threshold, high_threshold, thickness=1):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, low_threshold, high_threshold)

    # Apply thickness to edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    edges = cv2.dilate(edges, kernel)

    return edges

# def detect_edges(image_path, low_threshold=50, high_threshold=150):
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
#
#     if img is None:
#         raise ValueError("Image not found")
#
#     # Perform edge detection using OpenCV's CPU-based Canny method
#     edges = cv2.Canny(img, low_threshold, high_threshold)
#
#     return edges