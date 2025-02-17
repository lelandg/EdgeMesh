# Python | edge_detection.py
import cv2
import numpy as np

def detect_and_project_edges(image, low_threshold, high_threshold, thickness=1, project_on_original=False):
    """
    Detects edges in an image and returns an image where the edges are projected.

    :param image: Path to the input image. Can be an image instance or a path.
    :param low_threshold: Lower threshold for edge detection.
    :param high_threshold: Higher threshold for edge detection.
    :param thickness: Thickness of the edges.
    :param project_on_original: Whether to project edges on the original image or a black canvas.
    :return: RGB Image with projected edges.
    """
    # Read image (OpenCV defaults to BGR color space)
    if isinstance(image, str):
        original_image = cv2.imread(image, cv2.IMREAD_COLOR)
    else:
        original_image = image

    if original_image is None:
        raise ValueError("Could not read image. Check the image path.")

    if original_image.ndim == 3:
        # Convert Original Image to RGB (if required)
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif original_image.ndim == 2:
        # Convert Grayscale Image to RGB
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("Invalid image format. Please provide an image in RGB or Grayscale format.")

    is_grayscale = original_image.ndim == 2
    if is_grayscale:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Convert to grayscale for edge detection
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    # Perform edge detection
    edges = cv2.Canny(grayscale_image, low_threshold, high_threshold)

    # Apply thickness to edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
    thick_edges = cv2.dilate(edges, kernel)

    # Initialize output
    if project_on_original:
        if is_grayscale:
            # If the original image was grayscale, convert it to RGB
            result_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            result_image = rgb_image.copy()
        # Project edges in red color (RGB format: dark gray = [20, 20, 20])
        result_image[thick_edges > 0] = [20, 20, 20]
    else:
        # If not projecting on the original, use a black background
        result_image = np.zeros_like(rgb_image)
        # Project edges in white [255, 255, 255]
        result_image[thick_edges > 0] = [255, 255, 255]

    # Return the final image (explicitly RGB for display compatibility)
    return result_image

def detect_edges(image, low_threshold, high_threshold, thickness=1, project_on_original=False):
    return detect_and_project_edges(image, low_threshold, high_threshold, thickness, project_on_original)

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