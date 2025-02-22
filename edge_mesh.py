__author__ = "Leland Green"

from PyQt5 import QtWidgets

import image_processor
from _version import version
__version__ = version
__date_created__ = "2025-01-28"
__email__ = "lelandgreenproductions@gmail.com"

__license__ = "Commercial. License Required." # License of this script is free for all purposes.

import traceback

from flow_layout import FlowLayout
from smoothing_depth_map_utils import SmoothingDepthMapUtils

debug = True # Set False to disable debug messages. Yes. We do need this.
verbose = True # Not used, yet. When I add logging, this will also print messages to console when enabled.
visualize_images = False # Set to True to visualize images in a GUI window. Requires OpenCV.

from depth_to_3d import DepthTo3D
from edge_detection import detect_edges
from mesh_generator import MeshGenerator
from viewport_3d import ThreeDViewport

visualize_clustering = visualize_images # Visualization flags for MeshGenerator
visualize_depth = visualize_images
visualize_partitioning = visualize_images
visualize_edges = visualize_images

import datetime
f"""
Utilities using opencv for edge detection in a GUI. 
Also features depth map generation via select (implemented) methods via torch, etc. 
Then 3D mesh generation from the depth map.

Author: {__author__}
Version: {__version__}
Date: {__date_created__}
Last Updated: {datetime.datetime.now()}
Email: {__email__}

Requirements:
- Python 3.8 or newer
- Modules in python_requirements.txt. Run "pip install -r python_requirements.txt" to install.

How to Run:
Run this script using the command `python edge_mesh.py`.

License:
{__license__}

** Moved Version History to ReadMe.md
"""

import configparser
import os
import sys
import threading

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QSlider,
    QPushButton, QWidget, QHBoxLayout, QSizePolicy, QCheckBox, QComboBox, QLineEdit, QSpinBox
)
from PyQt5.QtCore import Qt, QSize, QEvent, QByteArray, QTimer
import PyQt5.QtGui as QtGui
from PyQt5.QtGui import QPixmap, QImage, QIntValidator, QIcon, QDoubleValidator, QResizeEvent

import cv2
import numpy as np
import open3d as o3d



def process_preview_image(image, is_grayscale=False, invert_colors=False):
    """
    Ensures preview images are handled consistently by converting to RGB color space.
    Applies optional grayscale and invert-color transformations.

    :param image: Input image (expected as BGR or grayscale).
    :param is_grayscale: Whether to convert the image to grayscale.
    :param invert_colors: Whether to invert the colors for the preview.
    :return: Processed image suitable for preview (RGB format).
    """
    # Handle None input
    if image is None:
        raise ValueError("Provided image is None.")

    # Convert to RGB if the image is in BGR color space
    if len(image.shape) == 3:  # Color image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply grayscale conversion if needed
    if is_grayscale:
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)  # Keep 3 channels for RGB compatibility

    # Invert colors if needed
    if invert_colors:
        image = cv2.bitwise_not(image)
    return image

class MainWindow_ImageProcessing(QMainWindow):
    def __init__(self):
        super().__init__()
        self._layout_references = []
        self._checkbox_references = []
        self.line_thickness = 2
        self.sensitivity = 150
        self.blend_amount = 100
        self.initialized = False
        self.resolution = 700
        self.depth_amount = 1.0
        self.max_depth = 50.0
        self.background_tolerance = 10  # Default value
        self.grayscale_enabled = False
        self.use_processed_image_enabled = True
        self.project_on_original = True
        self.invert_colors_enabled = False
        self.edge_thickness = 1  # Default line thickness
        self.flat_back_enabled = False
        self.edge_detection_enabled = True
        self.depth_labels = None
        self.depth_values = None
        self.depth_drop_percentage = 0.0  # Default percentage depth drop
        self.image_path = None  # To hold the currently loaded image path
        self.background_color = [0, 0, 0]  # Default background color = dark gray
        self.current_selected_color = (255, 255, 255)  # Default to white
        self.use_selected_color = False
        self.drop_background_enabled = False  # Initialize the variable
        self.previous_background_enabled = False  # Store the previous state
        self.three_d_viewport = None
        self.original_pixmap = None
        self.image = None
        self.processed_image = None
        self.extruded_edges = None
        self.eyedropper_active = False

        try:

            # Set the icon for the main window
            icon_path = os.path.join(os.getcwd(), "EdgeMesh.ico")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))  # Set the main window's icon

            self.setWindowTitle(f"3D Mesh Generator v{__version__}")
            self.setGeometry(50, 50, 929, 560) # Weird size, just looks good on my screen (today)

            os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"  # Enable OneDNN optimizations for TensorFlow
            # Initialize configuration
            # Config file path
            self.CONFIG_FILE_PATH = "config.ini"
            self.config = self.initialize_config()

            self._init_ui()
            self._load_config()

            o3d.visualization.webrtc_server.enable_webrtc()
            print(f"Open3D version: {o3d.__version__}")
        except Exception as e:
            print("Error initializing MainWindow:")
            print(traceback.format_exc())

    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout and Controls
        self.edge_sensitivity_layout = QVBoxLayout()  # For images and controls
        image_preview_layout = QHBoxLayout()  # To hold original and processed previews

        palette = self.palette()
        self.default_color = palette.color(QtGui.QPalette.Window)
        # Original Image Preview
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)
        self.original_label.setStyleSheet(f"background-color: {self.default_color.name()};")
        self.original_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.original_label.setMinimumSize(150,100)
        self.original_label.setToolTip("This is the original image. Load an image to start processing.\r\n"
                                      'Click "Depth Mesh" with "Use Processed Image" DISABLED to generate a 3D mesh '
                                      "from this image.")
        # self.original_label.setMaximumHeight(700)  # Limit height

        # Processed Image Preview
        self.preview_label = QLabel("Processed Image")
        self.preview_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)
        self.preview_label.setStyleSheet("background-color: default_color.name();")
        self.preview_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.preview_label.setMinimumSize(150,100)
        self.preview_label.setToolTip("This is the processed image, based on your selections below. \r\n"
                                      'Click "Save Image" to save what you see here..\r\n'
                                      'Click "Depth Mesh" with "Use Processed Image" enabled to generate a 3D mesh '
                                      "from this image.")

        # Add "Edge Detection" Option

        layout_edge, self.edge_detection_checkbox = self.create_label_and_checkbox_container(
            "Edge Detection", "Enable edge detection to highlight edges in the image.",
            self.toggle_edge_detection, True)

        # Add the "Project on Original" checkbox
        tooltip = "Project the processed image on the original image. This can create interesting effects."
        layout_project, self.project_on_original_checkbox = self.create_label_and_checkbox_container(
            "Project on Original", tooltip, self.toggle_project_on_original)

        # Add "Grayscale" Option
        tooltip = "Convert the image to grayscale before processing. This can help with edge detection."
        layout_grayscale, self.grayscale_checkbox = self.create_label_and_checkbox_container(
            "GrayScale", tooltip, self.toggle_grayscale)



        # Slider for Sensitivity
        label_sensitivity = QLabel("Edge Detection Sensitivity")
        label_sensitivity.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(200)
        self.sensitivity_slider.setValue(self.sensitivity)
        self.sensitivity_slider.setMinimumSize(100, 0)
        self.sensitivity_slider.setToolTip("Adjust the edge detection sensitivity. Higher values detect more edges.")
        self.sensitivity_slider.valueChanged.connect(self.update_sensitivity)

        # Slider for Line Thickness
        line_thickness_label = QLabel("Line Thickness")
        line_thickness_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setMinimum(1)
        self.line_thickness_slider.setMaximum(10)
        self.line_thickness_slider.setValue(self.edge_thickness)
        self.line_thickness_slider.setMinimumSize(100, 0)
        self.line_thickness_slider.setToolTip("Adjust the thickness of the lines drawn for detected edges.")
        self.line_thickness_slider.valueChanged.connect(self.update_line_thickness)

        blend_label = QLabel("Blend Amount")
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setMinimum(0)
        self.blend_slider.setMaximum(100)
        self.blend_slider.setValue(self.blend_amount)
        self.blend_slider.setMinimumSize(100, 0)
        self.blend_slider.setToolTip("Blend between the original and processed images. 0 = original, 100 = processed.\r\n"
                                     "Be sure \"Use Processed Image\" is enabled to see the effect in 3D meshes.\r\n"
                                     "\"Save Image\" always saves the processed image, as you see it on the right.")

        self.blend_slider.valueChanged.connect(self.update_blend)

        # Add "Invert Colors" Option
        tooltip = "Invert the colors of the processed image. Useful for edge detection and creates interesting effects."
        layout_invert, self.invert_checkbox = self.create_label_and_checkbox_container(
            "Invert Colors", tooltip, self.toggle_invert_colors, self.invert_colors_enabled)

        # 3D Viewport (Fill most of the window) - Moved below. Now created/updated when a new mesh is ready.
        # self.three_d_viewport = ThreeDViewport()
        # self.three_d_viewport.setStyleSheet("background-color: #ddd;")
        # self.three_d_viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # main_layout.addWidget(self.three_d_viewport)

        # Buttons for Load, Save, Depth Mesh, Mesh from 2D, and Export Mesh
        self.load_button = QPushButton("&Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setToolTip("Load an image to process.")

        self.save_button = QPushButton("&Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setToolTip("Save the processed image. What you see in the right hand image is what you get.")

        depth_method_label = QLabel("Depth Method")
        depth_method_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        # Create ComboBox
        self.depth_method_dropdown = QComboBox()
        self.depth_method_dropdown.addItems(["DepthAnythingV2", "DPT", "MiDaS"])  # Add items
        self.depth_method_dropdown.setCurrentIndex(0)
        self.depth_method_dropdown.setItemData(0, "DepthAnythingV2 is usually better for most images.", Qt.ToolTipRole)
        self.depth_method_dropdown.setItemData(1, "DPT can be good for architecture and hard surfaces.", Qt.ToolTipRole)
        self.depth_method_dropdown.setItemData(2, "MiDaS is sometimes good for general depth.", Qt.ToolTipRole)
        self.depth_method_dropdown.setToolTip("Depth estimation model to use.\r\n"
                                              "DepthAnythingV2 is usually better, but "
                                              "DPT is sometimes better for architecture or hard surfaces.\r\n"
                                              "MiDaS is good for general depth.\r\n"
                                              "Results are very different, so experiment!\r\n"
                                              "Remember, this works with Smoothing Method. "
                                              "So try different combinations!")

        depth_smoothing_label = QLabel("Depth Smoothing")
        depth_smoothing_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        self.smoothing_dropdown = QComboBox()
        self.smoothing_dropdown.addItems(["anisotropic", "gaussian", "bilateral", "median", "(none)"])
        self.smoothing_dropdown.setItemData(0, "Anisotropic diffusion (edge-aware smoothing). Usually best", Qt.ToolTipRole)
        self.smoothing_dropdown.setItemData(1, "Gaussian smoothing. Can be better for some images.", Qt.ToolTipRole)
        self.smoothing_dropdown.setItemData(2, "Bilateral smoothing.", Qt.ToolTipRole)
        self.smoothing_dropdown.setItemData(3, "Median smoothing.", Qt.ToolTipRole)
        self.smoothing_dropdown.setItemData(4, "No smoothing applied. Not recommended!! But you can try it! :-)", Qt.ToolTipRole)
        self.smoothing_dropdown.setToolTip("Select smoothing method for the depth map.")

        resolution_label = QLabel("Resolution")
        resolution_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        # Add the numeric input box for resolution
        self.resolution_input = QLineEdit()
        self.resolution_input.setValidator(QIntValidator(0, 10000))  # Allows numeric input in the range 0-10000
        self.resolution_input.setFixedWidth(50)
        self.resolution_input.setText("700")
        self.resolution_input.setToolTip("Enter the resolution for the depth map.\n"
                                         "0 = full size. Can take a while for large images!\n"
                                         "1000 is almost always good. Default: 700")
        self.resolution_input.textChanged.connect(self.update_resolution)

        depth_amount_label = QLabel("Depth Amount")
        depth_amount_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)

        # Add an input field for Depth Amount, with validation
        self.depth_amount_input = QLineEdit()
        self.depth_amount_input.setValidator(QDoubleValidator(0.0, 100.0, 5, self))
        self.depth_amount_input.setFixedWidth(100)
        self.depth_amount_input.setText("1.0")  # Default depth = 1.0 (current depth)
        self.depth_amount_input.setToolTip("Depth Amount multiplier.\nHigher values increase the depth.\n"
                                           "0.5 = half size. 2.0 = double size.\n"
                                           "0.0 = flat, generates a plane with image.\n"
                                           'Default: 1.0 = "normal" depth')

        min_depth_label = QLabel("Minimum Depth")
        min_depth_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)
        min_depth_label.setEnabled(False)  # Disable for now

        # Add an input field for Max Depth, validate from 0.0 to 1.0 (floating-point)
        self.min_depth_input = QLineEdit()
        double_validator = QDoubleValidator(1.0, 10000.0, 5, self)  # Ranges from 0.0 to 1000
        double_validator.setNotation(QDoubleValidator.StandardNotation)  # Use standard notation (no scientific)
        self.min_depth_input.setValidator(double_validator)
        self.min_depth_input.setFixedWidth(100)
        self.min_depth_input.setText("0.0")  # Default max depth = 1.0 (full range)
        self.min_depth_input.setToolTip("Minimum Depth. Min = 0.0 (unmodified), Max = 10000.0, which is crazy deep.\n"
                                        "Default: 0.0")
        self.min_depth_input.setEnabled(False)  # Disable for now

        percentage_label = QLabel(" Depth Drop (%):")
        percentage_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)
        self.percentage_input = QLineEdit(self)
        self.percentage_input.setFixedWidth(100)
        self.percentage_input.setToolTip("Set the percentage depth drop for the mesh. The farthest points will be dropped by this percentage")
        self.percentage_input.setText("0.0")  # Default percentage depth drop
        validator = QtGui.QIntValidator(0, 100)  # Limits input to 0 - 100
        self.percentage_input.setValidator(validator)
        self.percentage_input.editingFinished.connect(self.update_percentage_depth_drop)

        tooltip = "Enable this option to remove the background color from the image.\n"\
                  "This only works when all four corners are the same color."
        layout_remove_background, self.drop_background_checkbox = self.create_label_and_checkbox_container(
            "Remove Background", tooltip, self.toggle_drop_background)

        # Background tolerance input
        self.background_tolerance_label = QLabel("Background Tolerance:")
        self.background_tolerance_input = QSpinBox()
        self.background_tolerance_input.setToolTip(
            "Set the tolerance for background removal. 1 will remove only that color.\n"
            "Higher values will remove similar colors. 255 will remove all colors, so keep it low, usually.\n"
            "See ReadMe.md for more info.")
        self.background_tolerance_input.setRange(0, 255)  # Example range
        self.background_tolerance_input.setValue(self.background_tolerance)

        # Connect the value change to the handler
        self.background_tolerance_input.valueChanged.connect(self.update_background_tolerance)

        tooltip = "Enable this option to create a model with a flat back.\n"\
                  "This option is usually better for 3D printing.\n"\
                  "When disabled, the back will be shaped to match the front."

        layout_flat, self.flat_back_checkbox = self.create_label_and_checkbox_container(
            "Flat Back", tooltip, self.toggle_flat_back)

        tooltip = "Enable this to save and use the image on the right (the processed image) to generate the depth map. You can get crazy results with edge detection overlayed on original.\n"\
                  "When not enabled, the original image is always used, no matter what options you have set."
        layout_use_processed, self.use_processed_image_checkbox = self.create_label_and_checkbox_container("Use Processed Image", tooltip, self.toggle_use_processed_image)

        # Add Depth Mesh Button
        self.process_button = QPushButton("&Depth Mesh")
        self.process_button.setShortcut('Alt+D')
        self.process_button.setDefault(True)
        self.process_button.setToolTip("Generate a 3D mesh from the original or processed image.")
        self.process_button.clicked.connect(self.process_image)

        #TODO: Fix Mesh from 2D
        # Add Mesh From 2D Button
        self.generate_mesh_button = QPushButton("&Mesh From Edges")
        self.generate_mesh_button.setShortcut('Alt+M')
        self.generate_mesh_button.setToolTip("Use edge analysis to generate an approximated mesh from the processed image. (Currently broken.)")
        self.generate_mesh_button.clicked.connect(self.generate_mesh)

        # Additional "Export Mesh" button
        self.export_mesh_button = QPushButton("&Export Mesh")
        self.export_mesh_button.setShortcut('Alt+E')
        self.export_mesh_button.setToolTip("Export the generated 3D mesh as a .PLY file.\n**NOTE** You do NOT need to use this because the mesh is automagically saved in the same folder as the source image.")
        self.export_mesh_button.clicked.connect(self.export_mesh)

        self.reset_defaults_button = QPushButton("&Reset Defaults")  # Button labeled "Reset Defaults"
        self.reset_defaults_button.setToolTip("Reset all settings to their default values.")  # Set tooltip
        self.reset_defaults_button.setShortcut('Alt+R')  # Set keyboard shortcut
        self.reset_defaults_button.clicked.connect(self.reset_defaults)  # Connect button to reset_defaults method

        selected_color_label = QLabel("Selected Color")
        selected_color_label.setAlignment(Qt.AlignRight | Qt.AlignBaseline)
        # Add a color picker display (readonly text box with the color value)
        self.color_display = QLabel(self)
        self.color_display.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.color_display.setMinimumSize(QSize(50, 15))
        self.color_display.setMaximumSize(QSize(100, 15))
        self.color_display.setAutoFillBackground(True)
        self.color_display.setToolTip("The selected color for background removal.")
        self.update_color_swatch(self.current_selected_color)  # Initialize swatch color

        # Add a "Pick Color" button
        self.pick_color_button = QPushButton("&Pick Color")
        self.pick_color_button.setShortcut('Alt+P')
        self.pick_color_button.setToolTip("Click this button, then click anywhere on the screen to pick a color.")
        self.pick_color_button.clicked.connect(self.enable_color_picker_mode)

        self.clear_color_button = QPushButton("&Clear Color")
        self.clear_color_button.clicked.connect(self.clear_color_picker)
        self.clear_color_button.setToolTip("Clear the selected color.\n(Does not affect 'Remove Background' settings.) "
                                           "If no color is selected, a background color is detected if all four corners are the same color.")
        self.clear_color_button.setShortcut('Alt+C')

        # Finally, add everything to the GUI:
        # Groupbox for Images
        images_groupbox = QtWidgets.QGroupBox("Images")
        images_layout = QtWidgets.QHBoxLayout()
        images_layout.setContentsMargins(5, 5, 5, 5)
        images_layout.addWidget(self.original_label)
        images_layout.addWidget(self.preview_label)
        images_groupbox.setLayout(images_layout)

        image_processing_groupbox = QtWidgets.QGroupBox("Image Preprocessing")
        image_processing_layout = FlowLayout()
        # image_processing_layout.addWidget(self.grayscale_checkbox)
        # image_processing_layout.addWidget(self.invert_checkbox)
        # image_processing_layout.addWidget(self.project_on_original_checkbox)
        # image_processing_layout.addWidget(self.edge_detection_checkbox)
        image_processing_layout.addWidget(self.layout_to_container(layout_grayscale))
        image_processing_layout.addWidget(self.layout_to_container(layout_invert))
        image_processing_layout.addWidget(self.layout_to_container(layout_project))
        image_processing_layout.addWidget(self.layout_to_container(layout_edge))
        sensitivity_hbox = FlowLayout()
        sensitivity_hbox.addWidget(label_sensitivity)
        sensitivity_hbox.addWidget(self.sensitivity_slider)
        image_processing_layout.addWidget(self.layout_to_container(sensitivity_hbox))

        hlayout_line_thickness = QtWidgets.QHBoxLayout()
        hlayout_line_thickness.addWidget(line_thickness_label)
        line_thickness_hbox = FlowLayout()
        line_thickness_hbox.addWidget(line_thickness_label)
        line_thickness_hbox.addWidget(self.line_thickness_slider)
        image_processing_layout.addWidget(self.layout_to_container(line_thickness_hbox))

        blend_hbox =  FlowLayout()
        blend_hbox.addWidget(blend_label)
        blend_hbox.addWidget(self.blend_slider)
        image_processing_layout.addWidget(self.layout_to_container(blend_hbox))

        image_processing_groupbox.setLayout(image_processing_layout)
        image_processing_groupbox.setContentsMargins(5, 15, 5, 5)

        # # Groupbox for Image Processing
        # Groupbox for Depth Mapping
        depth_groupbox = QtWidgets.QGroupBox("Mesh Generation")
        depth_layout = FlowLayout()
        depth_layout.setContentsMargins(5, 5, 5, 5)
        hlayout_method = QtWidgets.QHBoxLayout()
        hlayout_method.addWidget(depth_method_label)
        hlayout_method.addWidget(self.depth_method_dropdown, alignment=Qt.AlignBaseline)
        depth_layout.addWidget(self.layout_to_container(hlayout_method))

        # Add depth smoothing method dropdown
        hlayout_smoothing = QtWidgets.QHBoxLayout()
        hlayout_smoothing.addWidget(depth_smoothing_label)
        hlayout_smoothing.addWidget(self.smoothing_dropdown, alignment=Qt.AlignBaseline)
        depth_layout.addWidget(self.layout_to_container(hlayout_smoothing))

        # Add resolution input
        hlayout_resolution = QtWidgets.QHBoxLayout()
        hlayout_resolution.addWidget(resolution_label)
        hlayout_resolution.addWidget(self.resolution_input)
        depth_layout.addWidget(self.layout_to_container(hlayout_resolution))

        # Add depth amount and max depth inputs
        hlayout_depth_amount = QtWidgets.QHBoxLayout()
        hlayout_depth_amount.addWidget(depth_amount_label)
        hlayout_depth_amount.addWidget(self.depth_amount_input)
        depth_layout.addWidget(self.layout_to_container(hlayout_depth_amount))

        hlayout_max_depth = QtWidgets.QHBoxLayout()
        hlayout_max_depth.addWidget(min_depth_label)
        hlayout_max_depth.addWidget(self.min_depth_input)
        depth_layout.addWidget(self.layout_to_container(hlayout_max_depth))

        # Add percentage depth drop input
        hlayout_percentage = QtWidgets.QHBoxLayout()
        hlayout_percentage.addWidget(percentage_label)
        hlayout_percentage.addWidget(self.percentage_input)
        depth_layout.addWidget(self.layout_to_container(hlayout_percentage))

        # Add background removal options
        # depth_layout.addWidget(self.flat_back_checkbox)
        depth_layout.addWidget(self.layout_to_container(layout_flat))
        # depth_layout.addWidget(self.drop_background_checkbox)
        depth_layout.addWidget(self.layout_to_container(layout_remove_background))

        hlayout_tolerance = QtWidgets.QHBoxLayout()
        hlayout_tolerance.addWidget(self.background_tolerance_label)
        hlayout_tolerance.addWidget(self.background_tolerance_input)
        depth_layout.addWidget(self.layout_to_container(hlayout_tolerance))

        # Add color options and picker buttons
        hlayout_color = QtWidgets.QHBoxLayout()
        hlayout_color.addWidget(selected_color_label)
        hlayout_color.addWidget(self.color_display)
        hlayout_color.addWidget(self.pick_color_button)
        hlayout_color.addWidget(self.clear_color_button)
        depth_layout.addWidget(self.layout_to_container(hlayout_color))

        depth_layout.addWidget(self.layout_to_container(layout_use_processed))

        depth_groupbox.setLayout(depth_layout)
        depth_groupbox.setContentsMargins(5, 15, 5, 5)

        # Button layout
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.reset_defaults_button)
        button_layout.addWidget(self.export_mesh_button)
        button_layout.addWidget(self.generate_mesh_button)
        button_layout.addWidget(self.process_button)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Add groupboxes and button layout to the main layout
        main_layout.addWidget(images_groupbox)
        main_layout.addWidget(image_processing_groupbox)
        main_layout.addWidget(depth_groupbox)
        main_layout.addLayout(button_layout)
        self.central_widget.setLayout(main_layout)
        self.process_button.setFocus()

        self.initialized = True
        self.load_ui_settings()
        self.load_last_used_image()
        self.central_widget.setMinimumSize(525, 475)
        self.setMinimumSize(525, 475)
        QTimer.singleShot(500, self.display_processed_image)  # Load last used image after 1 second
        QTimer.singleShot(500, self.display_original_image)  # Load last used image after 1 second


    def create_label_and_checkbox_container(self, label_text: str, tooltip: str = "", state_changed: callable = None,
                                            checked: bool = False) -> (QVBoxLayout, QCheckBox):
        """
        Creates a QWidget containing a QVBoxLayout with a label and a checkbox.

        Args:
            label_text (str): The text to display in the QLabel.
            tooltip (str): The tooltip text to display when hovering over the label.
            state_changed (callable): A callable function to be called when the checkbox state changes.
            checked (bool): The initial state of the QCheckBox.

        Returns:
            QWidget: A container with the QLabel above a QCheckBox.
        """
        # Create the container widget
        container = QWidget(self)  # Assign self as the parent, ensuring it's managed by the parent widget

        # Create a QVBoxLayout and set it for the container
        layout = QVBoxLayout(container)
        self._layout_references.append(layout)

        # Create the label and set its text
        label = QLabel(label_text, container)  # Explicitly assign the parent for the label
        if tooltip:
            label.setToolTip(tooltip)
        layout.addWidget(label)  # Add the label to the layout

        # Create the checkbox
        checkbox = QCheckBox(container)  # Assign parent to make it persistent
        checkbox.setChecked(checked)  # Set the initial state
        if state_changed:
            checkbox.stateChanged.connect(state_changed)
        if tooltip:
            checkbox.setToolTip(tooltip)
        layout.addWidget(checkbox)  # Add the checkbox to the layout

        # Save a reference to the checkbox within the class to avoid garbage collection
        self._checkbox_references.append(checkbox)

        # Return the container (the checkbox is now part of this container and has a parent)
        return layout, checkbox

    def layout_to_container(self, layout_widget ):
        layout_widget.setContentsMargins(2, 2, 2, 2)
        layout_widget.setSpacing(2)
        container = QWidget()
        container.setLayout(layout_widget)
        return container

    def clear_color_picker(self):
        """Clear the color picker display."""
        self.color_display.clear()
        self.use_selected_color = False
        self.update_color_swatch(self.default_color)  # Reset to white

    def enable_color_picker_mode(self):
        """
        Enables color picker mode by setting the cursor to a crosshair and waiting for a click.
        """
        self.eyedropper_active = True
        QApplication.setOverrideCursor(QtGui.QCursor(Qt.CrossCursor))  # Change the mouse cursor to crosshair
        self.installEventFilter(self)  # Start monitoring mouse events globally

    def eventFilter(self, source, event):
        """
        Captures mouse click in the window while the eyedropper is active.
        """
        try:
            if self.eyedropper_active and event.type() == QEvent.MouseButtonPress:
                # Disable the eyedropper
                self.eyedropper_active = False
                QApplication.restoreOverrideCursor()  # Restore the original cursor
                self.removeEventFilter(self)

                # Capture the mouse location and fetch color
                mouse_position = event.globalPos()
                color = self.get_color_under_mouse(mouse_position)
                if color:
                    # self.update_color_picker(color)
                    self.use_selected_color = True
                    self.update_color_swatch(color)
                return True  # Event was handled
            return super(QMainWindow, self).eventFilter(source, event)
        except Exception as e:
            print("Error in eventFilter:")
            traceback.print_exc()
            return False

    def get_color_under_mouse(self, global_pos):
        """
        Gets the color at the current mouse position using a QScreen grab.
        """
        screen = QApplication.primaryScreen()

        # Capture the screen at mouse position
        x, y = global_pos.x(), global_pos.y()
        pixmap = screen.grabWindow(0, x, y, 1, 1)
        image = pixmap.toImage()
        color = image.pixel(0, 0)
        qcolor = QtGui.QColor(color)
        return [qcolor.red(), qcolor.green(), qcolor.blue()]

    def update_color_swatch(self, color):
        """
        Updates the color swatch to display the newly selected color.
        """

        if not isinstance(color, QtGui.QColor):
            r, g, b = color
            color = QtGui.QColor(r, g, b)

        self.update_color_picker(color)
        palette = self.color_display.palette()
        palette.setColor(QtGui.QPalette.Window, color)
        self.color_display.setPalette(palette)

    def update_color_picker(self, color):
        """
        Updates the color picker display and stores the detected color.
        """
        self.current_selected_color = [color.red(), color.green(), color.blue()]
        print(f"Current selected color as RGB: {self.current_selected_color}")  # Output the selected color
        font_color = [255 - color.red(), 255 - color.green(), 255 - color.blue()]
        self.color_display.setStyleSheet("color: rgb({0}, {1}, {2});".format(*font_color))
        if self.use_selected_color:
            self.color_display.setText(f"[{color.red()}, {color.green()}, {color.blue()}]")
        else:
            self.color_display.setText(" <None> ")

    def update_background_tolerance(self, value):
        """
        Updates the background tolerance value when the input changes.
        """
        self.background_tolerance = value
        print(f"Background tolerance updated to: {self.background_tolerance}")

    def toggle_drop_background(self, state):
        """Toggle the Drop Background feature."""
        self.drop_background_enabled = state == Qt.Checked
        print(f"Background removal enabled: {self.drop_background_enabled}")

    def update_sensitivity(self):
        """Update the edge detection sensitivity based on the slider value."""
        self.sensitivity = self.sensitivity_slider.value()
        print(f"Edge Detection Sensitivity: {self.sensitivity}")
        self.update_preview()

    def update_percentage_depth_drop(self):
        """
        Callback to update the depth_drop_percentage value when the input changes.
        """
        try:
            if self.percentage_input.text() == "" or self.percentage_input.text() == "0":
                value = 0.0
            else:
                try:
                    # Get value from the text input box
                    value = float(self.percentage_input.text())
                except ValueError:
                    print(f"Invalid input: Could not convert {self.percentage_input.text()} to float.")
                    return
            self.percentage_input.setText(str(value))
            # Validate range (e.g., between 0–100)
            if 0.0 <= value <= 100.0:
                self.depth_drop_percentage = value
                print(f"Updated depth_drop_percentage to {value}")  # For debugging/logging
            else:
                self.depth_drop_percentage = 0.0  # Reset to default
                print("Invalid value: Percentage must be between 0 and 100.")
                # Optionally: Provide user feedback if out of range
        except ValueError:
            print("Invalid input: Could not convert to float.")
            # Optionally: Provide user feedback if input is invalid

    def toggle_project_on_original(self, state):
        """Handles the toggling of 'Project on Original' checkbox."""
        self.project_on_original = bool(state)  # Store the checkbox state in the attribute
        print(f"Project on Original: {self.project_on_original}")
        self.update_preview()

    def toggle_grayscale(self, state):
        """Enable or disable grayscale mode based on checkbox state."""
        self.grayscale_enabled = state == Qt.Checked
        print(f"Grayscale Enabled: {self.grayscale_enabled}")
        self.update_preview()

    def toggle_edge_detection(self, state):
        """Enable or disable edge detection based on the checkbox state."""
        self.edge_detection_enabled = state == Qt.Checked
        print(f"Edge Detection Enabled: {self.edge_detection_enabled}")
        # self.update_edge_sensitivity_layout()
        self.update_preview()

    def toggle_use_processed_image(self, state):
        """Enable or disable using the processed image based on the checkbox state."""
        self.use_processed_image_enabled = state == Qt.Checked
        print(f"Use Processed Image Enabled: {self.use_processed_image_enabled}")

    def toggle_flat_back(self, state):
        """Enable or disable Dynamic Depth based on the checkbox state."""
        self.flat_back_enabled = state == Qt.Checked
        print(f"Flat back Enabled: {self.flat_back_enabled}")

    def process_image(self):
        if not self.image_path:
            self.show_error("Please load an image first!")
            return

        try:
            self.resolution = int(self.resolution_input.text())
            self.depth_amount = float(self.depth_amount_input.text())
            self.max_depth = float(self.min_depth_input.text())

            # Validate depth_amount > 0 (no flat meshes)
            if self.depth_amount <= 0:
                self.show_error("Depth Amount must be greater than 0!")
                return

            # Validate max_depth within range [0,10000] -- somewhat arbitrary
            if not (0.0 <= self.max_depth <= 10000.0):
                self.show_error("Max Depth must be between 0.0 and 1.0!")
                return

            # Get smoothing method and model type
            smoothing_method = self.smoothing_dropdown.currentText()
            model_name = DepthTo3D().model_names[self.depth_method_dropdown.currentText()]
            self.depth_to_3d = DepthTo3D(model_type=model_name)

            # Determine image path
            image_to_use = self.image_path

            # Save processed image if needed
            if self.use_processed_image_enabled and self.processed_image is not None:
                processed_image_path = self._get_processed_image_path()
                cv2.imwrite(processed_image_path, self.processed_image)
                image_to_use = processed_image_path

            depth_amount = 1.0
            try:
                depth_amount = float(self.depth_amount_input.text())
            except ValueError:
                print("Invalid depth amount. Using default value of 1.0.")

            if self.use_selected_color:
                background_color = self.current_selected_color
            else:
                background_color = None
            # Pass depth_amount and max_depth into depth_to_3d processing
            self.output_mesh_obj, self.background_color = self.depth_to_3d.process_image(
                image_to_use,
                smoothing_method=smoothing_method,
                target_size=(self.resolution, self.resolution),
                flat_back=self.flat_back_enabled,
                grayscale_enabled=self.grayscale_enabled,
                edge_detection_enabled=self.edge_detection_enabled,
                invert_colors_enabled=self.invert_colors_enabled,
                depth_amount=depth_amount,
                depth_drop_percentage=self.depth_drop_percentage,
                project_on_original=self.project_on_original,
                background_removal=self.drop_background_enabled,
                background_tolerance=self.background_tolerance,
                background_color=background_color,
                # max_depth=max_depth,  # Pass max_depth
            )
            print(f"3D model generated successfully with Depth Amount: {depth_amount}, Max Depth: {self.max_depth}")
            self.depth_values = self.depth_to_3d.depth_values
            self.depth_labels = self.depth_to_3d.depth_labels
            print(f"Depth Range: {self.depth_labels[0]} - {self.depth_labels[-1]}")
            # Update 3D viewport
            if max(self.background_color) > 0:
                self.background_color = [c / 255.0 for c in self.background_color]
                self.update_3d_viewport(self.background_color)
            else:
                # Default background color is dark gray so black parts of models stand out.
                self.update_3d_viewport([10,10,10])

        except Exception as e:
            self.show_error(f"Error: {e}\r\n{traceback.format_exc()}")

    def _get_processed_image_path(self):
        """Construct the file path for saving the processed image."""
        base, ext = os.path.splitext(self.image_path)
        return f"{base}_processed{ext}"

    def resize_image_data(self, image_data, target_width, target_height):
        # Convert self.image (numpy.ndarray) to QImage
        height, width, channel = self.image.shape
        bytes_per_line = width * channel
        q_img = QImage(image_data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        scaled_image = q_img.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return scaled_image

    def resizeEvent(self, event):
        """Handles window resize events to adjust label sizes and re-scale images."""
        super().resizeEvent(event)
        if debug:
            print(f"Resize Event Triggered. New Size: {event.size()}. Window Size now: {self.size()}")

        if self.processed_image is not None and isinstance(self.processed_image, np.ndarray):
            self.display_processed_image()
            self.display_original_image()

    def update_resolution(self):
        """Update the stored resolution value from the input box."""
        try:
            self.resolution = int(self.resolution_input.text())
            print(f"Resolution set to: {self.resolution}")  # For debugging purposes
        except ValueError:
            self.resolution = None  # Clear or reset if invalid input
            print("Invalid resolution input.")  # For debugging purposes


    def sharpen_image(self, image):
        """Sharpen the processed image for more prominent edges."""
        # Convert image to float32 for precision
        image = image.astype(np.float32)

        # Step 1: Apply a Gaussian blur (Optional, to smooth subtle noise)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Step 2: Add weighted sharpening
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # Step 3: Further enhance edges with Laplacian (if needed)
        laplacian = cv2.Laplacian(sharpened, cv2.CV_32F)
        sharpened = sharpened - 0.2 * laplacian  # Adjust multiplier if necessary

        # Clip values & convert back to uint8
        sharpened = np.clip(sharpened, 0, 255)
        # return sharpened.astype(np.uint8)
        return sharpened

    # In MainWindow_ImageProcessing Class, replace generate_mesh()

    def generate_mesh(self):
        """ Generate a 3D mesh from the processed image. Actually called when "Generate 2D" is clicked. """
        if self.processed_image is None:
            self.show_error("No processed image available.")
            return

        try:
            # flipped_image = cv2.flip(self.processed_image, 0) # Flip vertically so 3D is upright

            # Initialize the MeshGenerator with visualization options
            visualizations = {
                "visualize_clustering": visualize_clustering,
                "visualize_depth": visualize_depth,
                "visualize_partitioning": visualize_partitioning,
                "visualize_edges": visualize_edges,
            }

            mesh_generator = MeshGenerator(visualizations)
            self.extruded_edges = mesh_generator.generate(self.processed_image)

            # Update the 3D viewport with the new mesh
            self.update_3d_viewport(self.background_color)

        except Exception as e:
            self.show_error(f"Error while generating mesh: {str(e)}")

    def toggle_invert_colors(self, state):
        """Toggle invert colors based on checkbox state."""
        # You can add any necessary logic here, such as applying inversion immediately
        # if self.invert_colors_enabled:
        #     self.invert_checkbox.setCheckState(Qt.Checked)
        # else:
        #     self.invert_checkbox.setCheckState(Qt.Unchecked)
        if not self.initialized:
            return
        self.invert_colors_enabled = state == Qt.Checked
        print(f"Toggle Invert Colors: {self.invert_colors_enabled}")
        self.update_preview()

    def invert_colors(self):
        """Invert colors of the original and processed image."""
        # if self.image is not None:
        #     # Invert the original image
        #     self.image = cv2.bitwise_not(self.image)
        #     self.display_original_image()
        if self.processed_image is not None:
            print("Inverting processed image.")
            # Invert the processed image
            self.processed_image = cv2.bitwise_not(self.processed_image)
            self.display_processed_image()

    def update_3d_viewport(self, background_color=None):
        if self.output_mesh_obj is None:
            print("Must set output_mesh_obj before calling update_3d_viewport.")
            return

        if self.three_d_viewport is None or not self.three_d_viewport.viewer.poll_events():
            self.three_d_viewport = ThreeDViewport(background_color=background_color)
            print("3D viewport created or reopened.")

        print(f"update_3d_viewport: Background color: {background_color}")

        if background_color is not None:
            self.three_d_viewport.viewer.get_render_option().background_color = background_color

        self.three_d_viewport.clear_geometries()
        self.three_d_viewport.load_mesh(self.output_mesh_obj, self.depth_labels)
        self.three_d_viewport.run()
        # self.three_d_viewport.show()

    def export_mesh(self):
        if not self.three_d_viewport.edges:
            self.show_error("No 3D mesh is generated to export.")
            return

        # Open save file dialog
        options = QFileDialog.Options()
        file_filter = "OBJ Files (*.obj);;STL Files (*.stl)"
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Mesh", "", file_filter, options=options
        )

        if save_path:
            try:
                # Determine export format based on selected file extension
                if selected_filter == "OBJ Files (*.obj)" or save_path.endswith(".obj"):
                    self.three_d_viewport.export_mesh_as_obj(save_path)
                elif selected_filter == "STL Files (*.stl)" or save_path.endswith(".stl"):
                    self.three_d_viewport.export_mesh_as_stl(save_path)
                else:
                    self.show_error("Unsupported file format.")
            except Exception as e:
                self.show_error(f"Failed to export mesh: {str(e)}")

    def update_preview(self):
        """Reprocess the preview and refresh the 3D viewport."""
        if not self.image_path:  # No image to process
            return

        try:
            fname = self.image_path
            if self.grayscale_enabled:
                self.processed_image = cv2.cvtColor(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
            else:
                self.processed_image = cv2.imread(fname, cv2.IMREAD_COLOR)
            if self.grayscale_enabled:
                grayscale = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                basename, ext = os.path.splitext(self.image_path)
                fname = f"{basename}_gray.{ext}"
                cv2.imwrite(fname, grayscale)
            if self.edge_detection_enabled:
                # Perform edge detection if enabled
                low_threshold = 200 - self.sensitivity

                if self.grayscale_enabled and self.project_on_original:
                    grayscale = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                    basename, ext = os.path.splitext(self.image_path)
                    fname = f"{basename}_gray.{ext}"
                    cv2.imwrite(fname, grayscale)
                self.processed_image = detect_edges(fname, low_threshold, low_threshold * 3,
                    thickness=self.edge_thickness, project_on_original=self.project_on_original)
            elif self.grayscale_enabled:
                # Otherwise, convert to grayscale if enabled
                self.processed_image = cv2.cvtColor(cv2.imread(fname, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)

            # Apply inversion if enabled (applies to all modes)
            if self.invert_colors_enabled:
                self.invert_colors()

            if self.blend_amount < 100:
                self.processed_image = image_processor.ImageProcessor.blend_images (self.image, self.processed_image, self.blend_amount)
            self.display_processed_image()
        except Exception as e:
            s = f"Error in update_preview(): {traceback.format_exc()}"
            print (s)
            self.show_error(s)

    def update_line_thickness(self):
        self.edge_thickness = self.line_thickness_slider.value()
        self.update_preview()

    def update_blend(self):
        self.blend_amount = self.blend_slider.value()
        self.update_preview()

    ### Drag-and-Drop Handling ###
    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event):
        event.setDropAction(Qt.CopyAction)
        event.accept()
        for url in event.mimeData().urls():
            self.image_path = url.toLocalFile()
            self.load_image(self.image_path)

    def load_image(self, path=None):
        if not path:
            options = QFileDialog.Options()
            path, _ = QFileDialog.getOpenFileName(
                self, "Load Image", "", "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            if not path:
                return

        self.image_path = path
        self._update_config("Settings", "last_used_image", path)

        self.image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if self.image is None:
            self.show_error("Failed to load image.")
            return
        self.display_original_image()
        self.update_preview()


    def display_original_image(self):
        if self.image is not None:
            height, width, channel = self.image.shape
            q_img = QImage(self.image.data, width, height, width * 3, QImage.Format_RGB888).rgbSwapped()
            self.original_pixmap = QPixmap.fromImage(q_img)
            self.original_label.setPixmap(
                self.original_pixmap.scaled(
                    self.original_label.width(), self.original_label.height(),
                    Qt.KeepAspectRatio
                )
            )

    def display_processed_image(self):
        if self.processed_image is not None:
            # self.processed_image = process_preview_image(self.image, is_grayscale=self.grayscale_enabled)
            if len(self.processed_image.shape) == 2:  # Grayscale image
                height, width = self.processed_image.shape
                bytes_per_line = width
                q_img = QImage(
                    self.processed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
                )

            else:  # Color image
                height, width, channels = self.processed_image.shape
                bytes_per_line = channels * width
                q_img = QImage(
                    self.processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888
                ).rgbSwapped()

            target_height = self.preview_label.height()
            target_width = self.preview_label.width()
            # Update the QLabel with the processed image
            self.preview_label.setPixmap(QPixmap.fromImage(q_img).scaled(target_width, target_height,Qt.KeepAspectRatio))

    def save_image(self):
        if not self.processed_image.any():
            self.show_error("No processed image to save.")
            return
        options = QFileDialog.Options()
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Processed Image", "", "Images (*.png *.jpg *.bmp)", options=options)
        if save_path:
            cv2.imwrite(save_path, self.processed_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def show_error(self, message):
        print(f"Error: {message}")

    def load_last_used_image(self):
        last_image_path = self.config.get("Settings", "last_used_image", fallback="")
        if last_image_path and os.path.exists(last_image_path):
            self.load_image(last_image_path)
        else:
            self.load_image("./Images/example.png")  # 1232x928

    def _load_config(self):
        self.config.read(self.CONFIG_FILE_PATH)
        self.image_path = self.config.get("Settings", "last_used_image", fallback="")
        self.load_ui_settings()

    def _save_config(self):
        self.config.set("Settings", "last_used_image", self.image_path)
        self.save_ui_settings()

    def _update_config(self, section, option, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, option, value)
        with open(self.CONFIG_FILE_PATH, "w") as configfile:
            self.config.write(configfile)
        self.save_ui_settings()

    def save_ui_settings(self):
        """Save all UI settings to the config.ini file."""
        if not self.config.has_section("UI_Settings"):
            self.config.add_section("UI_Settings")
        # Save geometry and state encoded in hexadecimal
        self.config.set("UI_Settings", "windowGeometry", self.saveGeometry().toHex().data().decode())
        self.config.set("UI_Settings", "windowState", self.saveState().toHex().data().decode())

        # Add settings to the section
        self.config.set("UI_Settings", "invert_colors", str(self.invert_colors_enabled))
        self.config.set("UI_Settings", "grayscale", str(self.grayscale_enabled))
        self.config.set("UI_Settings", "drop_background", str(self.drop_background_enabled))
        self.config.set("UI_Settings", "background_tolerance", str(self.background_tolerance))
        self.config.set("UI_Settings", "resolution", str(self.resolution))
        self.config.set("UI_Settings", "edge_detection", str(self.edge_detection_enabled))
        self.config.set("UI_Settings", "sensitivity", str(self.sensitivity_slider.value()))
        self.config.set("UI_Settings", "line_thickness", str(self.line_thickness_slider.value()))
        self.config.set("UI_Settings", "project_on_original", str(self.project_on_original))
        self.config.set("UI_Settings", "use_processed_image", str(self.use_processed_image_enabled))
        self.config.set("UI_Settings", "depth_amount", str(self.depth_amount_input.text()))
        self.config.set("UI_Settings", "flat_back", str(self.flat_back_enabled))
        self.config.set("UI_Settings", "depth_drop_percentage", str(self.depth_drop_percentage))
        self.config.set("UI_Settings", "blend_amount", str(self.blend_amount))

        # Add more settings as needed...

        # Write settings to the config file
        with open(self.CONFIG_FILE_PATH, "w") as configfile:
            self.config.write(configfile)

    def reset_defaults(self):
        self.invert_checkbox.setChecked(False)
        self.grayscale_checkbox.setChecked(False)
        self.drop_background_checkbox.setChecked(True)
        self.background_tolerance_input.setValue(10)
        self.resolution_input.setText("700")
        self.line_thickness_slider.setValue(1)  # Set the desired value
        self.edge_detection_checkbox.setChecked(True)
        self.project_on_original_checkbox.setChecked(True)
        self.use_processed_image_checkbox.setChecked(True)
        self.flat_back_checkbox.setChecked(False)
        self.percentage_input.setText("0")
        self.sensitivity_slider.setValue(150)
        self.percentage_input.setText("0")
        self.depth_amount_input.setText("1.0")
        self.min_depth_input.setText("100.0")
        self.blend_slider.setValue(100)
        self.update_preview()

    def load_ui_settings(self):
        """Load UI settings from the config.ini file."""
        if not self.initialized:
            return
        if os.path.exists(self.CONFIG_FILE_PATH):
            self.config.read(self.CONFIG_FILE_PATH)

            if self.config.has_section("UI_Settings"):
                # Load settings
                geometry = self.config.get("UI_Settings", "windowGeometry", fallback=None)
                # Decode the string to QByteArray
                self.restoreGeometry(QByteArray.fromHex(geometry.encode()))
                # Convert the state string to bytes before passing to restoreState
                state = self.config.get("UI_Settings", "windowState", fallback=None)
                self.restoreState(QByteArray.fromHex(state.encode()))
                self.invert_colors_enabled = self.config.getboolean("UI_Settings", "invert_colors", fallback=False)
                self.grayscale_enabled = self.config.getboolean("UI_Settings", "grayscale", fallback=False)
                self.drop_background_enabled = self.config.getboolean("UI_Settings", "drop_background", fallback=False)
                self.background_tolerance = self.config.getfloat("UI_Settings", "background_tolerance", fallback=20)
                self.depth_amount_input.setText(self.config.get("UI_Settings", "depth_amount", fallback="1.0"))
                self.resolution = self.config.getint("UI_Settings", "resolution", fallback=700)
                self.edge_detection_enabled = self.config.getboolean("UI_Settings", "edge_detection", fallback=False)
                self.sensitivity = self.config.getint("UI_Settings", "sensitivity", fallback=150)
                self.line_thickness = self.config.getint("UI_Settings", "line_thickness", fallback=2)
                self.project_on_original = self.config.getboolean("UI_Settings", "project_on_original", fallback=False)
                self.use_processed_image_enabled = self.config.getboolean("UI_Settings", "use_processed_image", fallback=False)
                self.flat_back_enabled = self.config.getboolean("UI_Settings", "flat_back", fallback=False)
                self.depth_drop_percentage = self.config.getfloat("UI_Settings", "depth_drop_percentage", fallback=0.0)
                self.blend_amount = self.config.getint("UI_Settings", "blend_amount", fallback=100)

                # Apply loaded values to the UI components
                self.invert_checkbox.setChecked(self.invert_colors_enabled)
                self.grayscale_checkbox.setChecked(self.grayscale_enabled)
                self.drop_background_checkbox.setChecked(self.drop_background_enabled)
                self.background_tolerance_input.setValue(int(self.background_tolerance))
                self.resolution_input.setText(str(self.resolution))
                self.edge_detection_checkbox.setChecked(self.edge_detection_enabled)
                self.project_on_original_checkbox.setChecked(self.project_on_original)
                self.use_processed_image_checkbox.setChecked(self.use_processed_image_enabled)
                self.flat_back_checkbox.setChecked(self.flat_back_enabled)
                self.percentage_input.setText(str(self.depth_drop_percentage))
                self.sensitivity_slider.setValue(self.sensitivity)
                self.line_thickness_slider.setValue(self.line_thickness)


    SETTINGS = [
        ("invert_colors_enabled", "invert_checkbox"),
        ("grayscale_enabled", "grayscale_checkbox"),
        ("drop_background_enabled", "drop_background_checkbox"),
        ("background_tolerance", "background_tolerance_input"),
        ("resolution", "resolution_input"),
        ("edge_detection_enabled", "edge_detection_checkbox"),
        ("sensitivity", "sensitivity_slider"),
        ("line_thickness", "line_thickness_slider"),
        ("project_on_original", "project_on_original_checkbox"),
        ("use_processed_image_enabled", "use_processed_image_checkbox"),
        ("flat_back_enabled", "flat_back_checkbox"),
        ("depth_drop_percentage", "percentage_input"),
        ("depth_amount", "depth_amount_input"),
        ("max_depth", "min_depth_input"),
        ("background_tolerance", "background_tolerance_input"),
        ("blend_amount", "blend_slider"),
    ]

    # Ensure configuration file exists with default settings
    def initialize_config(self):
        config = configparser.ConfigParser()
        if not os.path.exists(self.CONFIG_FILE_PATH):
            # Create default config.ini
            config["Settings"] = self.SETTINGS
            with open(self.CONFIG_FILE_PATH, "w") as configfile:
                config.write(configfile)
        return config

    def closeEvent(self, event):
        """
        This method is invoked when the window is about to be closed.
        """
        try:
            self._save_config()
            super(MainWindow_ImageProcessing, self).closeEvent(event)  # Ensure the close event proceeds
        except Exception as e:
            print(f"Error during close event: {traceback.format_exc()}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow_ImageProcessing()
    main_window.show()
    sys.exit(app.exec_())