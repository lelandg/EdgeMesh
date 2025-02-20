__author__ = "Leland Green"
__version__ = "0.5.5"
__date_created__ = "2025-01-28"
__last_updated__ = "2025-01-29"
__email__ = "lelandgreenproductions@gmail.com"

__license__ = "Commercial. License Required." # License of this script is free for all purposes.

import traceback

from flow_layout import FlowLayout

debug = True # Set False to disable debug messages. Yes. We do need this.
verbose = True # Not used, yet. When I add logging, this will also print messages to console when enabled.

from depth_to_3d import DepthTo3D
from edge_detection import detect_edges
from mesh_generator import MeshGenerator
from viewport_3d import ThreeDViewport

visualize_clustering = False
visualize_depth = False
visualize_partitioning = False
visualize_edges = False

f"""
Utilities using opencv for edge detection in a GUI. 
Also features depth map generation via select (implemented) methods via torch, etc. 
Then 3D mesh generation from the depth map.

Author: {__author__}
Version: {__version__}
Date: {__date_created__}
Last Updated: {__last_updated__}
Email: {__email__}

Requirements:
- Python 3.8 or newer
- Modules in python_requirements.txt. Run "pip install -r python_requirements.txt" to install.

How to Run:
Run this script using the command `python MainWindow_ImageProcessing.py`.

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
from PyQt5.QtCore import Qt, QSize
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
        self.initialized = False
        self.resolution = 700
        self.depth_amount = 1.0
        self.max_depth = 50.0
        self.background_tolerance = 1  # Default value
        self.use_processed_image_enabled = True
        self.project_on_original = True
        self.invert_colors_enabled = False
        self.edge_thickness = 1  # Default line thickness
        self.depth_labels = None
        self.depth_drop_percentage = 0.0  # Default percentage depth drop
        self.image_path = None  # To hold the currently loaded image path
        self.background_color = [0, 0, 0]  # Default background color = dark gray
        self.three_d_viewport = None
        self.original_pixmap = None
        self.image = None
        self.processed_image = None
        self.extruded_edges = None

        # Set the icon for the main window
        icon_path = os.path.join(os.getcwd(), "EdgeMesh.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))  # Set the main window's icon

        self.setWindowTitle(f"3D Mesh Generator v{__version__}")
        self.setGeometry(50, 50, 800, 600)
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Enable OneDNN optimizations for TensorFlow
        # Initialize configuration
        # Config file path
        self.CONFIG_FILE_PATH = "config.ini"
        self.config = self.initialize_config()

        self._init_ui()
        self._load_config()

        o3d.visualization.webrtc_server.enable_webrtc()
        print(f"Open3D version: {o3d.__version__}")


    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout and Controls
        main_layout = QVBoxLayout()
        self.edge_sensitivity_layout = QVBoxLayout()  # For images and controls
        image_preview_layout = QHBoxLayout()  # To hold original and processed previews
        bottom_controls = FlowLayout()

        # Original Image Preview
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: lightgray;")
        self.original_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.original_label.setMinimumSize(10,10)
        self.original_label.setToolTip("This is the original image. Load an image to start processing.")
        # self.original_label.setMaximumHeight(700)  # Limit height
        image_preview_layout.addWidget(self.original_label)

        # Processed Image Preview
        self.preview_label = QLabel("Processed Image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: lightgray;")
        self.preview_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        self.preview_label.setMinimumSize(10,10)
        self.preview_label.setToolTip("This is the processed image, based on your selections below. "
                                      "Click 'Depth Mesh' to generate a 3D mesh.")
        # self.preview_label.setMaximumHeight(700)  # Limit height
        image_preview_layout.addWidget(self.preview_label)

        # Add "Edge Detection" Option
        self.edge_detection_checkbox = QCheckBox("Edge Detection")
        self.edge_detection_checkbox.setChecked(True)  # Default: Enabled
        self.edge_detection_checkbox.stateChanged.connect(self.toggle_edge_detection)
        bottom_controls.addWidget(self.edge_detection_checkbox)

        # Initialize the variable to track the checkbox state
        self.edge_detection_enabled = True

        # Add the "Project on Original" checkbox
        self.project_on_original_checkbox = QCheckBox("Project on Original")
        self.project_on_original_checkbox.setChecked(False)  # Set default to False
        self.project_on_original_checkbox.stateChanged.connect(self.toggle_project_on_original)

        # Add the checkbox to the layout (example) - customize it based on your layout details
        bottom_controls.addWidget(self.project_on_original_checkbox)

        # Add "Grayscale" Option
        self.grayscale_checkbox = QCheckBox("Grayscale")
        self.grayscale_checkbox.setChecked(False)  # Default: Disabled
        self.grayscale_checkbox.stateChanged.connect(self.toggle_grayscale)
        bottom_controls.addWidget(self.grayscale_checkbox)

        # Initialize the variable to track checkbox state
        self.grayscale_enabled = False
        # Add images to the layout
        self.edge_sensitivity_layout.addLayout(image_preview_layout)

        # Slider for Sensitivity
        # sensitivity_hbox = QHBoxLayout()
        sensitivity_hbox = FlowLayout()
        slider_label = QLabel("Edge Detection Sensitivity")
        slider_label.setAlignment(Qt.AlignCenter)
        sensitivity_hbox.addWidget(slider_label)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(200)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.setMinimumSize(200, 0)
        self.sensitivity_slider.setToolTip("Adjust the edge detection sensitivity. Higher values detect more edges.")
        self.sensitivity_slider.valueChanged.connect(self.update_preview)
        sensitivity_hbox.addWidget(self.sensitivity_slider)

        self.edge_sensitivity_layout.addLayout(sensitivity_hbox)

        # Slider for Line Thickness
        # line_thickness_hbox = QHBoxLayout()
        line_thickness_hbox = FlowLayout()
        line_thickness_label = QLabel("Line Thickness")
        line_thickness_label.setAlignment(Qt.AlignCenter)
        line_thickness_hbox.addWidget(line_thickness_label)

        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setMinimum(1)
        self.line_thickness_slider.setMaximum(10)
        self.line_thickness_slider.setValue(self.edge_thickness)
        self.line_thickness_slider.setMinimumSize(200, 0)
        self.line_thickness_slider.setToolTip("Adjust the thickness of the lines drawn for detected edges.")
        self.line_thickness_slider.valueChanged.connect(self.update_line_thickness)
        line_thickness_hbox.addWidget(self.line_thickness_slider)

        self.edge_sensitivity_layout.addLayout(line_thickness_hbox)

        # Add "Invert Colors" Option
        self.invert_checkbox = QCheckBox("Invert Colors")
        self.invert_checkbox.setToolTip("Invert the colors of the image. Useful with edge detection, but can be fun.")
        self.invert_checkbox.stateChanged.connect(self.toggle_invert_colors)
        bottom_controls.addWidget(self.invert_checkbox)
        if self.invert_colors_enabled:
            self.invert_checkbox.setCheckState(Qt.Checked)
        else:
            self.invert_checkbox.setCheckState(Qt.Unchecked)

        # Add controls and buttons
        main_layout.addLayout(self.edge_sensitivity_layout)

        # 3D Viewport (Fill most of the window) - Moved below. Now created/updated when a new mesh is ready.
        # self.three_d_viewport = ThreeDViewport()
        # self.three_d_viewport.setStyleSheet("background-color: #ddd;")
        # self.three_d_viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # main_layout.addWidget(self.three_d_viewport)

        # Buttons for Load, Save, Depth Mesh, Mesh from 2D, and Export Mesh
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setToolTip("Load an image to process.")
        bottom_controls.addWidget(self.load_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        self.save_button.setToolTip("Save the processed image. What you see in the right hand image is what you get.")
        bottom_controls.addWidget(self.save_button)

        depth_hbox = FlowLayout()
        depth_method_label = QLabel("Depth Method")
        depth_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(depth_method_label)

        # Create ComboBox
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["DepthAnythingV2", "DPT", "MiDaS"])  # Add items
        self.model_dropdown.setToolTip("Depth estimation model to use.\r\n"
                                       "DepthAnythingV2 is usually better, but DPT can be good for architecture.\r\n"
                                       "MiDaS is good for general depth. Results are very different, so experiment!")
        self.model_dropdown.setToolTip("Select the depth estimation model to use.")
        depth_hbox.addWidget(self.model_dropdown)
        # main_layout.addLayout(depth_hbox)

        depth_smoothing_label = QLabel("Depth Smoothing")
        depth_smoothing_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(depth_smoothing_label)

        self.smoothing_dropdown = QComboBox()
        self.smoothing_dropdown.addItems(["anisotropic", "gaussian", "bilateral", "median", "(none)"])
        self.smoothing_dropdown.setToolTip("Select smoothing method for the depth map.")
        depth_hbox.addWidget(self.smoothing_dropdown)

        resolution_label = QLabel("Resolution")
        resolution_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(resolution_label)

        # Add the numeric input box for resolution
        self.resolution_input = QLineEdit()
        self.resolution_input.setValidator(QIntValidator(0, 10000))  # Allows numeric input in the range 0-10000
        self.resolution_input.setFixedWidth(50)
        self.resolution_input.setText("700")
        self.resolution_input.setToolTip("Enter the resolution for the depth map.\n"
                                         "0 = full size. Can take a while for large images!\n"
                                         "1000 is almost always good. Default: 700")
        self.resolution_input.textChanged.connect(self.update_resolution)
        depth_hbox.addWidget(self.resolution_input)

        # Add to the Depth Controls (depth_hbox)
        depth_amount_label = QLabel("Depth Amount")
        depth_amount_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        depth_hbox.addWidget(depth_amount_label)

        # Add an input field for Depth Amount, with validation
        self.depth_amount_input = QLineEdit()
        self.depth_amount_input.setValidator(QDoubleValidator(1.0, 10000.0, 5, self))
        self.depth_amount_input.setFixedWidth(100)
        self.depth_amount_input.setText("1.0")  # Default depth = 1.0 (current depth)
        self.depth_amount_input.setToolTip("Depth Amount. Higher values increase the depth as a multiplier.\n"
                                           "So 0.5 = half size, 2.0 = double size. Default: 1.0 = \"normal\" depth")
        depth_hbox.addWidget(self.depth_amount_input)

        max_depth_label = QLabel("Max Depth")
        max_depth_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        max_depth_label.setEnabled(False)  # Disable for now
        depth_hbox.addWidget(max_depth_label)

        # Add an input field for Max Depth, validate from 0.0 to 1.0 (floating-point)
        self.max_depth_input = QLineEdit()
        double_validator = QDoubleValidator(1.0, 10000.0, 5, self)  # Ranges from 0.0 to 10000.0, accuracy up to 5 decimals
        self.max_depth_input.setValidator(double_validator)
        self.max_depth_input.setFixedWidth(100)
        self.max_depth_input.setText("100.0")  # Default max depth = 1.0 (full range)
        self.max_depth_input.setToolTip("Maximum Depth. Min = 1.0 (basically flat), Max = 10000, which is crazy deep.\n"
                                        "Try 500-1000, or less for bas-relief. Default: 50.0")
        self.max_depth_input.setEnabled(False)  # Disable for now
        depth_hbox.addWidget(self.max_depth_input)

        # Add this where other UI elements are initialized (preferably with layout setup)
        percentage_label = QLabel(" Depth Drop (%):")
        self.percentage_input = QLineEdit(self)
        self.percentage_input.setFixedWidth(100)
        self.percentage_input.setToolTip("Set the percentage depth drop for the mesh. The farthest points will be dropped by this percentage")
        self.percentage_input.setText("0.0")  # Default percentage depth drop
        self.percentage_input.setValidator(QtGui.QDoubleValidator(0.0, 100.0, 2))  # Limits input to 0.00–100.00

        # Connect the input box to the update function
        self.percentage_input.editingFinished.connect(self.update_percentage_depth_drop)
        depth_hbox.addWidget(percentage_label)
        depth_hbox.addWidget(self.percentage_input)

        self.drop_background_checkbox = QCheckBox("Remove Background")
        self.drop_background_enabled = False  # Initialize the variable
        self.drop_background_checkbox.stateChanged.connect(self.toggle_drop_background)
        self.drop_background_checkbox.setToolTip("Enable this option to remove the background color from the image.\n"
                                                 "This only works when all four corners are the same color.")
        depth_hbox.addWidget(self.drop_background_checkbox)

        # Background tolerance input
        self.background_tolerance_label = QLabel("Background Tolerance:")
        self.background_tolerance_input = QSpinBox()
        self.background_tolerance_input.setToolTip(
            "Set the tolerance for background removal. 1 will remove only that color.\n"
            "Higher values will remove similar colors. 255 will remove all colors, so keep it low, usually.\n"
            "See ReadMe.md for more info.")
        self.background_tolerance_input.setValue(self.background_tolerance)
        self.background_tolerance_input.setRange(1, 255)  # Example range
        self.background_tolerance_input.setValue(self.background_tolerance)

        # Connect the value change to the handler
        self.background_tolerance_input.valueChanged.connect(self.update_background_tolerance)

        # Add to layout (assuming there's a layout for settings)
        depth_hbox.addWidget(self.background_tolerance_label)
        depth_hbox.addWidget(self.background_tolerance_input)

        # ming relevant layout exists
        # Add "Dynamic Depth" Checkbox to depth_hbox
        self.dynamic_depth_checkbox = QCheckBox("Flat Back")
        self.dynamic_depth_checkbox.setToolTip(
            "Enable this option to create a model with a flat back.\n"
            "This option is usually better for 3D printing.\n"
            "When disabled, the back will be shaped to match the front."
        )
        self.dynamic_depth_checkbox.stateChanged.connect(self.toggle_dynamic_depth)
        depth_hbox.addWidget(self.dynamic_depth_checkbox)

        # Initialize the variable to track the checkbox state
        self.dynamic_depth_enabled = False

        # Add the "Use Processed Image" checkbox to the depth_hbox layout
        self.use_processed_image_checkbox = QCheckBox("Use Processed Image")
        self.use_processed_image_checkbox.setToolTip(
            "Enable this to save and use the image on the right (the processed image) to generate the depth map. You can get crazy results with edge detection overlayed on original.\n"
            "When not enabled, the original image is always used, no matter what options you have set."
        )
        self.use_processed_image_checkbox.stateChanged.connect(self.toggle_use_processed_image)
        depth_hbox.addWidget(self.use_processed_image_checkbox)

        # Initialize the variable to track the checkbox state
        # Add Depth Mesh Button
        self.process_button = QPushButton("Depth Mesh")
        self.process_button.setToolTip("Generate a 3D mesh from the original or processed image.")
        self.process_button.clicked.connect(self.process_image)
        depth_hbox.addWidget(self.process_button)

        main_layout.addLayout(depth_hbox)

        #TODO: Fix Mesh from 2D
        # Add Mesh From 2D Button
        self.generate_mesh_button = QPushButton("Mesh From Edges")
        self.generate_mesh_button.setToolTip("Use edge analysis to generate an approximated mesh from the processed image. (Currently broken.)")
        self.generate_mesh_button.clicked.connect(self.generate_mesh)
        bottom_controls.addWidget(self.generate_mesh_button)

        # Additional "Export Mesh" button
        self.export_mesh_button = QPushButton("Export Mesh")
        self.export_mesh_button.setToolTip("Export the generated 3D mesh as a .PLY file.\n**NOTE** You do NOT need to use this because the mesh is automagically saved in the same folder as the source image.")
        self.export_mesh_button.clicked.connect(self.export_mesh)
        bottom_controls.addWidget(self.export_mesh_button)

        self.reset_defaults_button = QPushButton("Reset Defaults")  # Button labeled "Reset Defaults"
        self.reset_defaults_button.clicked.connect(self.reset_defaults)  # Connect button to reset_defaults method
        bottom_controls.addWidget(self.reset_defaults_button)  # Add button to layout

        main_layout.addLayout(bottom_controls)
        self.central_widget.setLayout(main_layout)

        self.initialized = True
        self.load_ui_settings()
        self.load_last_used_image()
        # self.update_preview()
        delay_ms = 500  # Delay in milliseconds
        delay_sec = delay_ms / 1000  # Convert to seconds
        timer = threading.Timer(delay_sec, self.scale_image_to_fit_original)
        timer.start()
        timer2 = threading.Timer(delay_sec, self.scale_image_to_fit_preview)
        timer2.start()

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

    def update_percentage_depth_drop(self):
        """
        Callback to update the depth_drop_percentage value when the input changes.
        """
        try:
            if self.percentage_input.text() == "" or self.percentage_input.text() == "0":
                value = None  # Default to 0.0 if input is empty
            else:
                # Get value from the text input box
                value = float(self.percentage_input.text())

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

    def update_edge_sensitivity_layout(self):
        """Enable or disable the edge sensitivity layout based on edge_detection state."""
        is_enabled = self.edge_detection_enabled
        for i in range(self.edge_sensitivity_layout.count()):
            widget = self.edge_sensitivity_layout.itemAt(i).widget()
            if widget:
                widget.setEnabled(is_enabled)  # Or use widget.setVisible(is_enabled) if you want to hide it

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

    def toggle_dynamic_depth(self, state):
        """Enable or disable Dynamic Depth based on the checkbox state."""
        self.dynamic_depth_enabled = state == Qt.Checked
        print(f"Dynamic Depth Enabled: {self.dynamic_depth_enabled}")

    def process_image(self):
        if not self.image_path:
            self.show_error("Please load an image first!")
            return

        try:
            self.resolution = int(self.resolution_input.text())
            self.depth_amount = float(self.depth_amount_input.text())
            self.max_depth = float(self.max_depth_input.text())

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
            model_name = DepthTo3D().model_names[self.model_dropdown.currentText()]
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

            # Pass depth_amount and max_depth into depth_to_3d processing
            self.output_mesh_obj, self.background_color = self.depth_to_3d.process_image(
                image_to_use,
                smoothing_method=smoothing_method,
                target_size=(self.resolution, self.resolution),
                dynamic_depth=self.dynamic_depth_enabled,
                grayscale_enabled=self.grayscale_enabled,
                edge_detection_enabled=self.edge_detection_enabled,
                invert_colors_enabled=self.invert_colors_enabled,
                depth_amount=depth_amount,
                depth_drop_percentage=self.depth_drop_percentage,
                project_on_original=self.project_on_original,
                background_removal=self.drop_background_enabled,
                background_tolerance=self.background_tolerance,
                # max_depth=max_depth,  # Pass max_depth
            )
            print(f"3D model generated successfully with Depth Amount: {depth_amount}, Max Depth: {self.max_depth}")
            self.depth_values = self.depth_to_3d.depth_values
            self.depth_labels = self.depth_to_3d.depth_labels
            print(f"Depth Range: {min(self.depth_values)}-{max(self.depth_values)}")
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

    def scale_image_to_fit_preview(self):
        if self.processed_image is None:
            self.update_preview()
        # Check if the processed image is valid
        if not isinstance(self.processed_image, np.ndarray):
            print("No processed image available to scale.")
            # Optionally, set a placeholder or clear the label
            self.preview_label.clear()
            self.preview_label.setText("No Image Loaded")
            return

        target_width = self.preview_label.width()
        target_height = self.preview_label.height()

        # Convert self.image (numpy.ndarray) to QImage
        height, width, channel = self.image.shape
        bytes_per_line = width * channel
        q_img = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        scaled_image = q_img.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(QPixmap.fromImage(scaled_image))

    def scale_image_to_fit_original(self):
        if self.image is not None:
            # Get the dimensions of the target QLabel
            target_width = self.original_label.width()
            target_height = self.original_label.height()

            # Convert self.image (numpy.ndarray) to QImage
            height, width, channel = self.image.shape
            bytes_per_line = width * channel
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Scale the QImage to fit within the QLabel dimensions
            scaled_image = q_img.scaled(target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Update the QLabel display with the scaled QPixmap
            self.original_label.setPixmap(QPixmap.fromImage(scaled_image))
        else:
            self.original_label.clear()  # Clear the QLabel if no image is loaded

    def resizeEvent(self, event):
        """Handles window resize events to adjust label sizes and re-scale images."""
        super().resizeEvent(event)

        # # Resize and scale the Original Image
        # if self.original_pixmap is not None:
        #     self.original_label.setPixmap(
        #         self.original_pixmap.scaled(
        #             self.original_label.width(),
        #             self.original_label.height(),
        #             Qt.KeepAspectRatio
        #         )
        #     )

        if self.processed_image is not None and isinstance(self.processed_image, np.ndarray):
            self.display_processed_image()
            self.display_original_image()
        # self.update_preview()
        #
        # # Resize and scale the Processed Image
        # if self.processed_image is not None and isinstance(self.processed_image, np.ndarray):
        #     height, width, depth = self.processed_image.shape
        #     bytes_per_line = width
        #     q_img = QImage(self.processed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        #     self.preview_label.setPixmap(
        #         QPixmap.fromImage(q_img).scaled(
        #             self.preview_label.width(),
        #             self.preview_label.height(),
        #             Qt.KeepAspectRatio
        #         )
        #     )

    def update_resolution(self):
        """Update the stored resolution value from the input box."""
        try:
            self.resolution = int(self.resolution_input.text())
            print(f"Resolution set to: {self.resolution}")  # For debugging purposes
        except ValueError:
            self.resolution = None  # Clear or reset if invalid input
            print("Invalid resolution input.")  # For debugging purposes

    def update_selected_label(self, text):
        # Update the label to show the selected item
        self.selected_label.setText(f"Selected: {text}")


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
                low_threshold = self.sensitivity_slider.value()
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

            self.display_processed_image()
        except Exception as e:
            s = f"Error in update_preview(): {traceback.format_exc()}"
            print (s)
            self.show_error(s)

    def update_line_thickness(self):
        self.edge_thickness = self.line_thickness_slider.value()
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
            self.scale_image_to_fit_original()

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
            self.scale_image_to_fit_preview()

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
            self.load_image()  # Open load image dialog on startup, or if the last_image_path doesn't exist.

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
        self.config.set("UI_Settings", "dynamic_depth", str(self.dynamic_depth_enabled))
        self.config.set("UI_Settings", "depth_drop_percentage", str(self.depth_drop_percentage))

        # Add more settings as needed...

        # Write settings to the config file
        with open(self.CONFIG_FILE_PATH, "w") as configfile:
            self.config.write(configfile)

    def reset_defaults(self):
        self.invert_checkbox.setChecked(False)
        self.grayscale_checkbox.setChecked(False)
        self.drop_background_checkbox.setChecked(True)
        self.background_tolerance_input.setValue(1)
        self.resolution_input.setText("700")
        self.line_thickness_slider.setValue(1)  # Set the desired value
        self.edge_detection_checkbox.setChecked(True)
        self.project_on_original_checkbox.setChecked(True)
        self.use_processed_image_checkbox.setChecked(True)
        self.dynamic_depth_checkbox.setChecked(False)
        self.percentage_input.setText("0")
        self.sensitivity_slider.setValue(50)
        self.depth_amount_input.setText("1.0")
        self.update_preview()

    def load_ui_settings(self):
        """Load UI settings from the config.ini file."""
        if not self.initialized:
            return
        if os.path.exists(self.CONFIG_FILE_PATH):
            self.config.read(self.CONFIG_FILE_PATH)

            if self.config.has_section("UI_Settings"):
                # Load settings
                self.invert_colors_enabled = self.config.getboolean("UI_Settings", "invert_colors", fallback=False)
                self.grayscale_enabled = self.config.getboolean("UI_Settings", "grayscale", fallback=False)
                self.drop_background_enabled = self.config.getboolean("UI_Settings", "drop_background", fallback=False)
                self.background_tolerance = self.config.getfloat("UI_Settings", "background_tolerance", fallback=0.5)
                self.depth_amount_input.setText(self.config.get("UI_Settings", "depth_amount", fallback="1.0"))
                self.resolution = self.config.get("UI_Settings", "resolution", fallback="Default")
                self.edge_detection_enabled = self.config.getboolean("UI_Settings", "edge_detection", fallback=False)
                self.sensitivity_slider.setValue(self.config.getint("UI_Settings", "sensitivity", fallback=50))
                self.line_thickness_slider.setValue(self.config.getint("UI_Settings", "line_thickness", fallback=2))
                self.project_on_original = self.config.getboolean("UI_Settings", "project_on_original", fallback=False)
                self.use_processed_image_enabled = self.config.getboolean("UI_Settings", "use_processed_image", fallback=False)
                self.dynamic_depth_enabled = self.config.getboolean("UI_Settings", "dynamic_depth", fallback=False)
                self.depth_drop_percentage = self.config.getfloat("UI_Settings", "depth_drop_percentage", fallback=0.0)

                # Apply loaded values to the UI components
                self.invert_checkbox.setChecked(self.invert_colors_enabled)
                self.grayscale_checkbox.setChecked(self.grayscale_enabled)
                self.drop_background_checkbox.setChecked(self.drop_background_enabled)
                self.background_tolerance_input.setValue(int(self.background_tolerance))
                self.resolution_input.setText(self.resolution)
                self.edge_detection_checkbox.setChecked(self.edge_detection_enabled)
                self.project_on_original_checkbox.setChecked(self.project_on_original)
                self.use_processed_image_checkbox.setChecked(self.use_processed_image_enabled)
                self.dynamic_depth_checkbox.setChecked(self.dynamic_depth_enabled)
                self.percentage_input.setText(str(self.depth_drop_percentage))
                # Apply other UI updates as needed...

    SETTINGS = [
        ("invert_colors_enabled", "invert_checkbox"),
        ("grayscale_enabled", "grayscale_checkbox"),
        ("drop_background_enabled", "drop_background_checkbox"),
        ("background_tolerance", "background_tolerance_input"),
        ("resolution", "resolution_input"),
        ("edge_detection_enabled", "edge_detection_checkbox"),
        ("sensitivity_slider", "sensitivity_slider"),
        ("line_thickness_slider", "line_thickness_slider"),
        ("project_on_original", "project_on_original_checkbox"),
        ("use_processed_image_enabled", "use_processed_image_checkbox"),
        ("dynamic_depth_enabled", "dynamic_depth_checkbox"),
        ("depth_drop_percentage", "percentage_input"),
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