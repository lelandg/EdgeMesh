__author__ = "Leland Green"
__version__ = "0.3.2"
__date_created__ = "2025-01-28"
__last_updated__ = "2025-01-29"
__email__ = "lelandgreenproductions@gmail.com"

__license__ = "Open Source" # License of this script is free for all purposes.

from depth_to_3d import DepthTo3D
from image_processor import ImageProcessor
from mesh_generator import MeshGenerator
from viewport_3d import ThreeDViewport

visualize_clustering = False
visualize_depth = False
visualize_partitioning = False
visualize_edges = False

from edge_detection import detect_edges

# At least early versions. Branch, clone, copy, download when ready.
# Just do not mutilate. If you must, always use your own bytes for that. :)
f"""
Utilities using opencv for edge detection in a GUI. 
Features depth map generation via select (implemented) methods via torch, etc. 
Then 3D mesh generation from the depth map.
Version 0.3.2 Reopens the 3D viewport when closed, and clears existing geometry before loading a new mesh. 
Version 0.3.1 Adds full color .PLY export support and a new depth map smoothing method.
              viewport_3d.py now has export_mesh_as_obj and export_mesh_as_stl methods, and a SUPPORTED_EXTENSIONS list.
                              It takes a list of mesh files as arguments and opens a viewport for each valid file. 
                              (One at a time.) Be careful with many files! *Especially* if they're large. 
                              Large files take a few seconds to minutes to load, depending on your system specs.
              Fixes 3D viewport update issue. (Now controlled by mouse.)
Version 0.3.0 adds depth map smoothing options and anisotropic diffusion.
Version 0.2.0 adds depth map generation from shading and light cues.
Version 0.1.0 adds edge detection and 3D mesh generation from edges.
---

Author: {__author__}
Version: {__version__}
Date: {__date_created__}
Last Updated: {__last_updated__}
Email: {__email__}

Requirements:
- Python 3.8 or newer
- tkinter (standard library)
- zipfile (standard library)

Install Python modules using `pip install -r python_requirements`

How to Run:
Run this script using the command `python MainWindow_ImageProcessing.py`.

License:
{__license__}

"""

# Yes. We do need this:
debug = True # Set to False to disable debug messages

import configparser
import os
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QSlider,
    QPushButton, QWidget, QHBoxLayout, QSizePolicy, QCheckBox, QComboBox, QLineEdit
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QIntValidator

import cv2
import numpy as np
import open3d as o3d

# Config file path
CONFIG_FILE_PATH = "config.ini"


# Ensure configuration file exists with default settings
def initialize_config():
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE_PATH):
        # Create default config.ini
        config["Settings"] = {
            "last_used_image": "",
            "invert_colors": "True"
        }
        with open(CONFIG_FILE_PATH, "w") as configfile:
            config.write(configfile)
    return config


class MainWindow_ImageProcessing(QMainWindow):
    def __init__(self):
        super().__init__()
        self.three_d_viewport = None
        self.original_pixmap = None
        self.edge_thickness = 2  # Default line thickness
        self.image_path = None  # To hold the currently loaded image path
        self.processed_image = None
        self.extruded_edges = None

        self.setWindowTitle("3D Mesh Generator")
        self.setGeometry(100, 100, 800, 500)

        # Initialize configuration
        self.config = initialize_config()
        self._load_config()

        self._init_ui()
        self.load_last_used_image()
        o3d.visualization.webrtc_server.enable_webrtc()
        print (f"Open3D version: {o3d.__version__}")

    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main Layout and Controls
        main_layout = QVBoxLayout()
        edge_sensitivity_layout = QVBoxLayout()  # For images and controls
        image_preview_layout = QHBoxLayout()  # To hold original and processed previews
        bottom_controls = QHBoxLayout()

        # Original Image Preview
        self.original_label = QLabel("Original Image")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: lightgray;")
        self.original_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        # self.original_label.setMaximumHeight(700)  # Limit height
        image_preview_layout.addWidget(self.original_label)

        # Processed Image Preview
        self.preview_label = QLabel("Processed Image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: lightgray;")
        self.preview_label.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        # self.preview_label.setMaximumHeight(700)  # Limit height
        image_preview_layout.addWidget(self.preview_label)

        # Add images to the layout
        edge_sensitivity_layout.addLayout(image_preview_layout)

        # Slider for Sensitivity
        sensitivity_hbox = QHBoxLayout()
        slider_label = QLabel("Edge Detection Sensitivity")
        slider_label.setAlignment(Qt.AlignCenter)
        sensitivity_hbox.addWidget(slider_label)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(10)
        self.sensitivity_slider.setMaximum(150)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_slider.valueChanged.connect(self.update_preview)
        sensitivity_hbox.addWidget(self.sensitivity_slider)

        edge_sensitivity_layout.addLayout(sensitivity_hbox)

        # Slider for Line Thickness
        line_thickness_hbox = QHBoxLayout()
        line_thickness_label = QLabel("Line Thickness")
        line_thickness_label.setAlignment(Qt.AlignCenter)
        line_thickness_hbox.addWidget(line_thickness_label)

        self.line_thickness_slider = QSlider(Qt.Horizontal)
        self.line_thickness_slider.setMinimum(1)
        self.line_thickness_slider.setMaximum(10)
        self.line_thickness_slider.setValue(3)
        self.line_thickness_slider.valueChanged.connect(self.update_line_thickness)
        line_thickness_hbox.addWidget(self.line_thickness_slider)

        edge_sensitivity_layout.addLayout(line_thickness_hbox)

        # Add "Invert Colors" Option
        self.invert_checkbox = QCheckBox("Invert Colors")
        self.invert_checkbox.stateChanged.connect(self.toggle_invert_colors)
        bottom_controls.addWidget(self.invert_checkbox)
        if self.invert_colors_enabled:
            self.invert_checkbox.setCheckState(Qt.Checked)
        else:
            self.invert_checkbox.setCheckState(Qt.Unchecked)

        # Add controls and buttons
        main_layout.addLayout(edge_sensitivity_layout)

        # 3D Viewport (Fill most of the window) - Moved below. Now created/updated when a new mesh is ready.
        # self.three_d_viewport = ThreeDViewport()
        # self.three_d_viewport.setStyleSheet("background-color: #ddd;")
        # self.three_d_viewport.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # main_layout.addWidget(self.three_d_viewport)

        # Buttons for Load, Save, Depth Mesh, Mesh from 2D, and Export Mesh
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        bottom_controls.addWidget(self.load_button)

        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        bottom_controls.addWidget(self.save_button)

        depth_hbox = QHBoxLayout()
        depth_method_label = QLabel("Depth Method")
        depth_method_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(depth_method_label)

        # Create ComboBox
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["DepthAnythingV2", "DPT", "MiDaS"])  # Add items
        depth_hbox.addWidget(self.model_dropdown)
        # main_layout.addLayout(depth_hbox)

        depth_smoothing_label = QLabel("Depth Smoothing")
        depth_smoothing_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(depth_smoothing_label)

        self.smoothing_dropdown = QComboBox()
        self.smoothing_dropdown.addItems(["anisotropic", "gaussian", "bilateral", "median", "(none)"])
        depth_hbox.addWidget(self.smoothing_dropdown)

        resolution_label = QLabel("Depth Resolution")
        resolution_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        depth_hbox.addWidget(resolution_label)

        # Add the numeric input box for resolution
        self.resolution_input = QLineEdit()
        self.resolution_input.setValidator(QIntValidator(0, 10000))  # Allows numeric input in the range 0-10000
        self.resolution_input.setFixedWidth(200)
        self.resolution_input.setText("700")
        self.resolution_input.setToolTip("Enter the resolution for the depth map. 0 = full size.")
        self.resolution_input.textChanged.connect(self.update_resolution)
        depth_hbox.addWidget(self.resolution_input)


        # Add Depth Mesh Button
        self.process_button = QPushButton("Depth Mesh")
        self.process_button.clicked.connect(self.process_image)
        depth_hbox.addWidget(self.process_button)

        main_layout.addLayout(depth_hbox)

        # Add Mesh From 2D Button
        self.generate_mesh_button = QPushButton("Mesh From 2D")
        self.generate_mesh_button.clicked.connect(self.generate_mesh)
        bottom_controls.addWidget(self.generate_mesh_button)

        # Additional "Export Mesh" button
        self.export_mesh_button = QPushButton("Export Mesh")
        self.export_mesh_button.clicked.connect(self.export_mesh)
        bottom_controls.addWidget(self.export_mesh_button)

        main_layout.addLayout(bottom_controls)
        self.central_widget.setLayout(main_layout)

    def resizeEvent(self, event):
        """Handles window resize events to adjust label sizes and re-scale images."""
        super().resizeEvent(event)

        # Resize and scale the Original Image
        if self.original_pixmap is not None:
            self.original_label.setPixmap(
                self.original_pixmap.scaled(
                    self.original_label.width(),
                    self.original_label.height(),
                    Qt.KeepAspectRatio
                )
            )

        # Resize and scale the Processed Image
        if self.processed_image is not None:
            height, width = self.processed_image.shape
            bytes_per_line = width
            q_img = QImage(
                self.processed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
            self.preview_label.setPixmap(
                QPixmap.fromImage(q_img).scaled(
                    self.preview_label.width(),
                    self.preview_label.height(),
                    Qt.KeepAspectRatio
                )
            )

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

    def process_image(self):
        # Get the selected model name
        value =  self.model_dropdown.currentText()
        model_name = DepthTo3D().model_names[value]
        self.depth_to_3d = DepthTo3D(model_type=model_name)  # Update DepthTo3D instance

        # Open file dialog to select an image
        if self.image_path:
            try:
                images = ImageProcessor().process_image(self.image_path,
                                                        ["edges", "depth_light", "depth_shading", "depth_combined",
                                                         "smooth_gaussian", "smooth_bilateral", "smooth_median"], None)
                                                        #os.path.split(self.image_path)[0])

                # Process the selected image to generate a 3D mesh
                resolution = int(self.resolution_input.text())
                self.output_mesh_obj = self.depth_to_3d.process_image(self.image_path, self.smoothing_dropdown.currentText(),
                                                           (resolution, resolution))
                print(f"3D model generated successfully for model: {value}")

                # dirname, fname = os.path.split(self.image_path)
                # basename, ext = fname.rsplit(".", 1)
                #
                # combined_name = os.path.join(dirname, f"{basename}-depth_combined.png")
                # self.output_mesh_obj = self.depth_to_3d.process_image(combined_name, self.smoothing_dropdown.currentText(),
                #                                            (resolution, resolution))
                # print(f"BONUS! 3D model generated successfully for model: {value}")

                # self.three_d_viewport.set_trimesh(self.output_mesh_obj)
                self.update_3d_viewport()
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print("No file selected.")

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
            self.update_3d_viewport()

        except Exception as e:
            self.show_error(f"Error while generating mesh: {str(e)}")

    def toggle_invert_colors(self, state):
        """Toggle invert colors based on checkbox state."""
        self.invert_colors_enabled = state == Qt.Checked
        # You can add any necessary logic here, such as applying inversion immediately
        if self.invert_colors_enabled:
            self.invert_checkbox.setCheckState(Qt.Checked)
        else:
            self.invert_checkbox.setCheckState(Qt.Unchecked)
        self.invert_colors()

    def invert_colors(self):
        """Invert colors of the original and processed image."""
        # if self.image is not None:
        #     # Invert the original image
        #     self.image = cv2.bitwise_not(self.image)
        #     self.display_original_image()
        if self.processed_image is not None:
            # Invert the processed image
            self.processed_image = cv2.bitwise_not(self.processed_image)
            self.display_processed_image()

    def update_3d_viewport(self):
        """
        Update or initialize the 3D viewport with the generated mesh.
        This ensures the viewport is reopened if it was closed, and clears existing geometry before adding a new one.
        """
        if self.output_mesh_obj is None:
            print("Must set output_mesh_obj before calling update_3d_viewport.")
            return

        if self.three_d_viewport is None or not self.three_d_viewport.viewer.poll_events():
            # Create a new 3D viewport instance if it doesn't exist or was closed
            self.three_d_viewport = ThreeDViewport()
            print("3D viewport created or reopened.")

        # Clear existing geometry in the viewport if already open
        self.three_d_viewport.viewer.clear_geometries()

        # Load the new mesh into the viewport
        self.three_d_viewport.load_mesh(self.output_mesh_obj)

        # Start the visualization; `run()` can block, but reopening happens only once
        self.three_d_viewport.run()

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
            # Process the image with the updated settings
            low_threshold = self.sensitivity_slider.value()
            self.processed_image = detect_edges(self.image_path, low_threshold, low_threshold * 3,
                                                thickness=self.edge_thickness)
            # Apply inversion if enabled
            if self.invert_colors_enabled:
                self.invert_colors()

            self.display_processed_image()  # Show the processed image

            # TODO: Update the 3D viewport with the new edges when enabled
            # Trigger the 3D viewport update
            # self.update_3d_viewport()

        except Exception as e:
            self.show_error(str(e))

    def update_line_thickness(self):
        self.edge_thickness = self.line_thickness_slider.value()
        self.update_preview()

    def display_processed_image(self):
        if self.processed_image is not None:
            height, width = self.processed_image.shape
            bytes_per_line = width
            q_img = QImage(
                self.processed_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
            self.preview_label.setPixmap(
                QPixmap.fromImage(q_img).scaled(
                    self.preview_label.width(), self.preview_label.height(),
                    Qt.KeepAspectRatio
                )
            )

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

        self.image = cv2.imread(path)
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

    def process_image1(self):
        if not self.image_path:
            self.show_error("Please load an image first!")
            return
        self.update_preview()
        self.save_button.setEnabled(True)

        self.update_3d_viewport()

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

    def _load_config(self):
        self.config.read(CONFIG_FILE_PATH)
        self.invert_colors_enabled = self.config.getboolean("Settings", "invert_colors")

    def _update_config(self, section, option, value):
        self.config.set(section, option, value)
        with open(CONFIG_FILE_PATH, "w") as configfile:
            self.config.write(configfile)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow_ImageProcessing()
    main_window.show()
    sys.exit(app.exec_())