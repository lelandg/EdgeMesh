"""!@file file_tools.py
@brief File tools module for file operations.
@details File tools module for file operations.
@version 0.1.0
@date_created 2025-02-26
@date_modified 2025-02-26
@author Leland Green
@license MIT
"""
import glob
import os
from typing import Any

from spinner import Spinner


def find_newest_file_in_directory(directory_path, supported_extensions):
    """!
    @brief Finds the most recently modified file in a directory with the specified extension.
    @param directory The directory to search in.
    @param extension The file extension to filter by (e.g., '.obj', '.stl').
    @return The name of the most recently modified file matching the criteria.
    """

    spinner = Spinner("{time} Scanning files...")
    # Scan the directory and collect files with allowed extensions
    files_with_timestamps = []
    for root, dirs, files in os.walk(directory_path):
        if root.startswith(".") or len(files) == 0:
            continue  # Skip hidden directories
        spinner.spin(f"Scanning {len(files)} files in {root}...")
        for file in files:
            # Check file extension
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                spinner.spin()
                file_path = os.path.join(root, file)
                modified_time = os.path.getmtime(file_path)  # Get the last modified timestamp
                files_with_timestamps.append((file_path, modified_time))

    # Check if any files are collected
    if not files_with_timestamps:
        return None  # Return None if no valid files are found
    spinner.spin(f"Found {len(files_with_timestamps)} matching files.")
    # Find the newest file
    newest_file = max(files_with_timestamps, key=lambda x: x[1])
    return newest_file[0]  # Return the file name of the newest file


def get_matching_files(patterns: list[str], supported_extensions: list[str]) -> list[str | bytes | Any]:
    """!
    @brief Retrieves a list of files in a directory matching specified pattern(s).
    @param patterns A list of file name patterns to match (e.g., '*.obj', '*.stl').
    @param supported_extensions A list of supported file extensions to filter by.
    @return A list of file names that match the given pattern.
    """

    matched_files = []
    for pattern in patterns:
        spinner = Spinner(f"Matching files for: {pattern}. Searching...")
        spinner.spin()
        # Resolve wildcard patterns
        for file in glob.glob(pattern, recursive=True):
            if os.path.isfile(file) and os.path.splitext(file)[1].lower() in supported_extensions:
                matched_files.append(file)
                spinner.spin("{time} Matched: " + file)

    spinner.spin(f"Found {len(matched_files)} matching files.")
    return matched_files
