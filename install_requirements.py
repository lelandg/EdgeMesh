#!/usr/bin/env python

import sys
import subprocess
import os

def install_requirements():
    print("Installing all requirements from requirements.txt...")

    # Install core requirements first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
        print("Core dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install core dependencies: {e}")
        return

    # Install Open3D specifically
    try:
        subprocess.check_call([sys.executable, "install_open3d.py"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install Open3D: {e}")

    # Install remaining requirements
    requirements_path = os.path.join("MeshTools", "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
            print("All requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install all requirements: {e}")
    else:
        print(f"Requirements file not found at {requirements_path}")

if __name__ == "__main__":
    install_requirements()
