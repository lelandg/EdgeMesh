#!/usr/bin/env python

import sys
import subprocess
import platform

def install_open3d():
    python_version = sys.version.split()[0]
    print(f"Installing Open3D for Python {python_version}...")

    # Check if Python version is 3.13 or higher
    major, minor = map(int, python_version.split('.')[:2])
    is_python_313_or_higher = (major == 3 and minor >= 13) or major > 3

    # For Python 3.13+, display compatibility warning
    if is_python_313_or_higher:
        print("\n" + "="*80)
        print("WARNING: Open3D is not currently compatible with Python 3.13+")
        print("="*80)
        print("Open3D 0.19.0 is known to work with Python 3.12 and earlier versions.")
        print("To use Open3D with this project, you have the following options:")
        print("1. Downgrade to Python 3.12 (recommended)")
        print("2. Wait for Open3D to release a version compatible with Python 3.13+")
        print("3. Use a virtual environment with Python 3.12")
        print("\nInstructions for creating a Python 3.12 virtual environment:")
        print("1. Install Python 3.12 from https://www.python.org/downloads/")
        print("2. Create a virtual environment: python3.12 -m venv venv")
        print("3. Activate the virtual environment:")
        if platform.system() == "Windows":
            print("   - Windows: venv\\Scripts\\activate")
        else:
            print("   - Linux/Mac: source venv/bin/activate")
        print("4. Install requirements: pip install -r requirements.txt")
        print("="*80 + "\n")

        # Still try to install in case a compatible version becomes available
        print("Attempting to install anyway in case a compatible version is available...")
        try:
            # Try the latest version first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d", "--force-reinstall"])
            print("Latest Open3D installed successfully!")
            return
        except subprocess.CalledProcessError:
            print("Failed to install latest Open3D.")
            print("As expected, Open3D is not currently compatible with Python 3.13+")
            print("Please follow the instructions above to use a compatible Python version.")
            return

    # For Python 3.12 and below, try the specified version first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d==0.19.0", "--force-reinstall"])
        print("Open3D 0.19.0 installed successfully!")
        return
    except subprocess.CalledProcessError:
        print("Failed to install Open3D 0.19.0 directly. Trying alternative approaches...")

    # Try to install from wheels if direct installation fails
    if platform.system() == "Windows":
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "--index-url=https://pypi.org/simple", 
                                  "--no-cache-dir", 
                                  "open3d==0.19.0"])
            print("Open3D 0.19.0 installed successfully from wheels!")
        except subprocess.CalledProcessError:
            print("Failed to install Open3D from wheels.")
    else:  # For Linux/MacOS
        try:
            # Install dependencies first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                  "--index-url=https://pypi.org/simple", 
                                  "--no-cache-dir", 
                                  "open3d==0.19.0"])
            print("Open3D 0.19.0 installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install Open3D. Please check if your Python version is supported.")

    # Verify installation
    try:
        import open3d
        print(f"Open3D version {open3d.__version__} is installed and working properly!")
    except ImportError:
        print("Open3D is still not installed correctly. Try running this script with administrator/sudo privileges.")

if __name__ == "__main__":
    install_open3d()
