#!/usr/bin/env python

import glob
import os
import platform
import subprocess
import sys

# PyPI's open3d releases top out at cp312 (0.19.0). For Python 3.13/3.14 the
# only wheels are on the rolling `main-devel` prerelease tag, whose URLs are
# overwritten in place as upstream main moves — so those wheels must be
# downloaded ONCE and installed from a pinned local file, never straight from
# the URL. Default pin location (override with EDGEMESH_OPEN3D_WHEEL):
WHEEL_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "edgemesh", "wheels")
MAIN_DEVEL_RELEASE = "https://github.com/isl-org/Open3D/releases/tag/main-devel"
MAIN_DEVEL_LINUX_X86_64_EXAMPLE = (
    "https://github.com/isl-org/Open3D/releases/download/main-devel/"
    "open3d-0.19.0-cp314-cp314-manylinux_2_35_x86_64.whl"
)


def find_pinned_wheel(major, minor):
    """Locate a pinned local Open3D wheel for this interpreter, if any."""
    explicit = os.environ.get("EDGEMESH_OPEN3D_WHEEL")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        print(f"EDGEMESH_OPEN3D_WHEEL is set but does not exist: {explicit}")
        return None
    pattern = os.path.join(WHEEL_CACHE, f"open3d-*-cp{major}{minor}-*.whl")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def install_open3d():
    python_version = sys.version.split()[0]
    print(f"Installing Open3D for Python {python_version}...")

    major, minor = map(int, python_version.split('.')[:2])

    # Python 3.13+: no PyPI wheels — install from a pinned local main-devel wheel
    if (major == 3 and minor >= 13) or major > 3:
        wheel = find_pinned_wheel(major, minor)
        if wheel:
            print(f"Installing pinned local wheel: {wheel}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", wheel])
                print("Open3D installed successfully from the pinned wheel!")
            except subprocess.CalledProcessError:
                print("Failed to install the pinned wheel. Check that its cp tag "
                      f"matches this interpreter (cp{major}{minor}) and your glibc "
                      "is >= the wheel's manylinux requirement.")
            return

        print("\n" + "=" * 80)
        print(f"Open3D on PyPI has no wheels for Python {major}.{minor} (PyPI stops at 3.12).")
        print("=" * 80)
        print("Upstream publishes development wheels for newer Pythons on the rolling")
        print(f"'main-devel' prerelease tag:\n  {MAIN_DEVEL_RELEASE}")
        print("Those URLs are overwritten in place as upstream main moves, so:")
        print(f"1. Download the cp{major}{minor} wheel for your platform ONCE, e.g.")
        print(f"   (Linux x86_64): {MAIN_DEVEL_LINUX_X86_64_EXAMPLE}")
        print(f"2. Save it under {WHEEL_CACHE}")
        print("   (or point EDGEMESH_OPEN3D_WHEEL at the file), and record its sha256.")
        print("3. Re-run this script — it will install from that pinned file.")
        print("Alternatively, use Python 3.12 and the released open3d==0.19.0 from PyPI.")
        print("=" * 80 + "\n")
        return

    # Python 3.12 and below: released PyPI version
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
