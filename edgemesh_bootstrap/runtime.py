"""Initialize native image-processing runtime before starting application jobs."""

import sys


def _platform_query_without_wmi(*args, **kwargs):
    raise OSError("EdgeMesh uses Python's Windows platform fallback instead of WMI")


def initialize_platform_runtime() -> None:
    """Avoid CPython 3.12 WMI worker handle corruption before library imports.

    A timed-out WMI query can leave a worker referring to expired stack data,
    causing delayed native crashes elsewhere in the process (CPython #134313).
    NumPy/SciPy can trigger this through platform.machine() during import.
    Python's platform module already has registry/environment fallbacks for an
    unavailable WMI query. Use those only in the affected Windows 3.12 runtime,
    including when platform was imported before this launcher. No Windows
    service or system setting is changed. Revisit when dropping Python 3.12.
    """
    if (sys.platform == "win32" and sys.version_info[:2] == (3, 12)
            and sys.implementation.name == "cpython"):
        import platform

        if hasattr(platform, "_wmi_query"):
            platform._wmi_query = _platform_query_without_wmi


def initialize_opencv_runtime() -> None:
    """Keep native parallelism when available, with a logged serial fallback.

    On Windows, getNumThreads initializes OpenCV's Concurrency scheduler. Some
    hosts fail here with an otherwise opaque C++ exception, including on valid
    cvtColor/Canny input. A single OpenCV thread bypasses that native scheduler.
    Call on the main thread before image processing or background jobs: changing
    OpenCV's thread configuration while operations are running is not safe.
    """
    import cv2

    try:
        cv2.getNumThreads()
    except cv2.error:
        from log_utils import get_logger

        get_logger().warning(
            "OpenCV parallel runtime could not initialize; using one OpenCV "
            "thread for this session. Image processing may be slower.",
            exc_info=True,
        )
        cv2.setNumThreads(1)
