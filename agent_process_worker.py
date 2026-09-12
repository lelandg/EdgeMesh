"""Own a runtime tree before launching it, including Windows venv redirectors.

The Windows caller runs this stdlib-only file with the base Python interpreter
and isolated startup. The worker joins its Job Object before creating any
descendant. Closing the worker closes the only job handle and stops the tree.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


class OwnedProcessTree:
    """A Windows Job Object owns descendants; POSIX launches use a new session."""

    def __init__(self, pid: int):
        self.pid = pid
        self.handle = None
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                        ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                        ("max_working", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                        ("scheduling", wintypes.DWORD)]

        class Counters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", Counters),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.SetInformationJobObject.restype = wintypes.BOOL
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        job = kernel.CreateJobObjectW(None, None)
        if not job:
            raise OSError("Could not create owned runtime job")
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        process = None
        try:
            if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise OSError("Could not configure owned runtime job")
            process = kernel.OpenProcess(0x0100 | 0x0001, False, pid)
            if not process or not kernel.AssignProcessToJobObject(job, process):
                raise OSError("Could not attach runtime to its job")
        except OSError:
            kernel.CloseHandle(job)
            raise
        finally:
            if process:
                kernel.CloseHandle(process)
        self.handle = job
        self._kernel = kernel

    def close(self):
        if self.handle:
            self._kernel.CloseHandle(self.handle)
            self.handle = None
        elif sys.platform != "win32" and self.pid:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.pid = 0


def main():
    if sys.platform != "win32" or len(sys.argv) < 3 or sys.argv[1] != "--":
        return 1
    try:
        # Retain the sole, non-inheritable job handle until ExitProcess. Closing
        # it explicitly would terminate this worker before its exit code is set.
        tree = OwnedProcessTree(os.getpid())
        child = subprocess.Popen(
            sys.argv[2:], stdin=sys.stdin.buffer, stdout=sys.stdout.buffer,
            stderr=sys.stderr.buffer, creationflags=subprocess.CREATE_NO_WINDOW)
        code = child.wait()
        if not tree.handle:
            raise OSError("Runtime ownership was lost")
    except (OSError, ValueError):
        print("The owned assistant runtime could not be started.", file=sys.stderr, flush=True)
        code = 1
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
