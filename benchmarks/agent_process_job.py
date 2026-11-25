"""Windows job ownership for benchmark CLI descendants, using only stdlib.

Assign a waiting launcher before releasing it to create the agent process.
This avoids the spawn/assignment race and kills descendants even when the
launcher exits successfully. This is lifecycle control, not a code sandbox.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes as w
import time


class WindowsJob:
    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel, self.handle = kernel, None
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            ("SetInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            ("AssignProcessToJobObject", [w.HANDLE, w.HANDLE], w.BOOL),
            ("TerminateJobObject", [w.HANDLE, w.UINT], w.BOOL),
            ("QueryInformationJobObject", [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p], w.BOOL),
            ("CloseHandle", [w.HANDLE], w.BOOL),
        ):
            function = getattr(kernel, name)
            function.argtypes, function.restype = args, result

        class BasicLimits(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                        ("flags", w.DWORD), ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t),
                        ("active_limit", w.DWORD), ("affinity", ctypes.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("basic", BasicLimits), ("io", ctypes.c_ulonglong * 6),
                        ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            kernel.CloseHandle(handle)
            self.handle = None
            raise ctypes.WinError(error)

    def assign(self, process) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle is None:
            return

        class Accounting(ctypes.Structure):
            _fields_ = [("times", ctypes.c_longlong * 4), ("faults", w.DWORD),
                        ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]

        try:
            if not self.kernel.TerminateJobObject(self.handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            deadline = time.monotonic() + 5
            while True:
                state = Accounting()
                if not self.kernel.QueryInformationJobObject(
                    self.handle, 1, ctypes.byref(state), ctypes.sizeof(state), None,
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if state.active == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("agent descendants did not terminate")
                time.sleep(0.01)
        finally:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
