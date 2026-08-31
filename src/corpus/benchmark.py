"""Optional, manually run benchmark for the offline stage.

This is a diagnostic tool, not part of the serving path, and it is never
exercised by the test suite.  Run it by hand when you want real numbers:

    python -m src.corpus.benchmark Archive.zip

It reports record count, distinct trigrams, total posting entries, build
time and - where the platform allows it without extra dependencies - the
process memory before and after initialization.
"""

import ctypes
import sys
import time

from src.corpus.initialization import initialize_corpus


def process_memory_bytes() -> int | None:
    """Best-effort resident memory of this process, or None if unavailable.

    Uses only the standard library: ``resource`` on Unix, the Win32 psapi
    call on Windows.  Any failure degrades to None rather than raising.
    """
    try:
        import resource
    except ImportError:
        pass
    else:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kilobytes, macOS reports bytes.
        return usage if sys.platform == "darwin" else usage * 1024

    try:
        class _MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")

        # The argument types matter: a process HANDLE is pointer sized, and
        # letting ctypes guess passes it as a 32-bit int, which makes the
        # call fail silently on 64-bit Python.
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int

        counters = _MemoryCounters()
        counters.cb = ctypes.sizeof(_MemoryCounters)

        succeeded = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if succeeded:
            return int(counters.WorkingSetSize)
    except (AttributeError, OSError):
        pass

    return None


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unavailable"
    return f"{value / (1024 * 1024):.1f} MiB"


def run_benchmark(archive_path: str) -> dict:
    """Initialize the corpus from ``archive_path`` and report statistics."""
    memory_before = process_memory_bytes()
    start_time = time.perf_counter()

    index = initialize_corpus(archive_path)

    total_seconds = time.perf_counter() - start_time
    memory_after = process_memory_bytes()

    report = dict(index.stats())
    report["total_initialization_seconds"] = total_seconds
    report["process_memory_before_bytes"] = memory_before
    report["process_memory_after_bytes"] = memory_after

    print(f"records                 : {report['record_count']:,}")
    print(f"distinct trigrams       : {report['distinct_trigrams']:,}")
    print(f"total posting entries   : {report['total_postings']:,}")
    print(f"index build time        : {report['build_seconds']:.2f} s")
    print(f"total initialization    : {total_seconds:.2f} s")
    print(
        "approximate index size  : "
        f"{_format_bytes(report['approximate_index_bytes'])}"
    )
    print(f"process memory before   : {_format_bytes(memory_before)}")
    print(f"process memory after    : {_format_bytes(memory_after)}")

    return report


if __name__ == "__main__":
    archive = sys.argv[1] if len(sys.argv) > 1 else "Archive.zip"
    run_benchmark(archive)
