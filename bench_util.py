"""Timing, environment capture, and reporting helpers."""

import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def timeCall(callable_, repeats=5):
    """Time callable_ and return statistics in seconds."""
    # Warm-up call: keeps import-time work and first-touch page faults out of
    # the measurement.
    callable_()
    samples = []
    # A collection triggered by one layout must not be charged to another.
    gcWasEnabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            callable_()
            samples.append(time.perf_counter() - start)
    finally:
        if gcWasEnabled:
            gc.enable()
    return {
        # The minimum is the sample least polluted by scheduler noise; the
        # median is kept so run-to-run variance stays visible.
        "best": min(samples),
        "median": statistics.median(samples),
        "worst": max(samples),
        "repeats": repeats,
    }


def sysctlValue(name):
    """Return a sysctl value on macOS, or None when it is unavailable."""
    try:
        output = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = output.stdout.strip()
    return value or None


def machineInfo():
    """Hardware and software context the numbers depend on."""
    info = {
        "platform": platform.platform(),
        "processor": sysctlValue("machdep.cpu.brand_string") or platform.processor(),
        "logicalCores": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    for label, key in (
        ("cacheLineBytes", "hw.cachelinesize"),
        ("l1DataCacheBytes", "hw.l1dcachesize"),
        ("l2CacheBytes", "hw.l2cachesize"),
        ("memoryBytes", "hw.memsize"),
    ):
        value = sysctlValue(key)
        if value is not None:
            info[label] = int(value)
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info


def printTable(title, headers, rows):
    """Print a fixed-width table, readable in a console screenshot."""
    widths = [len(str(head)) for head in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    line = "  ".join(str(head).ljust(widths[index]) for index, head in enumerate(headers))
    print(f"\n{title}")
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def saveResults(name, payload):
    """Persist results as JSON; the figures and the report read these back."""
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = dict(payload)
    payload["machine"] = machineInfo()
    path = RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nresults written to {path}")
    return path


def loadResults(name):
    return json.loads((RESULTS_DIR / f"{name}.json").read_text())
