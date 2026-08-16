"""Run the tests, the three experiments, and the figure generator."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = (
    ("Correctness tests", "tests/test_layouts.py"),
    ("Experiment 1 - data-structure layout", "benchmark_layouts.py"),
    ("Experiment 2 - stride and the cache line", "benchmark_stride.py"),
    ("Experiment 3 - cache blocking", "benchmark_blocking.py"),
    ("Figures", "make_figures.py"),
)


def main():
    for title, script in STEPS:
        print("\n" + "=" * 78, flush=True)
        print(f"== {title}  ({script})", flush=True)
        print("=" * 78, flush=True)
        result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
        if result.returncode != 0:
            print(f"\n{script} failed with exit code {result.returncode}")
            return result.returncode
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
