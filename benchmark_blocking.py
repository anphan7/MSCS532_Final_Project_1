"""Experiment 3: cache blocking (loop tiling) on a matrix transpose."""

import numpy as np

from bench_util import machineInfo, printTable, saveResults, timeCall

MATRIX_SIZE = 4096
BLOCK_SIZES = (8, 16, 32, 64, 128, 256, 512)
REPEATS = 3


def transposeNaive(source, destination):
    """Row-at-a-time transpose."""
    # Reading a row means writing a column: every write walks memory with a
    # full-row stride, so no cache line is ever reused before eviction.
    for i in range(source.shape[0]):
        destination[:, i] = source[i, :]
    return destination


def transposeBlocked(source, destination, blockSize):
    """Tile-at-a-time transpose (Lam, Rothberg, & Wolf, 1991)."""
    # Both streams stay inside one tile, so a tile small enough to be cache
    # resident is reused before it is evicted.
    rows, cols = source.shape
    for rowStart in range(0, rows, blockSize):
        rowStop = min(rowStart + blockSize, rows)
        for colStart in range(0, cols, blockSize):
            # min() clamps the edge tiles when the block size does not divide
            # the matrix evenly.
            colStop = min(colStart + blockSize, cols)
            destination[colStart:colStop, rowStart:rowStop] = source[
                rowStart:rowStop, colStart:colStop
            ].T
    return destination


def main():
    info = machineInfo()
    rng = np.random.default_rng(20250814)
    source = rng.random((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)
    destination = np.empty((MATRIX_SIZE, MATRIX_SIZE), dtype=np.float64)
    expected = source.T

    naive = timeCall(lambda: transposeNaive(source, destination), REPEATS)
    assert np.array_equal(destination, expected), "naive transpose is wrong"

    records = [{"variant": "naive (column writes)", "blockSize": None, "best": naive["best"]}]
    rows = [("naive (column writes)", "-", f"{naive['best'] * 1e3:9.1f}", "   1.0x", "-")]

    for blockSize in BLOCK_SIZES:
        destination.fill(0.0)
        timing = timeCall(
            lambda size=blockSize: transposeBlocked(source, destination, size), REPEATS
        )
        assert np.array_equal(destination, expected), f"blocked transpose {blockSize} is wrong"
        tileBytes = 2 * blockSize * blockSize * source.itemsize  # src + dst tile
        records.append(
            {
                "variant": f"blocked {blockSize}x{blockSize}",
                "blockSize": blockSize,
                "best": timing["best"],
                "tileWorkingSetBytes": tileBytes,
                "speedup": naive["best"] / timing["best"],
            }
        )
        rows.append(
            (
                f"blocked {blockSize}x{blockSize}",
                f"{blockSize}",
                f"{timing['best'] * 1e3:9.1f}",
                f"{naive['best'] / timing['best']:7.1f}x",
                f"{tileBytes / 1024:8.0f} KiB",
            )
        )

    # NumPy's own contiguous copy, as a reference point for the hand-tiled loop.
    library = timeCall(lambda: np.ascontiguousarray(source.T), REPEATS)
    records.append(
        {
            "variant": "NumPy ascontiguousarray",
            "blockSize": None,
            "best": library["best"],
            "speedup": naive["best"] / library["best"],
        }
    )
    rows.append(
        (
            "NumPy ascontiguousarray",
            "-",
            f"{library['best'] * 1e3:9.1f}",
            f"{naive['best'] / library['best']:7.1f}x",
            "internal",
        )
    )

    printTable(
        f"{MATRIX_SIZE}x{MATRIX_SIZE} float64 transpose "
        f"({source.nbytes / 1024 ** 2:.0f} MiB per matrix, "
        f"L1d = {info.get('l1DataCacheBytes', 0) // 1024} KiB, "
        f"L2 = {info.get('l2CacheBytes', 0) // 1024} KiB)",
        ("variant", "block", "time (ms)", "speedup", "tile working set"),
        rows,
    )
    saveResults(
        "blocking",
        {
            "matrixSize": MATRIX_SIZE,
            "itemBytes": int(source.itemsize),
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
