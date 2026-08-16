"""Experiment 2: what a growing access stride costs."""

import numpy as np

from bench_util import machineInfo, printTable, saveResults, timeCall

# Same number of elements in every configuration, so only the spacing between
# them varies. Reading one field of a 7-field record is a stride-7 access.
TOUCHED = 2_000_000
STRIDES = (1, 2, 4, 8, 16, 32, 64, 128)
ITEM_BYTES = 8               # float64
REPEATS = 7


def runStride(stride, repeats=REPEATS):
    """Sum TOUCHED elements separated by `stride` elements."""
    # The array grows with the stride so the element count stays fixed.
    data = np.ones(TOUCHED * stride, dtype=np.float64)
    view = data[::stride]
    timing = timeCall(lambda: float(view.sum()), repeats)
    return timing


def main():
    info = machineInfo()
    lineBytes = info.get("cacheLineBytes", 64)
    records = []
    rows = []
    baseline = None
    for stride in STRIDES:
        timing = runStride(stride)
        perElement = timing["best"] / TOUCHED * 1e9
        if baseline is None:
            baseline = perElement
        # Memory arrives one cache line at a time, so a read costs a whole
        # line once the stride reaches the line width.
        movedBytes = TOUCHED * min(stride * ITEM_BYTES, lineBytes)
        records.append(
            {
                "stride": stride,
                "strideBytes": stride * ITEM_BYTES,
                "best": timing["best"],
                "nsPerElement": perElement,
                "slowdown": perElement / baseline,
                "usefulFraction": min(1.0, lineBytes / (stride * ITEM_BYTES)),
                "effectiveGiBs": movedBytes / timing["best"] / (1024 ** 3),
            }
        )
        rows.append(
            (
                f"{stride:5d}",
                f"{stride * ITEM_BYTES:8d}",
                f"{perElement:9.2f}",
                f"{perElement / baseline:8.1f}x",
                f"{min(1.0, lineBytes / (stride * ITEM_BYTES)) * 100:8.1f}",
            )
        )

    printTable(
        f"Strided read of {TOUCHED:,} float64 values (cache line = {lineBytes} B)",
        ("stride", "bytes", "ns/elem", "slowdown", "line used %"),
        rows,
    )
    saveResults(
        "stride",
        {"touched": TOUCHED, "itemBytes": ITEM_BYTES, "records": records},
    )


if __name__ == "__main__":
    main()
