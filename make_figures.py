"""Render the report figures from the saved JSON results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench_util import loadResults

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
SHORT_NAMES = {
    "Linked list (pointer chasing)": "Linked list",
    "AoS: list of objects": "AoS objects",
    "AoS: __slots__ objects": "AoS __slots__",
    "SoA: array.array (interpreted loop)": "SoA array.array",
    "AoS: NumPy structured array": "AoS NumPy",
    "SoA: NumPy arrays (vectorised)": "SoA NumPy",
}
COLORS = ["#B03A2E", "#CB6D51", "#D9A05B", "#5B8C5A", "#3D6E9C", "#22405E"]


# Counts reported by Azad et al. (2023), Figure 2 and Section IV.
BUG_CATEGORIES = (
    ("Algorithm / data structure", 73),
    ("Micro-architectural", 58),
    ("Inefficient parallelization", 15),
    ("Memory management", 13),
    ("Missing parallelism", 12),
    ("Concurrency control", 7),
    ("Logical error", 3),
    ("I/O", 2),
    ("Communication overhead", 2),
    ("Compiler regression", 1),
)
FIX_CATEGORIES = (
    ("Micro-architecture specific", 64),
    ("Domain specific", 27),
    ("Guiding the compiler", 27),
    ("Algorithm / data structure", 17),
    ("Introduce parallelism", 17),
    ("Memory management", 13),
    ("Balance parallel load", 11),
    ("Remove synchronization", 7),
)


def figureStudyContext():
    """Figure 1: root-cause and fix distributions from the study."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    for axis, series, title, color in (
        (left, BUG_CATEGORIES, "Root causes of 186 HPC performance bugs", "#B03A2E"),
        (right, FIX_CATEGORIES, "Optimization techniques applied in the fixes", "#3D6E9C"),
    ):
        labels = [label for label, _ in series][::-1]
        counts = [count for _, count in series][::-1]
        bars = axis.barh(labels, counts, color=color)
        for bar, count in zip(bars, counts):
            axis.text(
                count + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({count / 186 * 100:.1f}%)",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(0, max(counts) * 1.35)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("commits")
        axis.grid(axis="x", alpha=0.3)
    save(figure, "fig1_study_context.png")


def figureLayoutSchematic():
    """Figure 2: how AoS and SoA place the same records into cache lines."""
    # vx is shaded to show how much of each fetched line one field uses.
    fieldColors = {
        "x": "#D5DBDB", "y": "#D5DBDB", "z": "#D5DBDB",
        "vx": "#B03A2E", "vy": "#E8C1B8", "vz": "#E8C1B8", "m": "#D5DBDB",
    }
    fields = ("x", "y", "z", "vx", "vy", "vz", "m")
    cellsPerLine = 16  # 128-byte cache line / 8-byte float64
    cellCount = 32
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 4.4))

    def drawRow(axis, labels, title, note):
        for index, label in enumerate(labels):
            field = label.split("[")[0].rstrip("0123456789")
            axis.add_patch(
                plt.Rectangle(
                    (index, 0), 1, 1,
                    facecolor=fieldColors.get(field, "#D5DBDB"),
                    edgecolor="white",
                )
            )
            axis.text(
                index + 0.5, 0.5, label, ha="center", va="center", fontsize=6.5,
                color="white" if field == "vx" else "#222",
            )
        for boundary in range(0, cellCount + 1, cellsPerLine):
            axis.axvline(boundary, color="#222", linewidth=1.6)
        for lineIndex in range(cellCount // cellsPerLine):
            axis.text(
                lineIndex * cellsPerLine + cellsPerLine / 2, 1.25,
                f"cache line {lineIndex} (128 B)", ha="center", fontsize=8, color="#222",
            )
        axis.set_xlim(0, cellCount)
        axis.set_ylim(-0.55, 1.6)
        axis.axis("off")
        axis.text(0, -0.42, note, fontsize=8.5, color="#444")
        axis.set_title(title, fontsize=10, loc="left")

    aosLabels = [f"{fields[i % 7]}{i // 7}" for i in range(cellCount)]
    drawRow(
        top, aosLabels,
        "Array of structures: one 56-byte record per particle",
        "Reading vx for every particle: 2 useful values out of 16 per line -> 12.5% of the "
        "traffic is used, no unit-stride SIMD.",
    )
    soaLabels = [f"vx[{i}]" for i in range(cellCount)]
    drawRow(
        bottom, soaLabels,
        "Structure of arrays: one contiguous buffer per field",
        "Reading vx for every particle: 16 useful values out of 16 per line -> 100% of the "
        "traffic is used, prefetcher and SIMD both engage.",
    )
    save(figure, "fig2_layout_schematic.png")


def figureLayouts():
    """Figure 3: ns per particle by layout and problem size (log scale)."""
    data = loadResults("layouts")
    sizes = data["sizes"]
    layouts = list(dict.fromkeys(record["layout"] for record in data["records"]))
    figure, axis = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(layouts)
    for index, layout in enumerate(layouts):
        series = [
            record["best"] / record["count"] * 1e9
            for record in data["records"]
            if record["layout"] == layout
        ]
        positions = [pos + index * width for pos in range(len(sizes))]
        bars = axis.bar(
            positions, series, width, label=SHORT_NAMES.get(layout, layout), color=COLORS[index]
        )
        for bar, value in zip(bars, series):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.1f}",
                ha="center",
                fontsize=6.5,
                rotation=90,
            )
    axis.set_yscale("log")
    axis.set_ylim(top=max(
        record["best"] / record["count"] * 1e9 for record in data["records"]
    ) * 6)
    axis.set_xticks([pos + 0.4 - width / 2 for pos in range(len(sizes))])
    axis.set_xticklabels([f"N = {size:,}" for size in sizes])
    axis.set_ylabel("nanoseconds per particle (log scale)")
    axis.legend(fontsize=8, ncol=3)
    axis.grid(axis="y", alpha=0.3)
    save(figure, "fig3_layouts.png")


def figureFootprint():
    """Figure 4: memory footprint per particle by layout."""
    data = loadResults("layouts")
    largest = max(record["count"] for record in data["records"])
    records = [record for record in data["records"] if record["count"] == largest]
    labels = [SHORT_NAMES.get(record["layout"], record["layout"]) for record in records]
    values = [record["bytesPerParticle"] for record in records]
    figure, axis = plt.subplots(figsize=(8, 4.2))
    bars = axis.barh(labels[::-1], values[::-1], color=COLORS[::-1])
    for bar, value in zip(bars, values[::-1]):
        axis.text(
            value + 4, bar.get_y() + bar.get_height() / 2,
            f"{value:.0f} B", va="center", fontsize=8,
        )
    axis.axvline(56, color="#444", linestyle="--", linewidth=1)
    axis.text(58, -0.45, "56 B = the payload (7 x float64)", fontsize=8, color="#444")
    axis.set_xlabel("bytes per particle")
    axis.grid(axis="x", alpha=0.3)
    save(figure, "fig4_footprint.png")


def figureStride():
    """Figure 5: access stride versus time per element and cache-line utilisation."""
    data = loadResults("stride")
    strides = [record["strideBytes"] for record in data["records"]]
    times = [record["nsPerElement"] for record in data["records"]]
    useful = [record["usefulFraction"] * 100 for record in data["records"]]
    lineBytes = data["machine"].get("cacheLineBytes", 64)

    figure, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot(strides, times, marker="o", color="#B03A2E", label="time per element")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("distance between consecutive reads (bytes)")
    axis.set_ylabel("nanoseconds per element read", color="#B03A2E")
    axis.axvline(lineBytes, color="#3D6E9C", linestyle="--", linewidth=1)
    axis.text(
        lineBytes * 1.1, max(times) * 0.5,
        f"cache line = {lineBytes} B", fontsize=8, color="#3D6E9C",
    )
    axis.grid(alpha=0.3)

    twin = axis.twinx()
    twin.plot(strides, useful, marker="s", color="#5B8C5A", label="cache line used")
    twin.set_ylabel("% of each fetched cache line actually used", color="#5B8C5A")
    twin.set_ylim(0, 105)
    save(figure, "fig5_stride.png")


def figureBlocking():
    """Figure 6: transpose time against tile size, with the naive baseline."""
    data = loadResults("blocking")
    naive = next(record for record in data["records"] if record["blockSize"] is None)
    blocked = [record for record in data["records"] if record["blockSize"]]
    library = [
        record
        for record in data["records"]
        if record["blockSize"] is None and record["variant"].startswith("NumPy")
    ]
    sizes = [record["blockSize"] for record in blocked]
    times = [record["best"] * 1e3 for record in blocked]
    l1 = data["machine"].get("l1DataCacheBytes")
    l2 = data["machine"].get("l2CacheBytes")

    figure, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot(sizes, times, marker="o", color="#3D6E9C", label="blocked transpose")
    axis.axhline(
        naive["best"] * 1e3, color="#B03A2E", linestyle="--",
        label="naive (column writes)",
    )
    if library:
        axis.axhline(
            library[0]["best"] * 1e3, color="#5B8C5A", linestyle=":",
            label="NumPy ascontiguousarray",
        )
    itemBytes = data["itemBytes"]
    for cacheBytes, label in ((l1, "L1d"), (l2, "L2")):
        if not cacheBytes:
            continue
        # Tile side at which two tiles stop fitting in this cache level.
        edge = (cacheBytes / (2 * itemBytes)) ** 0.5
        if min(sizes) <= edge <= max(sizes):
            axis.axvline(edge, color="#888", linewidth=1)
            axis.text(edge * 1.03, max(times) * 0.8, f"{label} limit", fontsize=8, color="#555")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("tile size (elements per side)")
    axis.set_ylabel("transpose time (ms)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    save(figure, "fig6_blocking.png")


def save(figure, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    figure.tight_layout()
    path = FIGURES_DIR / name
    figure.savefig(path, dpi=200)
    plt.close(figure)
    print(f"wrote {path}")


def main():
    figureStudyContext()
    figureLayoutSchematic()
    figureLayouts()
    figureFootprint()
    figureStride()
    figureBlocking()


if __name__ == "__main__":
    main()
