"""Experiment 1: what the data-structure layout costs per particle."""

import sys

from bench_util import printTable, saveResults, timeCall
from layouts import LAYOUTS, buildWithFootprint, seedValues

# 2.8 MB, 14 MB, and 56 MB of payload against a 4 MiB L2 cache: the smallest
# size fits in cache, the largest cannot.
SIZES = (50_000, 250_000, 1_000_000)
DT = 0.01
REPEATS = 5
TOLERANCE = 1e-9


def runSize(count, repeats=REPEATS):
    """Benchmark every layout at one problem size."""
    values = seedValues(count)
    records = []
    reference = None
    for layoutClass in LAYOUTS:
        instance, footprint = buildWithFootprint(layoutClass, values)
        # No layout is timed until it has reproduced the reference answer.
        energy = instance.advance(DT)
        if reference is None:
            reference = energy
        elif abs(energy - reference) > TOLERANCE * abs(reference):
            raise AssertionError(
                f"{layoutClass.name} produced {energy!r}, expected ~{reference!r}"
            )
        timing = timeCall(lambda inst=instance: inst.advance(DT), repeats)
        records.append(
            {
                "layout": layoutClass.name,
                "count": count,
                "best": timing["best"],
                "median": timing["median"],
                "bytesPerParticle": footprint / count,
                "energy": energy,
            }
        )
        del instance
    return records


def main():
    quick = "--quick" in sys.argv
    sizes = (50_000,) if quick else SIZES
    repeats = 3 if quick else REPEATS

    allRecords = []
    for count in sizes:
        records = runSize(count, repeats)
        allRecords.extend(records)
        baseline = records[0]["best"]  # the linked list is the unoptimised reference
        rows = []
        for record in records:
            record["speedupVsLinked"] = baseline / record["best"]
            rows.append(
                (
                    record["layout"],
                    f"{record['best'] * 1e3:9.2f}",
                    f"{record['best'] / count * 1e9:8.1f}",
                    f"{record['speedupVsLinked']:7.1f}x",
                    f"{record['bytesPerParticle']:6.0f}",
                )
            )
        printTable(
            f"Particle advance, N = {count:,} (best of {repeats} reps)",
            ("layout", "time (ms)", "ns/particle", "vs linked", "B/particle"),
            rows,
        )

    saveResults("layouts", {"sizes": list(sizes), "dt": DT, "records": allRecords})


if __name__ == "__main__":
    main()
