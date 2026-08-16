"""Correctness gate: an optimisation that changes the answer is not one."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmark_blocking import transposeBlocked, transposeNaive
from layouts import LAYOUTS, ArraySoa, LinkedParticles, NumpyAos, NumpySoa, seedValues

TOLERANCE = 1e-9


def collectState(instance):
    """Read x, y, z out of any layout into plain Python lists."""
    # Each layout stores its fields differently, so unpack per type.
    if isinstance(instance, LinkedParticles):
        state = []
        node = instance.head
        while node is not None:
            state.append((node.x, node.y, node.z))
            node = node.next
        return state
    if isinstance(instance, ArraySoa):
        return list(zip(instance.data["x"], instance.data["y"], instance.data["z"]))
    if isinstance(instance, (NumpyAos, NumpySoa)):
        return list(
            zip(
                instance.data["x"].tolist(),
                instance.data["y"].tolist(),
                instance.data["z"].tolist(),
            )
        )
    return [(item.x, item.y, item.z) for item in instance.items]


def testLayoutsAgree():
    values = seedValues(5_000)
    # The linked list runs first and becomes the reference answer.
    reference = None
    referenceState = None
    failures = []
    for layoutClass in LAYOUTS:
        instance = layoutClass(values)
        energy = instance.advance(0.01)
        state = collectState(instance)
        if reference is None:
            reference, referenceState = energy, state
            continue
        if abs(energy - reference) > TOLERANCE * abs(reference):
            failures.append(f"{layoutClass.name}: energy {energy} != {reference}")
        worst = max(
            abs(a - b)
            for got, want in zip(state, referenceState)
            for a, b in zip(got, want)
        )
        if worst > TOLERANCE:
            failures.append(f"{layoutClass.name}: position drift {worst}")
        print(f"  ok  {layoutClass.name:<34} energy = {energy:.9f}")
    return failures


def testBlockedTranspose():
    rng = np.random.default_rng(7)
    # Deliberately not a multiple of any block size, so every tile size hits
    # the ragged-edge path.
    source = rng.random((257, 129))
    expected = source.T.copy()
    failures = []
    naive = transposeNaive(source, np.empty((129, 257)))
    if not np.array_equal(naive, expected):
        failures.append("naive transpose mismatch")
    for blockSize in (8, 16, 32, 64, 128, 256):
        blocked = transposeBlocked(source, np.empty((129, 257)), blockSize)
        if not np.array_equal(blocked, expected):
            failures.append(f"blocked transpose mismatch at block {blockSize}")
        else:
            print(f"  ok  blocked transpose, block = {blockSize:<3} (ragged edges handled)")
    return failures


def main():
    print("Layout equivalence (5,000 particles, one 0.01 s step)")
    failures = testLayoutsAgree()
    print("\nBlocked transpose equivalence (257x129, ragged tiles)")
    failures += testBlockedTranspose()
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll correctness checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
