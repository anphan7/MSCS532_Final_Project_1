"""Six memory layouts for the same particle workload."""

import array
import random
import sys
import tracemalloc

import numpy as np

FIELDS = ("x", "y", "z", "vx", "vy", "vz", "m")


def seedValues(count, seed=20250814):
    # Fixed seed: every layout must process the identical numbers.
    rng = random.Random(seed)
    return [
        tuple(rng.uniform(-1.0, 1.0) for _ in range(6)) + (rng.uniform(0.5, 2.0),)
        for _ in range(count)
    ]


class LinkedParticles:
    """Pointer-chased nodes, the Python analogue of C++ std::forward_list."""

    class Node:
        __slots__ = FIELDS + ("next",)

        def __init__(self, values):
            (self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m) = values
            self.next = None

    name = "Linked list (pointer chasing)"

    def __init__(self, values):
        self.head = None
        self.count = len(values)
        previous = None
        # Each node is its own allocation, so consecutive particles end up at
        # unrelated addresses. This is the layout TileDB-d51b082 removed.
        for value in values:
            node = LinkedParticles.Node(value)
            if previous is None:
                self.head = node
            else:
                previous.next = node
            previous = node

    def advance(self, dt):
        # x += vx*dt for each coordinate, plus a kinetic-energy reduction.
        # Identical arithmetic in all six layouts; only the access differs.
        energy = 0.0
        node = self.head
        while node is not None:
            node.x += node.vx * dt
            node.y += node.vy * dt
            node.z += node.vz * dt
            energy += 0.5 * node.m * (
                node.vx * node.vx + node.vy * node.vy + node.vz * node.vz
            )
            # The prefetcher cannot guess this address: it only becomes known
            # once the current node has arrived from memory.
            node = node.next
        return energy

    def footprintBytes(self):
        node = self.head
        nodeBytes = sys.getsizeof(node) + sum(sys.getsizeof(getattr(node, f)) for f in FIELDS)
        return nodeBytes * self.count


class ObjectParticles:
    """Array of structures: a Python list of ordinary objects."""

    class Particle:
        def __init__(self, values):
            (self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m) = values

    name = "AoS: list of objects"

    def __init__(self, values):
        # The list is contiguous but holds pointers; each particle's fields
        # live in its own instance dictionary elsewhere on the heap.
        self.items = [ObjectParticles.Particle(value) for value in values]

    def advance(self, dt):
        energy = 0.0
        for particle in self.items:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.z += particle.vz * dt
            energy += 0.5 * particle.m * (
                particle.vx * particle.vx
                + particle.vy * particle.vy
                + particle.vz * particle.vz
            )
        return energy

    def footprintBytes(self):
        sample = self.items[0]
        perItem = (
            sys.getsizeof(sample)
            + sys.getsizeof(sample.__dict__)
            + sum(sys.getsizeof(getattr(sample, f)) for f in FIELDS)  # boxed floats
            + 8  # the pointer the enclosing list stores
        )
        return perItem * len(self.items)


class SlotParticles:
    """Array of structures without the per-instance dictionary."""

    class Particle:
        # __slots__ drops the instance dict, which separates the cost of that
        # dict from the cost of the layout itself.
        __slots__ = FIELDS

        def __init__(self, values):
            (self.x, self.y, self.z, self.vx, self.vy, self.vz, self.m) = values

    name = "AoS: __slots__ objects"

    def __init__(self, values):
        self.items = [SlotParticles.Particle(value) for value in values]

    def advance(self, dt):
        energy = 0.0
        for particle in self.items:
            particle.x += particle.vx * dt
            particle.y += particle.vy * dt
            particle.z += particle.vz * dt
            energy += 0.5 * particle.m * (
                particle.vx * particle.vx
                + particle.vy * particle.vy
                + particle.vz * particle.vz
            )
        return energy

    def footprintBytes(self):
        sample = self.items[0]
        perItem = (
            sys.getsizeof(sample)
            + sum(sys.getsizeof(getattr(sample, f)) for f in FIELDS)
            + 8
        )
        return perItem * len(self.items)


class ArraySoa:
    """Structure of arrays over the standard library's array.array."""

    name = "SoA: array.array (interpreted loop)"

    def __init__(self, values):
        self.count = len(values)
        # One contiguous buffer of raw C doubles per field: the layout a cache
        # wants, but reached from an interpreted loop.
        self.data = {
            field: array.array("d", [value[index] for value in values])
            for index, field in enumerate(FIELDS)
        }

    def advance(self, dt):
        x, y, z = self.data["x"], self.data["y"], self.data["z"]
        vx, vy, vz, m = (self.data[f] for f in ("vx", "vy", "vz", "m"))
        energy = 0.0
        for i in range(self.count):
            # Every read boxes a C double into a new Python float and every
            # write unboxes one. That cost is why this layout loses.
            velX, velY, velZ = vx[i], vy[i], vz[i]
            x[i] += velX * dt
            y[i] += velY * dt
            z[i] += velZ * dt
            energy += 0.5 * m[i] * (velX * velX + velY * velY + velZ * velZ)
        return energy

    def footprintBytes(self):
        return sum(buffer.buffer_info()[1] * buffer.itemsize for buffer in self.data.values())


class NumpyAos:
    """NumPy structured array: contiguous but interleaved."""

    name = "AoS: NumPy structured array"

    def __init__(self, values):
        # Records are stored x0,y0,...,m0,x1,y1,... so one field view has a
        # 56-byte stride: a 128-byte line yields two useful values, not sixteen.
        dtype = np.dtype([(field, np.float64) for field in FIELDS])
        self.data = np.array([tuple(value) for value in values], dtype=dtype)

    def advance(self, dt):
        # These operations run in compiled C, but on strided views, so the SIMD
        # units cannot issue unit-stride loads.
        data = self.data
        data["x"] += data["vx"] * dt
        data["y"] += data["vy"] * dt
        data["z"] += data["vz"] * dt
        return float(
            0.5
            * np.sum(
                data["m"]
                * (data["vx"] ** 2 + data["vy"] ** 2 + data["vz"] ** 2)
            )
        )

    def footprintBytes(self):
        return int(self.data.nbytes)


class NumpySoa:
    """Structure of arrays in NumPy: one unit-stride buffer per field."""

    name = "SoA: NumPy arrays (vectorised)"

    def __init__(self, values):
        columns = np.asarray(values, dtype=np.float64)
        # ascontiguousarray forces a real copy per field; a column slice of the
        # source would still carry the interleaved stride.
        self.data = {
            field: np.ascontiguousarray(columns[:, index])
            for index, field in enumerate(FIELDS)
        }

    def advance(self, dt):
        # Same compiled loops as NumpyAos, now over contiguous buffers: the
        # prefetcher and the SIMD units both engage.
        data = self.data
        vx, vy, vz = data["vx"], data["vy"], data["vz"]
        data["x"] += vx * dt
        data["y"] += vy * dt
        data["z"] += vz * dt
        return float(0.5 * np.sum(data["m"] * (vx * vx + vy * vy + vz * vz)))

    def footprintBytes(self):
        return sum(int(buffer.nbytes) for buffer in self.data.values())


# Ordered from least to most cache-friendly; the driver treats them uniformly
# through build(values), advance(dt), and footprintBytes().
LAYOUTS = (
    LinkedParticles,
    ObjectParticles,
    SlotParticles,
    ArraySoa,
    NumpyAos,
    NumpySoa,
)


def buildWithFootprint(layoutClass, values):
    """Construct a layout and measure the memory it actually allocated."""
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    instance = layoutClass(values)
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    # tracemalloc sees the object allocator; array.array and NumPy payloads
    # live outside it, so take whichever accounting is larger.
    return instance, max(after - before, instance.footprintBytes())
