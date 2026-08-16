# Final Project, Part 1 — Data Locality in HPC

A Python prototype of the optimization technique that Azad et al. (2023) found
most often in real HPC bug fixes: **data-locality optimization through
data-structure layout**. It runs one particle time step across six memory
layouts, measures what access stride costs, and applies cache blocking to a
matrix transpose.

## How to run

```bash
git clone https://github.com/anphan7/MSCS532_Final_Project_1.git
cd MSCS532_Final_Project_1
python3 -m venv .venv
source .venv/bin/activate          # needed in every new terminal
pip install -r requirements.txt

python3 run_all.py                 # tests, three experiments, figures
```

Skipping `source .venv/bin/activate` means `python3` is the system interpreter,
which fails with `ModuleNotFoundError: No module named 'numpy'`. Check with
`which python3` — it should end in `.venv/bin/python3`. Run `deactivate` to
exit. To skip activation entirely, call `.venv/bin/python run_all.py` instead.

Takes about four minutes on an Apple M1 and needs ~1 GB free memory. Results
land in `results/*.json`, figures in `figures/`.

## Findings

Measured on an Apple M1 (128-byte cache line, 64 KiB L1d, 4 MiB L2).

| Experiment | Result |
| --- | --- |
| Layout, N = 1M particles | Linked list 231 ms → NumPy SoA 6.4 ms (**36×**) |
| Same, execution mode held constant | NumPy AoS 18.9 ms → NumPy SoA 6.4 ms (**3.0×**) — the price of interleaving alone |
| Access stride | 0.21 ns/element sequential → 4.11 ns at a 1,024-byte stride (**19×**); the curve bends exactly at the 128-byte cache line |
| Cache blocking, 4096² transpose | 105 ms → 27 ms at a 256×256 tile (**4.0×**) |

Three results worth noting:

- **Layout is invisible in pure Python.** The linked list, the object list, and
  the `__slots__` list all land within a few percent of each other, because
  interpreter dispatch costs ~215 ns per particle and swamps any memory effect.
- **A cache-perfect layout can be slower.** The `array.array` structure of
  arrays has ideal locality but is the slowest of all six (343 ms), since every
  element access boxes or unboxes a Python float.
- **Theory is directionally right, numerically off.** A bytes-moved model
  predicts a 7× AoS penalty; the measurement is 3.0×. Classical blocking
  analysis points at an L1-sized tile (64 per side); the optimum is 256.

The takeaway: contiguity is necessary but not sufficient. It only pays when
paired with an execution mode that can exploit it.

## Files

```
layouts.py             the six memory layouts
bench_util.py          timing and environment capture
benchmark_layouts.py   Experiment 1: layout
benchmark_stride.py    Experiment 2: stride
benchmark_blocking.py  Experiment 3: cache blocking
tests/test_layouts.py  all layouts must give the same answer
make_figures.py        figures from results/*.json
run_all.py             runs all of the above
```
