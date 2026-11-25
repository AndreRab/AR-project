"""Porównanie czasu pełnych przebiegów Metropolisa na CPU."""

import csv
from math import pi
from statistics import median
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numba import config, set_num_threads

from model import RESULTS_DIR, Simulation, check_size, check_threads

SIZE = 256
TEMPERATURE = 1.0
THREAD_COUNTS = tuple(count for count in (1, 2, 4, 8) if count <= config.NUMBA_NUM_THREADS)
SWEEPS = 100
REPEATS = 5
SEED = 2026
MAX_ROTATION = pi / 2
OUTPUT_CSV = RESULTS_DIR / "benchmark.csv"
OUTPUT_PLOT = RESULTS_DIR / "scaling.png"


def time_sweeps(*, parallel):
    timings = []
    for _ in range(REPEATS):
        simulation = Simulation(
            SIZE, TEMPERATURE, MAX_ROTATION, np.random.default_rng(SEED),
            parallel=parallel,
        )
        # Inicjalizacja planszy i pierwsza kompilacja JIT są poza pomiarem.
        simulation.sweep()
        start = perf_counter()
        for _ in range(SWEEPS):
            simulation.sweep()
        timings.append(perf_counter() - start)
    return median(timings)


def plot_scaling(results):
    figure = Figure(figsize=(12, 4.1), layout="constrained")
    FigureCanvasAgg(figure)
    speedup_ax, efficiency_ax, karp_flatt_ax = figure.subplots(1, 3)
    threads, _, speedups, efficiencies, karp_flatt = zip(*results)

    speedup_ax.plot(threads, threads, "--", color="#7a8490", label="Idealne")
    speedup_ax.plot(threads, speedups, "o-", color="#007580", label="Pomiar")
    speedup_ax.legend(frameon=False)
    efficiency_ax.axhline(1, linestyle="--", color="#7a8490")
    efficiency_ax.plot(threads, efficiencies, "o-", color="#007580")
    karp_flatt_ax.axhline(0, linestyle="--", color="#7a8490")
    karp_threads = [count for count in threads if count > 1]
    if karp_threads:
        karp_flatt_ax.plot(
            karp_threads, [value for value in karp_flatt if value is not None],
            "o-", color="#007580",
        )
    else:
        karp_flatt_ax.text(
            0.5, 0.5, "Wymaga co najmniej 2 wątków",
            transform=karp_flatt_ax.transAxes, ha="center", va="center",
        )

    for axis, title, ylabel in (
        (speedup_ax, "Przyspieszenie", "S(p)"),
        (efficiency_ax, "Efektywność", "E(p)"),
        (karp_flatt_ax, "Metryka Karpa–Flatta", "e(p)"),
    ):
        axis.set(title=title, xlabel="Liczba wątków", ylabel=ylabel, xticks=threads)
        axis.grid(axis="y", alpha=0.25)

    figure.suptitle(f"Model XY na CPU · {SIZE} × {SIZE} · T = {TEMPERATURE:g}")
    figure.savefig(OUTPUT_PLOT, dpi=160)


def main():
    check_size(SIZE)
    if not THREAD_COUNTS or SWEEPS < 1 or REPEATS < 1:
        raise ValueError("Podaj liczby wątków oraz dodatnie SWEEPS i REPEATS.")
    thread_counts = sorted(set(THREAD_COUNTS))
    for threads in thread_counts:
        check_threads(threads)

    serial_time = time_sweeps(parallel=False)
    results = []
    for threads in thread_counts:
        set_num_threads(threads)
        seconds = time_sweeps(parallel=True)
        speedup = serial_time / seconds
        efficiency = speedup / threads
        karp_flatt = None
        if threads > 1:
            karp_flatt = (1 / speedup - 1 / threads) / (1 - 1 / threads)
        results.append((threads, seconds, speedup, efficiency, karp_flatt))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "size", "temperature", "backend", "threads", "sweeps", "repeats",
            "median_seconds", "speedup_vs_serial", "efficiency", "karp_flatt",
        ])
        writer.writerow([SIZE, TEMPERATURE, "serial", 1, SWEEPS, REPEATS, serial_time, 1, 1, None])
        for threads, seconds, speedup, efficiency, karp_flatt in results:
            writer.writerow([
                SIZE, TEMPERATURE, "parallel", threads, SWEEPS, REPEATS,
                seconds, speedup, efficiency, karp_flatt,
            ])

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plot_scaling(results)
    print(f"Zapisano {OUTPUT_CSV}")
    print(f"Zapisano {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
