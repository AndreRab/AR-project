"""Symulacja modelu XY metodą Metropolisa na periodycznej siatce kwadratowej."""

import csv
from dataclasses import dataclass
from math import cos, exp, hypot, pi, sin, sqrt
from pathlib import Path

import numpy as np
from numba import config, njit, prange, set_num_threads

SIZE = 32
TEMPERATURES = (0.5, 0.9, 1.5)
THREADS = min(4, config.NUMBA_NUM_THREADS)
TOTAL_SWEEPS = 20_000
BURN_IN = 2000
STRIDE = 5
CHAINS = 8
SEED = 2026
MAX_ROTATION = pi / 2
ORDERED_START = False
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_CSV = RESULTS_DIR / "magnetization.csv"
OUTPUT_PLOT = RESULTS_DIR / "spin_configurations.png"
TEMPERATURE_PLOT_TEMPLATE = "spin_configurations_T_{temperature}.png"

TAU = 2 * pi


def _half_sweep(angles, proposal, uniform, beta, max_rotation, color):
    size = angles.shape[0]
    for row in prange(size):
        above = (row - 1) % size
        below = (row + 1) % size
        for col in range((color + row) % 2, size, 2):
            left = (col - 1) % size
            right = (col + 1) % size
            old = angles[row, col]
            new = old + max_rotation * (2 * proposal[row, col] - 1)
            # Każde z czterech wiązań ma energię -cos(różnica kątów).
            neighbors = (
                angles[above, col], angles[below, col],
                angles[row, left], angles[row, right],
            )
            delta = 0.0
            for angle in neighbors:
                delta += cos(old - angle) - cos(new - angle)
            if delta <= 0 or uniform[row, col] < exp(-beta * delta):
                angles[row, col] = new % TAU


# Oba warianty wykonują identyczną pracę. Powrót z kernela stanowi
# barierę między kolorami, także gdy wykonuje go wiele wątków.
serial_half_sweep = njit(_half_sweep)
parallel_half_sweep = njit(_half_sweep, parallel=True)


@njit
def observables(angles):
    """Zwróć moduł magnetyzacji oraz energię, oba w przeliczeniu na spin."""
    size = angles.shape[0]
    mx = my = energy = 0.0
    for row in range(size):
        for col in range(size):
            angle = angles[row, col]
            mx += cos(angle)
            my += sin(angle)
            # Prawe i dolne wiązanie z każdego węzła liczymy po jednym razie.
            energy -= cos(angle - angles[row, (col + 1) % size])
            energy -= cos(angle - angles[(row + 1) % size, col])
    count = size * size
    return hypot(mx, my) / count, energy / count


def check_size(size):
    if size < 4 or size % 2:
        raise ValueError("Rozmiar L musi być parzysty i wynosić co najmniej 4.")


def check_threads(count):
    if not 1 <= count <= config.NUMBA_NUM_THREADS:
        raise ValueError(f"Liczba wątków musi należeć do przedziału [1, {config.NUMBA_NUM_THREADS}].")


class Simulation:
    def __init__(self, size, temperature, max_rotation, rng, *, parallel, ordered=False):
        check_size(size)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperatura musi być dodatnia i skończona.")
        if not 0 < max_rotation <= pi:
            raise ValueError("Maksymalny obrót musi należeć do przedziału (0, π].")
        self.beta = 1.0 / temperature
        if not np.isfinite(self.beta):
            raise ValueError("Temperatura jest zbyt mała dla arytmetyki float64.")
        self.max_rotation = max_rotation
        self.rng = rng
        shape = (size, size)
        self.angles = np.zeros(shape) if ordered else rng.uniform(0, TAU, size=shape)
        self.proposal = np.empty(shape)
        self.uniform = np.empty(shape)
        self.half_sweep = parallel_half_sweep if parallel else serial_half_sweep

    def sweep(self):
        """Wykonaj po jednej próbie aktualizacji każdego spinu."""
        self.rng.random(out=self.proposal)
        self.rng.random(out=self.uniform)
        # Każdy kolor czyta inną połowę liczb losowych z tych samych buforów.
        for color in (0, 1):
            self.half_sweep(
                self.angles, self.proposal, self.uniform, self.beta, self.max_rotation, color
            )


@dataclass(frozen=True)
class Estimate:
    mean: float
    stderr: float
    chain_means: tuple[float, ...]


def summarize(chain_means):
    """Oblicz błąd standardowy z niezależnych średnich łańcuchów."""
    return Estimate(
        float(np.mean(chain_means)),
        float(np.std(chain_means, ddof=1) / sqrt(len(chain_means))),
        tuple(chain_means),
    )


def measure(
    size, temperature, *, burn_in, samples, stride, chains, seed, max_rotation,
    parallel, ordered,
):
    if burn_in < 0 or samples < 1 or stride < 1 or chains < 2:
        raise ValueError("Wymagane: burn-in ≥ 0, samples ≥ 1, stride ≥ 1, chains ≥ 2.")

    magnetization_means = []
    energy_means = []
    for chain_index, chain_seed in enumerate(np.random.SeedSequence(seed).spawn(chains)):
        simulation = Simulation(
            size, temperature, max_rotation, np.random.default_rng(chain_seed),
            parallel=parallel, ordered=ordered,
        )
        if chain_index == 0:
            initial_angles = simulation.angles.copy()
        for _ in range(burn_in):
            simulation.sweep()

        magnetization_sum = energy_sum = 0.0
        for _ in range(samples):
            for _ in range(stride):
                simulation.sweep()
            magnetization, energy = observables(simulation.angles)
            magnetization_sum += magnetization
            energy_sum += energy
        magnetization_means.append(magnetization_sum / samples)
        energy_means.append(energy_sum / samples)
        if chain_index == 0:
            final_angles = simulation.angles.copy()
    return summarize(magnetization_means), summarize(energy_means), (initial_angles, final_angles)


def plot_spin_configurations(snapshots, output_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 3.5 * len(snapshots)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(snapshots), 2, squeeze=False)
    for (temperature, configurations), row in zip(snapshots, axes):
        for axis, angles, stage in zip(row, configurations, ("Początek", "Koniec")):
            size = angles.shape[0]
            step = (size + 31) // 32
            positions = np.arange(0, size, step)
            displayed_angles = angles[::step, ::step]
            axis.quiver(
                positions, positions, np.cos(displayed_angles), np.sin(displayed_angles),
                angles="xy", scale_units="xy", scale=1 / (0.8 * step),
                pivot="middle", color="#15536c",
            )
            magnetization, _ = observables(angles)
            axis.set(
                title=f"{stage}, T = {temperature:g}, |m| = {magnetization:.3f}",
                aspect="equal", xlim=(-step, size), ylim=(-step, size),
                xticks=[], yticks=[],
            )

    figure.suptitle(
        f"Kierunki spinów na siatce {SIZE} × {SIZE}\n"
        f"Pierwszy łańcuch przed aktualizacją i po {TOTAL_SWEEPS} przebiegach"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)


def main():
    check_size(SIZE)
    check_threads(THREADS)
    if not TEMPERATURES:
        raise ValueError("Podaj co najmniej jedną temperaturę.")
    if STRIDE < 1 or not 0 <= BURN_IN < TOTAL_SWEEPS:
        raise ValueError("Wymagane: STRIDE ≥ 1 oraz 0 ≤ BURN_IN < TOTAL_SWEEPS.")
    samples, remainder = divmod(TOTAL_SWEEPS - BURN_IN, STRIDE)
    if remainder:
        raise ValueError("Liczba przebiegów po rozgrzewce musi być podzielna przez STRIDE.")
    set_num_threads(THREADS)

    rows = []
    snapshots = []
    for temperature in TEMPERATURES:
        magnetization, energy, configurations = measure(
            SIZE, temperature, burn_in=BURN_IN, samples=samples, stride=STRIDE,
            chains=CHAINS, seed=SEED, max_rotation=MAX_ROTATION,
            parallel=THREADS > 1, ordered=ORDERED_START,
        )
        snapshots.append((temperature, configurations))
        rows.append([
            SIZE, temperature, THREADS, "ordered" if ORDERED_START else "random",
            SEED, MAX_ROTATION, BURN_IN, samples, STRIDE, CHAINS,
            magnetization.mean, magnetization.stderr, energy.mean, energy.stderr,
        ])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "size", "temperature", "threads", "initial_state", "seed", "max_rotation",
            "burn_in", "samples_per_chain", "stride", "chains", "magnetization",
            "magnetization_stderr", "energy_per_spin", "energy_stderr",
        ])
        writer.writerows(rows)
    print(f"Zapisano {OUTPUT_CSV}")
    plot_spin_configurations(snapshots, OUTPUT_PLOT)
    print(f"Zapisano {OUTPUT_PLOT}")
    for temperature, configurations in snapshots:
        output_path = OUTPUT_PLOT.with_name(
            TEMPERATURE_PLOT_TEMPLATE.format(temperature=temperature)
        )
        plot_spin_configurations([(temperature, configurations)], output_path)
        print(f"Zapisano {output_path}")


if __name__ == "__main__":
    main()
