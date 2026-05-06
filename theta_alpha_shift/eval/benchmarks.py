"""
Computational cost profiling for each method.

Measures wall-clock time and memory usage per method, then projects
total cost for full HBN dataset (~3000 subjects × ~100 epochs each).

Usage:
    from theta_alpha_shift.eval.benchmarks import benchmark_methods
    report = benchmark_methods()
"""

import time
import tracemalloc

import numpy as np

from theta_alpha_shift.sim.regimes import simulate_chirp
from theta_alpha_shift.eval.confusion import _get_method_runner, METHOD_RUNNERS


def benchmark_single(method_name, epoch, n_reps=3):
    """Benchmark a single method on one epoch.

    Parameters
    ----------
    method_name : str
    epoch : SimulatedEpoch
    n_reps : int
        Number of repetitions for timing.

    Returns
    -------
    timing : dict with wall_seconds_mean, wall_seconds_std, peak_memory_mb
    """
    run_fn = _get_method_runner(method_name)

    tracemalloc.start()
    run_fn(epoch)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_mem / (1024 * 1024)

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        run_fn(epoch)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return {
        "wall_seconds_mean": float(np.mean(times)),
        "wall_seconds_std": float(np.std(times)),
        "peak_memory_mb": float(peak_mb),
    }


def benchmark_methods(method_names=None, age=11, n_reps=3, seed=42):
    """Benchmark all methods and project HBN cost.

    Parameters
    ----------
    method_names : list of str, optional
        Methods to benchmark. None = all.
    age : int
        Age for simulation.
    n_reps : int
        Timing repetitions.
    seed : int

    Returns
    -------
    report : list of dict with keys: method, wall_seconds_mean,
        wall_seconds_std, peak_memory_mb, projected_hbn_hours
    """
    if method_names is None:
        method_names = list(METHOD_RUNNERS.keys())

    epoch = simulate_chirp(age=age, rng=np.random.default_rng(seed))

    hbn_epochs = 3000 * 100

    report = []
    for method_name in method_names:
        print(f"Benchmarking {method_name}...")
        try:
            timing = benchmark_single(method_name, epoch, n_reps)
            projected_seconds = timing["wall_seconds_mean"] * hbn_epochs
            projected_hours = projected_seconds / 3600
        except Exception as e:
            timing = {
                "wall_seconds_mean": float("nan"),
                "wall_seconds_std": float("nan"),
                "peak_memory_mb": float("nan"),
            }
            projected_hours = float("nan")
            print(f"  ERROR: {e}")

        entry = {
            "method": method_name,
            **timing,
            "projected_hbn_hours": float(projected_hours),
        }
        report.append(entry)
        print(f"  {timing['wall_seconds_mean']:.3f}s/epoch, "
              f"{timing['peak_memory_mb']:.1f} MB, "
              f"projected: {projected_hours:.1f}h for HBN")

    return report
