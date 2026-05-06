"""
Per-method confusion matrices across 5 regimes × 6 ages.

Runs each method on simulated epochs from each regime, collects headline
statistics, and generates confusion matrices showing whether each method
can identify the generating regime.

Usage:
    from theta_alpha_shift.eval.confusion import run_evaluation_grid, compute_confusion
    results = run_evaluation_grid(n_trials=10)
    cm = compute_confusion(results, 'specparam_baseline')
"""

import numpy as np

from theta_alpha_shift.eval import EvalResult
from theta_alpha_shift.sim.regimes import (
    simulate_chirp,
    simulate_mixture,
    simulate_drift,
    simulate_cooccur,
    simulate_narrowing,
)

REGIME_FUNCS = {
    "chirp": simulate_chirp,
    "mixture": simulate_mixture,
    "drift": simulate_drift,
    "cooccur": simulate_cooccur,
    "narrowing": simulate_narrowing,
}

REGIME_NAMES = list(REGIME_FUNCS.keys())

METHOD_RUNNERS = {
    "specparam_baseline": ("theta_alpha_shift.methods.specparam_baseline", "run_specparam"),
    "bycycle": ("theta_alpha_shift.methods.bycycle_wrap", "run_bycycle"),
    "itemd_if": ("theta_alpha_shift.methods.itemd_if", "run_itemd"),
    "cdl": ("theta_alpha_shift.methods.cdl_wrap", "run_cdl"),
    "threshold_bursts": ("theta_alpha_shift.methods.threshold_bursts", "run_threshold"),
    "hmm": ("theta_alpha_shift.methods.hmm_wrap", "run_hmm"),
}


def _get_method_runner(method_name):
    """Import and return a method's run function."""
    import importlib
    module_path, func_name = METHOD_RUNNERS[method_name]
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def run_evaluation_grid(method_names=None, ages=None, n_trials=10, seed=42,
                        verbose=True, method_kwargs=None):
    """Run methods across all regimes and ages.

    Parameters
    ----------
    method_names : list of str, optional
        Methods to evaluate. None = all.
    ages : list of int, optional
        Ages to test. None = [5, 8, 11, 14, 17, 20].
    n_trials : int
        Number of independent trials per regime × age.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.
    method_kwargs : dict, optional
        Per-method keyword arguments, e.g. {"specparam_baseline": {"psd_method": "meeglet"}}.

    Returns
    -------
    results : list of EvalResult
    """
    if method_names is None:
        method_names = list(METHOD_RUNNERS.keys())
    if ages is None:
        ages = [5, 8, 11, 14, 17, 20]
    if method_kwargs is None:
        method_kwargs = {}

    results = []
    total = len(method_names) * len(REGIME_NAMES) * len(ages) * n_trials

    runners = {m: _get_method_runner(m) for m in method_names}
    count = 0

    for method_name in method_names:
        run_fn = runners[method_name]
        kwargs = method_kwargs.get(method_name, {})
        if verbose:
            print(f"{'=' * 60}")
            print(f"Method: {method_name}")
            if kwargs:
                print(f"  kwargs: {kwargs}")

        for regime_name in REGIME_NAMES:
            sim_fn = REGIME_FUNCS[regime_name]

            for age in ages:
                for trial in range(n_trials):
                    trial_seed = seed + hash((regime_name, age, trial)) % (2**31)
                    rng = np.random.default_rng(trial_seed)

                    epoch = sim_fn(age=age, rng=rng)

                    try:
                        method_result = run_fn(epoch, **kwargs)
                        stat = method_result.headline_stat
                        stat_name = method_result.headline_stat_name
                        meta = {
                            "n_bursts_detected": len(method_result.detected_bursts),
                        }
                        meta.update({
                            k: v for k, v in method_result.metadata.items()
                            if isinstance(v, (int, float, str, bool))
                        })
                    except Exception as e:
                        stat = float("nan")
                        stat_name = "error"
                        meta = {"error": str(e)}

                    results.append(EvalResult(
                        method_name=method_name,
                        regime=regime_name,
                        age=age,
                        trial=trial,
                        headline_stat=stat,
                        headline_stat_name=stat_name,
                        metadata=meta,
                    ))

                    count += 1
                    if verbose and count % 50 == 0:
                        print(f"  {count}/{total} evaluations complete")

    if verbose:
        print(f"{'=' * 60}")
        print(f"Evaluation complete: {len(results)} results")

    return results


def results_to_array(results, method_name):
    """Extract headline stats into a structured array.

    Parameters
    ----------
    results : list of EvalResult
    method_name : str

    Returns
    -------
    stats : dict mapping (regime, age) → list of headline_stat values
    """
    stats = {}
    for r in results:
        if r.method_name != method_name:
            continue
        key = (r.regime, r.age)
        stats.setdefault(key, []).append(r.headline_stat)
    return stats


def compute_confusion(results, method_name, ages=None):
    """Compute confusion-style summary: mean headline stat per regime × age.

    Parameters
    ----------
    results : list of EvalResult
    method_name : str
    ages : list of int, optional

    Returns
    -------
    matrix : ndarray, shape (n_regimes, n_ages)
        Mean headline stat.
    std_matrix : ndarray, shape (n_regimes, n_ages)
        Std of headline stat.
    regime_labels : list of str
    age_labels : list of int
    """
    if ages is None:
        ages = sorted(set(r.age for r in results if r.method_name == method_name))

    stats = results_to_array(results, method_name)

    matrix = np.full((len(REGIME_NAMES), len(ages)), np.nan)
    std_matrix = np.full((len(REGIME_NAMES), len(ages)), np.nan)

    for i, regime in enumerate(REGIME_NAMES):
        for j, age in enumerate(ages):
            vals = stats.get((regime, age), [])
            if vals:
                matrix[i, j] = np.nanmean(vals)
                std_matrix[i, j] = np.nanstd(vals)

    return matrix, std_matrix, REGIME_NAMES, ages
