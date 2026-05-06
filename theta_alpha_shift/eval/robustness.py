"""
Robustness testing: re-run top methods with artifact injection.

Tests whether method discrimination degrades when EOG, EMG, and line noise
are added to simulated epochs.

Usage:
    from theta_alpha_shift.eval.robustness import run_robustness_test
    report = run_robustness_test(method_names=['bycycle', 'hmm'])
"""

import numpy as np

from theta_alpha_shift.eval import EvalResult
from theta_alpha_shift.eval.confusion import (
    _get_method_runner,
    REGIME_FUNCS,
    REGIME_NAMES,
)
from theta_alpha_shift.eval.discrimination import compute_auc
from theta_alpha_shift.sim.artifacts import inject_artifacts


ARTIFACT_CONFIGS = {
    "clean": None,
    "eog": {
        "artifacts": {
            "eog": {"enabled": True, "amplitude": 100e-6, "rate": 0.3},
            "emg": {"enabled": False},
            "line_noise": {"enabled": False},
        }
    },
    "emg": {
        "artifacts": {
            "eog": {"enabled": False},
            "emg": {"enabled": True, "amplitude": 5e-6, "freq_range": [20, 45]},
            "line_noise": {"enabled": False},
        }
    },
    "line_noise": {
        "artifacts": {
            "eog": {"enabled": False},
            "emg": {"enabled": False},
            "line_noise": {"enabled": True, "freq": 60.0, "amplitude": 1e-6},
        }
    },
    "all_artifacts": {
        "artifacts": {
            "eog": {"enabled": True, "amplitude": 100e-6, "rate": 0.3},
            "emg": {"enabled": True, "amplitude": 5e-6, "freq_range": [20, 45]},
            "line_noise": {"enabled": True, "freq": 60.0, "amplitude": 1e-6},
        }
    },
}


def run_robustness_test(method_names, ages=None, n_trials=10, seed=42,
                        conditions=None, verbose=True):
    """Run methods under artifact conditions and compare AUC.

    Parameters
    ----------
    method_names : list of str
    ages : list of int, optional
    n_trials : int
    seed : int
    conditions : list of str, optional
        Artifact condition names. None = all.
    verbose : bool

    Returns
    -------
    report : dict mapping (method, condition) → {auc_overall, auc_per_age}
    all_results : dict mapping condition → list of EvalResult
    """
    if ages is None:
        ages = [5, 8, 11, 14, 17, 20]
    if conditions is None:
        conditions = list(ARTIFACT_CONFIGS.keys())

    runners = {m: _get_method_runner(m) for m in method_names}
    report = {}
    all_results = {}

    for condition in conditions:
        art_config = ARTIFACT_CONFIGS[condition]
        if verbose:
            print(f"{'=' * 60}")
            print(f"Condition: {condition}")

        cond_results = []

        for method_name in method_names:
            run_fn = runners[method_name]

            for regime_name in REGIME_NAMES:
                sim_fn = REGIME_FUNCS[regime_name]

                for age in ages:
                    for trial in range(n_trials):
                        trial_seed = seed + hash((regime_name, age, trial)) % (2**31)
                        rng = np.random.default_rng(trial_seed)

                        epoch = sim_fn(age=age, rng=rng)

                        if art_config is not None:
                            epoch = inject_artifacts(epoch, config=art_config, rng=rng)

                        try:
                            mr = run_fn(epoch)
                            stat = mr.headline_stat
                            stat_name = mr.headline_stat_name
                        except Exception:
                            stat = float("nan")
                            stat_name = "error"

                        cond_results.append(EvalResult(
                            method_name=method_name,
                            regime=regime_name,
                            age=age,
                            trial=trial,
                            headline_stat=stat,
                            headline_stat_name=stat_name,
                        ))

        all_results[condition] = cond_results

        for method_name in method_names:
            auc_overall, auc_per_age = compute_auc(cond_results, method_name)
            report[(method_name, condition)] = {
                "auc_overall": auc_overall,
                "auc_per_age": auc_per_age,
            }
            if verbose:
                print(f"  {method_name}: AUC={auc_overall:.3f}")

    return report, all_results


def summarize_robustness(report, method_names, conditions=None):
    """Format robustness results as a comparison table.

    Parameters
    ----------
    report : dict from run_robustness_test
    method_names : list of str
    conditions : list of str, optional

    Returns
    -------
    table : list of dict with keys: method, condition, auc_overall, auc_drop
    """
    if conditions is None:
        conditions = list(ARTIFACT_CONFIGS.keys())

    table = []
    for method in method_names:
        clean_auc = report.get((method, "clean"), {}).get("auc_overall", float("nan"))

        for cond in conditions:
            entry = report.get((method, cond), {})
            auc = entry.get("auc_overall", float("nan"))
            drop = clean_auc - auc if np.isfinite(clean_auc) and np.isfinite(auc) else float("nan")
            table.append({
                "method": method,
                "condition": cond,
                "auc_overall": auc,
                "auc_drop": drop,
            })

    return table
