"""
AUC and bias/variance for binary Hypothesis A vs B discrimination.

Computes ROC AUC for each method's headline stat at distinguishing chirp
(Hypothesis A) from mixture (Hypothesis B) epochs. Also computes bias and
variance of recovered age trajectories vs ground truth.

Usage:
    from theta_alpha_shift.eval.discrimination import compute_auc, compute_bias_variance
    auc = compute_auc(results, 'bycycle')
"""

import numpy as np

from theta_alpha_shift.eval.confusion import results_to_array


def _roc_auc(scores_a, scores_b):
    """Compute AUC from two score arrays without sklearn.

    Uses the Mann-Whitney U statistic: AUC = U / (n_a * n_b).
    scores_a should be higher for method to discriminate correctly.

    Parameters
    ----------
    scores_a : array-like
        Headline stats for class A (chirp).
    scores_b : array-like
        Headline stats for class B (mixture).

    Returns
    -------
    auc : float
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    u = 0
    for ai in a:
        u += np.sum(ai > b) + 0.5 * np.sum(ai == b)

    auc = u / (len(a) * len(b))
    return max(auc, 1 - auc)


def compute_auc(results, method_name, ages=None):
    """Compute AUC for chirp vs mixture discrimination per age and overall.

    Parameters
    ----------
    results : list of EvalResult
    method_name : str
    ages : list of int, optional

    Returns
    -------
    auc_overall : float
        Overall AUC pooling all ages.
    auc_per_age : dict mapping age → AUC
    """
    stats = results_to_array(results, method_name)

    if ages is None:
        ages = sorted(set(
            r.age for r in results
            if r.method_name == method_name and r.regime in ("chirp", "mixture")
        ))

    all_chirp = []
    all_mixture = []
    auc_per_age = {}

    for age in ages:
        chirp_vals = stats.get(("chirp", age), [])
        mix_vals = stats.get(("mixture", age), [])
        all_chirp.extend(chirp_vals)
        all_mixture.extend(mix_vals)

        if chirp_vals and mix_vals:
            auc_per_age[age] = _roc_auc(chirp_vals, mix_vals)
        else:
            auc_per_age[age] = float("nan")

    auc_overall = _roc_auc(all_chirp, all_mixture)
    return auc_overall, auc_per_age


def compute_bias_variance(results, method_name, ages=None):
    """Compute bias and variance of headline stat trajectories vs ground truth.

    "Bias" = systematic offset from regime-specific expected trajectory.
    "Variance" = trial-to-trial variability at each age.

    Parameters
    ----------
    results : list of EvalResult
    method_name : str
    ages : list of int, optional

    Returns
    -------
    metrics : dict with keys per regime:
        {regime: {'mean_trajectory': list, 'std_trajectory': list,
                  'cv': float, 'trend_slope': float}}
    """
    stats = results_to_array(results, method_name)

    if ages is None:
        ages = sorted(set(r.age for r in results if r.method_name == method_name))

    regime_names = sorted(set(r.regime for r in results if r.method_name == method_name))
    metrics = {}

    for regime in regime_names:
        means = []
        stds = []
        for age in ages:
            vals = stats.get((regime, age), [])
            if vals:
                means.append(np.nanmean(vals))
                stds.append(np.nanstd(vals))
            else:
                means.append(float("nan"))
                stds.append(float("nan"))

        means_arr = np.array(means)
        stds_arr = np.array(stds)
        valid = np.isfinite(means_arr) & np.isfinite(stds_arr)

        if valid.sum() >= 2:
            ages_valid = np.array(ages)[valid]
            means_valid = means_arr[valid]
            slope = np.polyfit(ages_valid, means_valid, 1)[0]
            grand_mean = means_valid.mean()
            cv = stds_arr[valid].mean() / grand_mean if grand_mean != 0 else float("nan")
        else:
            slope = float("nan")
            cv = float("nan")

        metrics[regime] = {
            "mean_trajectory": means,
            "std_trajectory": stds,
            "ages": ages,
            "cv": float(cv),
            "trend_slope": float(slope),
        }

    return metrics


def summarize_discrimination(results, method_names=None):
    """Produce a summary table of discrimination metrics for all methods.

    Parameters
    ----------
    results : list of EvalResult
    method_names : list of str, optional

    Returns
    -------
    summary : list of dict with keys: method, auc_overall, auc_per_age,
        chirp_cv, mixture_cv, chirp_slope, mixture_slope
    """
    if method_names is None:
        method_names = sorted(set(r.method_name for r in results))

    summary = []
    for method in method_names:
        auc_overall, auc_per_age = compute_auc(results, method)
        bv = compute_bias_variance(results, method)

        summary.append({
            "method": method,
            "auc_overall": auc_overall,
            "auc_per_age": auc_per_age,
            "chirp_cv": bv.get("chirp", {}).get("cv", float("nan")),
            "mixture_cv": bv.get("mixture", {}).get("cv", float("nan")),
            "chirp_slope": bv.get("chirp", {}).get("trend_slope", float("nan")),
            "mixture_slope": bv.get("mixture", {}).get("trend_slope", float("nan")),
        })

    return summary
