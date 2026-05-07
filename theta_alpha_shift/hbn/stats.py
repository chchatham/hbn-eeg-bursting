"""
Full-scale HBN burst analysis statistics.

Computes correlations, effect sizes, split-half reliability, sex-stratified
analyses, and robust statistics on fullscale_results.csv.

Output: results/hbn/fullscale_stats.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


AGE_BIN_LABELS = ["5-6", "7-8", "9-10", "11-12", "13-15", "16-21"]
YOUNG_BINS = ["5-6", "7-8"]
OLD_BINS = ["13-15", "16-21"]

SEED = 42
N_BOOTSTRAP = 10000


def load_results(results_path):
    """Load full-scale results, drop rows with NaN in key columns."""
    df = pd.read_csv(results_path)
    key_cols = ["age", "bycycle_mean_period_slope", "hmm_mean_n_osc_states"]
    df = df.dropna(subset=key_cols)
    df = df[df["age_bin"].notna()].copy()
    return df


def _bootstrap_ci(data, stat_func, n_boot=N_BOOTSTRAP, seed=SEED, ci=0.95):
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = stat_func(sample)
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_stats, alpha * 100)), float(np.percentile(boot_stats, (1 - alpha) * 100))


def _bootstrap_corr_ci(x, y, method="spearman", n_boot=N_BOOTSTRAP, seed=SEED):
    """Bootstrap CI for a correlation coefficient."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = np.empty(n_boot)
    corr_func = sp_stats.spearmanr if method == "spearman" else sp_stats.pearsonr
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        r, _ = corr_func(x[idx], y[idx])
        boot_r[i] = r
    return float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))


def _cohens_d(group1, group2):
    """Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((m1 - m2) / pooled_sd)


def _cliffs_delta(group1, group2):
    """Cliff's delta (nonparametric effect size)."""
    n1, n2 = len(group1), len(group2)
    count = 0
    for x in group1:
        for y in group2:
            if x > y:
                count += 1
            elif x < y:
                count -= 1
    return float(count / (n1 * n2))


def compute_correlations(df):
    """Compute age correlations with bootstrap CIs."""
    age = df["age"].values
    bycycle = df["bycycle_mean_period_slope"].values
    hmm = df["hmm_mean_n_osc_states"].values
    paf = df["specparam_mean_peak_freq"].values

    rho_byc, p_byc = sp_stats.spearmanr(age, bycycle)
    rho_hmm, p_hmm = sp_stats.spearmanr(age, hmm)
    r_paf, p_paf = sp_stats.pearsonr(age, paf[~np.isnan(paf)] if np.any(np.isnan(paf)) else paf)

    paf_valid = ~np.isnan(paf)

    return {
        "bycycle_vs_age": {
            "spearman_rho": float(rho_byc),
            "p_value": float(p_byc),
            "ci_95": _bootstrap_corr_ci(age, bycycle, "spearman"),
            "n": len(age),
        },
        "hmm_vs_age": {
            "spearman_rho": float(rho_hmm),
            "p_value": float(p_hmm),
            "ci_95": _bootstrap_corr_ci(age, hmm, "spearman"),
            "n": len(age),
        },
        "paf_vs_age": {
            "pearson_r": float(sp_stats.pearsonr(age[paf_valid], paf[paf_valid])[0]),
            "p_value": float(sp_stats.pearsonr(age[paf_valid], paf[paf_valid])[1]),
            "ci_95": _bootstrap_corr_ci(age[paf_valid], paf[paf_valid], "pearson"),
            "n": int(paf_valid.sum()),
        },
    }


def compute_effect_sizes(df):
    """Compute Cohen's d for young vs old, and per-adjacent-bin."""
    young = df[df["age_bin"].isin(YOUNG_BINS)]
    old = df[df["age_bin"].isin(OLD_BINS)]

    result = {}
    for metric, col in [("bycycle", "bycycle_mean_period_slope"),
                        ("hmm", "hmm_mean_n_osc_states")]:
        y_vals = young[col].values
        o_vals = old[col].values
        d = _cohens_d(y_vals, o_vals)

        d_boot = []
        rng = np.random.default_rng(SEED)
        for _ in range(N_BOOTSTRAP):
            y_samp = rng.choice(y_vals, size=len(y_vals), replace=True)
            o_samp = rng.choice(o_vals, size=len(o_vals), replace=True)
            d_boot.append(_cohens_d(y_samp, o_samp))
        d_ci = (float(np.percentile(d_boot, 2.5)), float(np.percentile(d_boot, 97.5)))

        adjacent_d = {}
        for i in range(len(AGE_BIN_LABELS) - 1):
            bin1 = AGE_BIN_LABELS[i]
            bin2 = AGE_BIN_LABELS[i + 1]
            g1 = df[df["age_bin"] == bin1][col].values
            g2 = df[df["age_bin"] == bin2][col].values
            if len(g1) >= 5 and len(g2) >= 5:
                adjacent_d[f"{bin1}_vs_{bin2}"] = _cohens_d(g1, g2)

        result[metric] = {
            "cohens_d_young_vs_old": d,
            "ci_95": d_ci,
            "n_young": len(y_vals),
            "n_old": len(o_vals),
            "adjacent_bin_d": adjacent_d,
        }

    return result


def compute_by_age_bin(df):
    """Per-bin descriptive statistics."""
    result = {}
    for metric, col in [("bycycle", "bycycle_mean_period_slope"),
                        ("hmm", "hmm_mean_n_osc_states"),
                        ("specparam_paf", "specparam_mean_peak_freq")]:
        bins_data = {}
        for ab in AGE_BIN_LABELS:
            vals = df[df["age_bin"] == ab][col].dropna().values
            if len(vals) == 0:
                continue
            ci_lo, ci_hi = _bootstrap_ci(vals, np.mean)
            bins_data[ab] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals, ddof=1)),
                "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)),
                "n": len(vals),
                "ci_95_mean": [ci_lo, ci_hi],
            }
        result[metric] = bins_data
    return result


def split_half_reliability(df, n_splits=1000, seed=SEED):
    """Split-half reliability of age-bin trajectories."""
    rng = np.random.default_rng(seed)
    result = {}

    for metric, col in [("bycycle", "bycycle_mean_period_slope"),
                        ("hmm", "hmm_mean_n_osc_states")]:
        correlations = []
        for _ in range(n_splits):
            half1_means = []
            half2_means = []
            for ab in AGE_BIN_LABELS:
                bin_vals = df[df["age_bin"] == ab][col].values
                if len(bin_vals) < 4:
                    continue
                idx = rng.permutation(len(bin_vals))
                mid = len(bin_vals) // 2
                half1_means.append(np.mean(bin_vals[idx[:mid]]))
                half2_means.append(np.mean(bin_vals[idx[mid:]]))
            if len(half1_means) >= 3:
                r, _ = sp_stats.pearsonr(half1_means, half2_means)
                correlations.append(r)

        correlations = np.array(correlations)
        result[metric] = {
            "mean_r": float(np.mean(correlations)),
            "ci_95": [float(np.percentile(correlations, 2.5)),
                      float(np.percentile(correlations, 97.5))],
            "n_splits": n_splits,
        }

    return result


def sex_stratified(df):
    """Repeat key analyses stratified by sex."""
    result = {}

    for sex_label in ["M", "F"]:
        sub = df[df["sex"] == sex_label]
        if len(sub) < 30:
            continue

        age = sub["age"].values
        bycycle = sub["bycycle_mean_period_slope"].values
        hmm = sub["hmm_mean_n_osc_states"].values

        rho_byc, p_byc = sp_stats.spearmanr(age, bycycle)
        rho_hmm, p_hmm = sp_stats.spearmanr(age, hmm)

        young = sub[sub["age_bin"].isin(YOUNG_BINS)]
        old = sub[sub["age_bin"].isin(OLD_BINS)]

        result[sex_label] = {
            "n": len(sub),
            "bycycle_rho": float(rho_byc),
            "bycycle_p": float(p_byc),
            "hmm_rho": float(rho_hmm),
            "hmm_p": float(p_hmm),
            "bycycle_d": _cohens_d(
                young["bycycle_mean_period_slope"].values,
                old["bycycle_mean_period_slope"].values
            ) if len(young) >= 10 and len(old) >= 10 else None,
            "hmm_d": _cohens_d(
                young["hmm_mean_n_osc_states"].values,
                old["hmm_mean_n_osc_states"].values
            ) if len(young) >= 10 and len(old) >= 10 else None,
        }

    # Test for sex × age interaction via slope difference
    if "M" in result and "F" in result:
        result["interaction_bycycle_rho_diff"] = abs(
            result["M"]["bycycle_rho"] - result["F"]["bycycle_rho"])
        result["interaction_hmm_rho_diff"] = abs(
            result["M"]["hmm_rho"] - result["F"]["hmm_rho"])

    return result


def robust_statistics(df):
    """Robust measures: trimmed means, Cliff's delta, outlier counts."""
    result = {}

    for metric, col in [("bycycle", "bycycle_mean_period_slope"),
                        ("hmm", "hmm_mean_n_osc_states")]:
        bins_data = {}
        for ab in AGE_BIN_LABELS:
            vals = df[df["age_bin"] == ab][col].dropna().values
            if len(vals) < 5:
                continue
            trimmed = sp_stats.trim_mean(vals, proportiontocut=0.1)
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            n_outliers = int(np.sum((vals < q1 - 3 * iqr) | (vals > q3 + 3 * iqr)))
            bins_data[ab] = {
                "trimmed_mean_10pct": float(trimmed),
                "n_outliers_3iqr": n_outliers,
            }

        young = df[df["age_bin"].isin(YOUNG_BINS)][col].values
        old = df[df["age_bin"].isin(OLD_BINS)][col].values
        cliff_d = _cliffs_delta(young, old) if len(young) >= 10 and len(old) >= 10 else None

        result[metric] = {
            "by_bin": bins_data,
            "cliffs_delta_young_vs_old": cliff_d,
        }

    return result


def run_all_stats(results_path=None, output_dir=None):
    """Run all statistical analyses and save summary JSON."""
    if results_path is None:
        results_path = Path(__file__).resolve().parents[2] / "results" / "hbn" / "fullscale_results.csv"
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "results" / "hbn"
    output_dir = Path(output_dir)

    print("=" * 60)
    print("FULL-SCALE STATISTICAL ANALYSIS")
    print("=" * 60)

    df = load_results(results_path)
    print(f"Loaded {len(df)} subjects")

    print("Computing correlations...")
    correlations = compute_correlations(df)

    print("Computing effect sizes...")
    effects = compute_effect_sizes(df)

    print("Computing per-bin statistics...")
    by_bin = compute_by_age_bin(df)

    print("Computing split-half reliability...")
    reliability = split_half_reliability(df)

    print("Computing sex-stratified analyses...")
    sex = sex_stratified(df)

    print("Computing robust statistics...")
    robust = robust_statistics(df)

    summary = {
        "n_subjects": len(df),
        "age_range": [float(df["age"].min()), float(df["age"].max())],
        "sex_distribution": df["sex"].value_counts().to_dict(),
        "correlations": correlations,
        "effect_sizes": effects,
        "by_age_bin": by_bin,
        "split_half_reliability": reliability,
        "sex_stratified": sex,
        "robust": robust,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "fullscale_stats.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"  Bycycle vs age: rho={correlations['bycycle_vs_age']['spearman_rho']:.3f} "
          f"CI={correlations['bycycle_vs_age']['ci_95']}")
    print(f"  HMM vs age:     rho={correlations['hmm_vs_age']['spearman_rho']:.3f} "
          f"CI={correlations['hmm_vs_age']['ci_95']}")
    print(f"  Bycycle d (young vs old): {effects['bycycle']['cohens_d_young_vs_old']:.3f} "
          f"CI={effects['bycycle']['ci_95']}")
    print(f"  HMM d (young vs old):     {effects['hmm']['cohens_d_young_vs_old']:.3f} "
          f"CI={effects['hmm']['ci_95']}")
    print(f"  Split-half (bycycle): r={reliability['bycycle']['mean_r']:.3f} "
          f"CI={reliability['bycycle']['ci_95']}")
    print(f"  Split-half (HMM):     r={reliability['hmm']['mean_r']:.3f} "
          f"CI={reliability['hmm']['ci_95']}")

    return summary


if __name__ == "__main__":
    run_all_stats()
