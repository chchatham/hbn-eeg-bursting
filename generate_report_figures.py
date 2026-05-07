"""Generate publication-quality figures for the HBN empirical results report."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results/hbn")
OUTPUT_DIR = Path("theta_alpha_shift/app/static/figures")

AGE_BIN_ORDER = ['5-6', '7-8', '9-10', '11-12', '13-15', '16-21']
AGE_BIN_CENTERS = [5.5, 7.5, 9.5, 11.5, 14, 18.5]

PRIMARY = '#2c5282'
SECONDARY = '#c53030'
ACCENT = '#38a169'
LIGHT_BLUE = '#bee3f8'
LIGHT_RED = '#fed7d7'
GRAY = '#a0aec0'


def load_data():
    df = pd.read_csv(RESULTS_DIR / "fullscale_results.csv")
    with open(RESULTS_DIR / "fullscale_stats.json") as f:
        stats = json.load(f)
    return df, stats


def fig_empirical_trajectories(df, stats):
    """Figure 5: Three-panel developmental trajectory (bycycle, HMM, PAF)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')

    for ax_idx, (measure, label, ylabel, color) in enumerate([
        ('bycycle_mean_period_slope', 'bycycle', 'Mean Period Slope', SECONDARY),
        ('hmm_mean_n_osc_states', 'hmm', 'N Oscillatory States', PRIMARY),
        ('specparam_mean_peak_freq', 'specparam_paf', 'Peak Alpha Frequency (Hz)', ACCENT),
    ]):
        ax = axes[ax_idx]
        bin_stats = stats['by_age_bin'][label]

        means = [bin_stats[b]['mean'] for b in AGE_BIN_ORDER]
        medians = [bin_stats[b]['median'] for b in AGE_BIN_ORDER]
        ci_lo = [bin_stats[b]['ci_95_mean'][0] for b in AGE_BIN_ORDER]
        ci_hi = [bin_stats[b]['ci_95_mean'][1] for b in AGE_BIN_ORDER]
        ns = [bin_stats[b]['n'] for b in AGE_BIN_ORDER]

        for i, b in enumerate(AGE_BIN_ORDER):
            bin_data = df[df['age_bin'] == b][measure].dropna()
            jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(bin_data))
            ax.scatter(AGE_BIN_CENTERS[i] + jitter, bin_data, s=3, alpha=0.15,
                       color=GRAY, zorder=1, rasterized=True)

        yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
        yerr_hi = [hi - m for m, hi in zip(means, ci_hi)]
        ax.errorbar(AGE_BIN_CENTERS, means, yerr=[yerr_lo, yerr_hi],
                    fmt='o-', color=color, markersize=8, linewidth=2.5,
                    capsize=5, capthick=2, zorder=3, label='Mean (95% CI)')
        ax.plot(AGE_BIN_CENTERS, medians, 's--', color=color, markersize=6,
                linewidth=1.5, alpha=0.6, zorder=2, label='Median')

        data_min = min(ci_lo) - (max(ci_hi) - min(ci_lo)) * 0.15
        data_max = max(ci_hi) + (max(ci_hi) - min(ci_lo)) * 0.25
        ax.set_ylim(data_min, data_max)

        for i, n in enumerate(ns):
            ax.annotate(f'n={n}', (AGE_BIN_CENTERS[i], ci_hi[i]),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=8, color=GRAY)

        ax.set_xticks(AGE_BIN_CENTERS)
        ax.set_xticklabels(AGE_BIN_ORDER, fontsize=10)
        ax.set_xlabel('Age Bin (years)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Add correlation and effect size annotations
    bc = stats['correlations']['bycycle_vs_age']
    axes[0].set_title('A. Bycycle Period Slope', fontsize=13, fontweight='bold', pad=12)
    axes[0].text(0.03, 0.03,
                 f"ρ = {bc['spearman_rho']:.2f}, d = {stats['effect_sizes']['bycycle']['cohens_d_young_vs_old']:.2f}",
                 transform=axes[0].transAxes, fontsize=9, color=SECONDARY,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_RED, alpha=0.8))

    hm = stats['correlations']['hmm_vs_age']
    axes[1].set_title('B. HMM Oscillatory States', fontsize=13, fontweight='bold', pad=12)
    axes[1].text(0.03, 0.03,
                 f"ρ = {hm['spearman_rho']:.2f}, d = {stats['effect_sizes']['hmm']['cohens_d_young_vs_old']:.2f}",
                 transform=axes[1].transAxes, fontsize=9, color=PRIMARY,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=LIGHT_BLUE, alpha=0.8))

    paf = stats['correlations']['paf_vs_age']
    axes[2].set_title('C. Peak Alpha Frequency', fontsize=13, fontweight='bold', pad=12)
    axes[2].text(0.03, 0.97,
                 f"r = {paf['pearson_r']:.2f}",
                 transform=axes[2].transAxes, fontsize=9, color=ACCENT, va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#c6f6d5', alpha=0.8))

    fig.suptitle(f'HBN Developmental EEG (n = {stats["n_subjects"]:,}, eyes-closed)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "hbn_empirical_trajectories.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


def fig_sex_stratified(df, stats):
    """Figure 6: Sex-stratified bycycle and HMM trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white')

    for ax_idx, (measure, label, ylabel) in enumerate([
        ('bycycle_mean_period_slope', 'bycycle', 'Mean Period Slope'),
        ('hmm_mean_n_osc_states', 'hmm', 'N Oscillatory States'),
    ]):
        ax = axes[ax_idx]
        for sex, color, marker, offset in [('M', PRIMARY, 'o', -0.2), ('F', SECONDARY, 's', 0.2)]:
            sex_df = df[df['sex'] == sex]
            means, ci_lo, ci_hi = [], [], []
            for b in AGE_BIN_ORDER:
                bin_data = sex_df[sex_df['age_bin'] == b][measure].dropna()
                m = bin_data.mean()
                se = bin_data.std() / np.sqrt(len(bin_data))
                means.append(m)
                ci_lo.append(m - 1.96 * se)
                ci_hi.append(m + 1.96 * se)

            xs = [c + offset for c in AGE_BIN_CENTERS]
            yerr_lo = [m - lo for m, lo in zip(means, ci_lo)]
            yerr_hi = [hi - m for m, hi in zip(means, ci_hi)]
            n = len(sex_df)
            sex_stats = stats['sex_stratified'][sex]
            rho_key = f'{label}_rho'
            d_key = f'{label}_d'
            rho = sex_stats[rho_key]
            d = sex_stats[d_key]
            ax.errorbar(xs, means, yerr=[yerr_lo, yerr_hi],
                        fmt=f'{marker}-', color=color, markersize=7, linewidth=2,
                        capsize=4, capthick=1.5, label=f'{sex} (n={n}, ρ={rho:.2f}, d={d:.2f})')

        ax.set_xticks(AGE_BIN_CENTERS)
        ax.set_xticklabels(AGE_BIN_ORDER, fontsize=10)
        ax.set_xlabel('Age Bin (years)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    axes[0].set_title('A. Bycycle Period Slope by Sex', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_title('B. HMM Oscillatory States by Sex', fontsize=13, fontweight='bold', pad=12)

    fig.suptitle('Sex-Stratified Developmental Trajectories', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "hbn_sex_stratified.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


def fig_effect_sizes(stats):
    """Figure 7: Effect size summary — adjacent-bin Cohen's d with overall."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

    adjacent_labels = ['5-6\nvs\n7-8', '7-8\nvs\n9-10', '9-10\nvs\n11-12',
                       '11-12\nvs\n13-15', '13-15\nvs\n16-21']
    adj_keys = list(stats['effect_sizes']['bycycle']['adjacent_bin_d'].keys())

    for ax_idx, (label, color, light_color) in enumerate([
        ('bycycle', SECONDARY, LIGHT_RED),
        ('hmm', PRIMARY, LIGHT_BLUE),
    ]):
        ax = axes[ax_idx]
        adj_d = [stats['effect_sizes'][label]['adjacent_bin_d'][k] for k in adj_keys]
        overall_d = stats['effect_sizes'][label]['cohens_d_young_vs_old']
        overall_ci = stats['effect_sizes'][label]['ci_95']

        bars = ax.bar(range(len(adj_d)), adj_d, color=color, alpha=0.7, width=0.6,
                      edgecolor=color, linewidth=1.2)
        for bar, d_val in zip(bars, adj_d):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{d_val:.2f}', ha='center', va='bottom', fontsize=9, color=color)

        ax.axhline(y=0.2, color=GRAY, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(len(adj_d) - 0.5, 0.21, 'small effect (d=0.2)', fontsize=7, color=GRAY, ha='right')

        ax.axhline(y=overall_d, color=color, linestyle=':', alpha=0.8, linewidth=2)
        ax.text(len(adj_d) - 0.5, overall_d + 0.02,
                f'Overall d={overall_d:.2f} [{overall_ci[0]:.2f}, {overall_ci[1]:.2f}]',
                fontsize=8, color=color, ha='right', fontweight='bold')

        ax.set_xticks(range(len(adj_d)))
        ax.set_xticklabels(adjacent_labels, fontsize=9)
        ax.set_ylabel("Cohen's d", fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    axes[0].set_title('A. Bycycle — Adjacent-Bin Effect Sizes', fontsize=13, fontweight='bold', pad=12)
    axes[1].set_title('B. HMM — Adjacent-Bin Effect Sizes', fontsize=13, fontweight='bold', pad=12)

    fig.suptitle('Effect Size Progression Across Development', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "hbn_effect_sizes.png"
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    df, stats = load_data()
    fig_empirical_trajectories(df, stats)
    fig_sex_stratified(df, stats)
    fig_effect_sizes(stats)
    print("All figures generated.")
