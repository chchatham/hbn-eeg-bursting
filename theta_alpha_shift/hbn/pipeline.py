"""
HBN EEG burst analysis pipeline integration.

Loads preprocessed HBN epoch files from the parent pipeline, runs surviving
burst-detection methods (bycycle, HMM, specparam) per epoch, aggregates
results per subject, and merges with phenotypic metadata (age, sex).

Epoch files: ../hbn-eeg-pipeline/data/processed/{release}/{subject_id}_{condition}_epo.fif
Metadata:    ../hbn-eeg-pipeline/data/phenotypic/metadata_{condition}.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "hbn-eeg-pipeline" / "scripts"))
from utils.channel_config import get_scalp_channels, KEY_ELECTRODES

from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
from theta_alpha_shift.methods.hmm_wrap import run_hmm
from theta_alpha_shift.methods.specparam_baseline import run_specparam

mne.set_log_level("WARNING")

# ── Configuration ────────────────────────────────────────────────────────────

RELEASES = [f"R{i}" for i in range(1, 10)]

AGE_BINS = [(5, 7), (7, 9), (9, 11), (11, 13), (13, 16), (16, 22)]
AGE_BIN_LABELS = ["5-6", "7-8", "9-10", "11-12", "13-15", "16-21"]

POSTERIOR_CHANNELS = ["E75", "E70", "E83", "E62"]

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / ".." / "hbn-eeg-pipeline" / "data" / "processed"
DEFAULT_PHENOTYPIC_DIR = Path(__file__).resolve().parents[2] / ".." / "hbn-eeg-pipeline" / "data" / "phenotypic"

MIN_EPOCHS_PER_SUBJECT = 10


def assign_age_bin(age):
    """Assign an age to a bin label. Returns None if outside all bins."""
    for (lo, hi), label in zip(AGE_BINS, AGE_BIN_LABELS):
        if lo <= age < hi:
            return label
    return None


def discover_subjects(data_dir=None, condition="ec"):
    """Scan processed directory for available epoch files.

    Parameters
    ----------
    data_dir : Path, optional
        Root of processed data directory. Defaults to parent pipeline location.
    condition : str
        'ec' or 'eo'.

    Returns
    -------
    subjects : list of dict
        Each dict has keys: subject_id, release, epoch_path.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    subjects = []
    for release in RELEASES:
        release_dir = data_dir / release
        if not release_dir.exists():
            continue
        for fpath in sorted(release_dir.glob(f"*_{condition}_epo.fif")):
            subject_id = fpath.name.replace(f"_{condition}_epo.fif", "")
            subjects.append({
                "subject_id": subject_id,
                "release": release,
                "epoch_path": str(fpath),
            })
    return subjects


def load_metadata(phenotypic_dir=None, condition="ec"):
    """Load phenotypic metadata CSV.

    Returns
    -------
    df : DataFrame
        Columns: subject_id, release, age, sex, p_factor, attention,
        internalizing, externalizing, age_bin.
    """
    if phenotypic_dir is None:
        phenotypic_dir = DEFAULT_PHENOTYPIC_DIR
    csv_path = Path(phenotypic_dir) / f"metadata_{condition}.csv"
    df = pd.read_csv(csv_path)
    df["age_bin"] = df["age"].apply(assign_age_bin)
    return df


def load_subject_epochs(epoch_path):
    """Load one subject's epoch file and return data + info.

    Returns
    -------
    epochs : mne.Epochs
        Loaded epochs with scalp-only average reference re-applied.
    """
    epochs = mne.read_epochs(str(epoch_path), preload=True, verbose=False)
    scalp_chs = get_scalp_channels(list(epochs.ch_names))
    epochs.set_eeg_reference(ref_channels=scalp_chs, projection=False, verbose=False)
    return epochs


def select_posterior_data(epochs, channels=None):
    """Extract posterior channel data from epochs.

    Parameters
    ----------
    epochs : mne.Epochs
        Loaded epochs.
    channels : list of str, optional
        Channel names to select. Defaults to POSTERIOR_CHANNELS.

    Returns
    -------
    data : ndarray, shape (n_epochs, n_samples)
        Posterior-averaged epoch data.
    sfreq : float
        Sampling frequency.
    """
    if channels is None:
        channels = POSTERIOR_CHANNELS
    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available:
        available = [KEY_ELECTRODES["Oz"]]
    data = epochs.get_data(picks=available)
    data_avg = data.mean(axis=1)
    return data_avg, epochs.info["sfreq"]


def run_subject_burst_analysis(epoch_data, sfreq, methods=None):
    """Run burst-detection methods on all epochs of one subject.

    Parameters
    ----------
    epoch_data : ndarray, shape (n_epochs, n_samples)
        Posterior-averaged epoch time series.
    sfreq : float
        Sampling frequency.
    methods : list of str, optional
        Methods to run. Defaults to ['bycycle', 'hmm', 'specparam'].

    Returns
    -------
    results : dict
        Per-method aggregated results:
        - bycycle: mean_period_slope, std_period_slope, n_bursts_total, per_epoch_slopes
        - hmm: mean_n_osc_states, per_epoch_n_osc_states, bic_values
        - specparam: mean_n_peaks, per_epoch_n_peaks, mean_peak_freq
    """
    if methods is None:
        methods = ["bycycle", "hmm", "specparam"]

    n_epochs = epoch_data.shape[0]
    results = {}

    if "bycycle" in methods:
        slopes = []
        n_bursts = 0
        for i in range(n_epochs):
            r = run_bycycle((epoch_data[i], sfreq))
            slopes.append(r.headline_stat)
            n_bursts += r.metadata["n_bursts"]
        results["bycycle"] = {
            "mean_period_slope": float(np.mean(slopes)),
            "std_period_slope": float(np.std(slopes)),
            "n_bursts_total": n_bursts,
            "n_epochs": n_epochs,
            "per_epoch_slopes": slopes,
        }

    if "hmm" in methods:
        n_osc_states = []
        bic_2v3 = []
        for i in range(n_epochs):
            r = run_hmm((epoch_data[i], sfreq))
            n_osc_states.append(r.headline_stat)
            bic_2v3.append(r.metadata["bic_diff_2v3"])
        results["hmm"] = {
            "mean_n_osc_states": float(np.mean(n_osc_states)),
            "std_n_osc_states": float(np.std(n_osc_states)),
            "n_epochs": n_epochs,
            "per_epoch_n_osc_states": n_osc_states,
            "mean_bic_diff_2v3": float(np.mean(bic_2v3)),
        }

    if "specparam" in methods:
        n_peaks = []
        peak_freqs = []
        for i in range(n_epochs):
            r = run_specparam((epoch_data[i], sfreq), psd_method="welch")
            n_peaks.append(r.headline_stat)
            pf = r.metadata.get("peak_freq_hz", np.nan)
            if not np.isnan(pf):
                peak_freqs.append(pf)
        results["specparam"] = {
            "mean_n_peaks": float(np.mean(n_peaks)),
            "std_n_peaks": float(np.std(n_peaks)),
            "mean_peak_freq": float(np.mean(peak_freqs)) if peak_freqs else np.nan,
            "n_epochs": n_epochs,
            "per_epoch_n_peaks": n_peaks,
        }

    return results


def run_miniset(data_dir=None, phenotypic_dir=None, n_per_release=20,
                condition="ec", methods=None, seed=42):
    """Run burst analysis on an age-balanced miniset.

    Selects n_per_release subjects from each release, balanced across age bins
    where possible. Skips subjects with fewer than MIN_EPOCHS_PER_SUBJECT epochs.

    Parameters
    ----------
    data_dir : Path, optional
    phenotypic_dir : Path, optional
    n_per_release : int
        Target number of subjects per release.
    condition : str
        'ec' or 'eo'.
    methods : list of str, optional
    seed : int

    Returns
    -------
    results_df : DataFrame
        Subject-level results with metadata.
    """
    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("HBN BURST ANALYSIS — MINISET")
    print("=" * 60)

    subjects = discover_subjects(data_dir, condition)
    metadata = load_metadata(phenotypic_dir, condition)
    meta_lookup = metadata.set_index("subject_id").to_dict("index")

    print(f"Found {len(subjects)} subjects with {condition} epoch files")
    print(f"Metadata available for {len(metadata)} subjects")

    selected = _select_miniset_subjects(subjects, meta_lookup, n_per_release, rng)
    print(f"Selected {len(selected)} subjects for miniset")

    all_results = []
    for subj in tqdm(selected, desc="Processing subjects"):
        sid = subj["subject_id"]
        meta = meta_lookup.get(sid, {})
        age = meta.get("age", np.nan)

        try:
            epochs = load_subject_epochs(subj["epoch_path"])
        except Exception as e:
            print(f"  SKIP {sid}: failed to load epochs — {e}")
            continue

        epoch_data, sfreq = select_posterior_data(epochs)

        if epoch_data.shape[0] < MIN_EPOCHS_PER_SUBJECT:
            print(f"  SKIP {sid}: only {epoch_data.shape[0]} epochs (min {MIN_EPOCHS_PER_SUBJECT})")
            continue

        results = run_subject_burst_analysis(epoch_data, sfreq, methods)

        row = {
            "subject_id": sid,
            "release": subj["release"],
            "age": age,
            "sex": meta.get("sex", ""),
            "age_bin": assign_age_bin(age) if not np.isnan(age) else None,
            "n_epochs": epoch_data.shape[0],
            "condition": condition,
        }

        for method_name, method_results in results.items():
            for key, val in method_results.items():
                if not isinstance(val, list):
                    row[f"{method_name}_{key}"] = val

        all_results.append(row)

    results_df = pd.DataFrame(all_results)
    print(f"\nCompleted: {len(results_df)} subjects processed")
    if "age_bin" in results_df.columns:
        print("\nSubjects per age bin:")
        print(results_df["age_bin"].value_counts().sort_index().to_string())

    return results_df


def _select_miniset_subjects(subjects, meta_lookup, n_per_release, rng):
    """Select age-balanced subjects from each release."""
    by_release = {}
    for subj in subjects:
        r = subj["release"]
        by_release.setdefault(r, []).append(subj)

    selected = []
    for release, release_subjects in sorted(by_release.items()):
        with_age = []
        for s in release_subjects:
            meta = meta_lookup.get(s["subject_id"], {})
            age = meta.get("age", np.nan)
            if not np.isnan(age):
                s_copy = dict(s)
                s_copy["age"] = age
                s_copy["age_bin"] = assign_age_bin(age)
                with_age.append(s_copy)

        if not with_age:
            continue

        by_bin = {}
        for s in with_age:
            ab = s.get("age_bin")
            if ab:
                by_bin.setdefault(ab, []).append(s)

        n_bins = len(by_bin)
        if n_bins == 0:
            continue

        per_bin = max(1, n_per_release // n_bins)
        remainder = n_per_release - per_bin * n_bins

        for ab in AGE_BIN_LABELS:
            if ab not in by_bin:
                continue
            pool = by_bin[ab]
            n_take = min(per_bin, len(pool))
            if remainder > 0 and n_take < len(pool):
                n_take = min(n_take + 1, len(pool))
                remainder -= 1
            indices = rng.choice(len(pool), size=n_take, replace=False)
            for idx in indices:
                selected.append(pool[idx])

    return selected


def save_results(results_df, output_dir, prefix="miniset"):
    """Save subject-level results to CSV and summary to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    summary = {
        "n_subjects": len(results_df),
        "age_bins": results_df["age_bin"].value_counts().to_dict() if "age_bin" in results_df.columns else {},
    }

    for method in ["bycycle", "hmm", "specparam"]:
        stat_col = f"{method}_mean_period_slope" if method == "bycycle" else \
                   f"{method}_mean_n_osc_states" if method == "hmm" else \
                   f"{method}_mean_n_peaks"
        if stat_col in results_df.columns:
            summary[f"{method}_overall_mean"] = float(results_df[stat_col].mean())
            summary[f"{method}_overall_std"] = float(results_df[stat_col].std())

            if "age_bin" in results_df.columns:
                by_age = results_df.groupby("age_bin")[stat_col].agg(["mean", "std", "count"])
                summary[f"{method}_by_age_bin"] = by_age.to_dict("index")

    json_path = output_dir / f"{prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {json_path}")
