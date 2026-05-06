"""
Iterated EMD + within-cycle instantaneous frequency wrapper.

Uses iterated mask sift (Quinn et al. 2021) to decompose the signal into
intrinsic mode functions, then computes instantaneous frequency (IF) via
normalized Hilbert transform. The headline statistic is the mean absolute IF
slope during burst segments of the theta-alpha IMF.

Chirp bursts (Hypothesis A) produce systematic IF slopes (frequency changing
within each burst), while constant-frequency bursts (Hypothesis B) produce
near-zero IF slopes.

Usage:
    from theta_alpha_shift.methods.itemd_if import run_itemd
    result = run_itemd(epoch)
"""

import numpy as np

from theta_alpha_shift.methods import MethodResult


def _select_theta_alpha_imf(IF, IA, f_range=(4, 12)):
    """Select the IMF whose mean IF falls in the theta-alpha range.

    Parameters
    ----------
    IF : ndarray, shape (n_samples, n_imfs)
        Instantaneous frequency per IMF.
    IA : ndarray, shape (n_samples, n_imfs)
        Instantaneous amplitude per IMF.
    f_range : tuple of float
        Target frequency range (Hz).

    Returns
    -------
    imf_idx : int
        Index of selected IMF.
    """
    n_imfs = IF.shape[1]
    scores = np.zeros(n_imfs)

    for i in range(n_imfs):
        mean_if = np.median(IF[:, i])
        if f_range[0] <= mean_if <= f_range[1]:
            scores[i] = IA[:, i].mean()

    if scores.max() == 0:
        dists = np.array([
            abs(np.median(IF[:, i]) - np.mean(f_range)) for i in range(n_imfs)
        ])
        return int(np.argmin(dists))

    return int(np.argmax(scores))


def _compute_if_slopes(IF_imf, IA_imf, sfreq, amp_threshold_frac=0.3):
    """Compute IF slopes during high-amplitude (burst) segments.

    Parameters
    ----------
    IF_imf : ndarray, shape (n_samples,)
        Instantaneous frequency of selected IMF.
    IA_imf : ndarray, shape (n_samples,)
        Instantaneous amplitude of selected IMF.
    sfreq : float
        Sampling frequency.
    amp_threshold_frac : float
        Fraction of max amplitude for burst detection.

    Returns
    -------
    slopes : list of float
        IF slope (Hz/cycle) for each detected burst segment.
    burst_segments : list of dict
        Detected burst info.
    """
    threshold = amp_threshold_frac * np.max(IA_imf)
    is_burst = IA_imf >= threshold

    segments = []
    in_seg = False
    start = 0
    for i in range(len(is_burst)):
        if is_burst[i] and not in_seg:
            start = i
            in_seg = True
        elif not is_burst[i] and in_seg:
            if i - start > int(sfreq * 0.05):
                segments.append((start, i))
            in_seg = False
    if in_seg and len(is_burst) - start > int(sfreq * 0.05):
        segments.append((start, len(is_burst)))

    slopes = []
    burst_segments = []
    for start, end in segments:
        if_seg = IF_imf[start:end]
        valid = np.isfinite(if_seg) & (if_seg > 0) & (if_seg < sfreq / 2)
        if valid.sum() < 3:
            continue

        if_valid = if_seg[valid]
        t_valid = np.where(valid)[0] / sfreq

        if len(t_valid) > 1:
            slope = np.polyfit(t_valid, if_valid, 1)[0]
        else:
            slope = 0.0

        slopes.append(float(slope))
        burst_segments.append({
            "onset": start / sfreq,
            "offset": end / sfreq,
            "frequency": float(np.mean(if_valid)),
            "amplitude": float(np.mean(IA_imf[start:end])),
            "if_slope": float(slope),
            "n_samples": end - start,
        })

    return slopes, burst_segments


def run_itemd(epoch, f_range=(4, 12), max_imfs=5, amp_threshold_frac=0.3):
    """Run iterated mask sift EMD and compute IF slopes.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    f_range : tuple of float
        Theta-alpha frequency range (Hz).
    max_imfs : int
        Maximum number of IMFs to extract.
    amp_threshold_frac : float
        Amplitude threshold for burst detection.

    Returns
    -------
    result : MethodResult
    """
    import emd

    if isinstance(epoch, tuple):
        data, sfreq = epoch
    else:
        data = epoch.data
        sfreq = epoch.sfreq

    if data.ndim == 2:
        data = data.mean(axis=0)

    try:
        imfs = emd.sift.iterated_mask_sift(
            data, sample_rate=sfreq, max_imfs=max_imfs
        )
    except Exception:
        imfs = emd.sift.mask_sift(
            data, sample_rate=sfreq, max_imfs=max_imfs
        )

    try:
        IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, "nht")
    except (ValueError, np.linalg.LinAlgError):
        IP, IF, IA = emd.spectra.frequency_transform(imfs, sfreq, "hilbert")

    imf_idx = _select_theta_alpha_imf(IF, IA, f_range)
    IF_imf = IF[:, imf_idx]
    IA_imf = IA[:, imf_idx]

    slopes, burst_segments = _compute_if_slopes(
        IF_imf, IA_imf, sfreq, amp_threshold_frac
    )

    mean_if_slope = float(np.mean(np.abs(slopes))) if slopes else 0.0

    metadata = {
        "n_imfs": imfs.shape[1],
        "selected_imf_idx": imf_idx,
        "imf_mean_freqs": [float(np.median(IF[:, i])) for i in range(imfs.shape[1])],
        "n_burst_segments": len(burst_segments),
        "if_slopes": slopes,
        "mean_if_slope": mean_if_slope,
        "f_range": f_range,
    }

    return MethodResult(
        method_name="itemd_if",
        detected_bursts=burst_segments,
        headline_stat=mean_if_slope,
        headline_stat_name="mean_if_slope",
        metadata=metadata,
    )
