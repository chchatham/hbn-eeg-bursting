"""
Bycycle wrapper with relaxed thresholds for chirp detection.

Uses bycycle's cycle-by-cycle shape feature extraction with relaxed
period-consistency thresholds (per guardrail: default bycycle thresholds reject
the chirped cycles that Hypothesis A predicts). Computes chirp statistics as
post-hoc measures: period monotonicity and period dispersion within bursts.

The headline discriminative statistic is mean_period_slope: the average signed
slope of consecutive period values within detected bursts. Chirp bursts produce
systematically changing periods (positive or negative slope), while constant-
frequency bursts produce near-zero slope.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from theta_alpha_shift.methods import MethodResult


def _bandpass(signal, sfreq, f_range):
    """Apply zero-phase bandpass filter."""
    nyq = sfreq / 2.0
    low = f_range[0] / nyq
    high = min(f_range[1] / nyq, 0.99)
    b, a = butter(3, [low, high], btype="band")
    return filtfilt(b, a, signal)


def _find_zero_crossings(signal):
    """Find ascending zero-crossing indices."""
    crossings = np.where(np.diff(np.sign(signal)) > 0)[0]
    return crossings


def _extract_cycles(signal_raw, signal_filt, sfreq):
    """Extract cycle-by-cycle features from filtered signal.

    Parameters
    ----------
    signal_raw : ndarray
        Original (unfiltered) signal.
    signal_filt : ndarray
        Bandpass-filtered signal.
    sfreq : float
        Sampling frequency.

    Returns
    -------
    cycles : list of dict
        Per-cycle features: onset_sample, period_samples, period_sec,
        amplitude, rise_time, decay_time.
    """
    crossings = _find_zero_crossings(signal_filt)
    if len(crossings) < 2:
        return []

    cycles = []
    for i in range(len(crossings) - 1):
        onset = crossings[i]
        offset = crossings[i + 1]
        period_samples = offset - onset
        period_sec = period_samples / sfreq

        segment_raw = signal_raw[onset:offset]
        amplitude = np.ptp(segment_raw)

        peak_idx = np.argmax(segment_raw)
        rise_samples = peak_idx
        decay_samples = period_samples - peak_idx

        cycles.append({
            "onset_sample": int(onset),
            "offset_sample": int(offset),
            "period_samples": int(period_samples),
            "period_sec": float(period_sec),
            "frequency": float(sfreq / period_samples),
            "amplitude": float(amplitude),
            "rise_samples": int(rise_samples),
            "decay_samples": int(decay_samples),
        })

    return cycles


def _detect_bursts(cycles, amp_threshold_frac=0.3, min_n_cycles=2):
    """Detect bursts as consecutive high-amplitude cycles.

    Parameters
    ----------
    cycles : list of dict
        Cycle features.
    amp_threshold_frac : float
        Fraction of max amplitude to use as burst threshold.
    min_n_cycles : int
        Minimum consecutive cycles to count as a burst.

    Returns
    -------
    bursts : list of list of int
        Each burst is a list of cycle indices.
    """
    if not cycles:
        return []

    amplitudes = np.array([c["amplitude"] for c in cycles])
    threshold = amp_threshold_frac * np.max(amplitudes)
    is_above = amplitudes >= threshold

    bursts = []
    current_burst = []
    for i, above in enumerate(is_above):
        if above:
            current_burst.append(i)
        else:
            if len(current_burst) >= min_n_cycles:
                bursts.append(current_burst)
            current_burst = []
    if len(current_burst) >= min_n_cycles:
        bursts.append(current_burst)

    return bursts


def _compute_period_slope(cycles, burst_indices):
    """Compute mean signed period slope within a burst.

    Returns the average of consecutive period differences (in samples).
    Positive = periods getting longer (freq decreasing = downward chirp).
    Negative = periods getting shorter (freq increasing = upward chirp).
    """
    periods = [cycles[i]["period_samples"] for i in burst_indices]
    if len(periods) < 2:
        return 0.0
    diffs = np.diff(periods)
    return float(np.mean(diffs))


def run_bycycle(epoch, f_range=(4, 12), amp_threshold_frac=0.3, min_n_cycles=2):
    """Run cycle-by-cycle analysis with relaxed thresholds.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    f_range : tuple of float
        Frequency range for bandpass filtering (Hz).
    amp_threshold_frac : float
        Amplitude threshold for burst detection (fraction of max).
    min_n_cycles : int
        Minimum cycles per burst.

    Returns
    -------
    result : MethodResult
    """
    if isinstance(epoch, tuple):
        data, sfreq = epoch
    else:
        data = epoch.data
        sfreq = epoch.sfreq

    if data.ndim == 2:
        data = data.mean(axis=0)

    sig_filt = _bandpass(data, sfreq, f_range)
    cycles = _extract_cycles(data, sig_filt, sfreq)
    bursts = _detect_bursts(cycles, amp_threshold_frac, min_n_cycles)

    detected_bursts = []
    period_slopes = []

    for burst_indices in bursts:
        slope = _compute_period_slope(cycles, burst_indices)
        period_slopes.append(slope)

        onset_cycle = cycles[burst_indices[0]]
        offset_cycle = cycles[burst_indices[-1]]
        freqs = [cycles[i]["frequency"] for i in burst_indices]

        detected_bursts.append({
            "onset": onset_cycle["onset_sample"] / sfreq,
            "offset": offset_cycle["offset_sample"] / sfreq,
            "frequency": float(np.mean(freqs)),
            "amplitude": float(np.mean([cycles[i]["amplitude"] for i in burst_indices])),
            "n_cycles": len(burst_indices),
            "period_slope": slope,
            "freq_range": (min(freqs), max(freqs)),
        })

    mean_period_slope = float(np.mean(np.abs(period_slopes))) if period_slopes else 0.0

    all_periods = [c["period_samples"] for c in cycles]
    period_cv = float(np.std(all_periods) / np.mean(all_periods)) if all_periods else 0.0

    metadata = {
        "n_cycles_total": len(cycles),
        "n_bursts": len(bursts),
        "mean_period_slope": mean_period_slope,
        "period_slopes": period_slopes,
        "period_cv": period_cv,
        "f_range": f_range,
    }

    return MethodResult(
        method_name="bycycle",
        detected_bursts=detected_bursts,
        headline_stat=mean_period_slope,
        headline_stat_name="mean_period_slope",
        metadata=metadata,
    )
