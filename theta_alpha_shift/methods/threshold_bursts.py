"""
Threshold-based burst detection (fBOSC/PAPTO-style).

Detects bursts by thresholding bandpass-filtered analytic amplitude against
the aperiodic background. Per guardrail: this method has limited discriminative
capacity and is used for sensitivity analysis only, not primary hypothesis testing.

The headline statistic is burst_rate: detected bursts per second.

Usage:
    from theta_alpha_shift.methods.threshold_bursts import run_threshold
    result = run_threshold(epoch)
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from theta_alpha_shift.methods import MethodResult


def _analytic_amplitude(signal, sfreq, f_range):
    """Compute analytic amplitude in a frequency band."""
    nyq = sfreq / 2.0
    low = f_range[0] / nyq
    high = min(f_range[1] / nyq, 0.99)
    b, a = butter(3, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    analytic = hilbert(filtered)
    return np.abs(analytic)


def _estimate_threshold(amplitude, method="percentile", percentile=75):
    """Estimate burst detection threshold from amplitude envelope.

    Parameters
    ----------
    amplitude : ndarray
        Analytic amplitude envelope.
    method : str
        "percentile" or "median_mad".
    percentile : float
        Percentile for threshold (if method="percentile").

    Returns
    -------
    threshold : float
    """
    if method == "percentile":
        return np.percentile(amplitude, percentile)
    median = np.median(amplitude)
    mad = np.median(np.abs(amplitude - median))
    return median + 2 * mad


def run_threshold(epoch, f_range=(4, 12), min_cycles=2, threshold_pctl=75):
    """Detect bursts via amplitude thresholding.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    f_range : tuple of float
        Frequency band for burst detection (Hz).
    min_cycles : int
        Minimum number of cycles for a burst.
    threshold_pctl : float
        Percentile of amplitude for threshold.

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

    amplitude = _analytic_amplitude(data, sfreq, f_range)
    threshold = _estimate_threshold(amplitude, percentile=threshold_pctl)

    center_freq = np.mean(f_range)
    min_duration_s = min_cycles / center_freq
    min_samples = int(min_duration_s * sfreq)

    is_burst = amplitude >= threshold
    detected_bursts = []
    in_burst = False
    onset = 0

    for i in range(len(is_burst)):
        if is_burst[i] and not in_burst:
            onset = i
            in_burst = True
        elif not is_burst[i] and in_burst:
            duration = i - onset
            if duration >= min_samples:
                detected_bursts.append({
                    "onset": float(onset / sfreq),
                    "offset": float(i / sfreq),
                    "frequency": center_freq,
                    "amplitude": float(amplitude[onset:i].mean()),
                    "duration_s": float(duration / sfreq),
                    "n_cycles_approx": float(duration / sfreq * center_freq),
                })
            in_burst = False

    if in_burst:
        duration = len(is_burst) - onset
        if duration >= min_samples:
            detected_bursts.append({
                "onset": float(onset / sfreq),
                "offset": float(len(is_burst) / sfreq),
                "frequency": center_freq,
                "amplitude": float(amplitude[onset:].mean()),
                "duration_s": float(duration / sfreq),
                "n_cycles_approx": float(duration / sfreq * center_freq),
            })

    epoch_duration = len(data) / sfreq
    burst_rate = len(detected_bursts) / epoch_duration

    metadata = {
        "n_bursts": len(detected_bursts),
        "burst_rate": burst_rate,
        "threshold": float(threshold),
        "mean_amplitude": float(amplitude.mean()),
        "f_range": f_range,
        "min_cycles": min_cycles,
    }

    return MethodResult(
        method_name="threshold_bursts",
        detected_bursts=detected_bursts,
        headline_stat=burst_rate,
        headline_stat_name="burst_rate",
        metadata=metadata,
    )
