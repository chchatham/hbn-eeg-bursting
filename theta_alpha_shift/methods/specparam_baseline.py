"""
Specparam baseline method: PSD estimation + spectral parameterization.

Computes Welch PSD, fits specparam to decompose into aperiodic + periodic components,
and extracts peak frequency, bandwidth, and number of peaks in the theta-alpha range.
This is a spectral-domain method — it does not detect individual bursts.

The headline discriminative statistic is the number of peaks detected in the 4-12 Hz
band. If chirp (Hypothesis A) produces a single broad peak while mixture (Hypothesis B)
produces two distinct peaks, n_peaks will discriminate the two.

Usage:
    from theta_alpha_shift.methods.specparam_baseline import run_specparam
    result = run_specparam(epoch)
"""

import numpy as np
from scipy.signal import welch

from theta_alpha_shift.methods import MethodResult


def _compute_psd(data, sfreq):
    """Compute Welch PSD for a 1D or 2D signal.

    Parameters
    ----------
    data : ndarray
        Signal, shape (n_samples,) or (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).

    Returns
    -------
    freqs : ndarray
        Frequency vector.
    psd : ndarray
        Power spectral density. For multi-channel, averaged across channels.
    """
    n_samples = data.shape[-1]
    nperseg = min(n_samples, int(sfreq))
    if data.ndim == 2:
        freqs, psds = welch(data, fs=sfreq, nperseg=nperseg)
        psd = psds.mean(axis=0)
    else:
        freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
    return freqs, psd


def run_specparam(epoch, freq_range=(1, 45), peak_width_limits=(2.0, 6.0),
                  max_n_peaks=6, peak_threshold=0.5):
    """Run specparam on a SimulatedEpoch and return a MethodResult.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    freq_range : tuple of float
        Frequency range for specparam fitting (Hz).
    peak_width_limits : tuple of float
        Min and max peak width (Hz) for specparam.
    max_n_peaks : int
        Maximum number of peaks to fit.
    peak_threshold : float
        Minimum peak height threshold (in units of spectral power).

    Returns
    -------
    result : MethodResult
    """
    from specparam import SpectralModel

    if isinstance(epoch, tuple):
        data, sfreq = epoch
    else:
        data = epoch.data
        sfreq = epoch.sfreq

    freqs, psd = _compute_psd(data, sfreq)

    sm = SpectralModel(
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=peak_threshold,
        aperiodic_mode="fixed",
    )
    sm.fit(freqs, psd, freq_range)

    peak_params = sm.get_params("peak")
    aperiodic_params = sm.get_params("aperiodic")
    fit_error = sm.get_metrics("error")

    if peak_params.ndim == 1:
        peak_params = peak_params.reshape(1, -1) if len(peak_params) > 0 else np.empty((0, 3))

    theta_alpha_mask = (peak_params[:, 0] >= 4.0) & (peak_params[:, 0] <= 12.0)
    peaks_in_range = peak_params[theta_alpha_mask]
    n_peaks = len(peaks_in_range)

    if n_peaks > 0:
        dominant_idx = np.argmax(peaks_in_range[:, 1])
        peak_freq = float(peaks_in_range[dominant_idx, 0])
        peak_bandwidth = float(peaks_in_range[dominant_idx, 2])
    else:
        peak_freq = np.nan
        peak_bandwidth = np.nan

    metadata = {
        "peak_params": peak_params.tolist(),
        "aperiodic_params": aperiodic_params.tolist(),
        "fit_error": float(fit_error),
        "n_peaks_total": len(peak_params),
        "n_peaks_theta_alpha": n_peaks,
        "peak_freq_hz": peak_freq,
        "peak_bandwidth_hz": peak_bandwidth,
        "peaks_in_theta_alpha": peaks_in_range.tolist(),
        "freqs": freqs.tolist(),
        "psd": psd.tolist(),
    }

    return MethodResult(
        method_name="specparam_baseline",
        detected_bursts=[],
        headline_stat=float(n_peaks),
        headline_stat_name="n_peaks_theta_alpha",
        metadata=metadata,
    )
