"""
Specparam baseline method: PSD estimation + spectral parameterization.

Computes PSD (Welch or meeglet), fits specparam to decompose into aperiodic + periodic
components, and extracts peak frequency, bandwidth, and number of peaks in the
theta-alpha range. This is a spectral-domain method — it does not detect individual
bursts.

The headline discriminative statistic is the number of peaks detected in the 4-12 Hz
band. If chirp (Hypothesis A) produces a single broad peak while mixture (Hypothesis B)
produces two distinct peaks, n_peaks will discriminate the two.
"""

import warnings

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import welch
import mne
from meeglet import compute_spectral_features
from specparam import SpectralModel

from theta_alpha_shift.methods import MethodResult

mne.set_log_level("WARNING")

FOI_START = 1
FOI_END = 45
BW_OCT = 0.35
DENSITY = 12


def _compute_psd_welch(data, sfreq):
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


def _data_to_mne_epochs(data, sfreq):
    """Wrap a numpy array in an MNE EpochsArray for meeglet consumption."""
    if data.ndim == 1:
        data_3d = data.reshape(1, 1, -1)
        ch_names = ["Oz"]
    else:
        data_3d = data.reshape(1, data.shape[0], data.shape[1])
        ch_names = [f"E{i+1}" for i in range(data.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.EpochsArray(data_3d, info, verbose=False)


def _compute_psd_meeglet(data, sfreq):
    """Compute meeglet wavelet PSD for a 1D or 2D signal.

    Parameters
    ----------
    data : ndarray
        Signal, shape (n_samples,) or (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).

    Returns
    -------
    freqs : ndarray
        Log-spaced frequency vector (meeglet foi grid).
    psd : ndarray
        Power spectral density (linear). For multi-channel, averaged across channels.
    """
    epochs = _data_to_mne_epochs(data, sfreq)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, info = compute_spectral_features(
            epochs, foi_start=FOI_START, foi_end=FOI_END, bw_oct=BW_OCT,
            features=("pow",), density=DENSITY, verbose=False,
        )
    freqs = info.foi
    psd = out.pow.mean(axis=0) if out.pow.ndim == 2 else out.pow
    return freqs, psd


def _interpolate_to_linear(freqs_log, psd_log, freq_range=(1, 45), n_points=None):
    """Interpolate log-spaced PSD onto a linear frequency grid for specparam."""
    if n_points is None:
        n_points = int(freq_range[1] - freq_range[0]) + 1
    freqs_lin = np.linspace(freq_range[0], freq_range[1], n_points)
    interp_fn = interp1d(freqs_log, psd_log, kind="cubic", bounds_error=False,
                         fill_value="extrapolate")
    psd_lin = interp_fn(freqs_lin)
    psd_lin = np.maximum(psd_lin, np.finfo(float).tiny)
    return freqs_lin, psd_lin


def _compute_psd(data, sfreq, method="welch"):
    """Compute PSD using the specified method."""
    if method == "meeglet":
        return _compute_psd_meeglet(data, sfreq)
    return _compute_psd_welch(data, sfreq)


def run_specparam(epoch, freq_range=(1, 45), peak_width_limits=(2.0, 6.0),
                  max_n_peaks=6, peak_threshold=0.5, psd_method="welch"):
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
    psd_method : str
        PSD estimation method: 'welch' or 'meeglet'.

    Returns
    -------
    result : MethodResult
    """
    if isinstance(epoch, tuple):
        data, sfreq = epoch
    else:
        data = epoch.data
        sfreq = epoch.sfreq

    freqs_raw, psd_raw = _compute_psd(data, sfreq, method=psd_method)

    if psd_method == "meeglet":
        freqs, psd = _interpolate_to_linear(freqs_raw, psd_raw, freq_range)
    else:
        freqs, psd = freqs_raw, psd_raw

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
        "psd_method": psd_method,
        "freqs_raw": freqs_raw.tolist() if psd_method == "meeglet" else None,
        "psd_raw": psd_raw.tolist() if psd_method == "meeglet" else None,
    }

    return MethodResult(
        method_name="specparam_baseline",
        detected_bursts=[],
        headline_stat=float(n_peaks),
        headline_stat_name="n_peaks_theta_alpha",
        metadata=metadata,
    )
