"""
Artifact injection wrappers for simulated EEG epochs.

Adds realistic artifact contamination (EOG blinks, EMG muscle noise, 60 Hz line noise)
to simulated signals. Each artifact type is independently togglable via config.
Ground-truth burst metadata is preserved — only the signal is modified.

Usage:
    from theta_alpha_shift.sim.artifacts import inject_artifacts
    clean_epoch = simulate_chirp(age=8)
    noisy_epoch = inject_artifacts(clean_epoch)
"""

import numpy as np
from scipy.signal import butter, filtfilt

from theta_alpha_shift.sim.params import get_random_seed, load_config


def inject_artifacts(epoch, config=None, rng=None):
    """Apply all enabled artifact layers to a SimulatedEpoch.

    Parameters
    ----------
    epoch : SimulatedEpoch
        Clean simulated epoch.
    config : dict, optional
        Pre-loaded config. Loaded from file if None.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
        New epoch with artifacts injected (ground truth preserved).
    """
    from theta_alpha_shift.sim.regimes import SimulatedEpoch

    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    art_cfg = cfg.get("artifacts", {})
    signal = epoch.data.copy()

    if art_cfg.get("eog", {}).get("enabled", False):
        signal = _inject_eog(signal, epoch.sfreq, art_cfg["eog"], rng)

    if art_cfg.get("emg", {}).get("enabled", False):
        signal = _inject_emg(signal, epoch.sfreq, art_cfg["emg"], rng)

    if art_cfg.get("line_noise", {}).get("enabled", False):
        signal = _inject_line_noise(signal, epoch.sfreq, art_cfg["line_noise"], rng)

    return SimulatedEpoch(
        data=signal,
        sfreq=epoch.sfreq,
        age=epoch.age,
        regime=epoch.regime,
        params={**epoch.params, "artifacts_applied": True},
        burst_times=epoch.burst_times,
        burst_freqs=epoch.burst_freqs,
        burst_labels=epoch.burst_labels,
    )


def _inject_eog(signal, sfreq, eog_cfg, rng):
    """Add eye blink artifacts as stereotypical half-sinusoid pulses.

    Parameters
    ----------
    signal : ndarray
        Input signal, shape (n_samples,) or (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).
    eog_cfg : dict
        EOG config with keys: amplitude, rate.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    signal : ndarray
        Signal with blink artifacts added.
    """
    amplitude = eog_cfg["amplitude"]
    rate = eog_cfg["rate"]

    is_2d = signal.ndim == 2
    n_samples = signal.shape[-1]
    duration = n_samples / sfreq

    blink_duration_s = 0.25
    blink_samples = int(round(blink_duration_s * sfreq))
    t_blink = np.arange(blink_samples) / sfreq
    blink_template = amplitude * np.sin(np.pi * t_blink / blink_duration_s)

    t = rng.exponential(1.0 / rate) if rate > 0 else duration + 1
    while t < duration:
        onset = int(round(t * sfreq))
        end = min(onset + blink_samples, n_samples)
        segment = blink_template[:end - onset]

        if is_2d:
            signal[:, onset:end] += segment[np.newaxis, :]
        else:
            signal[onset:end] += segment

        t += blink_duration_s + rng.exponential(1.0 / rate)

    return signal


def _inject_emg(signal, sfreq, emg_cfg, rng):
    """Add broadband muscle artifact as band-limited Gaussian noise.

    Parameters
    ----------
    signal : ndarray
        Input signal, shape (n_samples,) or (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).
    emg_cfg : dict
        EMG config with keys: amplitude, freq_range.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    signal : ndarray
        Signal with EMG artifact added.
    """
    amplitude = emg_cfg["amplitude"]
    freq_lo, freq_hi = emg_cfg["freq_range"]

    is_2d = signal.ndim == 2
    n_samples = signal.shape[-1]

    nyquist = sfreq / 2.0
    low = freq_lo / nyquist
    high = min(freq_hi / nyquist, 0.99)
    b, a = butter(4, [low, high], btype="band")

    if is_2d:
        n_channels = signal.shape[0]
        for ch in range(n_channels):
            noise = rng.standard_normal(n_samples)
            emg = filtfilt(b, a, noise)
            emg = emg / np.std(emg) * amplitude
            signal[ch, :] += emg
    else:
        noise = rng.standard_normal(n_samples)
        emg = filtfilt(b, a, noise)
        emg = emg / np.std(emg) * amplitude
        signal += emg

    return signal


def _inject_line_noise(signal, sfreq, line_cfg, rng):
    """Add power line noise as a sinusoid at the mains frequency.

    Parameters
    ----------
    signal : ndarray
        Input signal, shape (n_samples,) or (n_channels, n_samples).
    sfreq : float
        Sampling frequency (Hz).
    line_cfg : dict
        Line noise config with keys: freq, amplitude.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    signal : ndarray
        Signal with line noise added.
    """
    freq = line_cfg["freq"]
    amplitude = line_cfg["amplitude"]

    n_samples = signal.shape[-1]
    t = np.arange(n_samples) / sfreq
    phase = rng.uniform(0, 2 * np.pi)
    line = amplitude * np.sin(2 * np.pi * freq * t + phase)

    if signal.ndim == 2:
        signal += line[np.newaxis, :]
    else:
        signal += line

    return signal
