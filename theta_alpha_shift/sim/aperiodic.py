"""
Age-parameterized aperiodic (1/f + knee) background signal generation.

Generates colored noise with a spectral profile matching the developmental
trajectory from Cellier et al. 2021: exponent declines from 1.7 (age 5)
to 1.2 (age 20), with a fixed knee at ~5 Hz.

Usage:
    from theta_alpha_shift.sim.aperiodic import generate_aperiodic
    signal = generate_aperiodic(age=8, duration=2.0, sfreq=500)
"""

import numpy as np

from theta_alpha_shift.sim.params import (
    aperiodic_exponent,
    aperiodic_offset,
    knee_freq,
    load_config,
)


def _lorentzian_psd(freqs, knee, exponent, offset):
    """Compute Lorentzian PSD: P(f) = 10^offset / (f^exponent + knee^exponent).

    Parameters
    ----------
    freqs : ndarray
        Frequency vector (Hz). Must not contain 0.
    knee : float
        Knee frequency (Hz).
    exponent : float
        Spectral exponent (positive = steeper falloff).
    offset : float
        Log10 power offset.

    Returns
    -------
    psd : ndarray
        Power spectral density (linear scale).
    """
    psd = (10 ** offset) / (freqs ** exponent + knee ** exponent)
    return psd


def generate_aperiodic(age, duration=None, sfreq=None, config=None, rng=None):
    """Generate aperiodic background signal with age-appropriate spectral shape.

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    duration : float, optional
        Signal duration in seconds. Defaults to config epoch_duration.
    sfreq : float, optional
        Sampling frequency in Hz. Defaults to config sfreq.
    config : dict, optional
        Pre-loaded config dict. Loaded from file if None.
    rng : numpy.random.Generator, optional
        Random number generator. Created from config seed if None.

    Returns
    -------
    signal : ndarray, shape (n_samples,)
        Aperiodic time series.
    """
    cfg = config or load_config()

    if duration is None:
        duration = cfg["epoch_duration"]
    if sfreq is None:
        sfreq = cfg["sfreq"]
    if rng is None:
        rng = np.random.default_rng(cfg["random_seed"])

    n_samples = int(duration * sfreq)

    exp = aperiodic_exponent(age, config=cfg)
    offset = aperiodic_offset(config=cfg)
    knee = knee_freq(config=cfg)

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)

    # Build target amplitude spectrum from Lorentzian PSD
    # Skip DC (index 0) to avoid division issues
    amplitudes = np.zeros(len(freqs))
    amplitudes[1:] = np.sqrt(_lorentzian_psd(freqs[1:], knee, exp, offset))

    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    phases[0] = 0  # DC has no phase

    # Construct complex spectrum and inverse FFT
    spectrum = amplitudes * np.exp(1j * phases)
    signal = np.fft.irfft(spectrum, n=n_samples)

    return signal


def generate_aperiodic_psd(age, sfreq=None, n_freqs=1000, config=None):
    """Return the theoretical aperiodic PSD for a given age (no noise realization).

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    sfreq : float, optional
        Sampling frequency in Hz. Defaults to config sfreq.
    n_freqs : int
        Number of frequency points between 0.5 and sfreq/2.
    config : dict, optional
        Pre-loaded config dict.

    Returns
    -------
    freqs : ndarray
        Frequency vector (Hz).
    psd : ndarray
        Power spectral density (linear scale).
    """
    cfg = config or load_config()

    if sfreq is None:
        sfreq = cfg["sfreq"]

    exp = aperiodic_exponent(age, config=cfg)
    offset = aperiodic_offset(config=cfg)
    knee = knee_freq(config=cfg)

    freqs = np.linspace(0.5, sfreq / 2, n_freqs)
    psd = _lorentzian_psd(freqs, knee, exp, offset)

    return freqs, psd
