"""
Generative regime implementations for theta-alpha shift simulations.

Five regimes model competing hypotheses for the developmental "theta shelf":
  Regime 1 (chirp): within-burst frequency sweeps from theta to alpha
  Regime 2 (mixture): distinct theta and alpha burst populations
  Regime 3 (drift): single oscillator at PAF, no chirp (control)
  Regime 4 (cooccur): theta + alpha bursts with controlled co-occurrence
  Regime 5 (narrowing): single oscillator with age-dependent bandwidth

Each regime function returns a SimulatedEpoch with ground-truth burst metadata.

Usage:
    from theta_alpha_shift.sim.regimes import simulate_chirp
    epoch = simulate_chirp(age=8)
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import windows

from theta_alpha_shift.sim.aperiodic import generate_aperiodic
from theta_alpha_shift.sim.params import (
    burst_n_cycles,
    burst_rate,
    burst_snr,
    chirp_fraction,
    get_epoch_duration,
    get_random_seed,
    get_sfreq,
    load_config,
    mixture_theta_weight,
    narrowing_bandwidth,
    paf,
)


@dataclass
class SimulatedEpoch:
    data: np.ndarray
    sfreq: float
    age: int
    regime: str
    params: dict
    burst_times: np.ndarray
    burst_freqs: np.ndarray
    burst_labels: np.ndarray


def _poisson_burst_onsets(duration, rate, n_cycles_each, freq_each, sfreq, rng):
    """Generate burst onset times via Poisson process, avoiding overlaps.

    Parameters
    ----------
    duration : float
        Epoch duration in seconds.
    rate : float
        Mean burst rate (bursts/second).
    n_cycles_each : callable
        Function returning number of cycles for each burst.
    freq_each : callable
        Function returning representative frequency for duration estimation.
    sfreq : float
        Sampling frequency.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    onsets : list of float
        Burst onset times in seconds.
    n_cycles_list : list of int
        Number of cycles per burst.
    """
    onsets = []
    n_cycles_list = []
    t = rng.exponential(1.0 / rate) if rate > 0 else duration + 1

    while t < duration:
        nc = n_cycles_each()
        f = freq_each()
        burst_dur = nc / f
        if t + burst_dur <= duration:
            onsets.append(t)
            n_cycles_list.append(nc)
        t += burst_dur + rng.exponential(1.0 / rate)

    return onsets, n_cycles_list


def _make_chirped_burst(n_cycles, f_low, f_high, sfreq):
    """Generate a single chirped burst: linear frequency sweep with Gaussian envelope.

    Parameters
    ----------
    n_cycles : int
        Number of oscillation cycles in the burst.
    f_low : float
        Start frequency (Hz).
    f_high : float
        End frequency (Hz).
    sfreq : float
        Sampling frequency.

    Returns
    -------
    burst : ndarray
        Burst waveform.
    cycle_freqs : ndarray
        Frequency at each cycle.
    """
    cycle_freqs = np.linspace(f_low, f_high, n_cycles)

    samples = []
    for freq in cycle_freqs:
        n_samples_cycle = int(round(sfreq / freq))
        t = np.arange(n_samples_cycle) / sfreq
        cycle = np.sin(2 * np.pi * freq * t)
        samples.append(cycle)

    burst = np.concatenate(samples)
    envelope = windows.gaussian(len(burst), std=len(burst) / 6.0)
    burst = burst * envelope

    return burst, cycle_freqs


def _make_constant_burst(n_cycles, freq, sfreq, freq_jitter=0.0, rng=None):
    """Generate a burst at a nominal frequency with optional cycle-to-cycle jitter.

    Parameters
    ----------
    n_cycles : int
        Number of oscillation cycles.
    freq : float
        Center oscillation frequency (Hz).
    sfreq : float
        Sampling frequency.
    freq_jitter : float
        Cycle-to-cycle frequency jitter (std, Hz).
    rng : numpy.random.Generator, optional
        Random number generator for jitter.

    Returns
    -------
    burst : ndarray
        Burst waveform.
    """
    if freq_jitter > 0 and rng is not None:
        samples = []
        for _ in range(n_cycles):
            cycle_freq = max(freq + rng.normal(0, freq_jitter), 1.0)
            n_samp = int(round(sfreq / cycle_freq))
            t = np.arange(n_samp) / sfreq
            samples.append(np.sin(2 * np.pi * cycle_freq * t))
        burst = np.concatenate(samples)
    else:
        n_samples_cycle = int(round(sfreq / freq))
        n_samples = n_samples_cycle * n_cycles
        t = np.arange(n_samples) / sfreq
        burst = np.sin(2 * np.pi * freq * t)

    envelope = windows.gaussian(len(burst), std=len(burst) / 6.0)
    burst = burst * envelope
    return burst


def _add_burst_to_signal(signal, burst, onset_sample, snr_db, rng):
    """Add a burst to the signal at a given onset, scaled by SNR.

    Parameters
    ----------
    signal : ndarray
        Background signal (modified in place).
    burst : ndarray
        Burst waveform to add.
    onset_sample : int
        Sample index for burst onset.
    snr_db : float
        Target SNR in dB relative to local background RMS.
    rng : numpy.random.Generator
        Random number generator (unused, kept for interface consistency).

    Returns
    -------
    end_sample : int
        Sample index of burst end.
    """
    end_sample = min(onset_sample + len(burst), len(signal))
    burst_segment = burst[:end_sample - onset_sample]

    local_bg = signal[onset_sample:end_sample]
    bg_rms = np.sqrt(np.mean(local_bg ** 2)) if np.any(local_bg) else 1e-6
    burst_rms = np.sqrt(np.mean(burst_segment ** 2)) if np.any(burst_segment) else 1e-6

    target_rms = bg_rms * (10 ** (snr_db / 20.0))
    scale = target_rms / burst_rms if burst_rms > 0 else 1.0

    signal[onset_sample:end_sample] += burst_segment * scale
    return end_sample


def simulate_chirp(age, config=None, rng=None):
    """Regime 1 (Hypothesis A): within-burst chirp between theta and alpha.

    Chirp direction is controlled by config chirp.direction:
      "up"   — theta→alpha (frequency rises within burst)
      "down" — alpha→theta (frequency falls within burst)
      "both" — each burst randomly assigned up or down

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    sfreq = get_sfreq(cfg)
    duration = get_epoch_duration(cfg)
    f_low = cfg["chirp"]["f_low"]
    f_high = paf(age, config=cfg)
    direction = cfg["chirp"].get("direction", "both")
    rate = burst_rate(cfg)
    snr_db = burst_snr(age, config=cfg)
    cf = chirp_fraction(age, config=cfg)
    freq_jitter = cfg["burst"].get("freq_jitter_hz", 0.0)

    signal = generate_aperiodic(age, duration=duration, sfreq=sfreq,
                                config=cfg, rng=rng)

    mean_freq = (f_low + f_high) / 2.0
    onsets, n_cycles_list = _poisson_burst_onsets(
        duration, rate,
        n_cycles_each=lambda: burst_n_cycles(age=age, rng=rng, config=cfg),
        freq_each=lambda: mean_freq,
        sfreq=sfreq, rng=rng,
    )

    burst_times = []
    burst_freqs = []
    burst_labels = []

    for onset_t, nc in zip(onsets, n_cycles_list):
        is_chirp = rng.random() < cf

        if is_chirp:
            if direction == "up":
                sweep_up = True
            elif direction == "down":
                sweep_up = False
            else:
                sweep_up = rng.random() < 0.5

            if sweep_up:
                burst, cycle_freqs = _make_chirped_burst(nc, f_low, f_high, sfreq)
                label = "chirp_up"
            else:
                burst, cycle_freqs = _make_chirped_burst(nc, f_high, f_low, sfreq)
                label = "chirp_down"
        else:
            burst = _make_constant_burst(nc, f_high, sfreq,
                                         freq_jitter=freq_jitter, rng=rng)
            cycle_freqs = np.full(nc, f_high)
            label = "stable_alpha"

        onset_sample = int(round(onset_t * sfreq))

        if onset_sample + len(burst) > len(signal):
            continue

        _add_burst_to_signal(signal, burst, onset_sample, snr_db, rng)

        burst_times.append(onset_t)
        burst_freqs.append(cycle_freqs)
        burst_labels.append(label)

    params = {
        "f_low": f_low,
        "f_high": f_high,
        "direction": direction,
        "chirp_fraction": cf,
        "burst_rate": rate,
        "snr_db": snr_db,
        "n_bursts": len(burst_times),
        "n_chirp": sum(1 for l in burst_labels if l.startswith("chirp")),
        "n_stable": sum(1 for l in burst_labels if l == "stable_alpha"),
    }

    return SimulatedEpoch(
        data=signal,
        sfreq=sfreq,
        age=age,
        regime="chirp",
        params=params,
        burst_times=np.array(burst_times),
        burst_freqs=np.array(burst_freqs, dtype=object),
        burst_labels=np.array(burst_labels),
    )


def simulate_mixture(age, config=None, rng=None):
    """Regime 2 (Hypothesis B): distinct theta and alpha burst populations.

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    sfreq = get_sfreq(cfg)
    duration = get_epoch_duration(cfg)
    theta_freq = cfg["mixture"]["theta_freq"]
    alpha_freq = paf(age, config=cfg)
    rate = burst_rate(cfg)
    snr_db = burst_snr(age, config=cfg)

    theta_weight = mixture_theta_weight(age, config=cfg)
    freq_jitter = cfg["burst"].get("freq_jitter_hz", 0.0)

    signal = generate_aperiodic(age, duration=duration, sfreq=sfreq,
                                config=cfg, rng=rng)

    mean_freq = theta_weight * theta_freq + (1 - theta_weight) * alpha_freq
    onsets, n_cycles_list = _poisson_burst_onsets(
        duration, rate,
        n_cycles_each=lambda: burst_n_cycles(age=age, rng=rng, config=cfg),
        freq_each=lambda: mean_freq,
        sfreq=sfreq, rng=rng,
    )

    burst_times = []
    burst_freqs = []
    burst_labels = []

    for onset_t, nc in zip(onsets, n_cycles_list):
        is_theta = rng.random() < theta_weight
        freq = theta_freq if is_theta else alpha_freq
        label = "theta" if is_theta else "alpha"

        burst = _make_constant_burst(nc, freq, sfreq, freq_jitter=freq_jitter, rng=rng)
        onset_sample = int(round(onset_t * sfreq))

        if onset_sample + len(burst) > len(signal):
            continue

        _add_burst_to_signal(signal, burst, onset_sample, snr_db, rng)

        burst_times.append(onset_t)
        burst_freqs.append(np.full(nc, freq))
        burst_labels.append(label)

    params = {
        "theta_freq": theta_freq,
        "alpha_freq": alpha_freq,
        "theta_weight": theta_weight,
        "burst_rate": rate,
        "snr_db": snr_db,
        "n_bursts": len(burst_times),
        "n_theta": sum(1 for l in burst_labels if l == "theta"),
        "n_alpha": sum(1 for l in burst_labels if l == "alpha"),
    }

    return SimulatedEpoch(
        data=signal,
        sfreq=sfreq,
        age=age,
        regime="mixture",
        params=params,
        burst_times=np.array(burst_times),
        burst_freqs=np.array(burst_freqs, dtype=object),
        burst_labels=np.array(burst_labels),
    )


def simulate_drift(age, config=None, rng=None):
    """Regime 3 (control): single oscillator at PAF(age), no chirp.

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    sfreq = get_sfreq(cfg)
    duration = get_epoch_duration(cfg)
    center_freq = paf(age, config=cfg)
    freq_jitter = cfg["drift"]["freq_jitter_hz"]
    rate = burst_rate(cfg)
    snr_db = burst_snr(age, config=cfg)

    signal = generate_aperiodic(age, duration=duration, sfreq=sfreq,
                                config=cfg, rng=rng)

    onsets, n_cycles_list = _poisson_burst_onsets(
        duration, rate,
        n_cycles_each=lambda: burst_n_cycles(age=age, rng=rng, config=cfg),
        freq_each=lambda: center_freq,
        sfreq=sfreq, rng=rng,
    )

    burst_times = []
    burst_freqs = []
    burst_labels = []

    for onset_t, nc in zip(onsets, n_cycles_list):
        freq = center_freq + rng.normal(0, freq_jitter)
        freq = max(freq, 1.0)

        cycle_jitter = cfg["burst"].get("freq_jitter_hz", 0.0)
        burst = _make_constant_burst(nc, freq, sfreq, freq_jitter=cycle_jitter, rng=rng)
        onset_sample = int(round(onset_t * sfreq))

        if onset_sample + len(burst) > len(signal):
            continue

        _add_burst_to_signal(signal, burst, onset_sample, snr_db, rng)

        burst_times.append(onset_t)
        burst_freqs.append(np.full(nc, freq))
        burst_labels.append("drift")

    params = {
        "center_freq": center_freq,
        "freq_jitter": freq_jitter,
        "burst_rate": rate,
        "snr_db": snr_db,
        "n_bursts": len(burst_times),
    }

    return SimulatedEpoch(
        data=signal,
        sfreq=sfreq,
        age=age,
        regime="drift",
        params=params,
        burst_times=np.array(burst_times),
        burst_freqs=np.array(burst_freqs, dtype=object),
        burst_labels=np.array(burst_labels),
    )


def simulate_cooccur(age, config=None, rng=None):
    """Regime 4: theta and alpha bursts with controlled temporal co-occurrence.

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    sfreq = get_sfreq(cfg)
    duration = get_epoch_duration(cfg)
    theta_freq = cfg["cooccur"]["theta_freq"]
    alpha_freq = paf(age, config=cfg)
    cooccur_prob = cfg["cooccur"]["cooccurrence_prob"]
    rate = burst_rate(cfg)
    snr_db = burst_snr(age, config=cfg)
    freq_jitter = cfg["burst"].get("freq_jitter_hz", 0.0)

    signal = generate_aperiodic(age, duration=duration, sfreq=sfreq,
                                config=cfg, rng=rng)

    onsets, n_cycles_list = _poisson_burst_onsets(
        duration, rate,
        n_cycles_each=lambda: burst_n_cycles(age=age, rng=rng, config=cfg),
        freq_each=lambda: alpha_freq,
        sfreq=sfreq, rng=rng,
    )

    burst_times = []
    burst_freqs = []
    burst_labels = []

    for onset_t, nc in zip(onsets, n_cycles_list):
        burst_alpha = _make_constant_burst(nc, alpha_freq, sfreq,
                                           freq_jitter=freq_jitter, rng=rng)
        onset_sample = int(round(onset_t * sfreq))

        if onset_sample + len(burst_alpha) > len(signal):
            continue

        _add_burst_to_signal(signal, burst_alpha, onset_sample, snr_db, rng)
        burst_times.append(onset_t)
        burst_freqs.append(np.full(nc, alpha_freq))
        burst_labels.append("alpha")

        if rng.random() < cooccur_prob:
            nc_theta = burst_n_cycles(age=age, rng=rng, config=cfg)
            burst_theta = _make_constant_burst(nc_theta, theta_freq, sfreq,
                                               freq_jitter=freq_jitter, rng=rng)

            if onset_sample + len(burst_theta) <= len(signal):
                _add_burst_to_signal(signal, burst_theta, onset_sample, snr_db, rng)
                burst_times.append(onset_t)
                burst_freqs.append(np.full(nc_theta, theta_freq))
                burst_labels.append("theta_cooccur")

    params = {
        "theta_freq": theta_freq,
        "alpha_freq": alpha_freq,
        "cooccurrence_prob": cooccur_prob,
        "burst_rate": rate,
        "snr_db": snr_db,
        "n_bursts": len(burst_times),
        "n_alpha": sum(1 for l in burst_labels if l == "alpha"),
        "n_theta_cooccur": sum(1 for l in burst_labels if l == "theta_cooccur"),
    }

    return SimulatedEpoch(
        data=signal,
        sfreq=sfreq,
        age=age,
        regime="cooccur",
        params=params,
        burst_times=np.array(burst_times),
        burst_freqs=np.array(burst_freqs, dtype=object),
        burst_labels=np.array(burst_labels),
    )


def _make_broadband_burst(n_cycles, center_freq, bandwidth_hz, sfreq, rng):
    """Generate a broadband burst with frequency jitter around center_freq.

    Parameters
    ----------
    n_cycles : int
        Number of oscillation cycles.
    center_freq : float
        Center frequency (Hz).
    bandwidth_hz : float
        Frequency bandwidth (std of IF jitter, Hz).
    sfreq : float
        Sampling frequency.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    burst : ndarray
        Burst waveform.
    mean_freq : float
        Mean instantaneous frequency of the burst.
    """
    n_samples_cycle = int(round(sfreq / center_freq))
    n_samples = n_samples_cycle * n_cycles
    t = np.arange(n_samples) / sfreq

    freq_mod = center_freq + rng.normal(0, bandwidth_hz, size=n_samples)
    freq_mod = np.maximum(freq_mod, 1.0)
    phase = np.cumsum(2 * np.pi * freq_mod / sfreq)
    burst = np.sin(phase)

    envelope = windows.gaussian(n_samples, std=n_samples / 6.0)
    burst = burst * envelope

    return burst, float(np.mean(freq_mod))


def simulate_narrowing(age, config=None, rng=None):
    """Regime 5: single oscillator with age-dependent bandwidth narrowing.

    Parameters
    ----------
    age : int or float
        Simulated age in years.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    sfreq = get_sfreq(cfg)
    duration = get_epoch_duration(cfg)
    center_freq = paf(age, config=cfg)
    bw = narrowing_bandwidth(age, config=cfg)
    rate = burst_rate(cfg)
    snr_db = burst_snr(age, config=cfg)

    signal = generate_aperiodic(age, duration=duration, sfreq=sfreq,
                                config=cfg, rng=rng)

    onsets, n_cycles_list = _poisson_burst_onsets(
        duration, rate,
        n_cycles_each=lambda: burst_n_cycles(age=age, rng=rng, config=cfg),
        freq_each=lambda: center_freq,
        sfreq=sfreq, rng=rng,
    )

    burst_times = []
    burst_freqs = []
    burst_labels = []

    for onset_t, nc in zip(onsets, n_cycles_list):
        burst, mean_freq = _make_broadband_burst(nc, center_freq, bw, sfreq, rng)
        onset_sample = int(round(onset_t * sfreq))

        if onset_sample + len(burst) > len(signal):
            continue

        _add_burst_to_signal(signal, burst, onset_sample, snr_db, rng)

        burst_times.append(onset_t)
        burst_freqs.append(np.array([mean_freq]))
        burst_labels.append("narrowing")

    params = {
        "center_freq": center_freq,
        "bandwidth_hz": bw,
        "burst_rate": rate,
        "snr_db": snr_db,
        "n_bursts": len(burst_times),
    }

    return SimulatedEpoch(
        data=signal,
        sfreq=sfreq,
        age=age,
        regime="narrowing",
        params=params,
        burst_times=np.array(burst_times),
        burst_freqs=np.array(burst_freqs, dtype=object),
        burst_labels=np.array(burst_labels),
    )
