"""
128-channel forward model projecting simulated source signals to scalp electrodes.

Uses a simplified spherical head model with occipital dipole sources and the
GSN-HydroCel-129 montage matching HBN data. Projects 1D simulated epochs to
multi-channel signals with realistic spatial structure.

Usage:
    from theta_alpha_shift.sim.forward_model import forward_project
    epoch_1d = simulate_chirp(age=8)
    epoch_128ch = forward_project(epoch_1d)
"""

import numpy as np

from theta_alpha_shift.sim.params import get_random_seed, load_config

_cached_lead_field = None
_cached_lead_field_key = None


def _get_montage_positions(montage_name):
    """Get electrode positions from an MNE standard montage.

    Parameters
    ----------
    montage_name : str
        MNE montage name (e.g., 'GSN-HydroCel-129').

    Returns
    -------
    ch_names : list of str
        Channel names.
    positions : ndarray, shape (n_channels, 3)
        Electrode positions in meters.
    """
    import mne
    montage = mne.channels.make_standard_montage(montage_name)
    pos_dict = montage.get_positions()["ch_pos"]
    ch_names = list(pos_dict.keys())
    positions = np.array([pos_dict[ch] for ch in ch_names])
    return ch_names, positions


def _get_occipital_sources(n_sources, electrode_positions):
    """Define occipital dipole source positions inside the head.

    Places sources at ~70% depth along the posterior midline and lateral
    occipital positions, matching the typical generator location of posterior
    alpha rhythms.

    Parameters
    ----------
    n_sources : int
        Number of dipole sources.
    electrode_positions : ndarray, shape (n_channels, 3)
        Electrode positions (used to determine head geometry).

    Returns
    -------
    source_positions : ndarray, shape (n_sources, 3)
        Dipole positions in the same coordinate frame as electrodes.
    """
    head_radius = np.max(np.linalg.norm(electrode_positions, axis=1))
    depth_fraction = 0.7

    posterior_y = electrode_positions[:, 1].min()
    center_z = np.mean(electrode_positions[:, 2])

    base_pos = np.array([0.0, posterior_y * depth_fraction, center_z])

    if n_sources == 1:
        return base_pos.reshape(1, 3)

    lateral_spread = head_radius * 0.2
    vertical_spread = head_radius * 0.1

    sources = [base_pos]
    if n_sources >= 2:
        sources.append(base_pos + np.array([-lateral_spread, 0.0, vertical_spread]))
    if n_sources >= 3:
        sources.append(base_pos + np.array([lateral_spread, 0.0, vertical_spread]))
    for i in range(3, n_sources):
        angle = 2 * np.pi * i / n_sources
        offset = np.array([
            lateral_spread * np.cos(angle),
            0.0,
            vertical_spread * np.sin(angle),
        ])
        sources.append(base_pos + offset)

    return np.array(sources[:n_sources])


def _compute_lead_field(electrode_positions, source_positions):
    """Compute lead field matrix using inverse-square distance law.

    Simplified spherical head model: gain from each dipole to each electrode
    is proportional to 1/r^2 where r is the Euclidean distance.

    Parameters
    ----------
    electrode_positions : ndarray, shape (n_channels, 3)
        Electrode positions.
    source_positions : ndarray, shape (n_sources, 3)
        Dipole source positions.

    Returns
    -------
    lead_field : ndarray, shape (n_channels, n_sources)
        Gain matrix mapping sources to electrodes.
    """
    n_channels = electrode_positions.shape[0]
    n_sources = source_positions.shape[0]
    lead_field = np.zeros((n_channels, n_sources))

    for s in range(n_sources):
        distances = np.linalg.norm(
            electrode_positions - source_positions[s], axis=1
        )
        distances = np.maximum(distances, 1e-6)
        lead_field[:, s] = 1.0 / (distances ** 2)

    max_gain = lead_field.max()
    if max_gain > 0:
        lead_field /= max_gain

    return lead_field


def get_lead_field(config=None):
    """Get or compute cached lead field matrix.

    Parameters
    ----------
    config : dict, optional
        Pre-loaded config.

    Returns
    -------
    lead_field : ndarray, shape (n_channels, n_sources)
        Gain matrix.
    ch_names : list of str
        Channel names.
    """
    global _cached_lead_field, _cached_lead_field_key

    cfg = config or load_config()
    fwd_cfg = cfg["forward"]
    cache_key = (fwd_cfg["montage"], fwd_cfg["n_sources"], fwd_cfg["n_channels"])

    if _cached_lead_field is not None and _cached_lead_field_key == cache_key:
        return _cached_lead_field

    ch_names, electrode_positions = _get_montage_positions(fwd_cfg["montage"])

    n_channels = fwd_cfg["n_channels"]
    ch_names = ch_names[:n_channels]
    electrode_positions = electrode_positions[:n_channels]

    source_positions = _get_occipital_sources(
        fwd_cfg["n_sources"], electrode_positions
    )
    lead_field = _compute_lead_field(electrode_positions, source_positions)

    _cached_lead_field = (lead_field, ch_names)
    _cached_lead_field_key = cache_key
    return lead_field, ch_names


def forward_project(epoch, config=None, rng=None):
    """Project a 1D simulated epoch to multi-channel scalp EEG.

    Each source dipole receives the same base signal with small independent
    temporal jitter, then the lead field maps sources to 128 scalp channels.
    Independent sensor noise is added per channel.

    Parameters
    ----------
    epoch : SimulatedEpoch
        Single-channel simulated epoch.
    config : dict, optional
        Pre-loaded config.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    epoch : SimulatedEpoch
        Multi-channel epoch with shape (n_channels, n_samples).
    """
    from theta_alpha_shift.sim.regimes import SimulatedEpoch

    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng(get_random_seed(cfg))

    fwd_cfg = cfg["forward"]
    n_sources = fwd_cfg["n_sources"]

    lead_field, ch_names = get_lead_field(config=cfg)

    source_signal = epoch.data if epoch.data.ndim == 1 else epoch.data[0]
    n_samples = len(source_signal)

    source_signals = np.zeros((n_sources, n_samples))
    for s in range(n_sources):
        shift = rng.integers(-2, 3)
        source_signals[s] = np.roll(source_signal, shift)
        source_signals[s] += rng.normal(0, np.std(source_signal) * 0.05, n_samples)

    scalp_data = lead_field @ source_signals

    noise_level = np.std(scalp_data) * 0.02
    scalp_data += rng.normal(0, noise_level, scalp_data.shape)

    params = {
        **epoch.params,
        "n_channels": len(ch_names),
        "n_sources": n_sources,
        "forward_projected": True,
    }

    return SimulatedEpoch(
        data=scalp_data,
        sfreq=epoch.sfreq,
        age=epoch.age,
        regime=epoch.regime,
        params=params,
        burst_times=epoch.burst_times,
        burst_freqs=epoch.burst_freqs,
        burst_labels=epoch.burst_labels,
    )
