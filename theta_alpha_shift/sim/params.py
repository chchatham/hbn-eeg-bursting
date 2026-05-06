"""
Developmental parameter curves for theta-alpha shift simulations.

Loads simulation parameters from configs/sim_params.yaml and provides
interpolation functions for age-dependent quantities (PAF, aperiodic
exponent, mixture weights, bandwidth).

Usage:
    from theta_alpha_shift.sim.params import load_config, paf, aperiodic_exponent
    cfg = load_config()
    freq = paf(age=8)
"""

from pathlib import Path

import numpy as np
import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "sim_params.yaml"
_cached_config = None


def load_config(path=None):
    """Load simulation parameters from YAML config."""
    global _cached_config
    if path is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    if _cached_config is None:
        with open(_CONFIG_PATH) as f:
            _cached_config = yaml.safe_load(f)
    return _cached_config


def _interp(age, ages, values):
    """Linear interpolation with clamping at boundaries."""
    return float(np.interp(age, ages, values))


def paf(age, config=None):
    """Peak alpha frequency (Hz) from Freschl et al. 2022 meta-analytic curve."""
    cfg = config or load_config()
    curve = cfg["paf_curve"]
    return _interp(age, curve["ages"], curve["freqs"])


def aperiodic_exponent(age, config=None):
    """Aperiodic exponent from Cellier et al. 2021 developmental trajectory."""
    cfg = config or load_config()
    ap = cfg["aperiodic"]["exponent"]
    return _interp(age, ap["ages"], ap["values"])


def aperiodic_offset(config=None):
    """Aperiodic offset (log10 power units)."""
    cfg = config or load_config()
    return cfg["aperiodic"]["offset"]


def knee_freq(config=None):
    """Knee frequency (Hz) for aperiodic spectrum."""
    cfg = config or load_config()
    return cfg["aperiodic"]["knee_freq"]


def mixture_theta_weight(age, config=None):
    """Theta burst weight for Regime 2 (mixture) at a given age."""
    cfg = config or load_config()
    mw = cfg["mixture"]["theta_weight"]
    return _interp(age, mw["ages"], mw["weights"])


def narrowing_bandwidth(age, config=None):
    """Oscillation bandwidth (Hz) for Regime 5 (broadband narrowing) at a given age."""
    cfg = config or load_config()
    bw = cfg["narrowing"]["bandwidth"]
    return _interp(age, bw["ages"], bw["sigma_hz"])


def burst_n_cycles(age=None, rng=None, config=None):
    """Sample number of cycles per burst from uniform distribution.

    Supports both scalar [lo, hi] and age-dependent config formats.
    """
    cfg = config or load_config()
    if rng is None:
        rng = np.random.default_rng()
    ncr = cfg["burst"]["n_cycles_range"]
    if isinstance(ncr, dict):
        if age is None:
            age = 11
        lo = int(round(_interp(age, ncr["ages"], ncr["lo"])))
        hi = int(round(_interp(age, ncr["ages"], ncr["hi"])))
    else:
        lo, hi = ncr
    return rng.integers(lo, hi + 1)


def burst_snr(age=None, config=None):
    """Burst SNR in dB. Supports scalar or age-dependent config."""
    cfg = config or load_config()
    snr = cfg["burst"]["snr_db"]
    if isinstance(snr, dict):
        if age is None:
            age = 11
        return _interp(age, snr["ages"], snr["values"])
    return float(snr)


def chirp_fraction(age, config=None):
    """Fraction of bursts that chirp (vs. stable at PAF) in Regime 1."""
    cfg = config or load_config()
    cf = cfg["chirp"].get("chirp_fraction")
    if cf is None:
        return 1.0
    if isinstance(cf, dict):
        return _interp(age, cf["ages"], cf["values"])
    return float(cf)


def burst_rate(config=None):
    """Mean burst rate (bursts/second)."""
    cfg = config or load_config()
    return cfg["burst"]["rate"]


def get_ages(config=None):
    """Return list of simulated ages."""
    cfg = config or load_config()
    return cfg["ages"]


def get_sfreq(config=None):
    """Return sampling frequency (Hz)."""
    cfg = config or load_config()
    return cfg["sfreq"]


def get_epoch_duration(config=None):
    """Return epoch duration (seconds)."""
    cfg = config or load_config()
    return cfg["epoch_duration"]


def get_random_seed(config=None):
    """Return global random seed."""
    cfg = config or load_config()
    return cfg["random_seed"]
