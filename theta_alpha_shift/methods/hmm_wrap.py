"""
Time-Delay Embedded Hidden Markov Model (TDE-HMM) wrapper using hmmlearn.

Implements the TDE-HMM approach (Vidaurre et al. 2018) for detecting latent
oscillatory states. Fits 2-state and 3-state Gaussian HMMs to time-delay
embedded EEG, compares model fit via BIC, and computes per-state spectra.

The headline discriminative statistic is n_oscillatory_states: the number of
HMM states whose per-state spectrum has a clear peak in 4-12 Hz. Mixture
(Hypothesis B) should produce 2 distinct oscillatory states (theta + alpha),
while chirp (Hypothesis A) should produce 1 broadband oscillatory state.
"""

import numpy as np
from scipy.signal import welch
from hmmlearn.hmm import GaussianHMM

from theta_alpha_shift.methods import MethodResult


def _time_delay_embed(signal, n_lags=7):
    """Create time-delay embedded matrix from 1-D signal.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
    n_lags : int
        Number of lag copies (including zero-lag).

    Returns
    -------
    embedded : ndarray, shape (n_samples - n_lags + 1, n_lags)
    """
    n = len(signal)
    out = np.empty((n - n_lags + 1, n_lags))
    for lag in range(n_lags):
        out[:, lag] = signal[lag:n - n_lags + 1 + lag]
    return out


def _pca_reduce(X, n_components=None, variance_retained=0.9):
    """PCA dimensionality reduction with z-scoring.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    n_components : int, optional
        Fixed number of components. Overrides variance_retained.
    variance_retained : float
        Fraction of variance to retain when n_components is None.

    Returns
    -------
    X_reduced : ndarray, shape (n_samples, n_components)
    """
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    explained = (S ** 2) / (S ** 2).sum()

    if n_components is None:
        cumvar = np.cumsum(explained)
        n_components = int(np.searchsorted(cumvar, variance_retained) + 1)
        n_components = max(n_components, 2)

    n_components = min(n_components, X.shape[1])
    X_proj = X_centered @ Vt[:n_components].T
    std = X_proj.std(axis=0)
    std[std < 1e-12] = 1.0
    return X_proj / std


def _fit_hmm(X, n_states, n_iter=100, seed=42):
    """Fit a Gaussian HMM and return model + BIC.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    n_states : int
    n_iter : int
    seed : int

    Returns
    -------
    model : GaussianHMM
    bic : float
    """
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=seed,
        verbose=False,
        min_covar=1e-3,
    )
    model.fit(X)
    log_likelihood = model.score(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_params = (
        n_states - 1                                # start probs
        + n_states * (n_states - 1)                 # transition matrix
        + n_states * n_features                     # means
        + n_states * n_features                     # covariances (diag)
    )
    bic = -2 * log_likelihood + n_params * np.log(n_samples)
    return model, bic


def _per_state_spectrum(signal, state_seq, n_states, sfreq, f_range=(4, 12)):
    """Compute PSD for each HMM state's time segments.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Original (non-embedded) signal, trimmed to match state_seq length.
    state_seq : ndarray, shape (n_samples,)
        Viterbi state assignments.
    n_states : int
    sfreq : float
    f_range : tuple of float

    Returns
    -------
    state_spectra : list of dict
        Per-state spectral info.
    """
    nperseg = min(len(signal), int(sfreq))
    state_spectra = []

    for s in range(n_states):
        mask = state_seq == s
        occupancy = mask.sum() / len(state_seq)

        seg = signal[mask]
        if len(seg) < nperseg:
            nperseg_s = max(len(seg), 16)
        else:
            nperseg_s = nperseg

        if len(seg) < 4:
            state_spectra.append({
                "state": s,
                "occupancy": float(occupancy),
                "peak_freq": 0.0,
                "peak_power": 0.0,
                "has_theta_alpha_peak": False,
                "mean_amplitude": 0.0,
            })
            continue

        freqs, psd = welch(seg, fs=sfreq, nperseg=nperseg_s)
        band_mask = (freqs >= f_range[0]) & (freqs <= f_range[1])

        if band_mask.any() and psd[band_mask].max() > 0:
            peak_idx = np.argmax(psd[band_mask])
            peak_freq = float(freqs[band_mask][peak_idx])
            peak_power = float(psd[band_mask][peak_idx])
            broadband_power = psd[(freqs >= 1) & (freqs <= 30)].mean() if (freqs <= 30).any() else psd.mean()
            has_peak = peak_power > 2.0 * broadband_power
        else:
            peak_freq = 0.0
            peak_power = 0.0
            has_peak = False

        state_spectra.append({
            "state": s,
            "occupancy": float(occupancy),
            "peak_freq": peak_freq,
            "peak_power": peak_power,
            "has_theta_alpha_peak": has_peak,
            "mean_amplitude": float(np.std(seg)),
        })

    return state_spectra


def _count_oscillatory_states(state_spectra, min_freq_sep=1.5):
    """Count spectrally distinct oscillatory states.

    Two states are "distinct" if both have theta-alpha peaks and their
    peak frequencies differ by more than min_freq_sep Hz.

    Parameters
    ----------
    state_spectra : list of dict
    min_freq_sep : float
        Minimum frequency separation (Hz) to count as distinct.

    Returns
    -------
    n_osc : int
        Number of distinct oscillatory states.
    """
    osc_states = [s for s in state_spectra if s["has_theta_alpha_peak"]]
    if len(osc_states) <= 1:
        return len(osc_states)

    osc_states.sort(key=lambda s: s["peak_freq"])
    distinct = [osc_states[0]]
    for s in osc_states[1:]:
        if s["peak_freq"] - distinct[-1]["peak_freq"] >= min_freq_sep:
            distinct.append(s)

    return len(distinct)


def run_hmm(epoch, n_lags=7, n_pca=None, n_states_list=(2, 3),
            f_range=(4, 12), seed=42):
    """Run TDE-HMM and analyze oscillatory states.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    n_lags : int
        Number of time-delay embedding lags.
    n_pca : int, optional
        Number of PCA components. None = auto (90% variance).
    n_states_list : tuple of int
        Number of states to try. Best by BIC is selected.
    f_range : tuple of float
        Theta-alpha frequency range for state classification.
    seed : int
        Random seed.

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

    X_embed = _time_delay_embed(data, n_lags)
    X_pca = _pca_reduce(X_embed, n_components=n_pca)

    best_model = None
    best_bic = np.inf
    best_n = 0
    all_bics = {}

    for n_states in n_states_list:
        model, bic = _fit_hmm(X_pca, n_states, seed=seed)
        all_bics[n_states] = float(bic)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n = n_states

    state_seq = best_model.predict(X_pca)

    trim_offset = n_lags - 1
    signal_trimmed = data[trim_offset:trim_offset + len(state_seq)]

    state_spectra = _per_state_spectrum(
        signal_trimmed, state_seq, best_n, sfreq, f_range
    )

    n_osc = _count_oscillatory_states(state_spectra)

    detected_bursts = []
    for s_info in state_spectra:
        if not s_info["has_theta_alpha_peak"]:
            continue
        state_idx = s_info["state"]
        mask = state_seq == state_idx
        changes = np.diff(mask.astype(int))
        onsets = np.where(changes == 1)[0]
        offsets = np.where(changes == -1)[0]

        for j, onset in enumerate(onsets):
            offset = offsets[j] if j < len(offsets) else len(state_seq) - 1
            if offset <= onset:
                continue
            detected_bursts.append({
                "onset": float((onset + trim_offset) / sfreq),
                "offset": float((offset + trim_offset) / sfreq),
                "frequency": s_info["peak_freq"],
                "amplitude": s_info["mean_amplitude"],
                "state": state_idx,
            })

    bic_diff = all_bics.get(2, 0) - all_bics.get(3, 0) if len(all_bics) > 1 else 0.0

    metadata = {
        "best_n_states": best_n,
        "bic_values": all_bics,
        "bic_diff_2v3": float(bic_diff),
        "state_spectra": state_spectra,
        "n_oscillatory_states": n_osc,
        "n_lags": n_lags,
        "n_pca_components": X_pca.shape[1],
        "f_range": list(f_range),
    }

    return MethodResult(
        method_name="hmm",
        detected_bursts=detected_bursts,
        headline_stat=float(n_osc),
        headline_stat_name="n_oscillatory_states",
        metadata=metadata,
    )
