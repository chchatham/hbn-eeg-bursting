"""
Convolutional Dictionary Learning (CDL) wrapper using alphacsc.

Learns a set of temporal atoms from the signal. The headline discriminative
statistic is the number of atoms whose center frequency falls in the theta-alpha
range (4-12 Hz). Mixture hypothesis should produce 2 distinct atoms (theta + alpha),
while chirp should produce 1 broadband or chirped atom.

Usage:
    from theta_alpha_shift.methods.cdl_wrap import run_cdl
    result = run_cdl(epoch)
"""

import numpy as np
from scipy.signal import welch

from theta_alpha_shift.methods import MethodResult


def _atom_center_freq(atom, sfreq):
    """Compute center frequency of a learned atom via PSD peak."""
    freqs, psd = welch(atom, fs=sfreq, nperseg=min(len(atom), 64))
    valid = freqs > 0
    if not valid.any():
        return 0.0
    peak_idx = np.argmax(psd[valid])
    return float(freqs[valid][peak_idx])


def _atom_bandwidth(atom, sfreq):
    """Estimate bandwidth as half-power width of atom's PSD."""
    freqs, psd = welch(atom, fs=sfreq, nperseg=min(len(atom), 64))
    valid = freqs > 0
    if not valid.any():
        return 0.0
    psd_valid = psd[valid]
    freqs_valid = freqs[valid]
    peak_power = psd_valid.max()
    half_power = peak_power / 2
    above = psd_valid >= half_power
    if above.sum() < 2:
        return 0.0
    indices = np.where(above)[0]
    return float(freqs_valid[indices[-1]] - freqs_valid[indices[0]])


def run_cdl(epoch, n_atoms=4, n_times_atom=None, reg=0.1, n_iter=10,
            f_range=(4, 12)):
    """Run Convolutional Dictionary Learning and analyze learned atoms.

    Parameters
    ----------
    epoch : SimulatedEpoch or tuple of (data, sfreq)
        Input signal.
    n_atoms : int
        Number of atoms to learn.
    n_times_atom : int, optional
        Atom length in samples. Defaults to ~3 cycles at center of f_range.
    reg : float
        Sparsity regularization parameter.
    n_iter : int
        Number of CDL iterations.
    f_range : tuple of float
        Theta-alpha frequency range for counting atoms.

    Returns
    -------
    result : MethodResult
    """
    from alphacsc import BatchCDL

    if isinstance(epoch, tuple):
        data, sfreq = epoch
    else:
        data = epoch.data
        sfreq = epoch.sfreq

    if data.ndim == 1:
        data = data.reshape(1, 1, -1)
    elif data.ndim == 2:
        data = data.reshape(1, data.shape[0], -1)

    if n_times_atom is None:
        center_freq = np.mean(f_range)
        n_times_atom = int(3 * sfreq / center_freq)

    cdl = BatchCDL(
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        reg=reg,
        n_iter=n_iter,
        n_jobs=1,
        solver_z="l-bfgs",
        verbose=0,
    )
    cdl.fit(data)

    n_channels = data.shape[1]
    atoms_temporal = cdl.v_hat_

    atom_info = []
    atoms_in_range = 0

    for i in range(n_atoms):
        atom = atoms_temporal[i]
        cf = _atom_center_freq(atom, sfreq)
        bw = _atom_bandwidth(atom, sfreq)
        in_range = f_range[0] <= cf <= f_range[1]
        if in_range:
            atoms_in_range += 1
        atom_info.append({
            "atom_idx": i,
            "center_freq": cf,
            "bandwidth": bw,
            "in_theta_alpha": in_range,
        })

    activations = cdl.z_hat_
    detected_bursts = []
    if activations is not None:
        for i in range(n_atoms):
            if not atom_info[i]["in_theta_alpha"]:
                continue
            act = activations[0, i, :]
            threshold = np.std(act) * 2
            above = act > threshold
            changes = np.diff(above.astype(int))
            onsets = np.where(changes == 1)[0]
            offsets = np.where(changes == -1)[0]

            for j, onset in enumerate(onsets):
                offset = offsets[j] if j < len(offsets) else len(act) - 1
                if offset <= onset:
                    continue
                detected_bursts.append({
                    "onset": float(onset / sfreq),
                    "offset": float(offset / sfreq),
                    "frequency": atom_info[i]["center_freq"],
                    "amplitude": float(act[onset:offset].max()),
                    "atom_idx": i,
                })

    metadata = {
        "n_atoms": n_atoms,
        "n_atoms_theta_alpha": atoms_in_range,
        "atom_info": atom_info,
        "n_times_atom": n_times_atom,
        "reg": reg,
    }

    return MethodResult(
        method_name="cdl",
        detected_bursts=detected_bursts,
        headline_stat=float(atoms_in_range),
        headline_stat_name="n_cdl_atoms_theta_alpha",
        metadata=metadata,
    )
