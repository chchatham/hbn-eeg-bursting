"""Flask backend for the theta-alpha shift interactive report and simulator."""

import copy
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.signal import welch

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from theta_alpha_shift.sim.aperiodic import generate_aperiodic
from theta_alpha_shift.sim.params import paf, aperiodic_exponent, load_config
from theta_alpha_shift.sim.regimes import (
    simulate_chirp,
    simulate_mixture,
    simulate_drift,
    simulate_cooccur,
    simulate_narrowing,
)
from theta_alpha_shift.methods.bycycle_wrap import run_bycycle

app = Flask(__name__)

SFREQ = 500.0
EPOCH_DURATION = 2.0
MAX_EPOCHS = 50
APERIODIC_SEED_OFFSET = 100000

REGIME_FUNCS = {
    'chirp': simulate_chirp,
    'mixture': simulate_mixture,
    'drift': simulate_drift,
    'cooccur': simulate_cooccur,
    'narrowing': simulate_narrowing,
}


def _build_config(params):
    """Build a simulation config from request parameters."""
    config = copy.deepcopy(load_config())
    config['burst']['rate'] = float(params.get('burst_rate', 2.0))
    config['burst']['snr_db'] = float(params.get('snr_db', 8.0))
    config['burst']['n_cycles_range'] = [
        int(params.get('n_cycles_min', 2)),
        int(params.get('n_cycles_max', 5)),
    ]
    config['epoch_duration'] = EPOCH_DURATION
    config['sfreq'] = SFREQ
    return config


def _run_multi_epoch(regime, age, config, n_epochs, base_seed):
    """Generate n_epochs, return concatenated signal, averaged PSD, aperiodic PSD, and burst info."""
    regime_func = REGIME_FUNCS[regime]
    all_data = []
    all_bursts = []
    psd_accum = None
    psd_aperiodic_accum = None

    for i in range(n_epochs):
        config['random_seed'] = base_seed + i
        epoch = regime_func(age=age, config=config)
        data = epoch.data if epoch.data.ndim == 1 else epoch.data[0]
        all_data.append(data)

        nperseg = min(len(data), int(SFREQ))
        freqs, psd_i = welch(data, fs=SFREQ, nperseg=nperseg)
        if psd_accum is None:
            psd_accum = psd_i
        else:
            psd_accum += psd_i

        rng_ap = np.random.default_rng(APERIODIC_SEED_OFFSET + base_seed + i)
        ap_signal = generate_aperiodic(
            age, duration=EPOCH_DURATION, sfreq=SFREQ, config=config, rng=rng_ap
        )
        _, psd_ap_i = welch(ap_signal, fs=SFREQ, nperseg=nperseg)
        if psd_aperiodic_accum is None:
            psd_aperiodic_accum = psd_ap_i
        else:
            psd_aperiodic_accum += psd_ap_i

        time_offset = i * EPOCH_DURATION
        if epoch.burst_times is not None:
            for j in range(len(epoch.burst_times)):
                info = {'onset': float(epoch.burst_times[j]) + time_offset}
                if epoch.burst_freqs is not None and j < len(epoch.burst_freqs):
                    freq_val = epoch.burst_freqs[j]
                    if hasattr(freq_val, '__len__'):
                        info['freq'] = [float(f) for f in freq_val]
                    else:
                        info['freq'] = float(freq_val)
                if epoch.burst_labels is not None and j < len(epoch.burst_labels):
                    info['label'] = str(epoch.burst_labels[j])
                all_bursts.append(info)

    psd_avg = psd_accum / n_epochs
    psd_aperiodic_avg = psd_aperiodic_accum / n_epochs
    freq_mask = freqs <= 45
    freqs = freqs[freq_mask]
    psd_avg = psd_avg[freq_mask]
    psd_aperiodic_avg = psd_aperiodic_avg[freq_mask]

    signal = np.concatenate(all_data)
    t = np.arange(len(signal)) / SFREQ

    return t, signal, freqs, psd_avg, psd_aperiodic_avg, all_bursts


def _compute_period_slopes(signal, sfreq):
    """Run bycycle on the signal and return period slope info."""
    try:
        result = run_bycycle((signal, sfreq))
        slopes = result.metadata.get("period_slopes", [])
        return {
            'mean_period_slope': float(result.headline_stat),
            'period_slopes': [float(s) for s in slopes],
            'n_bursts_bycycle': result.metadata.get("n_bursts", 0),
            'n_cycles_total': result.metadata.get("n_cycles_total", 0),
        }
    except Exception:
        return {
            'mean_period_slope': 0.0,
            'period_slopes': [],
            'n_bursts_bycycle': 0,
            'n_cycles_total': 0,
        }


@app.route('/')
def about():
    return render_template('about.html')


@app.route('/simulator')
def simulator():
    config = load_config()
    return render_template('simulator.html', config=config)


@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    params = request.get_json()
    age = int(params.get('age', 11))
    regime = params.get('regime', 'chirp')
    n_epochs = min(int(params.get('n_epochs', 20)), MAX_EPOCHS)
    seed = int(params.get('seed', 42))

    if regime not in REGIME_FUNCS:
        return jsonify({'error': f'Unknown regime: {regime}'}), 400
    if not 5 <= age <= 20:
        return jsonify({'error': 'Age must be between 5 and 20'}), 400
    if n_epochs < 1:
        return jsonify({'error': 'n_epochs must be >= 1'}), 400

    start = time.time()
    config = _build_config(params)
    t, signal, freqs, psd_avg, psd_ap_avg, burst_info = _run_multi_epoch(
        regime, age, config, n_epochs, seed
    )

    log_psd = np.log10(psd_avg)
    log_aperiodic = np.log10(psd_ap_avg)
    residual = log_psd - log_aperiodic

    bycycle_stats = _compute_period_slopes(signal, SFREQ)

    elapsed = time.time() - start

    paf_val = paf(age)
    exponent = aperiodic_exponent(age)

    return jsonify({
        'time': t.tolist(),
        'signal': signal.tolist(),
        'freqs': freqs.tolist(),
        'psd': log_psd.tolist(),
        'psd_aperiodic': log_aperiodic.tolist(),
        'residual': residual.tolist(),
        'bursts': burst_info,
        'bycycle': bycycle_stats,
        'params': {
            'age': age,
            'regime': regime,
            'paf': paf_val,
            'exponent': exponent,
            'burst_rate': float(params.get('burst_rate', 2.0)),
            'snr_db': float(params.get('snr_db', 8.0)),
            'n_cycles': [
                int(params.get('n_cycles_min', 2)),
                int(params.get('n_cycles_max', 5)),
            ],
            'n_epochs': n_epochs,
            'seed': seed,
        },
        'stats': {
            'n_bursts': len(burst_info),
            'mean_freq': float(np.mean([
                b['freq'] if isinstance(b.get('freq'), (int, float))
                else np.mean(b['freq'])
                for b in burst_info
            ])) if burst_info else 0.0,
            'elapsed_s': round(elapsed, 3),
            'total_duration_s': round(n_epochs * EPOCH_DURATION, 1),
        },
    })


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """Run two regimes side-by-side for comparison mode."""
    params = request.get_json()
    results = []
    for i, regime_params in enumerate(params.get('regimes', [])):
        if i >= 2:
            break
        age = int(regime_params.get('age', 11))
        regime = regime_params.get('regime', 'chirp')
        seed = int(regime_params.get('seed', 42))
        n_epochs = min(int(regime_params.get('n_epochs', 20)), MAX_EPOCHS)

        if regime not in REGIME_FUNCS:
            return jsonify({'error': f'Unknown regime: {regime}'}), 400

        config = _build_config(regime_params)
        t, signal, freqs, psd_avg, psd_ap_avg, burst_info = _run_multi_epoch(
            regime, age, config, n_epochs, seed
        )

        log_psd = np.log10(psd_avg)
        log_aperiodic = np.log10(psd_ap_avg)
        residual = log_psd - log_aperiodic

        bycycle_stats = _compute_period_slopes(signal, SFREQ)

        results.append({
            'time': t.tolist(),
            'signal': signal.tolist(),
            'freqs': freqs.tolist(),
            'psd': log_psd.tolist(),
            'psd_aperiodic': log_aperiodic.tolist(),
            'residual': residual.tolist(),
            'bursts': burst_info,
            'bycycle': bycycle_stats,
            'regime': regime,
            'age': age,
        })

    return jsonify({'results': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
