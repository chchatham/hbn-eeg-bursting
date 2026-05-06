"""Flask backend for the theta-alpha shift interactive report and simulator."""

import copy
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', '..')))

from theta_alpha_shift.sim.params import paf, aperiodic_exponent, load_config
from theta_alpha_shift.sim.regimes import (
    simulate_chirp,
    simulate_mixture,
    simulate_drift,
    simulate_cooccur,
    simulate_narrowing,
)

app = Flask(__name__)

REGIME_FUNCS = {
    'chirp': simulate_chirp,
    'mixture': simulate_mixture,
    'drift': simulate_drift,
    'cooccur': simulate_cooccur,
    'narrowing': simulate_narrowing,
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
    burst_rate = float(params.get('burst_rate', 3.0))
    snr_db = float(params.get('snr_db', 8.0))
    n_cycles_min = int(params.get('n_cycles_min', 2))
    n_cycles_max = int(params.get('n_cycles_max', 5))
    epoch_duration = float(params.get('epoch_duration', 2.0))
    sfreq = 500.0
    seed = int(params.get('seed', 42))

    if regime not in REGIME_FUNCS:
        return jsonify({'error': f'Unknown regime: {regime}'}), 400
    if not 5 <= age <= 20:
        return jsonify({'error': 'Age must be between 5 and 20'}), 400
    if epoch_duration > 5.0:
        return jsonify({'error': 'Epoch duration must be <= 5s for interactive use'}), 400

    start = time.time()

    config = copy.deepcopy(load_config())
    config['burst']['rate'] = burst_rate
    config['burst']['snr_db'] = snr_db
    config['burst']['n_cycles_range'] = [n_cycles_min, n_cycles_max]
    config['epoch_duration'] = epoch_duration
    config['sfreq'] = sfreq
    config['random_seed'] = seed

    regime_func = REGIME_FUNCS[regime]
    epoch = regime_func(age=age, config=config)

    data = epoch.data if epoch.data.ndim == 1 else epoch.data[0]
    n_samples = len(data)
    t = np.arange(n_samples) / sfreq

    from scipy.signal import welch
    nperseg = min(n_samples, int(sfreq))
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg)
    freq_mask = freqs <= 45
    freqs = freqs[freq_mask]
    psd = psd[freq_mask]

    paf_val = paf(age)
    exponent = aperiodic_exponent(age)

    burst_info = []
    if epoch.burst_times is not None and len(epoch.burst_times) > 0:
        for i in range(len(epoch.burst_times)):
            info = {'onset': float(epoch.burst_times[i])}
            if epoch.burst_freqs is not None and i < len(epoch.burst_freqs):
                freq_val = epoch.burst_freqs[i]
                if hasattr(freq_val, '__len__'):
                    info['freq'] = [float(f) for f in freq_val]
                else:
                    info['freq'] = float(freq_val)
            if epoch.burst_labels is not None and i < len(epoch.burst_labels):
                info['label'] = str(epoch.burst_labels[i])
            burst_info.append(info)

    elapsed = time.time() - start

    return jsonify({
        'time': t.tolist(),
        'signal': data.tolist(),
        'freqs': freqs.tolist(),
        'psd': np.log10(psd).tolist(),
        'bursts': burst_info,
        'params': {
            'age': age,
            'regime': regime,
            'paf': paf_val,
            'exponent': exponent,
            'burst_rate': burst_rate,
            'snr_db': snr_db,
            'n_cycles': [n_cycles_min, n_cycles_max],
            'epoch_duration': epoch_duration,
            'seed': seed,
        },
        'stats': {
            'n_bursts': len(burst_info),
            'mean_freq': float(np.mean([b['freq'] if isinstance(b.get('freq'), (int, float)) else np.mean(b['freq']) for b in burst_info])) if burst_info else 0.0,
            'elapsed_s': round(elapsed, 3),
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

        import copy
        config = copy.deepcopy(load_config())
        config['burst']['rate'] = float(regime_params.get('burst_rate', 3.0))
        config['burst']['snr_db'] = float(regime_params.get('snr_db', 8.0))
        config['burst']['n_cycles_range'] = [
            int(regime_params.get('n_cycles_min', 2)),
            int(regime_params.get('n_cycles_max', 5)),
        ]
        config['epoch_duration'] = float(regime_params.get('epoch_duration', 2.0))
        config['sfreq'] = 500.0
        config['random_seed'] = seed

        if regime not in REGIME_FUNCS:
            return jsonify({'error': f'Unknown regime: {regime}'}), 400

        epoch = REGIME_FUNCS[regime](age=age, config=config)
        data = epoch.data if epoch.data.ndim == 1 else epoch.data[0]
        n_samples = len(data)
        t = np.arange(n_samples) / 500.0

        from scipy.signal import welch
        nperseg = min(n_samples, 500)
        freqs, psd = welch(data, fs=500.0, nperseg=nperseg)
        freq_mask = freqs <= 45
        freqs = freqs[freq_mask]
        psd = psd[freq_mask]

        results.append({
            'time': t.tolist(),
            'signal': data.tolist(),
            'freqs': freqs.tolist(),
            'psd': np.log10(psd).tolist(),
            'regime': regime,
            'age': age,
        })

    return jsonify({'results': results})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', '0') == '1')
