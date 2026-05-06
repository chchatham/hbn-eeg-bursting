document.addEventListener('DOMContentLoaded', function() {
    const regimeSelect = document.getElementById('regime-select');
    const ageSlider = document.getElementById('age-slider');
    const ageValue = document.getElementById('age-value');
    const pafDisplay = document.getElementById('paf-display');
    const expDisplay = document.getElementById('exp-display');
    const rateSlider = document.getElementById('burst-rate-slider');
    const rateValue = document.getElementById('rate-value');
    const snrSlider = document.getElementById('snr-slider');
    const snrValue = document.getElementById('snr-value');
    const cyclesMin = document.getElementById('cycles-min');
    const cyclesMax = document.getElementById('cycles-max');
    const cyclesValue = document.getElementById('cycles-value');
    const durationSlider = document.getElementById('duration-slider');
    const durationValue = document.getElementById('duration-value');
    const seedInput = document.getElementById('seed-input');
    const randomSeedBtn = document.getElementById('random-seed-btn');
    const runBtn = document.getElementById('run-btn');
    const compareToggle = document.getElementById('compare-toggle');
    const compareControls = document.getElementById('compare-controls');
    const compareBtn = document.getElementById('compare-btn');
    const loading = document.getElementById('loading');
    const statsPanel = document.getElementById('stats-panel');
    const statsContent = document.getElementById('stats-content');
    const regimeHelp = document.getElementById('regime-help');

    const REGIME_DESCRIPTIONS = {
        'chirp': 'Within-burst frequency sweep from theta to alpha',
        'mixture': 'Two burst populations (theta + alpha) with age-dependent mixing',
        'drift': 'Single oscillator at PAF with trial-to-trial jitter',
        'cooccur': 'Alpha bursts with probabilistic theta co-occurrence',
        'narrowing': 'Single oscillator, bandwidth narrows with age'
    };

    const PAF_CURVE = {5: 6.5, 8: 8.0, 11: 9.2, 14: 9.8, 17: 10.1, 20: 10.2};
    const EXP_CURVE = {5: 1.65, 20: 1.35};

    function interpolatePAF(age) {
        const ages = [5, 8, 11, 14, 17, 20];
        const freqs = [6.5, 8.0, 9.2, 9.8, 10.1, 10.2];
        if (age <= 5) return 6.5;
        if (age >= 20) return 10.2;
        for (let i = 0; i < ages.length - 1; i++) {
            if (age >= ages[i] && age <= ages[i+1]) {
                const t = (age - ages[i]) / (ages[i+1] - ages[i]);
                return freqs[i] + t * (freqs[i+1] - freqs[i]);
            }
        }
        return 9.2;
    }

    function interpolateExp(age) {
        const t = (age - 5) / 15;
        return 1.65 + t * (1.35 - 1.65);
    }

    function updateDisplays() {
        const age = parseInt(ageSlider.value);
        ageValue.textContent = age;
        pafDisplay.textContent = interpolatePAF(age).toFixed(1);
        expDisplay.textContent = interpolateExp(age).toFixed(2);
        rateValue.textContent = parseFloat(rateSlider.value).toFixed(1);
        snrValue.textContent = parseFloat(snrSlider.value).toFixed(1);
        cyclesValue.innerHTML = cyclesMin.value + '&ndash;' + cyclesMax.value;
        durationValue.textContent = parseFloat(durationSlider.value).toFixed(1);
        regimeHelp.textContent = REGIME_DESCRIPTIONS[regimeSelect.value] || '';
    }

    function getParams() {
        return {
            regime: regimeSelect.value,
            age: parseInt(ageSlider.value),
            burst_rate: parseFloat(rateSlider.value),
            snr_db: parseFloat(snrSlider.value),
            n_cycles_min: parseInt(cyclesMin.value),
            n_cycles_max: parseInt(cyclesMax.value),
            epoch_duration: parseFloat(durationSlider.value),
            seed: parseInt(seedInput.value)
        };
    }

    function showLoading(show) {
        loading.classList.toggle('hidden', !show);
        runBtn.disabled = show;
    }

    function plotTimeseries(container, time, signal, bursts, title, regime) {
        const traces = [{
            x: time,
            y: signal,
            type: 'scatter',
            mode: 'lines',
            line: {color: '#4a5568', width: 0.8},
            name: 'Signal'
        }];

        const shapes = [];
        if (bursts && bursts.length > 0) {
            bursts.forEach(function(b) {
                shapes.push({
                    type: 'line',
                    x0: b.onset,
                    x1: b.onset,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {color: 'rgba(229, 62, 62, 0.4)', width: 1, dash: 'dot'}
                });
            });
        }

        const layout = {
            title: {text: title, font: {size: 13}},
            xaxis: {title: 'Time (s)', titlefont: {size: 11}},
            yaxis: {title: 'Amplitude', titlefont: {size: 11}},
            shapes: shapes,
            margin: {t: 40, b: 50, l: 60, r: 20},
            height: 260,
            font: {size: 10}
        };

        Plotly.newPlot(container, traces, layout, {responsive: true});
    }

    function plotPSD(container, freqs, psd, title, params) {
        const traces = [{
            x: freqs,
            y: psd,
            type: 'scatter',
            mode: 'lines',
            line: {color: '#2c5282', width: 1.5},
            name: 'PSD'
        }];

        const paf = params ? interpolatePAF(params.age) : 9.2;

        const shapes = [
            {type: 'rect', x0: 4, x1: 7, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(159, 122, 234, 0.1)', line: {width: 0}},
            {type: 'rect', x0: 7, x1: 13, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(66, 153, 225, 0.1)', line: {width: 0}},
            {type: 'line', x0: paf, x1: paf, y0: 0, y1: 1, yref: 'paper',
             line: {color: 'rgba(197, 48, 48, 0.6)', width: 1, dash: 'dash'}}
        ];

        const annotations = [
            {x: 5.5, y: 1.02, yref: 'paper', text: 'θ', showarrow: false,
             font: {size: 10, color: '#805ad5'}},
            {x: 10, y: 1.02, yref: 'paper', text: 'α', showarrow: false,
             font: {size: 10, color: '#2b6cb0'}},
            {x: paf, y: 1.05, yref: 'paper', text: 'PAF', showarrow: false,
             font: {size: 9, color: '#c53030'}}
        ];

        const layout = {
            title: {text: title, font: {size: 13}},
            xaxis: {title: 'Frequency (Hz)', titlefont: {size: 11}, range: [0, 45]},
            yaxis: {title: 'log₁₀ Power', titlefont: {size: 11}},
            shapes: shapes,
            annotations: annotations,
            margin: {t: 40, b: 50, l: 60, r: 20},
            height: 260,
            font: {size: 10}
        };

        Plotly.newPlot(container, traces, layout, {responsive: true});
    }

    function updateStats(data) {
        statsPanel.classList.remove('hidden');
        const p = data.params;
        const s = data.stats;
        statsContent.innerHTML =
            '<strong>Regime:</strong> ' + p.regime + '<br>' +
            '<strong>Age:</strong> ' + p.age + ' yr (PAF=' + p.paf.toFixed(1) + ' Hz)<br>' +
            '<strong>Bursts detected:</strong> ' + s.n_bursts + '<br>' +
            '<strong>Mean burst freq:</strong> ' + s.mean_freq.toFixed(1) + ' Hz<br>' +
            '<strong>Compute time:</strong> ' + s.elapsed_s + 's';
    }

    async function runSimulation() {
        showLoading(true);
        const params = getParams();

        try {
            const resp = await fetch('/api/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            const data = await resp.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            const regimeLabel = regimeSelect.options[regimeSelect.selectedIndex].text;
            plotTimeseries('plot-timeseries', data.time, data.signal, data.bursts,
                          regimeLabel + ' — Time Series (age ' + params.age + ')', params.regime);
            plotPSD('plot-psd', data.freqs, data.psd,
                    regimeLabel + ' — Power Spectral Density', params);
            updateStats(data);
        } catch (e) {
            alert('Simulation failed: ' + e.message);
        } finally {
            showLoading(false);
        }
    }

    async function runComparison() {
        showLoading(true);
        const params1 = getParams();
        const params2 = Object.assign({}, params1, {
            regime: document.getElementById('regime2-select').value
        });

        try {
            const resp = await fetch('/api/compare', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({regimes: [params1, params2]})
            });
            const data = await resp.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            const r = data.results;
            if (r.length >= 2) {
                document.getElementById('compare-row').style.display = 'flex';

                const colors = ['#c53030', '#2c5282'];
                const names = [r[0].regime, r[1].regime];

                // Overlay time series
                const tsTraces = r.map(function(result, i) {
                    return {
                        x: result.time,
                        y: result.signal,
                        type: 'scatter',
                        mode: 'lines',
                        line: {color: colors[i], width: 0.8},
                        name: names[i],
                        opacity: 0.7
                    };
                });
                Plotly.newPlot('plot-compare-ts', tsTraces, {
                    title: {text: 'Comparison — Time Series', font: {size: 13}},
                    xaxis: {title: 'Time (s)'},
                    yaxis: {title: 'Amplitude'},
                    margin: {t: 40, b: 50, l: 60, r: 20},
                    height: 260,
                    font: {size: 10}
                }, {responsive: true});

                // Overlay PSDs
                const psdTraces = r.map(function(result, i) {
                    return {
                        x: result.freqs,
                        y: result.psd,
                        type: 'scatter',
                        mode: 'lines',
                        line: {color: colors[i], width: 1.5},
                        name: names[i]
                    };
                });
                Plotly.newPlot('plot-compare-psd', psdTraces, {
                    title: {text: 'Comparison — PSD (overlaid)', font: {size: 13}},
                    xaxis: {title: 'Frequency (Hz)', range: [0, 45]},
                    yaxis: {title: 'log₁₀ Power'},
                    margin: {t: 40, b: 50, l: 60, r: 20},
                    height: 260,
                    font: {size: 10}
                }, {responsive: true});
            }
        } catch (e) {
            alert('Comparison failed: ' + e.message);
        } finally {
            showLoading(false);
        }
    }

    // Parse URL presets
    function loadPresets() {
        const params = new URLSearchParams(window.location.search);
        if (params.has('regime')) regimeSelect.value = params.get('regime');
        if (params.has('age')) ageSlider.value = params.get('age');
        if (params.has('burst_rate')) rateSlider.value = params.get('burst_rate');
        if (params.has('snr_db')) snrSlider.value = params.get('snr_db');
        if (params.has('n_cycles_min')) cyclesMin.value = params.get('n_cycles_min');
        if (params.has('n_cycles_max')) cyclesMax.value = params.get('n_cycles_max');
        if (params.has('duration')) durationSlider.value = params.get('duration');
        if (params.has('seed')) seedInput.value = params.get('seed');
        if (params.has('compare')) {
            compareToggle.checked = true;
            compareControls.classList.remove('hidden');
            document.getElementById('regime2-select').value = params.get('compare');
        }
        updateDisplays();
    }

    // Debounce: wait for pause in rapid input (e.g. slider drag) before firing
    let _debounceTimer = null;
    function debounceSimulation(delay) {
        if (_debounceTimer) clearTimeout(_debounceTimer);
        _debounceTimer = setTimeout(function() {
            _debounceTimer = null;
            if (compareToggle.checked) {
                runComparison();
            } else {
                runSimulation();
            }
        }, delay);
    }

    function onParamChange() {
        updateDisplays();
        debounceSimulation(150);
    }

    // Event listeners — every parameter change triggers a new simulation
    ageSlider.addEventListener('input', onParamChange);
    rateSlider.addEventListener('input', onParamChange);
    snrSlider.addEventListener('input', onParamChange);
    cyclesMin.addEventListener('input', onParamChange);
    cyclesMax.addEventListener('input', onParamChange);
    durationSlider.addEventListener('input', onParamChange);
    regimeSelect.addEventListener('change', onParamChange);
    seedInput.addEventListener('input', onParamChange);

    runBtn.addEventListener('click', runSimulation);

    randomSeedBtn.addEventListener('click', function() {
        seedInput.value = Math.floor(Math.random() * 10000);
        debounceSimulation(0);
    });

    compareToggle.addEventListener('change', function() {
        compareControls.classList.toggle('hidden', !this.checked);
        if (!this.checked) {
            document.getElementById('compare-row').style.display = 'none';
            runSimulation();
        } else {
            runComparison();
        }
    });

    document.getElementById('regime2-select').addEventListener('change', function() {
        if (compareToggle.checked) debounceSimulation(0);
    });

    compareBtn.addEventListener('click', runComparison);

    // Initialize
    loadPresets();
    if (compareToggle.checked) {
        runComparison();
    } else {
        runSimulation();
    }
});
