document.addEventListener('DOMContentLoaded', function() {
    var regimeSelect = document.getElementById('regime-select');
    var ageSlider = document.getElementById('age-slider');
    var ageValue = document.getElementById('age-value');
    var pafDisplay = document.getElementById('paf-display');
    var expDisplay = document.getElementById('exp-display');
    var rateSlider = document.getElementById('burst-rate-slider');
    var rateValue = document.getElementById('rate-value');
    var snrSlider = document.getElementById('snr-slider');
    var snrValue = document.getElementById('snr-value');
    var cyclesMin = document.getElementById('cycles-min');
    var cyclesMax = document.getElementById('cycles-max');
    var cyclesValue = document.getElementById('cycles-value');
    var epochsSlider = document.getElementById('epochs-slider');
    var epochsValue = document.getElementById('epochs-value');
    var totalDuration = document.getElementById('total-duration');
    var seedInput = document.getElementById('seed-input');
    var randomSeedBtn = document.getElementById('random-seed-btn');
    var runBtn = document.getElementById('run-btn');
    var compareToggle = document.getElementById('compare-toggle');
    var compareControls = document.getElementById('compare-controls');
    var compareBtn = document.getElementById('compare-btn');
    var loading = document.getElementById('loading');
    var statsPanel = document.getElementById('stats-panel');
    var statsContent = document.getElementById('stats-content');
    var regimeHelp = document.getElementById('regime-help');

    var REGIME_DESCRIPTIONS = {
        'chirp': 'Within-burst frequency sweep from theta to alpha',
        'mixture': 'Two burst populations (theta + alpha) with age-dependent mixing',
        'drift': 'Single oscillator at PAF with trial-to-trial jitter',
        'cooccur': 'Alpha bursts with probabilistic theta co-occurrence',
        'narrowing': 'Single oscillator, bandwidth narrows with age'
    };

    function interpolatePAF(age) {
        var ages = [5, 8, 11, 14, 17, 20];
        var freqs = [6.5, 8.0, 9.2, 9.8, 10.1, 10.2];
        if (age <= 5) return 6.5;
        if (age >= 20) return 10.2;
        for (var i = 0; i < ages.length - 1; i++) {
            if (age >= ages[i] && age <= ages[i+1]) {
                var t = (age - ages[i]) / (ages[i+1] - ages[i]);
                return freqs[i] + t * (freqs[i+1] - freqs[i]);
            }
        }
        return 9.2;
    }

    function interpolateExp(age) {
        var t = (age - 5) / 15;
        return 1.65 + t * (1.35 - 1.65);
    }

    function updateDisplays() {
        var age = parseInt(ageSlider.value);
        ageValue.textContent = age;
        pafDisplay.textContent = interpolatePAF(age).toFixed(1);
        expDisplay.textContent = interpolateExp(age).toFixed(2);
        rateValue.textContent = parseFloat(rateSlider.value).toFixed(1);
        snrValue.textContent = parseFloat(snrSlider.value).toFixed(1);
        cyclesValue.innerHTML = cyclesMin.value + '&ndash;' + cyclesMax.value;
        var nEpochs = parseInt(epochsSlider.value);
        epochsValue.textContent = nEpochs;
        totalDuration.textContent = (nEpochs * 2).toString();
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
            n_epochs: parseInt(epochsSlider.value),
            seed: parseInt(seedInput.value)
        };
    }

    function showLoading(show) {
        loading.classList.toggle('hidden', !show);
        runBtn.disabled = show;
    }

    // --- Plot functions ---

    function plotTimeseries(container, time, signal, bursts, title) {
        var totalDur = time[time.length - 1];
        var viewEnd = Math.min(4.0, totalDur);
        var traces = [{
            x: time, y: signal, type: 'scatter', mode: 'lines',
            line: {color: '#4a5568', width: 0.8}, name: 'Signal'
        }];
        var shapes = [];
        if (bursts && bursts.length > 0) {
            bursts.forEach(function(b) {
                if (b.onset <= viewEnd + 2) {
                    shapes.push({
                        type: 'line', x0: b.onset, x1: b.onset,
                        y0: 0, y1: 1, yref: 'paper',
                        line: {color: 'rgba(229, 62, 62, 0.4)', width: 1, dash: 'dot'}
                    });
                }
            });
        }
        Plotly.newPlot(container, traces, {
            title: {text: title, font: {size: 13}},
            xaxis: {title: 'Time (s)', titlefont: {size: 11},
                    range: [0, viewEnd], rangeslider: {visible: totalDur > 5}},
            yaxis: {title: 'Amplitude', titlefont: {size: 11}},
            shapes: shapes,
            margin: {t: 40, b: totalDur > 5 ? 80 : 50, l: 60, r: 20},
            height: totalDur > 5 ? 300 : 260, font: {size: 10}
        }, {responsive: true});
    }

    function plotPSD(container, freqs, psd, psdAperiodic, title, age, nEpochs) {
        var paf = interpolatePAF(age);
        var epochLabel = nEpochs > 1 ? ' (avg ' + nEpochs + ' epochs)' : '';
        var traces = [
            {x: freqs, y: psd, type: 'scatter', mode: 'lines',
             line: {color: '#2c5282', width: 1.5}, name: 'Full PSD'},
            {x: freqs, y: psdAperiodic, type: 'scatter', mode: 'lines',
             line: {color: '#a0aec0', width: 1, dash: 'dash'}, name: 'Aperiodic only'}
        ];
        var shapes = [
            {type: 'rect', x0: 4, x1: 7, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(159, 122, 234, 0.1)', line: {width: 0}},
            {type: 'rect', x0: 7, x1: 13, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(66, 153, 225, 0.1)', line: {width: 0}},
            {type: 'line', x0: paf, x1: paf, y0: 0, y1: 1, yref: 'paper',
             line: {color: 'rgba(197, 48, 48, 0.6)', width: 1, dash: 'dash'}}
        ];
        var annotations = [
            {x: 5.5, y: 1.02, yref: 'paper', text: 'θ', showarrow: false,
             font: {size: 10, color: '#805ad5'}},
            {x: 10, y: 1.02, yref: 'paper', text: 'α', showarrow: false,
             font: {size: 10, color: '#2b6cb0'}},
            {x: paf, y: 1.05, yref: 'paper', text: 'PAF', showarrow: false,
             font: {size: 9, color: '#c53030'}}
        ];
        Plotly.newPlot(container, traces, {
            title: {text: title + epochLabel, font: {size: 13}},
            xaxis: {title: 'Frequency (Hz)', titlefont: {size: 11}, range: [0, 45]},
            yaxis: {title: 'log₁₀ Power', titlefont: {size: 11}},
            shapes: shapes, annotations: annotations,
            margin: {t: 40, b: 50, l: 60, r: 20}, height: 260, font: {size: 10}
        }, {responsive: true});
    }

    function plotResidual(container, freqs, residual, title, age) {
        var paf = interpolatePAF(age);
        var traces = [{
            x: freqs, y: residual, type: 'scatter', mode: 'lines',
            line: {color: '#805ad5', width: 1.5}, name: 'Residual', fill: 'tozeroy',
            fillcolor: 'rgba(128, 90, 213, 0.15)'
        }];
        var shapes = [
            {type: 'rect', x0: 4, x1: 7, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(159, 122, 234, 0.08)', line: {width: 0}},
            {type: 'rect', x0: 7, x1: 13, y0: 0, y1: 1, yref: 'paper',
             fillcolor: 'rgba(66, 153, 225, 0.08)', line: {width: 0}},
            {type: 'line', x0: paf, x1: paf, y0: 0, y1: 1, yref: 'paper',
             line: {color: 'rgba(197, 48, 48, 0.5)', width: 1, dash: 'dash'}},
            {type: 'line', x0: 0, x1: 45, y0: 0, y1: 0,
             line: {color: '#a0aec0', width: 1, dash: 'dot'}}
        ];
        Plotly.newPlot(container, traces, {
            title: {text: title, font: {size: 13}},
            xaxis: {title: 'Frequency (Hz)', titlefont: {size: 11}, range: [1, 25]},
            yaxis: {title: 'Residual (log₁₀)', titlefont: {size: 11}},
            shapes: shapes,
            margin: {t: 40, b: 50, l: 60, r: 20}, height: 240, font: {size: 10}
        }, {responsive: true});
    }

    function plotPeriodSlopes(container, bycycle, title) {
        var slopes = bycycle.period_slopes || [];
        if (slopes.length === 0) {
            Plotly.newPlot(container, [], {
                title: {text: title + ' — no bursts detected', font: {size: 13}},
                margin: {t: 40, b: 50, l: 60, r: 20}, height: 220, font: {size: 10}
            }, {responsive: true});
            return;
        }
        var absSlopes = slopes.map(function(s) { return Math.abs(s); });
        var traces = [{
            x: absSlopes, type: 'histogram',
            marker: {color: 'rgba(44, 82, 130, 0.7)', line: {color: '#2c5282', width: 1}},
            name: 'Period slope'
        }];
        var meanSlope = bycycle.mean_period_slope;
        Plotly.newPlot(container, traces, {
            title: {text: title + ' (mean=' + meanSlope.toFixed(1) + ')', font: {size: 13}},
            xaxis: {title: '|Period slope| (samples/cycle)', titlefont: {size: 11}},
            yaxis: {title: 'Count', titlefont: {size: 11}},
            shapes: [{
                type: 'line', x0: meanSlope, x1: meanSlope, y0: 0, y1: 1, yref: 'paper',
                line: {color: '#c53030', width: 2, dash: 'dash'}
            }],
            margin: {t: 40, b: 50, l: 60, r: 20}, height: 220, font: {size: 10},
            bargap: 0.05
        }, {responsive: true});
    }

    // --- Stats panel ---

    function updateStats(data) {
        statsPanel.classList.remove('hidden');
        var p = data.params;
        var s = data.stats;
        var b = data.bycycle;
        statsContent.innerHTML =
            '<strong>Regime:</strong> ' + p.regime + '<br>' +
            '<strong>Age:</strong> ' + p.age + ' yr (PAF=' + p.paf.toFixed(1) + ' Hz)<br>' +
            '<strong>Epochs:</strong> ' + p.n_epochs + ' × 2 s = ' + s.total_duration_s + ' s<br>' +
            '<strong>Bursts (ground truth):</strong> ' + s.n_bursts + '<br>' +
            '<strong>Bursts (bycycle):</strong> ' + b.n_bursts_bycycle +
                ' (' + b.n_cycles_total + ' cycles)<br>' +
            '<strong>Mean period slope:</strong> ' + b.mean_period_slope.toFixed(1) +
                ' samples/cycle<br>' +
            '<strong>Compute time:</strong> ' + s.elapsed_s + ' s';
    }

    // --- Simulation runners ---

    async function runSimulation() {
        showLoading(true);
        var params = getParams();
        try {
            var resp = await fetch('/api/simulate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(params)
            });
            var data = await resp.json();
            if (data.error) { alert('Error: ' + data.error); return; }

            var label = regimeSelect.options[regimeSelect.selectedIndex].text;
            plotTimeseries('plot-timeseries', data.time, data.signal, data.bursts,
                          label + ' — Time Series (age ' + params.age + ')');
            plotPSD('plot-psd', data.freqs, data.psd, data.psd_aperiodic,
                    label + ' — PSD', params.age, params.n_epochs);
            plotResidual('plot-residual', data.freqs, data.residual,
                         label + ' — Residual (periodic component)', params.age);
            plotPeriodSlopes('plot-slopes', data.bycycle,
                            label + ' — Burst Period Slopes');
            updateStats(data);
        } catch (e) {
            alert('Simulation failed: ' + e.message);
        } finally {
            showLoading(false);
        }
    }

    async function runComparison() {
        showLoading(true);
        var params1 = getParams();
        var params2 = Object.assign({}, params1, {
            regime: document.getElementById('regime2-select').value
        });
        try {
            var resp = await fetch('/api/compare', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({regimes: [params1, params2]})
            });
            var data = await resp.json();
            if (data.error) { alert('Error: ' + data.error); return; }

            var r = data.results;
            if (r.length >= 2) {
                // Single-regime plots show regime 1
                var label1 = regimeSelect.options[regimeSelect.selectedIndex].text;
                plotTimeseries('plot-timeseries', r[0].time, r[0].signal,
                              r[0].bursts || [],
                              label1 + ' — Time Series (age ' + params1.age + ')');
                plotPSD('plot-psd', r[0].freqs, r[0].psd, r[0].psd_aperiodic,
                        label1 + ' — PSD', params1.age, params1.n_epochs);
                plotResidual('plot-residual', r[0].freqs, r[0].residual,
                             label1 + ' — Residual', params1.age);
                plotPeriodSlopes('plot-slopes', r[0].bycycle,
                                label1 + ' — Period Slopes');

                // Comparison overlay plots
                document.getElementById('compare-row').style.display = '';
                var colors = ['#c53030', '#2c5282'];
                var names = [r[0].regime, r[1].regime];
                var totalDur = r[0].time[r[0].time.length - 1];
                var viewEnd = Math.min(4.0, totalDur);

                // Overlaid time series
                Plotly.newPlot('plot-compare-ts', r.map(function(result, i) {
                    return {x: result.time, y: result.signal, type: 'scatter',
                            mode: 'lines', line: {color: colors[i], width: 0.8},
                            name: names[i], opacity: 0.7};
                }), {
                    title: {text: 'Comparison — Time Series (overlaid)', font: {size: 13}},
                    xaxis: {title: 'Time (s)', range: [0, viewEnd],
                            rangeslider: {visible: totalDur > 5}},
                    yaxis: {title: 'Amplitude'},
                    margin: {t: 40, b: totalDur > 5 ? 80 : 50, l: 60, r: 20},
                    height: totalDur > 5 ? 300 : 260, font: {size: 10}
                }, {responsive: true});

                // Overlaid PSDs
                var nEp = params1.n_epochs;
                var epLbl = nEp > 1 ? ' (avg ' + nEp + ' epochs)' : '';
                Plotly.newPlot('plot-compare-psd', r.map(function(result, i) {
                    return {x: result.freqs, y: result.psd, type: 'scatter',
                            mode: 'lines', line: {color: colors[i], width: 1.5},
                            name: names[i]};
                }), {
                    title: {text: 'Comparison — PSD (overlaid)' + epLbl, font: {size: 13}},
                    xaxis: {title: 'Frequency (Hz)', range: [0, 45]},
                    yaxis: {title: 'log₁₀ Power'},
                    margin: {t: 40, b: 50, l: 60, r: 20}, height: 260, font: {size: 10}
                }, {responsive: true});

                // Overlaid residuals
                Plotly.newPlot('plot-compare-residual', r.map(function(result, i) {
                    return {x: result.freqs, y: result.residual, type: 'scatter',
                            mode: 'lines', line: {color: colors[i], width: 1.5},
                            name: names[i], fill: i === 0 ? 'tozeroy' : undefined,
                            fillcolor: i === 0 ? 'rgba(197,48,48,0.1)' : undefined};
                }), {
                    title: {text: 'Comparison — Residual (periodic component)', font: {size: 13}},
                    xaxis: {title: 'Frequency (Hz)', range: [1, 25]},
                    yaxis: {title: 'Residual (log₁₀)'},
                    shapes: [{type: 'line', x0: 0, x1: 25, y0: 0, y1: 0,
                              line: {color: '#a0aec0', width: 1, dash: 'dot'}}],
                    margin: {t: 40, b: 50, l: 60, r: 20}, height: 240, font: {size: 10}
                }, {responsive: true});

                // Side-by-side period slopes
                var s0 = (r[0].bycycle.period_slopes || []).map(Math.abs);
                var s1 = (r[1].bycycle.period_slopes || []).map(Math.abs);
                var m0 = r[0].bycycle.mean_period_slope;
                var m1 = r[1].bycycle.mean_period_slope;
                Plotly.newPlot('plot-compare-slopes', [
                    {x: s0, type: 'histogram', name: names[0] + ' (mean=' + m0.toFixed(1) + ')',
                     marker: {color: 'rgba(197,48,48,0.5)'}, opacity: 0.7},
                    {x: s1, type: 'histogram', name: names[1] + ' (mean=' + m1.toFixed(1) + ')',
                     marker: {color: 'rgba(44,82,130,0.5)'}, opacity: 0.7}
                ], {
                    title: {text: 'Comparison — Burst Period Slopes', font: {size: 13}},
                    xaxis: {title: '|Period slope| (samples/cycle)', titlefont: {size: 11}},
                    yaxis: {title: 'Count', titlefont: {size: 11}},
                    barmode: 'overlay',
                    margin: {t: 40, b: 50, l: 60, r: 20}, height: 220, font: {size: 10}
                }, {responsive: true});
            }
        } catch (e) {
            alert('Comparison failed: ' + e.message);
        } finally {
            showLoading(false);
        }
    }

    // --- Presets and event wiring ---

    function loadPresets() {
        var params = new URLSearchParams(window.location.search);
        if (params.has('regime')) regimeSelect.value = params.get('regime');
        if (params.has('age')) ageSlider.value = params.get('age');
        if (params.has('burst_rate')) rateSlider.value = params.get('burst_rate');
        if (params.has('snr_db')) snrSlider.value = params.get('snr_db');
        if (params.has('n_cycles_min')) cyclesMin.value = params.get('n_cycles_min');
        if (params.has('n_cycles_max')) cyclesMax.value = params.get('n_cycles_max');
        if (params.has('n_epochs')) epochsSlider.value = params.get('n_epochs');
        if (params.has('seed')) seedInput.value = params.get('seed');
        if (params.has('compare')) {
            compareToggle.checked = true;
            compareControls.classList.remove('hidden');
            document.getElementById('regime2-select').value = params.get('compare');
        }
        updateDisplays();
    }

    var _debounceTimer = null;
    function debounceSimulation(delay) {
        if (_debounceTimer) clearTimeout(_debounceTimer);
        _debounceTimer = setTimeout(function() {
            _debounceTimer = null;
            if (compareToggle.checked) runComparison();
            else runSimulation();
        }, delay);
    }

    function onParamChange() {
        updateDisplays();
        debounceSimulation(300);
    }

    ageSlider.addEventListener('input', onParamChange);
    rateSlider.addEventListener('input', onParamChange);
    snrSlider.addEventListener('input', onParamChange);
    cyclesMin.addEventListener('input', onParamChange);
    cyclesMax.addEventListener('input', onParamChange);
    epochsSlider.addEventListener('input', onParamChange);
    regimeSelect.addEventListener('change', onParamChange);
    seedInput.addEventListener('input', onParamChange);

    runBtn.addEventListener('click', function() {
        if (compareToggle.checked) runComparison();
        else runSimulation();
    });

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

    loadPresets();
    if (compareToggle.checked) runComparison();
    else runSimulation();
});
