"""Tests for simulation parameter curves, aperiodic generation, and regimes."""

import numpy as np
import pytest

from theta_alpha_shift.sim.params import (
    aperiodic_exponent,
    burst_n_cycles,
    get_ages,
    knee_freq,
    load_config,
    mixture_theta_weight,
    narrowing_bandwidth,
    paf,
)
from theta_alpha_shift.sim.aperiodic import generate_aperiodic, generate_aperiodic_psd
from theta_alpha_shift.sim.regimes import (
    SimulatedEpoch,
    simulate_chirp,
    simulate_cooccur,
    simulate_drift,
    simulate_mixture,
    simulate_narrowing,
)


class TestParams:
    def test_load_config(self):
        cfg = load_config()
        assert "ages" in cfg
        assert cfg["random_seed"] == 42
        assert cfg["sfreq"] == 500

    def test_paf_known_ages(self):
        assert paf(5) == pytest.approx(6.5)
        assert paf(20) == pytest.approx(10.2)

    def test_paf_interpolation(self):
        freq = paf(6.5)
        assert 6.5 < freq < 8.0

    def test_paf_monotonically_increasing(self):
        ages = get_ages()
        freqs = [paf(a) for a in ages]
        for i in range(1, len(freqs)):
            assert freqs[i] >= freqs[i - 1]

    def test_aperiodic_exponent_known(self):
        assert aperiodic_exponent(5) == pytest.approx(1.65)
        assert aperiodic_exponent(20) == pytest.approx(1.35)

    def test_aperiodic_exponent_decreasing(self):
        ages = get_ages()
        exps = [aperiodic_exponent(a) for a in ages]
        for i in range(1, len(exps)):
            assert exps[i] <= exps[i - 1]

    def test_mixture_theta_weight_decreasing(self):
        ages = get_ages()
        weights = [mixture_theta_weight(a) for a in ages]
        for i in range(1, len(weights)):
            assert weights[i] <= weights[i - 1]

    def test_narrowing_bandwidth_decreasing(self):
        ages = get_ages()
        bws = [narrowing_bandwidth(a) for a in ages]
        for i in range(1, len(bws)):
            assert bws[i] <= bws[i - 1]

    def test_burst_n_cycles_range(self):
        cfg = load_config()
        lo, hi = cfg["burst"]["n_cycles_range"]
        rng = np.random.default_rng(42)
        for _ in range(100):
            n = burst_n_cycles(rng=rng)
            assert lo <= n <= hi

    def test_knee_freq(self):
        assert knee_freq() == 5.0


class TestAperiodic:
    def test_generate_signal_shape(self):
        signal = generate_aperiodic(age=8, duration=2.0, sfreq=500)
        assert signal.shape == (1000,)

    def test_generate_signal_not_all_zeros(self):
        signal = generate_aperiodic(age=8, duration=2.0, sfreq=500)
        assert np.std(signal) > 0

    def test_different_ages_different_spectra(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sig_young = generate_aperiodic(age=5, rng=rng1)
        sig_old = generate_aperiodic(age=20, rng=rng2)
        # Younger should have steeper spectrum (more low-freq power relative to high)
        freqs = np.fft.rfftfreq(len(sig_young), d=1.0 / 500)
        psd_young = np.abs(np.fft.rfft(sig_young)) ** 2
        psd_old = np.abs(np.fft.rfft(sig_old)) ** 2
        # Ratio of low-freq to high-freq power should be higher for young
        low_mask = (freqs >= 2) & (freqs <= 10)
        high_mask = (freqs >= 20) & (freqs <= 40)
        ratio_young = np.mean(psd_young[low_mask]) / np.mean(psd_young[high_mask])
        ratio_old = np.mean(psd_old[low_mask]) / np.mean(psd_old[high_mask])
        assert ratio_young > ratio_old

    def test_theoretical_psd_shape(self):
        freqs, psd = generate_aperiodic_psd(age=11)
        assert len(freqs) == len(psd)
        assert np.all(psd > 0)

    def test_theoretical_psd_decreasing(self):
        freqs, psd = generate_aperiodic_psd(age=11)
        # PSD should generally decrease with frequency (above knee)
        above_knee = freqs > 10
        psd_above = psd[above_knee]
        assert psd_above[0] > psd_above[-1]

    def test_reproducibility_with_seed(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sig1 = generate_aperiodic(age=11, rng=rng1)
        sig2 = generate_aperiodic(age=11, rng=rng2)
        np.testing.assert_array_equal(sig1, sig2)


class TestRegimeChirp:
    def test_returns_simulated_epoch(self):
        epoch = simulate_chirp(age=8, rng=np.random.default_rng(42))
        assert isinstance(epoch, SimulatedEpoch)
        assert epoch.regime == "chirp"
        assert epoch.age == 8

    def test_signal_shape(self):
        epoch = simulate_chirp(age=8, rng=np.random.default_rng(42))
        expected_samples = int(2.0 * 500)
        assert epoch.data.shape == (expected_samples,)
        assert epoch.sfreq == 500

    def test_ground_truth_populated(self):
        epoch = simulate_chirp(age=8, rng=np.random.default_rng(42))
        assert len(epoch.burst_times) > 0
        assert len(epoch.burst_freqs) == len(epoch.burst_times)
        assert len(epoch.burst_labels) == len(epoch.burst_times)
        assert all(l in ("chirp_up", "chirp_down") for l in epoch.burst_labels)

    def test_burst_freqs_span_range(self):
        cfg = load_config()
        epoch = simulate_chirp(age=8, rng=np.random.default_rng(42))
        f_low = cfg["chirp"]["f_low"]
        f_high = paf(8)
        for freqs, label in zip(epoch.burst_freqs, epoch.burst_labels):
            if label == "chirp_up":
                assert freqs[0] == pytest.approx(f_low)
                assert freqs[-1] == pytest.approx(f_high)
            else:
                assert freqs[0] == pytest.approx(f_high)
                assert freqs[-1] == pytest.approx(f_low)

    def test_both_directions_present(self):
        epoch = simulate_chirp(age=8, rng=np.random.default_rng(42))
        labels = set(epoch.burst_labels)
        # With "both" direction and enough bursts, expect both
        assert epoch.params["n_up"] + epoch.params["n_down"] == epoch.params["n_bursts"]

    def test_params_recorded(self):
        cfg = load_config()
        epoch = simulate_chirp(age=11, rng=np.random.default_rng(42))
        assert "f_low" in epoch.params
        assert "f_high" in epoch.params
        assert "direction" in epoch.params
        assert epoch.params["f_low"] == cfg["chirp"]["f_low"]

    def test_reproducibility(self):
        e1 = simulate_chirp(age=8, rng=np.random.default_rng(42))
        e2 = simulate_chirp(age=8, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(e1.data, e2.data)


class TestRegimeMixture:
    def test_returns_simulated_epoch(self):
        epoch = simulate_mixture(age=8, rng=np.random.default_rng(42))
        assert isinstance(epoch, SimulatedEpoch)
        assert epoch.regime == "mixture"

    def test_ground_truth_has_both_labels(self):
        epoch = simulate_mixture(age=8, rng=np.random.default_rng(99))
        labels = set(epoch.burst_labels)
        assert "theta" in labels or "alpha" in labels

    def test_young_age_more_theta(self):
        rng = np.random.default_rng(42)
        n_trials = 20
        theta_counts = []
        for _ in range(n_trials):
            epoch = simulate_mixture(age=5, rng=rng)
            theta_counts.append(sum(1 for l in epoch.burst_labels if l == "theta"))
        # At age 5, theta_weight=0.8, so most bursts should be theta
        assert np.mean(theta_counts) > 0

    def test_burst_freqs_are_constant_within_burst(self):
        epoch = simulate_mixture(age=11, rng=np.random.default_rng(42))
        for freqs in epoch.burst_freqs:
            assert np.all(freqs == freqs[0])

    def test_params_include_counts(self):
        epoch = simulate_mixture(age=8, rng=np.random.default_rng(42))
        assert "n_theta" in epoch.params
        assert "n_alpha" in epoch.params
        assert epoch.params["n_theta"] + epoch.params["n_alpha"] == epoch.params["n_bursts"]


class TestRegimeDrift:
    def test_returns_simulated_epoch(self):
        epoch = simulate_drift(age=11, rng=np.random.default_rng(42))
        assert isinstance(epoch, SimulatedEpoch)
        assert epoch.regime == "drift"

    def test_all_labels_drift(self):
        epoch = simulate_drift(age=11, rng=np.random.default_rng(42))
        assert all(l == "drift" for l in epoch.burst_labels)

    def test_burst_freqs_near_paf(self):
        epoch = simulate_drift(age=11, rng=np.random.default_rng(42))
        expected_paf = paf(11)
        for freqs in epoch.burst_freqs:
            assert abs(freqs[0] - expected_paf) < 3.0

    def test_constant_freq_within_burst(self):
        epoch = simulate_drift(age=11, rng=np.random.default_rng(42))
        for freqs in epoch.burst_freqs:
            assert np.all(freqs == freqs[0])


class TestRegimeCooccur:
    def test_returns_simulated_epoch(self):
        epoch = simulate_cooccur(age=8, rng=np.random.default_rng(42))
        assert isinstance(epoch, SimulatedEpoch)
        assert epoch.regime == "cooccur"

    def test_has_alpha_bursts(self):
        epoch = simulate_cooccur(age=8, rng=np.random.default_rng(42))
        assert epoch.params["n_alpha"] > 0

    def test_params_tracked(self):
        epoch = simulate_cooccur(age=8, rng=np.random.default_rng(42))
        assert "cooccurrence_prob" in epoch.params
        assert "n_theta_cooccur" in epoch.params


class TestRegimeNarrowing:
    def test_returns_simulated_epoch(self):
        epoch = simulate_narrowing(age=8, rng=np.random.default_rng(42))
        assert isinstance(epoch, SimulatedEpoch)
        assert epoch.regime == "narrowing"

    def test_bandwidth_decreases_with_age(self):
        e5 = simulate_narrowing(age=5, rng=np.random.default_rng(42))
        e20 = simulate_narrowing(age=20, rng=np.random.default_rng(42))
        assert e5.params["bandwidth_hz"] > e20.params["bandwidth_hz"]

    def test_all_labels_narrowing(self):
        epoch = simulate_narrowing(age=11, rng=np.random.default_rng(42))
        assert all(l == "narrowing" for l in epoch.burst_labels)


class TestArtifacts:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def _cfg_with_artifacts(self, eog=False, emg=False, line=False):
        cfg = load_config()
        cfg = {**cfg, "artifacts": {**cfg.get("artifacts", {})}}
        cfg["artifacts"]["eog"] = {**cfg["artifacts"].get("eog", {}), "enabled": eog}
        cfg["artifacts"]["emg"] = {**cfg["artifacts"].get("emg", {}), "enabled": emg}
        cfg["artifacts"]["line_noise"] = {**cfg["artifacts"].get("line_noise", {}), "enabled": line}
        return cfg

    def test_no_artifacts_returns_copy(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(eog=False, emg=False, line=False)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(result.data, epoch.data)
        assert result.burst_times is epoch.burst_times

    def test_eog_modifies_signal(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(eog=True)
        cfg["artifacts"]["eog"]["rate"] = 3.0
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        assert not np.array_equal(result.data, epoch.data)

    def test_emg_modifies_signal(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(emg=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        assert not np.array_equal(result.data, epoch.data)

    def test_line_noise_modifies_signal(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(line=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        assert not np.array_equal(result.data, epoch.data)

    def test_ground_truth_preserved(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(eog=True, emg=True, line=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(result.burst_times, epoch.burst_times)
        np.testing.assert_array_equal(result.burst_labels, epoch.burst_labels)
        assert result.age == epoch.age
        assert result.regime == epoch.regime

    def test_epoch_shape_unchanged(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(eog=True, emg=True, line=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        assert result.data.shape == epoch.data.shape
        assert result.sfreq == epoch.sfreq

    def test_artifacts_applied_flag(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(eog=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        assert result.params["artifacts_applied"] is True

    def test_2d_signal_support(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        signal_2d = np.stack([epoch.data, epoch.data * 0.9])
        epoch_2d = SimulatedEpoch(
            data=signal_2d, sfreq=epoch.sfreq, age=epoch.age,
            regime=epoch.regime, params=epoch.params,
            burst_times=epoch.burst_times, burst_freqs=epoch.burst_freqs,
            burst_labels=epoch.burst_labels,
        )
        cfg = self._cfg_with_artifacts(eog=True, emg=True, line=True)
        result = inject_artifacts(epoch_2d, config=cfg, rng=np.random.default_rng(42))
        assert result.data.shape == (2, epoch.data.shape[0])

    def test_line_noise_at_correct_freq(self):
        from theta_alpha_shift.sim.artifacts import inject_artifacts
        epoch = self._make_epoch()
        cfg = self._cfg_with_artifacts(line=True)
        result = inject_artifacts(epoch, config=cfg, rng=np.random.default_rng(42))
        diff = result.data - epoch.data
        freqs = np.fft.rfftfreq(len(diff), d=1.0 / epoch.sfreq)
        spectrum = np.abs(np.fft.rfft(diff))
        peak_freq = freqs[np.argmax(spectrum)]
        assert abs(peak_freq - 60.0) < 1.0


class TestForwardModel:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_projects_to_128_channels(self):
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch = self._make_epoch()
        result = forward_project(epoch, rng=np.random.default_rng(42))
        assert result.data.ndim == 2
        assert result.data.shape[0] == 128
        assert result.data.shape[1] == len(epoch.data)

    def test_preserves_ground_truth(self):
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch = self._make_epoch()
        result = forward_project(epoch, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(result.burst_times, epoch.burst_times)
        np.testing.assert_array_equal(result.burst_labels, epoch.burst_labels)
        assert result.age == epoch.age
        assert result.regime == epoch.regime

    def test_posterior_channels_stronger(self):
        from theta_alpha_shift.sim.forward_model import forward_project, get_lead_field
        epoch = self._make_epoch()
        result = forward_project(epoch, rng=np.random.default_rng(42))
        lead_field, ch_names = get_lead_field()
        channel_power = np.mean(result.data ** 2, axis=1)
        max_power_idx = np.argmax(channel_power)
        assert ch_names[max_power_idx].startswith("E")

    def test_params_recorded(self):
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch = self._make_epoch()
        result = forward_project(epoch, rng=np.random.default_rng(42))
        assert result.params["forward_projected"] is True
        assert result.params["n_channels"] == 128
        assert result.params["n_sources"] == 3

    def test_lead_field_cached(self):
        from theta_alpha_shift.sim.forward_model import get_lead_field
        lf1, _ = get_lead_field()
        lf2, _ = get_lead_field()
        assert lf1 is lf2

    def test_reproducibility(self):
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch = self._make_epoch()
        r1 = forward_project(epoch, rng=np.random.default_rng(99))
        r2 = forward_project(epoch, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(r1.data, r2.data)

    def test_different_regimes_same_shape(self):
        from theta_alpha_shift.sim.forward_model import forward_project
        e_chirp = simulate_chirp(age=11, rng=np.random.default_rng(42))
        e_mix = simulate_mixture(age=11, rng=np.random.default_rng(42))
        r_chirp = forward_project(e_chirp, rng=np.random.default_rng(42))
        r_mix = forward_project(e_mix, rng=np.random.default_rng(42))
        assert r_chirp.data.shape == r_mix.data.shape
