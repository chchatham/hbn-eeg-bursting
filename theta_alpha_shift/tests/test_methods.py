"""Tests for burst detection and spectral analysis method wrappers."""

import numpy as np
import pytest

from theta_alpha_shift.methods import MethodResult
from theta_alpha_shift.sim.regimes import (
    simulate_chirp,
    simulate_mixture,
    simulate_drift,
)


class TestMethodResult:
    def test_dataclass_fields(self):
        r = MethodResult(
            method_name="test",
            detected_bursts=[],
            headline_stat=1.0,
            headline_stat_name="n_peaks",
        )
        assert r.method_name == "test"
        assert r.headline_stat == 1.0
        assert r.metadata == {}

    def test_metadata_default(self):
        r = MethodResult("m", [], 0.0, "stat")
        assert isinstance(r.metadata, dict)


class TestSpecparamBaseline:
    def _make_epochs(self):
        return (
            simulate_chirp(age=8, rng=np.random.default_rng(42)),
            simulate_mixture(age=8, rng=np.random.default_rng(42)),
        )

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam(epoch)
        assert isinstance(result, MethodResult)
        assert result.method_name == "specparam_baseline"

    def test_headline_stat_is_n_peaks(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam(epoch)
        assert result.headline_stat_name == "n_peaks_theta_alpha"
        assert result.headline_stat >= 0

    def test_no_detected_bursts(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam(epoch)
        assert result.detected_bursts == []

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam(epoch)
        assert "peak_params" in result.metadata
        assert "aperiodic_params" in result.metadata
        assert "fit_error" in result.metadata
        assert "peak_freq_hz" in result.metadata
        assert "peak_bandwidth_hz" in result.metadata

    def test_tuple_input(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_multichannel_input(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch, _ = self._make_epochs()
        epoch_128 = forward_project(epoch, rng=np.random.default_rng(42))
        result = run_specparam(epoch_128)
        assert isinstance(result, MethodResult)
        assert result.headline_stat >= 0

    def test_detects_peaks_in_theta_alpha(self):
        from theta_alpha_shift.methods.specparam_baseline import run_specparam
        epoch, _ = self._make_epochs()
        result = run_specparam(epoch)
        assert result.headline_stat >= 1
        assert not np.isnan(result.metadata["peak_freq_hz"])
        assert 4.0 <= result.metadata["peak_freq_hz"] <= 12.0


class TestBycycleWrap:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
        result = run_bycycle(self._make_epoch())
        assert isinstance(result, MethodResult)
        assert result.method_name == "bycycle"

    def test_headline_stat_name(self):
        from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
        result = run_bycycle(self._make_epoch())
        assert result.headline_stat_name == "mean_period_slope"
        assert result.headline_stat >= 0

    def test_detects_bursts(self):
        from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
        result = run_bycycle(self._make_epoch())
        assert len(result.detected_bursts) > 0
        burst = result.detected_bursts[0]
        assert "onset" in burst
        assert "offset" in burst
        assert "period_slope" in burst

    def test_tuple_input(self):
        from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
        epoch = self._make_epoch()
        result = run_bycycle((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.bycycle_wrap import run_bycycle
        result = run_bycycle(self._make_epoch())
        assert "n_cycles_total" in result.metadata
        assert "n_bursts" in result.metadata
        assert "period_cv" in result.metadata


class TestItemdIF:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.itemd_if import run_itemd
        result = run_itemd(self._make_epoch())
        assert isinstance(result, MethodResult)
        assert result.method_name == "itemd_if"

    def test_headline_stat_name(self):
        from theta_alpha_shift.methods.itemd_if import run_itemd
        result = run_itemd(self._make_epoch())
        assert result.headline_stat_name == "mean_if_slope"
        assert result.headline_stat >= 0

    def test_selects_theta_alpha_imf(self):
        from theta_alpha_shift.methods.itemd_if import run_itemd
        result = run_itemd(self._make_epoch())
        imf_freq = result.metadata["imf_mean_freqs"][result.metadata["selected_imf_idx"]]
        assert 3.0 <= imf_freq <= 15.0

    def test_tuple_input(self):
        from theta_alpha_shift.methods.itemd_if import run_itemd
        epoch = self._make_epoch()
        result = run_itemd((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.itemd_if import run_itemd
        result = run_itemd(self._make_epoch())
        assert "n_imfs" in result.metadata
        assert "selected_imf_idx" in result.metadata
        assert "if_slopes" in result.metadata
        assert "n_burst_segments" in result.metadata


class TestCDLWrap:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.cdl_wrap import run_cdl
        result = run_cdl(self._make_epoch())
        assert isinstance(result, MethodResult)
        assert result.method_name == "cdl"

    def test_headline_stat_name(self):
        from theta_alpha_shift.methods.cdl_wrap import run_cdl
        result = run_cdl(self._make_epoch())
        assert result.headline_stat_name == "n_cdl_atoms_theta_alpha"
        assert result.headline_stat >= 0

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.cdl_wrap import run_cdl
        result = run_cdl(self._make_epoch())
        assert "n_atoms" in result.metadata
        assert "n_atoms_theta_alpha" in result.metadata
        assert "atom_info" in result.metadata

    def test_tuple_input(self):
        from theta_alpha_shift.methods.cdl_wrap import run_cdl
        epoch = self._make_epoch()
        result = run_cdl((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_atom_info_structure(self):
        from theta_alpha_shift.methods.cdl_wrap import run_cdl
        result = run_cdl(self._make_epoch())
        for info in result.metadata["atom_info"]:
            assert "atom_idx" in info
            assert "center_freq" in info
            assert "bandwidth" in info
            assert "in_theta_alpha" in info


class TestThresholdBursts:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.threshold_bursts import run_threshold
        result = run_threshold(self._make_epoch())
        assert isinstance(result, MethodResult)
        assert result.method_name == "threshold_bursts"

    def test_headline_stat_name(self):
        from theta_alpha_shift.methods.threshold_bursts import run_threshold
        result = run_threshold(self._make_epoch())
        assert result.headline_stat_name == "burst_rate"
        assert result.headline_stat >= 0

    def test_detects_bursts(self):
        from theta_alpha_shift.methods.threshold_bursts import run_threshold
        result = run_threshold(self._make_epoch())
        assert len(result.detected_bursts) > 0
        burst = result.detected_bursts[0]
        assert "onset" in burst
        assert "offset" in burst
        assert "duration_s" in burst

    def test_tuple_input(self):
        from theta_alpha_shift.methods.threshold_bursts import run_threshold
        epoch = self._make_epoch()
        result = run_threshold((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.threshold_bursts import run_threshold
        result = run_threshold(self._make_epoch())
        assert "n_bursts" in result.metadata
        assert "burst_rate" in result.metadata
        assert "threshold" in result.metadata


class TestHMMWrap:
    def _make_epoch(self):
        return simulate_chirp(age=8, rng=np.random.default_rng(42))

    def test_returns_method_result(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        result = run_hmm(self._make_epoch())
        assert isinstance(result, MethodResult)
        assert result.method_name == "hmm"

    def test_headline_stat_name(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        result = run_hmm(self._make_epoch())
        assert result.headline_stat_name == "n_oscillatory_states"
        assert result.headline_stat >= 0

    def test_metadata_keys(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        result = run_hmm(self._make_epoch())
        assert "best_n_states" in result.metadata
        assert "bic_values" in result.metadata
        assert "state_spectra" in result.metadata
        assert "n_oscillatory_states" in result.metadata

    def test_tuple_input(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        epoch = self._make_epoch()
        result = run_hmm((epoch.data, epoch.sfreq))
        assert isinstance(result, MethodResult)

    def test_state_spectra_structure(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        result = run_hmm(self._make_epoch())
        for s in result.metadata["state_spectra"]:
            assert "state" in s
            assert "occupancy" in s
            assert "peak_freq" in s
            assert "has_theta_alpha_peak" in s

    def test_multichannel_averages(self):
        from theta_alpha_shift.methods.hmm_wrap import run_hmm
        from theta_alpha_shift.sim.forward_model import forward_project
        epoch = self._make_epoch()
        epoch_128 = forward_project(epoch, rng=np.random.default_rng(42))
        result = run_hmm(epoch_128)
        assert isinstance(result, MethodResult)
