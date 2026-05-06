# Time-resolved Theta–Alpha Shift Hypotheses

## Session Recovery (READ THIS FIRST)
If you are starting a new session, recovering from compaction, or running in a Ralph loop:
1. Read `.ralph/ralph_task.md` — the anchor. Has all checkboxes. Defines "done."
2. Read `.ralph/progress.md` — what exists, what is broken, what to do next.
3. Read `.ralph/guardrails.md` — learned constraints. Follow every sign.
4. Do NOT re-read the full codebase unless progress.md says something is broken. Trust the files.
5. Pick up the "Current Focus" from progress.md and work on it.
6. Before exiting or if context feels heavy: update progress.md with what you did and what is next.

## Adaptive Integration Protocol
This project extends the HBN EEG spectral analysis pipeline at `../hbn-eeg-pipeline/`.
Read that project's `CLAUDE.md` for full data schemas, epoch format spec, and key findings.
Before writing ANY code:
1. Scan the parent project for coding conventions: import ordering, docstring format,
   logging setup, naming conventions, directory layout.
2. Check the parent project's plotting configuration: colormaps, font sizes, axis labels,
   figure sizes, DPI, file format, style sheets.
3. Search the parent project for existing utility functions that overlap with what you need
   (PSD computation, channel selection, epoch rejection, filtering, montage setup, figure
   saving). Import and wrap rather than rewrite.
4. If the parent project uses specific MNE, specparam, or meeglet calling conventions
   (parameter choices, preprocessing steps), adopt them here.
5. Document all discovered conventions in `.ralph/progress.md` under "Parent Project
   Conventions" on the first iteration.

When in doubt, find the closest analog module in the parent project and use it as a
structural template.

## Compaction Instructions
When compacting this conversation, preserve:
- Current task and its completion state
- Any new guardrails discovered this session
- Any new known issues
- The exact next step to take
Do NOT preserve: file contents already read, full API/command outputs, failed approaches
(log failures to .ralph/errors.log instead).

## Project Purpose
This project tests two competing hypotheses about the developmental theta–alpha shift
observed in HBN EEG data. The "theta shelf" — a broadened left shoulder of the posterior
alpha peak in younger children — could arise from:

- **Hypothesis A (chirp):** within-burst frequency sweeps from theta to alpha, producing
  spectral smear when averaged across bursts.
- **Hypothesis B (mixture):** distinct theta-frequency and alpha-frequency burst
  populations co-exist, with the mixing ratio shifting toward alpha dominance with age.

**Stage 1** builds a NeuroDSP-based forward model to evaluate which burst-detection methods
discriminate these hypotheses. **Stage 2** applies the winners to real HBN EEG data. The
goal is seamless integration back into the parent HBN analysis project.

## Architecture
theta_alpha_shift/
├── sim/
│   ├── init.py
│   ├── aperiodic.py          # age-parameterized 1/f + knee backbone
│   ├── regimes.py             # 5 generative regime implementations
│   ├── artifacts.py           # artifact injection wrappers (EOG, EMG, line noise)
│   ├── forward_model.py       # 128-channel HBN-like forward projection
│   └── params.py              # developmental parameter curves (Freschl PAF, Cellier exponent)
├── methods/
│   ├── init.py
│   ├── specparam_baseline.py  # meeglet-PSD + specparam wrapper
│   ├── threshold_bursts.py    # fBOSC / PAPTO wrapper
│   ├── bycycle_wrap.py        # bycycle with relaxed thresholds + chirp statistics
│   ├── itemd_if.py            # itEMD + within-cycle IF wrapper
│   ├── cdl_wrap.py            # alphacsc CDL wrapper
│   ├── hmm_wrap.py            # osl-dynamics TDE-HMM wrapper
│   └── mp_wrap.py             # BEAD / chirped-MP wrapper (optional)
├── eval/
│   ├── init.py
│   ├── confusion.py           # per-method confusion matrices
│   ├── discrimination.py      # AUC, bias, variance calculations
│   ├── benchmarks.py          # computational cost profiling
│   └── robustness.py          # artifact injection + re-evaluation
├── hbn/
│   ├── init.py
│   ├── pipeline.py            # integration with parent HBN preprocessing
│   ├── miniset.py             # miniset (20 subjects/release) runner
│   ├── fullscale.py           # cluster-scale runner
│   └── stats.py               # mixed-effects models and effect-size tests
├── plots/
│   ├── init.py
│   └── figures.py             # all plotting — MUST match parent project style
├── tests/
│   ├── test_sim.py
│   ├── test_methods.py
│   └── test_eval.py
├── configs/
│   └── sim_params.yaml        # all simulation hyperparameters
└── notebooks/
├── 01_regime_sanity.ipynb
├── 02_method_eval.ipynb
└── 03_hbn_results.ipynb

## Key Schemas / Interfaces

### SimulatedEpoch (output of sim/regimes.py)
```python
@dataclass
class SimulatedEpoch:
    data: np.ndarray          # shape (n_channels, n_samples) or (n_samples,) for 1-ch
    sfreq: float              # sampling frequency in Hz
    age: int                  # simulated age in years
    regime: str               # 'chirp' | 'mixture' | 'drift' | 'cooccur' | 'narrowing'
    params: dict              # full parameter dict used to generate this epoch
    burst_times: np.ndarray   # ground-truth burst onset times (seconds)
    burst_freqs: np.ndarray   # ground-truth burst frequencies (or trajectories for chirp)
    burst_labels: np.ndarray  # ground-truth class labels (for mixture: 'theta' / 'alpha')
```

### MethodResult (output of each methods/ wrapper)
```python
@dataclass
class MethodResult:
    method_name: str
    detected_bursts: list     # list of dicts: {onset, offset, frequency, amplitude, ...}
    headline_stat: float      # single discriminative statistic
    headline_stat_name: str   # e.g. 'mean_if_slope', 'n_hmm_states', 'n_cdl_atoms'
    metadata: dict            # method-specific additional outputs
```

### Contract
Every method wrapper in methods/ accepts either a SimulatedEpoch or a tuple of
(raw_data: np.ndarray, sfreq: float) and returns a MethodResult. The eval/ module
iterates over methods generically via this interface.

## Environment
- Python >= 3.10
- Core: neurodsp, mne (>= 1.6), specparam (>= 1.1), alphacsc, osl-dynamics, emd, bycycle
- Testing: pytest
- Config: PyYAML for sim_params.yaml
- Plotting: matplotlib (parent project style), seaborn (sparingly)
- All versions pinned in requirements.txt at project creation

## Design Principles
1. **Conform to parent project.** Import ordering, logging, plotting style, docstrings,
   directory naming — all must match. When starting a new module, find the closest analog
   in the parent project and use it as a template.
2. **Parameters externalized.** All simulation parameters live in configs/sim_params.yaml.
   No magic numbers in code. Every function that uses a parameter loads it from config or
   receives it as an argument.
3. **Uniform method interface.** SimulatedEpoch in, MethodResult out. This enables generic
   evaluation loops and makes adding new methods trivial.
4. **Ground truth always available.** In simulation, always populate burst_times/burst_freqs/burst_labels in SimulatedEpoch.
   Never discard ground truth. Evaluation depends on it.
5. **Stage gate.** Stage 1 touches NO real data. Stage 2 is gated on Stage 1 triage results.
6. **Reproducibility.** Random seeds set everywhere. Package versions pinned. All parameters
   logged. Any result must be reproducible from config + code + seed.
7. **Adaptive discovery.** Before implementing any analysis step, consult the parent project
   to check whether a similar function already exists. Reuse and wrap rather than rewrite.
