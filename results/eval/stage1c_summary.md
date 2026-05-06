# Stage 1C — Method Discrimination and Triage

## Date: 2026-05-05

## Single-Epoch AUC (5 regimes × 6 ages × 5 trials, chirp vs mixture)

| Method            | AUC   | Notes                                                |
|-------------------|-------|------------------------------------------------------|
| specparam         | 0.543 | n_peaks too discrete for single-epoch ROC            |
| bycycle           | 0.527 | High variance on 2s epochs, signal emerges with aggregation |
| itEMD             | 0.585 | Discriminates in WRONG direction (mode mixing artifact) |
| CDL               | 0.557 | n_atoms too discrete; 5.1s/epoch prohibitive for HBN |
| HMM               | 0.533 | n_osc_states near-discrete; weak on single epochs     |
| threshold_bursts  | 0.533 | No discriminative capacity (as predicted by guardrail) |

## Multi-Epoch Aggregation (30 subjects × 50 epochs, chirp vs mixture)

### Bycycle (mean_period_slope)
| Age | AUC   | Cohen d | Chirp mean±sd | Mixture mean±sd |
|-----|-------|---------|---------------|-----------------|
|   5 | 0.660 |  +0.61  |  7.48±0.91    |  7.00±0.68      |
|   8 | 0.951 |  +2.30  |  8.87±0.87    |  7.19±0.57      |
|  11 | 0.954 |  +2.55  |  9.17±0.83    |  7.16±0.74      |
|  14 | 0.957 |  +2.53  |  9.71±1.05    |  7.25±0.89      |
|  17 | 0.980 |  +2.85  |  9.52±0.93    |  7.04±0.81      |
|  20 | 0.947 |  +2.26  |  9.18±1.07    |  7.09±0.75      |

### HMM (n_oscillatory_states)
| Age | AUC   | Cohen d | Chirp mean±sd | Mixture mean±sd |
|-----|-------|---------|---------------|-----------------|
|   5 | 0.604 |  +0.38  |  1.53±0.08    |  1.50±0.06      |
|   8 | 0.716 |  +0.79  |  1.58±0.06    |  1.52±0.08      |
|  11 | 0.528 |  +0.07  |  1.59±0.07    |  1.59±0.07      |
|  14 | 0.776 |  -1.10  |  1.56±0.07    |  1.63±0.06      |
|  17 | 0.763 |  -1.03  |  1.58±0.07    |  1.65±0.06      |
|  20 | 0.779 |  -1.21  |  1.56±0.06    |  1.64±0.07      |

## Computational Benchmarks

| Method           | Time/epoch | Memory  | HBN projected |
|------------------|-----------|---------|---------------|
| bycycle          | <0.001s   | <1 MB   | <0.1h         |
| threshold_bursts | <0.001s   | <1 MB   | <0.1h         |
| specparam        | 0.001s    | 39.5 MB | 0.1h          |
| HMM              | 0.053s    | 9.4 MB  | 4.4h          |
| itEMD            | 0.097s    | 10.8 MB | 8.1h          |
| CDL              | 5.074s    | 30.4 MB | 422.9h        |

## Method Triage Decision

### Survivors → Stage 2
1. **bycycle (mean_period_slope)** — PRIMARY discriminator
   - AUC 0.95+ for ages 8-20 with multi-epoch aggregation
   - Cohen's d 2.3-2.9 (very large effect)
   - Negligible compute cost (<0.001s/epoch)
   - Scientific rationale: chirp bursts have systematic within-burst period variation
     (monotonically changing cycle durations), while mixture bursts have constant-frequency
     cycles. Period slope captures this directly.
   - Limitation: weak at age 5 (AUC=0.66) where theta-alpha overlap is maximal

2. **HMM (n_oscillatory_states)** — SECONDARY/confirmatory
   - AUC 0.72-0.78 for ages 14-20 (moderate discrimination)
   - Moderate compute cost (0.05s/epoch, ~4h for HBN)
   - Direction flips with age: at young ages, chirp shows slightly more states;
     at older ages, mixture shows more states. Interpretation requires care.

3. **specparam (n_peaks_theta_alpha)** — BASELINE measure
   - Not discriminative on its own, but needed for spectral characterization
   - Confirms theta-shelf presence and tracks peak frequency with age
   - Negligible compute cost

### Dropped
4. **itEMD (mean_if_slope)** — DROPPED
   - Discriminates in WRONG direction: mixture shows higher IF slopes than chirp
   - Cause: mode mixing between theta (~5 Hz) and alpha (~8 Hz) creates spurious
     IF jumps in the mixture case. Known EMD limitation when components are less
     than an octave apart (see guardrails).
   - Could be revisited with different IMF selection or masking strategies.

5. **CDL (n_cdl_atoms_theta_alpha)** — DROPPED
   - 423h projected for full HBN — computationally infeasible
   - Discrete atom count has low discriminative resolution
   - Would need multi-epoch concatenation to learn meaningful atoms

6. **threshold_bursts (burst_rate)** — KEPT only for burst-rate metrics
   - No discriminative capacity for hypothesis testing (AUC=0.53)
   - Confirms guardrail: threshold methods detect bursts but don't discriminate chirp vs mixture

## Decision Rule Assessment
- Required: AUC ≥ 0.85 AND consistency across ≥ 2 orthogonal method families
- Bycycle: AUC ≥ 0.85 ✓ (ages 8-20)
- HMM: AUC 0.72-0.78 ✗ (below threshold, but provides independent evidence)
- Assessment: Bycycle alone meets the AUC criterion. HMM provides orthogonal
  confirmatory evidence below the threshold. Proceed to Stage 2 with bycycle
  as primary and HMM as secondary, but note that the "2 method families" criterion
  is only partially met.

## Key Scientific Insights
1. Single 2s epochs are insufficient for reliable chirp-vs-mixture discrimination
   with ANY method. Multi-epoch aggregation (~50 epochs) is essential.
2. Bycycle's period slope is the strongest discriminator because it directly measures
   within-burst frequency structure without spectral decomposition.
3. Mode mixing between close-frequency components (theta/alpha < 1 octave apart)
   invalidates EMD-based IF analysis for this specific application.
4. The theta-alpha frequency proximity (5 Hz vs 6.5-10 Hz) is the fundamental challenge:
   methods that rely on spectral separation (specparam, CDL) struggle because the
   two hypotheses produce overlapping spectra by design.

## Stage 2 Plan

### Primary Analysis (bycycle)
1. Compute mean_period_slope per subject (aggregate across all available EC epochs)
2. Run on HBN miniset first (20 subjects/release, age-balanced): confirm signal replicates
3. Scale to full HBN (~3,000 subjects) or age-stratified subsample (n~500/bin)
4. Mixed-effects model: period_slope ~ age + (1|site), test for age interaction
5. Hypothesis adjudication: if period_slope decreases with age → supports chirp (A);
   if period_slope is flat across ages → supports mixture (B)

### Secondary Analysis (HMM)
1. Fit 2-state and 3-state TDE-HMM per subject (concatenate all EC epochs)
2. Extract per-state spectra; count states with clear theta-alpha peaks
3. Test whether n_oscillatory_states correlates with age and agrees with bycycle

### Baseline (specparam)
1. Confirm theta-shelf presence in youngest cohorts
2. Track n_peaks and peak_freq across age bins
3. Cross-validate against existing parent project meeglet+specparam outputs

### Decision Criteria
- If bycycle shows large (d≥0.3) age-related decline in period_slope AND HMM confirms
  fewer distinct states at older ages: SUPPORT Hypothesis A (chirp)
- If bycycle shows flat period_slope AND HMM shows persistent 2+ states: SUPPORT
  Hypothesis B (mixture)
- If results are mixed or effect sizes < 0.2: INCONCLUSIVE — report as hybrid/null

### Parallel Deliverable
Build and deploy interactive web app (Railway) with About page (scholarly report) and
Simulator page (interactive parameter exploration) concurrently with Stage 2 analysis.
