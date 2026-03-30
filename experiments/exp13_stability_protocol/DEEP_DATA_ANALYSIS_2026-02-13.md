# Deep Data Analysis (2026-02-13)

## 1. Dataset geometry

- Train: `221 x 37` (35 features + `event_id`, `time_to_hit_hours`, `event`)
- Test: `95 x 35`
- Event rate: `69/221 = 31.2%`
- Missingness: none in train/test feature columns
- Strong zero inflation in dynamic features:
  - `closing_speed_m_per_h`, `dist_change_ci_0_5h`, `projected_advance_m`: `~91.9%` zeros
  - `dist_std_ci_0_5h`, `dist_fit_r2_0_5h`: `~91.4%` zeros
  - Growth-direction features often `~88%` zeros

Interpretation:

- Most samples are low-information snapshots (single/noisy early perimeter dynamics).
- Model learns strong rules from a few high-signal variables; many engineered dynamics are sparse.

## 2. Censoring and label mechanics

- Horizon-eligible sample counts drop fast:
  - `12h: 215` eligible (`49` positive)
  - `24h: 196` eligible (`63` positive)
  - `48h: 166` eligible (`66` positive)
  - `72h: 69` eligible (`69` positive, degenerate)
- Censoring is concentrated in the tail:
  - `[60,66)`: `54` censored, `0` events
  - `[66,72)`: `27` censored, `1` event

Interpretation:

- `72h` has almost no learning signal and acts mostly as a postprocess constant.
- Score is effectively dominated by 12h ranking + 24/48h calibration.

## 3. Event-time process shape

Empirical discrete hazard (events / risk set, 6-hour bins):

- `[0,6)`: `0.199`
- `[6,12)`: `0.029`
- `[12,24)`: `~0.04-0.05`
- Later bins mostly near zero with sparse long tail

Interpretation:

- Process is heavily front-loaded (early-hit or never-hit profile).
- This supports split strategies by early-vs-late regime and robust handling of tail censoring.

## 4. Dominant predictor: distance (near-deterministic regime split)

Observed from train labels:

- For all `h in {12,24,48}`, when `dist_min_ci_0_5h > 5000`, positive rate is `0.0` in train.
- Around 4.5–5.5 km there is a sharp phase transition:
  - `4500-5500m`: mixed outcomes (small n)
  - `>5500m`: effectively all negatives in train

Test composition:

- `dist > 5000`: `67/95 = 70.5%` of test
- `dist > 4500`: `69/95 = 72.6%` of test

Interpretation:

- The task is almost a gated problem:
  - Stage 1: distance-driven regime assignment
  - Stage 2: near-range risk ordering/calibration
- OOF can look very strong if gate matches train, but LB may drop if private has positives beyond train range.

## 5. Secondary signal in near-range subset (<= 5km)

Inside `dist <= 5000`, top discriminators (12h) are:

- `dt_first_last_0_5h` (`abs AUC ~0.81`)
- `num_perimeters_0_5h` (`~0.81`)
- `low_temporal_resolution_0_5h` (inverse signal, `~0.80`)
- `alignment_abs` (`~0.78`)

Interpretation:

- Distance saturates globally; inside near-range, temporal coverage and directional alignment drive ranking.
- This is where feature engineering and model blending actually matter.

## 6. Model error concentration (OOF RSF, event_time split)

Worst calibration/Brier segment appears consistently:

- `dist 1500-3000m` + `low_temporal_resolution=1` + `num_perimeters<2`
- This segment is more frequent in test than train:
  - Train: `19/221 = 8.6%`
  - Test: `11/95 = 11.6%`

Interpretation:

- This is the current generalization bottleneck.
- Improving this subgroup can yield larger real-world gain than global micro-tuning.

## 7. Train-test shift

Moderate but meaningful shift:

- More low-information cases in test:
  - `low_temporal_resolution=1`: `+5.0%`
  - `num_perimeters=1`: `+5.5%`
- Distance regime shift:
  - `>4545m`: `+3.4%`
  - Near bands (`3000-4545`, `1500-3000`) are reduced

Out-of-range test values exist but small (`~1.1%` on 4 features):

- `dist_min_ci_0_5h` includes one test sample below train minimum
- `num_perimeters_0_5h` includes values up to 19 (train max 17)

Interpretation:

- LB variance is likely from subgroup composition + hard-threshold behavior, not random noise only.

## 8. Redundancy and effective dimensionality

High-correlation pairs (`|r| >= 0.95`): 14 pairs, including exact/near-exact duplicates:

- `area_growth_rel_0_5h` ↔ `relative_growth_0_5h`
- `dist_change_ci_0_5h` ↔ `projected_advance_m`
- `dist_change_ci_0_5h` ↔ `closing_speed_m_per_h`
- Multiple centroid/radial duplicates

Interpretation:

- Effective independent feature count is much lower than raw column count.
- Aggressive complexity increases overfit risk on `n=221`.

## 9. Why OOF micro-gains often fail on LB

Observed facts:

- CV noise floor for Hybrid is around `~2e-4 to 3e-4`.
- Many candidate gains are in `+1e-4 to +4e-4` range.
- Test has higher share of hard/ambiguous low-information segments.

Conclusion:

- OOF improvements below noise gate and without subgroup robustness checks should be treated as non-actionable.

## 10. Actionable modeling roadmap

Priority 1:

- Keep `event_time` composite stratification as default.
- Stop forcing fixed RSF/EST caps by default; treat cap as optional policy.
- Use multi-seed stability gate for acceptance (`delta > 1 std noise`).

Priority 2:

- Build a two-stage model:
  - Stage A: soft distance gate (not hard zero), calibrated for `>4.5km` uncertainty.
  - Stage B: near-range expert model (`<=5km`) emphasizing `dt_first_last`, `num_perimeters`, `alignment`, resolution flags.

Priority 3:

- Segment-aware calibration:
  - Calibrate separately for key uncertain segment (`1500-3000m`, low-res, perimeter=1).
  - Apply conservative floor/temperature for far-distance probabilities to reduce threshold brittleness.

Priority 4:

- Feature policy:
  - Keep redundancy-pruned core.
  - Add only features with demonstrated near-range subgroup lift.
  - Avoid global complexity expansion without subgroup payoff.

