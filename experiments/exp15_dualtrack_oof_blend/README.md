# Exp15: OOF-Learned Dual-Track Blend

## Goal
- Implement a production-ready pipeline with:
  - OOF weight learning
  - full-retrain dual-track fusion
  - explicit low-weight constraint on EST

## Design
- `Track-Rank`: `RSF + EST` with `EST <= est_cap`, tuned on 12h C-index.
- `Track-Calib`: `RSF + LogNormalAFT` with per-horizon weights.
  - 12h by C-index
  - 24h/48h by horizon Brier
  - 72h fixed (final postprocess sets 1.0)
- `Dual-Track`: per-horizon alpha blend of the two tracks.

## Run
```bash
python -m experiments.exp15_dualtrack_oof_blend.train
```

Common override example:
```bash
python -m experiments.exp15_dualtrack_oof_blend.train \
  --n-repeats 10 \
  --est-cap 0.15 \
  --lognormal-cap 0.30 \
  --weight-step 0.05
```

## Outputs
- Submission: `experiments/exp15_dualtrack_oof_blend/submission.csv`
- Report: `experiments/exp15_dualtrack_oof_blend/RESULTS_2026-02-13.md`

