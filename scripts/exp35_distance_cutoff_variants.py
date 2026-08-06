"""Exp35: distance-cutoff post-processing variants for WiDS 2026 submissions.

The training labels in this competition are perfectly separated by
``dist_min_ci_0_5h < 5000``: all 69 hits are already inside 5 km during the
first-five-hour feature window, and all censored rows are outside it.  This
script turns that observation into auditable submission variants without
retraining models.

It intentionally does not submit to Kaggle.  It only writes validated CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "event_id"
DIST_COL = "dist_min_ci_0_5h"
PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]


def enforce_monotonic(probs: pd.DataFrame) -> pd.DataFrame:
    """Clip to [0, 1] and enforce cumulative row-wise monotonicity."""
    arr = probs[PROB_COLS].to_numpy(dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.maximum.accumulate(arr, axis=1)
    out = probs.copy()
    out[PROB_COLS] = arr
    return out


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Validate Kaggle schema, IDs, bounds, duplicates, and monotonicity."""
    expected_cols = [ID_COL, *PROB_COLS]
    if list(sub.columns) != expected_cols:
        raise ValueError(f"bad columns: {list(sub.columns)} != {expected_cols}")
    if sub[ID_COL].duplicated().any():
        raise ValueError("duplicate event_id values")
    if set(sub[ID_COL]) != set(sample[ID_COL]):
        missing = set(sample[ID_COL]) - set(sub[ID_COL])
        extra = set(sub[ID_COL]) - set(sample[ID_COL])
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
    vals = sub[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise ValueError("non-finite probability values")
    if ((vals < -1e-12) | (vals > 1 + 1e-12)).any():
        raise ValueError("probability outside [0, 1]")
    if (np.diff(vals, axis=1) < -1e-12).any():
        raise ValueError("row-wise monotonicity violation")


def train_distance_report(train: pd.DataFrame, test: pd.DataFrame) -> str:
    """Return compact evidence for the 5 km cutoff assumption."""
    hit = train["event"].eq(1)
    cens = train["event"].eq(0)
    lines = [
        "Distance evidence:",
        f"- train hits: n={int(hit.sum())}, max_dist={train.loc[hit, DIST_COL].max():.3f} m",
        f"- train censored: n={int(cens.sum())}, min_dist={train.loc[cens, DIST_COL].min():.3f} m",
        f"- test near <5km: n={int((test[DIST_COL] < 5000).sum())}",
        f"- test far >=5km: n={int((test[DIST_COL] >= 5000).sum())}",
    ]
    return "\n".join(lines)


def make_variant(
    base: pd.DataFrame,
    test: pd.DataFrame,
    threshold: float,
    far_multiplier: float,
    power24: float | None,
) -> pd.DataFrame:
    """Apply distance gating and optional 24h power calibration."""
    sub = base.copy()
    sub[PROB_COLS] = sub[PROB_COLS].astype(float)

    if power24 is not None:
        sub["prob_24h"] = np.clip(sub["prob_24h"].to_numpy(dtype=float), 0.0, 1.0) ** power24

    far = test[DIST_COL].to_numpy(dtype=float) >= threshold
    sub.loc[far, PROB_COLS] = sub.loc[far, PROB_COLS].to_numpy(dtype=float) * far_multiplier
    return enforce_monotonic(sub)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/submission_exp34_full.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--out-dir", default="submissions")
    args = parser.parse_args()

    base_path = Path(args.base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    sample = pd.read_csv(args.sample)
    base = pd.read_csv(base_path)

    validate_submission(enforce_monotonic(base), sample)
    print(train_distance_report(train, test))

    variants = [
        ("hard5km", 5000.0, 0.0, None),
        ("soft5km_x001", 5000.0, 0.01, None),
        ("hard10km", 10000.0, 0.0, None),
        ("hard5km_p24_110", 5000.0, 0.0, 1.10),
        ("soft5km_x001_p24_110", 5000.0, 0.01, 1.10),
    ]

    for name, threshold, multiplier, power24 in variants:
        sub = make_variant(base, test, threshold, multiplier, power24)
        validate_submission(sub, sample)
        out = out_dir / f"{base_path.stem}_exp35_{name}.csv"
        sub.to_csv(out, index=False)
        far = test[DIST_COL].to_numpy(dtype=float) >= threshold
        means = sub.loc[far, PROB_COLS].mean().round(6).to_dict()
        print(f"wrote {out} | threshold={threshold:g} multiplier={multiplier:g} power24={power24} far_means={means}")


if __name__ == "__main__":
    main()
