"""Exp30: Deep learning survival models via pycox.

Step 1: DeepSurv (pycox.CoxPH) — low-variance baseline.
Step 2: LogisticHazard — discrete-time, native 12/24/48/72.
Step 3: DeepHitSingle — only if Step 1-2 show signal.

Constraints:
  - v96624 16 features
  - Small net: 2 layers [64,64], dropout=0.3, weight_decay=1e-4
  - Early stopping patience=10
  - Same CV protocol: 5-fold 10-repeat stratified
  - Stop gate: OOF hybrid must beat baseline by +0.003
"""

import sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from pycox.models import CoxPH as PycoxCoxPH
from pycox.models import LogisticHazard

from src.config import (
    TRAIN_PATH, TEST_PATH, ID_COL, TIME_COL, EVENT_COL,
    HORIZONS, N_SPLITS, N_REPEATS, RANDOM_STATE,
)
from src.features import add_engineered, get_feature_set
from src.evaluation import hybrid_score, c_index, horizon_brier_score
from src.train import _strat_labels

# ---------- Config ----------
FEATURE_LEVEL = "v96624"
NET_LAYERS = [64, 64]
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
LR = 1e-3
BATCH_SIZE = 256
EPOCHS = 200
PATIENCE = 10
STOP_GATE = 0.003  # OOF hybrid must beat baseline by this much


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train = add_engineered(train)
    test = add_engineered(test)
    return train, test


def make_net(in_features, out_features=1):
    """Small MLP with dropout + batch norm."""
    return tt.practical.MLPVanilla(
        in_features, NET_LAYERS, out_features,
        batch_norm=True, dropout=DROPOUT,
    )


# ---------- DeepSurv (CoxPH) ----------

def deepsurv_fit_predict(X_tr, y_tr, X_va, horizons):
    """Train DeepSurv on (X_tr, y_tr), predict P(T<=h) on X_va."""
    in_f = X_tr.shape[1]
    net = make_net(in_f, 1)
    model = PycoxCoxPH(net, tt.optim.Adam(weight_decay=WEIGHT_DECAY))

    # pycox expects float32 arrays
    x_tr = X_tr.values.astype('float32')
    x_va = X_va.values.astype('float32')
    y_time = y_tr[0].astype('float32')
    y_event = y_tr[1].astype('float32')

    # Validation split for early stopping: use last 15% of training
    n = len(x_tr)
    n_val = max(int(n * 0.15), 50)
    idx = np.random.permutation(n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    val = (x_tr[val_idx], (y_time[val_idx], y_event[val_idx]))

    model.fit(
        x_tr[tr_idx], (y_time[tr_idx], y_event[tr_idx]),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[tt.callbacks.EarlyStopping(patience=PATIENCE)],
        val_data=val, verbose=False,
    )

    # Compute baseline hazards on full training set
    model.compute_baseline_hazards(input=x_tr, target=(y_time, y_event))

    # Predict survival function on validation set
    surv = model.predict_surv_df(x_va)

    # Extract P(T <= h) = 1 - S(h) for each horizon
    preds = {}
    times = surv.index.values
    for h in horizons:
        if h <= times.min():
            preds[h] = np.zeros(len(x_va))
        elif h >= times.max():
            preds[h] = 1.0 - surv.iloc[-1].values
        else:
            # Interpolate
            idx_after = np.searchsorted(times, h, side='right')
            idx_before = max(idx_after - 1, 0)
            if idx_after >= len(times):
                preds[h] = 1.0 - surv.iloc[-1].values
            else:
                t0, t1 = times[idx_before], times[idx_after]
                w = (h - t0) / (t1 - t0) if t1 > t0 else 0.5
                s_interp = (1 - w) * surv.iloc[idx_before].values + w * surv.iloc[idx_after].values
                preds[h] = np.clip(1.0 - s_interp, 0, 1)

    return preds


# ---------- LogisticHazard ----------

def loghaz_fit_predict(X_tr, y_tr, X_va, horizons, cuts):
    """Train LogisticHazard, predict P(T<=h) on X_va."""
    in_f = X_tr.shape[1]
    out_f = len(cuts)
    net = make_net(in_f, out_f)
    model = LogisticHazard(net, tt.optim.Adam(weight_decay=WEIGHT_DECAY))

    x_tr = X_tr.values.astype('float32')
    x_va = X_va.values.astype('float32')

    # Discretize targets
    labtrans = LogisticHazard.label_transform(cuts)
    y_disc = labtrans.fit_transform(y_tr[0].astype('float64'), y_tr[1].astype('float64'))

    # Internal val split for early stopping
    n = len(x_tr)
    n_val = max(int(n * 0.15), 50)
    idx = np.random.permutation(n)
    vi, ti = idx[:n_val], idx[n_val:]

    val_y = (y_disc[0][vi], y_disc[1][vi])
    val = (x_tr[vi], val_y)

    model.fit(
        x_tr[ti], (y_disc[0][ti], y_disc[1][ti]),
        batch_size=BATCH_SIZE, epochs=EPOCHS,
        callbacks=[tt.callbacks.EarlyStopping(patience=PATIENCE)],
        val_data=val, verbose=False,
    )

    surv = model.predict_surv_df(x_va)
    preds = {}
    times = surv.index.values
    for h in horizons:
        if h >= times.max():
            preds[h] = 1.0 - surv.iloc[-1].values
        elif h <= times.min():
            preds[h] = 1.0 - surv.iloc[0].values
        else:
            idx_after = np.searchsorted(times, h, side='right')
            idx_before = max(idx_after - 1, 0)
            if idx_after >= len(times):
                preds[h] = 1.0 - surv.iloc[-1].values
            else:
                t0, t1 = times[idx_before], times[idx_after]
                w = (h - t0) / (t1 - t0) if t1 > t0 else 0.5
                s_interp = (1 - w) * surv.iloc[idx_before].values + w * surv.iloc[idx_after].values
                preds[h] = np.clip(1.0 - s_interp, 0, 1)
    return preds


# ---------- CV Runner ----------

def run_dl_cv(train, feature_cols, model_type="deepsurv"):
    """Run repeated stratified K-fold CV for a DL model."""
    X = train[feature_cols]
    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values
    n = len(train)

    oof = {h: np.zeros(n) for h in HORIZONS}
    oof_counts = np.zeros(n)

    strat = _strat_labels(y_time, y_event)
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE,
    )

    # LogisticHazard needs time cuts
    cuts = np.array([0, 6, 12, 18, 24, 36, 48, 60, 72, 96, 120, 168], dtype='float64')

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X, strat):
        fold_idx += 1
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X.iloc[tr_idx]),
                            columns=feature_cols, index=X.iloc[tr_idx].index)
        X_va = pd.DataFrame(scaler.transform(X.iloc[va_idx]),
                            columns=feature_cols, index=X.iloc[va_idx].index)
        yt_tr = y_time[tr_idx]
        ye_tr = y_event[tr_idx]

        np.random.seed(RANDOM_STATE + fold_idx)
        torch.manual_seed(RANDOM_STATE + fold_idx)

        if model_type == "deepsurv":
            preds = deepsurv_fit_predict(X_tr, (yt_tr, ye_tr), X_va, HORIZONS)
        elif model_type == "loghaz":
            preds = loghaz_fit_predict(X_tr, (yt_tr, ye_tr), X_va, HORIZONS, cuts)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        for h in HORIZONS:
            oof[h][va_idx] += preds[h]
        oof_counts[va_idx] += 1

        if fold_idx % N_SPLITS == 0:
            rep = fold_idx // N_SPLITS
            # Partial OOF score
            mask = oof_counts > 0
            partial = {h: np.where(mask, oof[h] / oof_counts, 0.5) for h in HORIZONS}
            ci = c_index(y_time[mask], y_event[mask], partial[12][mask])
            print(f"  Repeat {rep}/{N_REPEATS} done  (partial CI={ci:.4f})")

    # Average
    mask = oof_counts > 0
    for h in HORIZONS:
        oof[h][mask] /= oof_counts[mask]

    return oof


# ---------- Main ----------

def main():
    print("=== Exp30: Deep Learning Survival Models ===")
    train, test = load_data()
    feature_cols = get_feature_set(train, level=FEATURE_LEVEL)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    y_time = train[TIME_COL].values
    y_event = train[EVENT_COL].values

    # Load anchor baseline for comparison
    ref_path = "submissions/submission_0.96624.csv"
    ref = pd.read_csv(ref_path)

    # ---- Step 1: DeepSurv ----
    print("\n" + "="*60)
    print("  STEP 1: DeepSurv (pycox.CoxPH)")
    print("="*60)

    oof_ds = run_dl_cv(train, feature_cols, model_type="deepsurv")

    # Force 72h = 1.0
    oof_ds[72] = np.ones(len(train))

    score_ds, det_ds = hybrid_score(y_time, y_event, oof_ds)
    print(f"\n  DeepSurv OOF: Hybrid={score_ds:.4f}  CI={det_ds['c_index']:.4f}  "
          f"WBrier={det_ds['weighted_brier']:.4f}")
    for h in HORIZONS:
        vals = oof_ds[h]
        print(f"    {h}h: min={vals.min():.4f} median={np.median(vals):.4f} max={vals.max():.4f}")

    # Spearman vs anchor
    print("\n  Spearman vs anchor (0.96624):")
    # Need OOF from current pipeline for comparison — use train.py baseline
    # For now, compare distribution shape
    for h in [12, 24, 48]:
        col = f"prob_{h}h"
        # Can't directly compare OOF vs test submission, but show distribution
        print(f"    {h}h OOF: p10={np.percentile(oof_ds[h], 10):.4f} "
              f"p50={np.median(oof_ds[h]):.4f} p90={np.percentile(oof_ds[h], 90):.4f}")

    # ---- Step 2: LogisticHazard ----
    print("\n" + "="*60)
    print("  STEP 2: LogisticHazard (discrete-time)")
    print("="*60)

    oof_lh = run_dl_cv(train, feature_cols, model_type="loghaz")
    oof_lh[72] = np.ones(len(train))

    score_lh, det_lh = hybrid_score(y_time, y_event, oof_lh)
    print(f"\n  LogisticHazard OOF: Hybrid={score_lh:.4f}  CI={det_lh['c_index']:.4f}  "
          f"WBrier={det_lh['weighted_brier']:.4f}")
    for h in HORIZONS:
        vals = oof_lh[h]
        print(f"    {h}h: min={vals.min():.4f} median={np.median(vals):.4f} max={vals.max():.4f}")

    # ---- Summary ----
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  DeepSurv:       Hybrid={score_ds:.4f}  CI={det_ds['c_index']:.4f}  WBrier={det_ds['weighted_brier']:.4f}")
    print(f"  LogisticHazard: Hybrid={score_lh:.4f}  CI={det_lh['c_index']:.4f}  WBrier={det_lh['weighted_brier']:.4f}")

    # Baseline reference (from experiments.md: RSF+EST OOF ~0.9050 hybrid)
    # The exact baseline will be printed for manual comparison
    print("\n  Compare against current RSF+EST baseline in experiments.md")
    print(f"  Stop gate: need +{STOP_GATE:.3f} hybrid improvement to continue DL direction")

    best_dl = max(score_ds, score_lh)
    best_name = "DeepSurv" if score_ds >= score_lh else "LogisticHazard"
    print(f"\n  Best DL model: {best_name} (Hybrid={best_dl:.4f})")


if __name__ == "__main__":
    main()
