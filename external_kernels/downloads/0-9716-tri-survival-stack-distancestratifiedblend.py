#!/usr/bin/env python3
"""
WiDS Global Datathon 2026 — Tri-Survival Stack v7
===================================================
Changes vs v6 (LB=0.97235):

  CALIBRATION EXPERIMENT
  ----------------------
  Submission A (baseline):  v6 logic + proven +0.09 right-tail shift
  Submission B (isotonic):  same + isotonic calibration on far-zone OOF (24h, 48h)
                            + rule-based left-tail push for far-zone

  Philosophy: simplicity over complexity. No new models, no new weight tuning.
  All v4 near/far weights locked. isotonic only applied where OOF is reliable
  (152 far-zone samples). LB decides whether isotonic helps.

  Order of operations (both submissions):
    1. Blend (zone-stratified, locked weights)
    2. POWER_CAL_24 = 1.1
    3. enforce_monotonicity
    4. +0.09 right-tail shift (p >= 0.9 -> p + 0.09)
    5. [Submission B only] isotonic on far-zone 24h/48h + left-tail push
    6. clip to [0, 1]
    7. enforce_monotonicity again (safety pass)

  Diagnostics:
    - OOF Brier per zone per horizon printed after each calibration step
    - Prediction distribution histogram (near vs far, per horizon)
    - Isotonic OOF gain reported — if < 0.0003, flagged as noise
"""

import os, warnings
warnings.filterwarnings("ignore")

DATA_DIR      = "/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26"
OUTPUT_PATH_A = "/kaggle/working/submission_A_baseline.csv"
OUTPUT_PATH_B = "/kaggle/working/submission_B_isotonic.csv"

RUN_MODE    = "full"
DO_OOF      = True
CV_BAG_TEST = True
STRAT_THR   = 5000
TIMING_MODEL_WEIGHT = 0.0

GBSA_SEEDS_FULL = (
    123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
    279, 239, 70, 77, 31, 2024, 2077, 3077, 123456, 654321,
    4640, 841, 7755, 8525, 2701, 8817, 8864, 4085, 8919, 934,
    4746, 1699, 7401, 7826, 4098, 2921, 1204, 2752, 8384, 1284,
)
GBSA_SEEDS_FAST = tuple(range(42, 52))
COX_SEEDS_FULL  = (123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
                    279, 239, 70, 77, 31, 2024, 2077, 3077, 123456, 654321)
COX_SEEDS_FAST  = (123, 456, 789)
RSF_SEEDS_FULL  = (123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
                    279, 239, 70, 77, 31)
RSF_SEEDS_FAST  = (123, 456, 789)
LGB_NEAR_SEEDS_FULL = (123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
                        279, 239, 70, 77, 31, 2024, 2077, 3077, 123456, 654321,
                        2034, 2035, 2036, 1984, 1991, 3255, 1011, 6241, 2790, 6847)
LGB_FAR_SEEDS_FULL  = (8141, 7752, 432, 906, 6217, 7785, 1603, 7609, 965, 2506,
                        3771, 7080, 4963, 7939, 2751, 473, 339, 3675, 5535, 4760,
                        123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033)
LGB_NEAR_SEEDS_FAST = tuple(range(42, 52))
LGB_FAR_SEEDS_FAST  = tuple(range(52, 62))

# ── Near-zone: LOCKED to v4 (best LB=0.97167). DO NOT change. ─────────────
W_GBSA_NEAR_12 = 0.76;  W_COX_NEAR_12 = 0.12;  W_RSF_NEAR_12 = 0.02;  W_LGB_NEAR_12 = 0.10
W_GBSA_NEAR_24 = 0.82;  W_COX_NEAR_24 = 0.14;  W_RSF_NEAR_24 = 0.02;  W_LGB_NEAR_24 = 0.02
W_GBSA_NEAR_48 = 0.73;  W_COX_NEAR_48 = 0.16;  W_RSF_NEAR_48 = 0.03;  W_LGB_NEAR_48 = 0.08

# ── Far-zone: v4 values ────────────────────────────────────────────────────
W_GBSA_FAR_24 = 0.62;  W_COX_FAR_24 = 0.25;  W_RSF_FAR_24 = 0.06;  W_LGB_FAR_24 = 0.07
W_GBSA_FAR_48 = 0.35;  W_COX_FAR_48 = 0.22;  W_RSF_FAR_48 = 0.06;  W_LGB_FAR_48 = 0.37

POWER_CAL_24     = 1.1
RIGHT_TAIL_SHIFT = 0.09   # applied to any p >= 0.9 across all horizons
LEFT_TAIL_CAP    = 0.01   # far-zone predictions pushed down to this
LEFT_TAIL_THR    = 0.03   # far-zone threshold below which left-tail push applies
P72_MODE         = "constant1"
HORIZONS_PRED    = [12, 24, 48, 72]

# ─────────────────────────────────────────────────────────────────────────────
# Installs + imports
# ─────────────────────────────────────────────────────────────────────────────
def _install(pkg, import_name=None):
    name = import_name or pkg
    try: __import__(name)
    except Exception:
        print(f"[INSTALL] {pkg}")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_install("scikit-survival", "sksurv")
_install("lightgbm")
_install("scikit-learn", "sklearn")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats

train_df   = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df    = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

print("Train:", train_df.shape, " | Test:", test_df.shape)
print("Events:", train_df["event"].value_counts().to_dict())
print(f"Near (<5km): {(train_df.dist_min_ci_0_5h < 5000).sum()} "
      f"| Far (>=5km): {(train_df.dist_min_ci_0_5h >= 5000).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
def create_features(df, fit_df=None):
    ref = fit_df if fit_df is not None else df
    result = df.copy()
    dist       = result["dist_min_ci_0_5h"].clip(lower=1)
    speed      = result["closing_speed_m_per_h"]
    perimeters = result["num_perimeters_0_5h"]
    area_first = result["area_first_ha"]

    result["log_distance"]    = np.log1p(dist)
    result["inv_distance"]    = 1 / (dist / 1000 + 0.1)
    result["inv_distance_sq"] = result["inv_distance"] ** 2
    result["sqrt_distance"]   = np.sqrt(dist)
    result["dist_km"]         = dist / 1000
    result["dist_km_sq"]      = (dist / 1000) ** 2
    result["dist_km_cb"]      = (dist / 1000) ** 3
    result["dist_rank"]       = dist.rank(pct=True)

    fire_radius               = np.sqrt(area_first * 10000 / np.pi)
    result["fire_radius_km"]  = fire_radius / 1000
    result["radius_to_dist"]  = fire_radius / dist
    result["area_to_dist_ratio"]  = area_first / (dist / 1000 + 0.1)
    result["log_area_dist_ratio"] = np.log1p(area_first) - np.log1p(dist)

    result["has_movement"] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    result["eta_hours"]    = np.where(closing_pos > 0.01, dist / closing_pos, 9999).clip(max=9999)
    result["log_eta"]      = np.log1p(result["eta_hours"].clip(0, 9999))
    radial_growth = result["radial_growth_rate_m_per_h"].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    result["effective_closing_speed"] = effective_closing
    result["eta_effective"]  = np.where(effective_closing > 0.01, dist / effective_closing, 9999).clip(max=9999)
    result["threat_score"]   = result["alignment_abs"] * speed / np.log1p(dist)
    result["threat_score_sq"]= result["threat_score"] ** 2
    result["fire_urgency"]    = perimeters * speed
    result["growth_intensity"]= result["area_growth_rate_ha_per_h"] * perimeters

    result["zone_near"]    = (dist < 5000).astype(float)
    result["zone_warning"] = ((dist >= 5000) & (dist < 10000)).astype(float)
    result["zone_far"]     = (dist >= 10000).astype(float)

    ref_near_mask = ref["dist_min_ci_0_5h"].clip(lower=1) < 5000
    ref_far_mask  = ~ref_near_mask
    near_speed_ref = ref.loc[ref_near_mask, "closing_speed_m_per_h"].values
    far_threat_ref = (ref.loc[ref_far_mask, "alignment_abs"] *
                      ref.loc[ref_far_mask, "closing_speed_m_per_h"] /
                      np.log1p(ref.loc[ref_far_mask, "dist_min_ci_0_5h"].clip(lower=1))).values

    def rank_against_ref(vals, ref_vals):
        return np.array([(ref_vals < v).mean() for v in vals])

    cur_near_mask = dist < 5000
    cur_far_mask  = ~cur_near_mask
    near_speed_rank = np.zeros(len(result))
    far_threat_rank = np.zeros(len(result))
    if cur_near_mask.sum() > 0:
        near_speed_rank[cur_near_mask.values] = rank_against_ref(
            speed[cur_near_mask].values, near_speed_ref)
    if cur_far_mask.sum() > 0:
        threat_cur_far = (result.loc[cur_far_mask, "alignment_abs"] *
                          speed[cur_far_mask] /
                          np.log1p(dist[cur_far_mask])).values
        far_threat_rank[cur_far_mask.values] = rank_against_ref(
            threat_cur_far, far_threat_ref)
    result["near_speed_rank"] = near_speed_rank
    result["far_threat_rank"] = far_threat_rank

    result["is_summer"]    = result["event_start_month"].isin([6, 7, 8]).astype(float)
    result["is_afternoon"] = ((result["event_start_hour"] >= 12) &
                               (result["event_start_hour"] < 20)).astype(float)

    drop_cols = [
        "relative_growth_0_5h", "projected_advance_m",
        "centroid_displacement_m", "centroid_speed_m_per_h",
        "closing_speed_abs_m_per_h", "area_growth_abs_0_5h",
    ]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result


train_processed = create_features(train_df, fit_df=train_df)
test_processed  = create_features(test_df,  fit_df=train_df)

print("Engineered features:", len([c for c in train_processed.columns
                                    if c not in ["event_id","event","time_to_hit_hours"]]))

# ─────────────────────────────────────────────────────────────────────────────
# Competition metric (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def compute_c_index(time, event, risk):
    n = len(time); concordant = comparable = 0
    for i in range(n):
        if event[i] != 1: continue
        for j in range(n):
            if i==j or time[i]>=time[j]: continue
            comparable += 1
            if risk[i] > risk[j]:    concordant += 1
            elif risk[i]==risk[j]:   concordant += 0.5
    return concordant/comparable if comparable > 0 else 0.5

def compute_brier(time, event, prob, horizon):
    valid  = ~((event==0) & (time<horizon))
    if valid.sum() == 0: return 0.25
    y_true = ((event==1) & (time<=horizon)).astype(float)[valid]
    return float(np.mean((np.clip(prob[valid],0,1) - y_true)**2))

def compute_hybrid_score(time, event, p24, p48, p72):
    risk  = 0.3*p24 + 0.4*p48 + 0.3*p72
    c_idx = compute_c_index(time, event, risk)
    b24 = compute_brier(time, event, p24, 24)
    b48 = compute_brier(time, event, p48, 48)
    b72 = compute_brier(time, event, p72, 72)
    wb  = 0.3*b24 + 0.4*b48 + 0.3*b72
    return 0.3*c_idx + 0.7*(1-wb), c_idx, wb

def enforce_monotonicity(preds):
    result = np.clip(preds, 0, 1)
    for i in range(1, result.shape[1]):
        result[:,i] = np.maximum(result[:,i], result[:,i-1])
    return result

def get_surv_predictions(model, X):
    surv_fns = model.predict_survival_function(X)
    preds = np.empty((len(surv_fns), len(HORIZONS_PRED)), dtype=float)
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.domain
        preds[i,:] = fn(np.clip(HORIZONS_PRED, t_min, t_max))
    return 1.0 - preds

def make_binary_target(time_vals, event_vals, horizon):
    unknown = (event_vals==0) & (time_vals<horizon)
    y = ((event_vals==1) & (time_vals<=horizon)).astype(float)
    return y, ~unknown

def compute_ipcw_weights(times, events, horizon):
    unique_t = np.sort(np.unique(times)); surv = np.ones(len(unique_t))
    for i, t in enumerate(unique_t):
        at_risk = (times>=t).sum()
        cens = ((times==t) & (events==0)).sum()
        if at_risk > 0: surv[i] = 1 - cens/at_risk
        if i > 0: surv[i] *= surv[i-1]
    def G(t):
        idx = np.searchsorted(unique_t, t, side="right")-1
        return max(surv[idx], 0.01) if idx >= 0 else 1.0
    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i]==1 and times[i]<=horizon:   weights[i] = 1.0/G(times[i])
        elif times[i]>=horizon:                   weights[i] = 1.0/G(horizon)
    return weights

# Shared survival data
y_surv       = Surv.from_arrays(event=train_df["event"].astype(bool),
                                  time=train_df["time_to_hit_hours"])
event_values = train_df["event"].values
time_values  = train_df["time_to_hit_hours"].values
dist_train   = train_df["dist_min_ci_0_5h"].values
dist_test    = test_df["dist_min_ci_0_5h"].values
near_train   = dist_train < STRAT_THR
near_test    = dist_test  < STRAT_THR
far_train    = ~near_train
far_test     = ~near_test

print(f"Survival structure: {y_surv.shape} | Near train: {near_train.sum()} | Far train: {far_train.sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# Model 1: GBSA Survival Ensemble (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
X_gbsa_train = train_df.drop(columns=["event_id","event","time_to_hit_hours"])
X_gbsa_test  = test_df.drop(columns=["event_id"])

gbsa_configs = [
    {"learning_rate":0.01,  "subsample":0.70, "max_depth":3, "min_samples_leaf":12, "min_samples_split":3, "n_estimators":1200, "dropout_rate":0.0},
    {"learning_rate":0.01,  "subsample":0.85, "max_depth":3, "min_samples_leaf":15, "min_samples_split":3, "n_estimators":1200, "dropout_rate":0.0},
    {"learning_rate":0.01,  "subsample":0.60, "max_depth":3, "min_samples_leaf":12, "min_samples_split":3, "n_estimators":1200, "dropout_rate":0.0},
    {"learning_rate":0.005, "subsample":0.85, "max_depth":3, "min_samples_leaf":12, "min_samples_split":3, "n_estimators":2000, "dropout_rate":0.0},
    {"learning_rate":0.01,  "subsample":0.85, "max_depth":3, "min_samples_leaf":20, "min_samples_split":3, "n_estimators":1400, "dropout_rate":0.0},
    {"learning_rate":0.008, "subsample":0.75, "max_depth":2, "min_samples_leaf":15, "min_samples_split":4, "n_estimators":1500, "dropout_rate":0.0},
    {"learning_rate":0.015, "subsample":0.70, "max_depth":3, "min_samples_leaf":10, "min_samples_split":3, "n_estimators":1000, "dropout_rate":0.0},
    {"learning_rate":0.005, "subsample":0.90, "max_depth":3, "min_samples_leaf":18, "min_samples_split":5, "n_estimators":2500, "dropout_rate":0.0},
    {"learning_rate":0.01,  "subsample":0.80, "max_depth":4, "min_samples_leaf":12, "min_samples_split":3, "n_estimators":1200, "dropout_rate":0.0},
    {"learning_rate":0.02,  "subsample":0.65, "max_depth":3, "min_samples_leaf":10, "min_samples_split":3, "n_estimators":800,  "dropout_rate":0.0},
]

GBSA_SEEDS = GBSA_SEEDS_FULL if RUN_MODE=="full" else GBSA_SEEDS_FAST

oof_gbsa  = np.zeros((len(X_gbsa_train), 4))
test_gbsa = np.zeros((len(X_gbsa_test), 4))

print(f"\nGBSA: {len(gbsa_configs)} configs x {len(GBSA_SEEDS)} seeds x 5-fold CV-bag")
for cfg_idx, cfg in enumerate(gbsa_configs, 1):
    cfg_oof  = np.zeros((len(X_gbsa_train), 4))
    cfg_test = np.zeros((len(X_gbsa_test), 4))
    for seed in GBSA_SEEDS:
        seed_oof  = np.zeros((len(X_gbsa_train), 4))
        seed_test = np.zeros((len(X_gbsa_test), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(X_gbsa_train, event_values):
            m = GradientBoostingSurvivalAnalysis(**{**cfg, "random_state": seed})
            m.fit(X_gbsa_train.iloc[tr_idx], y_surv[tr_idx])
            seed_oof[va_idx] = get_surv_predictions(m, X_gbsa_train.iloc[va_idx])
            if CV_BAG_TEST:
                seed_test += get_surv_predictions(m, X_gbsa_test) / 5
        cfg_oof  += seed_oof  / len(GBSA_SEEDS)
        cfg_test += seed_test / len(GBSA_SEEDS)
    oof_gbsa  += cfg_oof  / len(gbsa_configs)
    test_gbsa += cfg_test / len(gbsa_configs)
    print(f"  cfg {cfg_idx}/{len(gbsa_configs)} done")

oof_gbsa_raw  = oof_gbsa.copy()
test_gbsa_raw = test_gbsa.copy()
print("GBSA done.")

# ─────────────────────────────────────────────────────────────────────────────
# Model 2: CoxPH Survival Ensemble (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
COX_FEATURES_LIST_ENG = [
    "dist_km", "log_distance", "inv_distance",
    "closing_speed_m_per_h", "radial_growth_rate_m_per_h",
    "alignment_abs", "threat_score", "log_eta", "eta_effective",
    "area_to_dist_ratio", "fire_radius_km",
    "num_perimeters_0_5h", "has_movement",
    "near_speed_rank", "far_threat_rank",
    "is_summer", "is_afternoon",
    "zone_near", "zone_far",
]

X_cox_train = train_processed[[c for c in COX_FEATURES_LIST_ENG if c in train_processed.columns]].copy()
X_cox_test  = test_processed[[c for c in COX_FEATURES_LIST_ENG if c in test_processed.columns]].copy()

scaler = StandardScaler()
X_cox_train_sc = pd.DataFrame(scaler.fit_transform(X_cox_train),
                                columns=X_cox_train.columns, index=X_cox_train.index)
X_cox_test_sc  = pd.DataFrame(scaler.transform(X_cox_test),
                                columns=X_cox_test.columns,  index=X_cox_test.index)

cox_alphas = [0.001, 0.01, 0.05, 0.10, 0.50, 1.00, 2.00]
COX_SEEDS  = COX_SEEDS_FULL if RUN_MODE=="full" else COX_SEEDS_FAST

oof_cox  = np.zeros((len(X_cox_train_sc), 4))
test_cox = np.zeros((len(X_cox_test_sc), 4))

print(f"\nCoxPH: {len(cox_alphas)} alphas x {len(COX_SEEDS)} seeds x 5-fold CV-bag")
for alpha_idx, alpha in enumerate(cox_alphas, 1):
    alpha_oof  = np.zeros((len(X_cox_train_sc), 4))
    alpha_test = np.zeros((len(X_cox_test_sc), 4))
    for seed in COX_SEEDS:
        seed_oof  = np.zeros((len(X_cox_train_sc), 4))
        seed_test = np.zeros((len(X_cox_test_sc), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(X_cox_train_sc, event_values):
            m = CoxPHSurvivalAnalysis(alpha=alpha)
            try:
                m.fit(X_cox_train_sc.iloc[tr_idx], y_surv[tr_idx])
                seed_oof[va_idx] = get_surv_predictions(m, X_cox_train_sc.iloc[va_idx])
                if CV_BAG_TEST:
                    seed_test += get_surv_predictions(m, X_cox_test_sc) / 5
            except:
                seed_oof[va_idx] = 0.5
                if CV_BAG_TEST: seed_test += 0.5 / 5
        alpha_oof  += seed_oof  / len(COX_SEEDS)
        alpha_test += seed_test / len(COX_SEEDS)
    oof_cox  += alpha_oof  / len(cox_alphas)
    test_cox += alpha_test / len(cox_alphas)
    print(f"  alpha={alpha} done")

print("CoxPH done.")

# ─────────────────────────────────────────────────────────────────────────────
# Model 3: Random Survival Forest (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
X_rsf_train = train_df.drop(columns=["event_id","event","time_to_hit_hours"])
X_rsf_test  = test_df.drop(columns=["event_id"])

rsf_configs = [
    {"n_estimators": 200, "min_samples_leaf": 12, "max_features": "sqrt", "max_depth": None},
    {"n_estimators": 200, "min_samples_leaf": 18, "max_features": "sqrt", "max_depth": None},
    {"n_estimators": 200, "min_samples_leaf": 12, "max_features": 0.5,    "max_depth": 5},
]

RSF_SEEDS = RSF_SEEDS_FULL if RUN_MODE=="full" else RSF_SEEDS_FAST

oof_rsf  = np.zeros((len(X_rsf_train), 4))
test_rsf = np.zeros((len(X_rsf_test), 4))

print(f"\nRSF: {len(rsf_configs)} configs x {len(RSF_SEEDS)} seeds x 5-fold CV-bag")
for cfg_idx, cfg in enumerate(rsf_configs, 1):
    cfg_oof  = np.zeros((len(X_rsf_train), 4))
    cfg_test = np.zeros((len(X_rsf_test), 4))
    for seed in RSF_SEEDS:
        seed_oof  = np.zeros((len(X_rsf_train), 4))
        seed_test = np.zeros((len(X_rsf_test), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, va_idx in cv.split(X_rsf_train, event_values):
            m = RandomSurvivalForest(**{**cfg, "random_state": seed, "n_jobs": -1})
            m.fit(X_rsf_train.iloc[tr_idx], y_surv[tr_idx])
            seed_oof[va_idx] = get_surv_predictions(m, X_rsf_train.iloc[va_idx])
            if CV_BAG_TEST:
                seed_test += get_surv_predictions(m, X_rsf_test) / 5
        cfg_oof  += seed_oof  / len(RSF_SEEDS)
        cfg_test += seed_test / len(RSF_SEEDS)
    oof_rsf  += cfg_oof  / len(rsf_configs)
    test_rsf += cfg_test / len(rsf_configs)
    print(f"  cfg {cfg_idx}/{len(rsf_configs)} done")

print("RSF done.")

# ─────────────────────────────────────────────────────────────────────────────
# Near-Zone Timing Model (diagnostic only, weight=0.0)
# ─────────────────────────────────────────────────────────────────────────────
TIMING_FEATURES = [
    "closing_speed_m_per_h", "radial_growth_rate_m_per_h",
    "alignment_abs", "num_perimeters_0_5h", "area_growth_rate_ha_per_h",
    "eta_effective", "log_eta", "dist_km", "threat_score",
    "near_speed_rank", "event_start_hour", "is_afternoon", "fire_urgency",
    "area_first_ha", "fire_radius_km",
]

near_hit_mask  = (dist_train < STRAT_THR) & (event_values == 1)
near_hit_idx   = np.where(near_hit_mask)[0]
near_all_mask  = dist_train < STRAT_THR
near_test_mask = dist_test  < STRAT_THR

avail_timing  = [f for f in TIMING_FEATURES if f in train_processed.columns]
X_timing_full = train_processed[avail_timing].values
X_timing_test = test_processed[[f for f in avail_timing if f in test_processed.columns]].values
y_timing_log  = np.log(time_values[near_hit_mask] + 1e-6)

timing_lgb_cfg = {
    "max_depth": 2, "learning_rate": 0.05, "n_estimators": 150,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 4,
    "reg_alpha": 1.0, "reg_lambda": 2.0, "num_leaves": 4,
}
TIMING_SEEDS = (123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033,
                279, 239, 70, 77, 31) if RUN_MODE == "full" else (123, 456, 789)

oof_timing_log_accum = np.zeros(len(train_processed))
oof_timing_log_count = np.zeros(len(train_processed))
all_test_timing_log  = np.zeros(near_test_mask.sum())
all_oof_residuals    = []

for seed in TIMING_SEEDS:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_v, va_v in cv.split(near_hit_idx, np.zeros(len(near_hit_idx))):
        tr_near, va_near = near_hit_idx[tr_v], near_hit_idx[va_v]
        m = lgb.LGBMRegressor(**timing_lgb_cfg, random_state=seed, verbose=-1)
        m.fit(X_timing_full[tr_near], y_timing_log[tr_v])
        pred_va = m.predict(X_timing_full[va_near])
        oof_timing_log_accum[va_near] += pred_va
        oof_timing_log_count[va_near] += 1
        all_oof_residuals.extend((y_timing_log[va_v] - pred_va).tolist())
    mf = lgb.LGBMRegressor(**timing_lgb_cfg, random_state=seed, verbose=-1)
    mf.fit(X_timing_full[near_hit_idx], y_timing_log)
    all_test_timing_log += mf.predict(X_timing_test[near_test_mask]) / len(TIMING_SEEDS)

valid_oof          = oof_timing_log_count > 0
oof_timing_log_avg = np.where(valid_oof, oof_timing_log_accum / np.maximum(oof_timing_log_count, 1), 0)
timing_sigma       = max(np.std(all_oof_residuals) if len(all_oof_residuals) > 5 else 0.5, 0.2)

def lognormal_horizon_prob(t_hat_log, sigma, horizons):
    probs = []
    for h in horizons:
        z = (np.log(h + 1e-6) - t_hat_log) / (sigma + 1e-8)
        probs.append(scipy_stats.norm.cdf(z))
    return np.stack(probs, axis=1)

HORIZONS_TIMING   = [12, 24, 48]
oof_timing_probs  = np.zeros((len(train_processed), 3))
test_timing_probs = np.zeros((len(test_df), 3))

if valid_oof[near_hit_idx].any():
    oof_timing_probs[near_hit_idx] = lognormal_horizon_prob(
        oof_timing_log_avg[near_hit_idx], timing_sigma, HORIZONS_TIMING)
if near_test_mask.sum() > 0:
    test_timing_probs[near_test_mask] = lognormal_horizon_prob(
        all_test_timing_log, timing_sigma, HORIZONS_TIMING)

print(f"\nTiming model sigma: {timing_sigma:.4f} (weight=0.0, diagnostic only)")

# ─────────────────────────────────────────────────────────────────────────────
# Zone-Split LGB IPCW Calibrators (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
NEAR_LGB_FEATURES = [
    "closing_speed_m_per_h", "radial_growth_rate_m_per_h",
    "alignment_abs", "num_perimeters_0_5h", "area_growth_rate_ha_per_h",
    "eta_effective", "log_eta", "dist_km", "threat_score",
    "near_speed_rank", "event_start_hour", "is_afternoon", "fire_urgency",
    "area_first_ha", "fire_radius_km",
]
FAR_LGB_FEATURES = [
    "dist_km", "log_distance", "inv_distance",
    "closing_speed_m_per_h", "alignment_abs",
    "threat_score", "log_eta", "eta_effective",
    "area_to_dist_ratio", "num_perimeters_0_5h",
    "far_threat_rank", "is_summer", "zone_far",
]

avail_near_lgb = [f for f in NEAR_LGB_FEATURES if f in train_processed.columns]
avail_far_lgb  = [f for f in FAR_LGB_FEATURES  if f in train_processed.columns]

X_near_lgb_train = train_processed[avail_near_lgb]
X_near_lgb_test  = test_processed[[f for f in avail_near_lgb if f in test_processed.columns]]
X_far_lgb_train  = train_processed[avail_far_lgb]
X_far_lgb_test   = test_processed[[f for f in avail_far_lgb if f in test_processed.columns]]

near_lgb_cfgs = {
    24: {"max_depth":2, "learning_rate":0.04, "n_estimators":200,
         "subsample":0.8, "colsample_bytree":0.8, "min_child_samples":4,
         "reg_alpha":0.5, "reg_lambda":1.5, "num_leaves":4},
    48: {"max_depth":2, "learning_rate":0.05, "n_estimators":150,
         "subsample":0.8, "colsample_bytree":0.8, "min_child_samples":3,
         "reg_alpha":0.3, "reg_lambda":1.0, "num_leaves":4},
}
far_lgb_cfgs = {
    24: {"max_depth":2, "learning_rate":0.03, "n_estimators":200,
         "subsample":0.7, "colsample_bytree":0.7, "min_child_samples":8,
         "reg_alpha":1.0, "reg_lambda":3.0, "num_leaves":4},
    48: {"max_depth":2, "learning_rate":0.05, "n_estimators":150,
         "subsample":0.8, "colsample_bytree":0.8, "min_child_samples":6,
         "reg_alpha":0.5, "reg_lambda":2.0, "num_leaves":4},
}

LGB_NEAR_SEEDS = LGB_NEAR_SEEDS_FULL if RUN_MODE=="full" else LGB_NEAR_SEEDS_FAST
LGB_FAR_SEEDS  = LGB_FAR_SEEDS_FULL  if RUN_MODE=="full" else LGB_FAR_SEEDS_FAST
lgb_near_oof, lgb_near_test = {}, {}
lgb_far_oof,  lgb_far_test  = {}, {}

print(f"\nZone-Split LGB: {len(LGB_NEAR_SEEDS)} near seeds | {len(LGB_FAR_SEEDS)} far seeds")
for horizon in [24, 48]:
    y_bin, mask = make_binary_target(time_values, event_values, horizon)
    valid_idx   = np.where(mask)[0]
    for zone, cfg_d, X_tr, X_te, seeds, oof_d, test_d in [
        ("near", near_lgb_cfgs, X_near_lgb_train, X_near_lgb_test, LGB_NEAR_SEEDS, lgb_near_oof, lgb_near_test),
        ("far",  far_lgb_cfgs,  X_far_lgb_train,  X_far_lgb_test,  LGB_FAR_SEEDS,  lgb_far_oof,  lgb_far_test),
    ]:
        cfg      = cfg_d[horizon]
        all_oof  = np.zeros(len(X_tr))
        all_test = np.zeros(len(X_te))
        for seed in seeds:
            seed_oof = np.zeros(len(X_tr)); last_m = None
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for tr_v, va_v in cv.split(valid_idx, y_bin[mask]):
                tr_idx, va_idx = valid_idx[tr_v], valid_idx[va_v]
                ipcw_w = compute_ipcw_weights(time_values[tr_idx], event_values[tr_idx], horizon)
                m = lgb.LGBMClassifier(**cfg, objective="binary", random_state=seed, verbose=-1)
                m.fit(X_tr.iloc[tr_idx], y_bin[tr_idx], sample_weight=ipcw_w)
                seed_oof[va_idx] = m.predict_proba(X_tr.iloc[va_idx])[:,1]; last_m = m
            cens_idx = np.where(~mask)[0]
            if len(cens_idx)>0 and last_m is not None:
                seed_oof[cens_idx] = last_m.predict_proba(X_tr.iloc[cens_idx])[:,1]
            all_oof += seed_oof
            ipcw_w_full = compute_ipcw_weights(time_values[valid_idx], event_values[valid_idx], horizon)
            mf = lgb.LGBMClassifier(**cfg, objective="binary", random_state=seed, verbose=-1)
            mf.fit(X_tr.iloc[valid_idx], y_bin[valid_idx], sample_weight=ipcw_w_full)
            all_test += mf.predict_proba(X_te)[:,1]
        oof_d[horizon]  = all_oof  / len(seeds)
        test_d[horizon] = all_test / len(seeds)
    b_near = compute_brier(time_values, event_values, np.clip(lgb_near_oof[horizon],0,1), horizon)
    b_far  = compute_brier(time_values, event_values, np.clip(lgb_far_oof[horizon],0,1),  horizon)
    print(f"  {horizon}h -> Near-LGB B={b_near:.5f} | Far-LGB B={b_far:.5f}")

print("Zone-split LGB done.")

# ─────────────────────────────────────────────────────────────────────────────
# Raw blend (shared between A and B)
# ─────────────────────────────────────────────────────────────────────────────
def blend_zone(gbsa, cox, rsf, lgb_p, timing_p, wg, wc, wr, wl, wt=0.0):
    return wg*gbsa + wc*cox + wr*rsf + wl*lgb_p + wt*timing_p

def zone_blend_h(gbsa, cox, rsf, lgb_near_p, lgb_far_p, timing_p, near_mask,
                  wg_n, wc_n, wr_n, wl_n, wt_n, wg_f, wc_f, wr_f, wl_f):
    near = blend_zone(gbsa, cox, rsf, lgb_near_p, timing_p, wg_n, wc_n, wr_n, wl_n, wt_n)
    far  = blend_zone(gbsa, cox, rsf, lgb_far_p,  timing_p, wg_f, wc_f, wr_f, wl_f, 0.0)
    return np.where(near_mask, near, far)

oof_gbsa  = oof_gbsa_raw.copy()
test_gbsa = test_gbsa_raw.copy()
if POWER_CAL_24 != 1.0:
    oof_gbsa[:,1]  = np.clip(oof_gbsa[:,1]  ** POWER_CAL_24, 0, 1)
    test_gbsa[:,1] = np.clip(test_gbsa[:,1] ** POWER_CAL_24, 0, 1)

oof_blend = np.zeros_like(oof_gbsa)
oof_blend[:,0] = zone_blend_h(
    oof_gbsa[:,0], oof_cox[:,0], oof_rsf[:,0],
    lgb_near_oof[24], lgb_far_oof[24], oof_timing_probs[:,0], near_train,
    W_GBSA_NEAR_12, W_COX_NEAR_12, W_RSF_NEAR_12, W_LGB_NEAR_12, 0.0,
    1.0, 0.0, 0.0, 0.0)
oof_blend[:,1] = zone_blend_h(
    oof_gbsa[:,1], oof_cox[:,1], oof_rsf[:,1],
    lgb_near_oof[24], lgb_far_oof[24], oof_timing_probs[:,1], near_train,
    W_GBSA_NEAR_24, W_COX_NEAR_24, W_RSF_NEAR_24, W_LGB_NEAR_24, 0.0,
    W_GBSA_FAR_24,  W_COX_FAR_24,  W_RSF_FAR_24,  W_LGB_FAR_24)
oof_blend[:,2] = zone_blend_h(
    oof_gbsa[:,2], oof_cox[:,2], oof_rsf[:,2],
    lgb_near_oof[48], lgb_far_oof[48], oof_timing_probs[:,2], near_train,
    W_GBSA_NEAR_48, W_COX_NEAR_48, W_RSF_NEAR_48, W_LGB_NEAR_48, 0.0,
    W_GBSA_FAR_48,  W_COX_FAR_48,  W_RSF_FAR_48,  W_LGB_FAR_48)
oof_blend[:,3] = 1.0

test_blend = np.zeros_like(test_gbsa)
test_blend[:,0] = zone_blend_h(
    test_gbsa[:,0], test_cox[:,0], test_rsf[:,0],
    lgb_near_test[24], lgb_far_test[24], test_timing_probs[:,0], near_test,
    W_GBSA_NEAR_12, W_COX_NEAR_12, W_RSF_NEAR_12, W_LGB_NEAR_12, 0.0,
    1.0, 0.0, 0.0, 0.0)
test_blend[:,1] = zone_blend_h(
    test_gbsa[:,1], test_cox[:,1], test_rsf[:,1],
    lgb_near_test[24], lgb_far_test[24], test_timing_probs[:,1], near_test,
    W_GBSA_NEAR_24, W_COX_NEAR_24, W_RSF_NEAR_24, W_LGB_NEAR_24, 0.0,
    W_GBSA_FAR_24,  W_COX_FAR_24,  W_RSF_FAR_24,  W_LGB_FAR_24)
test_blend[:,2] = zone_blend_h(
    test_gbsa[:,2], test_cox[:,2], test_rsf[:,2],
    lgb_near_test[48], lgb_far_test[48], test_timing_probs[:,2], near_test,
    W_GBSA_NEAR_48, W_COX_NEAR_48, W_RSF_NEAR_48, W_LGB_NEAR_48, 0.0,
    W_GBSA_FAR_48,  W_COX_FAR_48,  W_RSF_FAR_48,  W_LGB_FAR_48)
test_blend[:,3] = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Calibration utilities
# ─────────────────────────────────────────────────────────────────────────────
def right_tail_shift(preds, shift=RIGHT_TAIL_SHIFT, threshold=0.9):
    """Proven +0.09 shift for any p >= 0.9. Applied uniformly across all horizons."""
    return np.where(preds >= threshold, preds + shift, preds).clip(0, 1)

def left_tail_push(preds, far_mask, threshold=LEFT_TAIL_THR, cap=LEFT_TAIL_CAP):
    """Push far-zone predictions below threshold down to cap. Skips 72h col."""
    result = preds.copy()
    for col in range(preds.shape[1] - 1):   # skip col 3 (72h = 1.0)
        row_mask = far_mask & (preds[:, col] < threshold)
        result[row_mask, col] = cap
    return result

def fit_isotonic_far_zone(oof_preds, time_vals, event_vals, far_mask, horizon, col_idx):
    """
    Fit isotonic regression on far-zone OOF predictions for a given horizon.
    Returns (fitted IsotonicRegression, OOF Brier gain).
    """
    mask_valid = ~((event_vals == 0) & (time_vals < horizon))
    y_true     = ((event_vals == 1) & (time_vals <= horizon)).astype(float)
    fit_mask   = far_mask & mask_valid

    X_iso = oof_preds[fit_mask, col_idx]
    y_iso = y_true[fit_mask]

    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(X_iso, y_iso)

    b_before = compute_brier(time_vals[far_mask], event_vals[far_mask],
                              oof_preds[far_mask, col_idx], horizon)
    oof_cal  = iso.predict(oof_preds[far_mask, col_idx])
    b_after  = compute_brier(time_vals[far_mask], event_vals[far_mask], oof_cal, horizon)
    gain     = b_before - b_after

    print(f"  Isotonic {horizon}h far-zone OOF: B={b_before:.5f} -> {b_after:.5f} "
          f"(gain={gain:+.5f}{'  <<NOISE' if gain < 0.0003 else ''})")
    return iso, gain

# ─────────────────────────────────────────────────────────────────────────────
# Submission A — baseline (monotonicity + right-tail shift only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUBMISSION A — baseline calibration")
print("="*60)

oof_A  = enforce_monotonicity(oof_blend.copy())
oof_A  = right_tail_shift(oof_A)
oof_A  = enforce_monotonicity(oof_A)

test_A = enforce_monotonicity(test_blend.copy())
test_A = right_tail_shift(test_A)
test_A = enforce_monotonicity(test_A)

hybrid_A, c_idx_A, wb_A = compute_hybrid_score(
    time_values, event_values, oof_A[:,1], oof_A[:,2], oof_A[:,3])
b12_A = compute_brier(time_values, event_values, oof_A[:,0], 12)
b24_A = compute_brier(time_values, event_values, oof_A[:,1], 24)
b48_A = compute_brier(time_values, event_values, oof_A[:,2], 48)

print(f"OOF Hybrid: {hybrid_A:.5f}  C-Index: {c_idx_A:.4f}  WBrier: {wb_A:.5f}")
print(f"  B12={b12_A:.5f}  B24={b24_A:.5f}  B48={b48_A:.5f}  B72=0.00000")

sub_A = pd.DataFrame({
    "event_id": test_df["event_id"].values,
    "prob_12h" : test_A[:,0], "prob_24h" : test_A[:,1],
    "prob_48h" : test_A[:,2], "prob_72h" : test_A[:,3],
})
sub_A = sample_sub[["event_id"]].merge(sub_A, on="event_id", how="left")
sub_A.to_csv(OUTPUT_PATH_A, index=False)
print(f"Saved: {OUTPUT_PATH_A}")

# ─────────────────────────────────────────────────────────────────────────────
# Submission B — isotonic on far-zone + left-tail push
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUBMISSION B — isotonic calibration (far-zone 24h + 48h) + left-tail push")
print("="*60)

print("\nFitting isotonic on far-zone OOF (n_far_train={})".format(far_train.sum()))
iso_24, gain_24 = fit_isotonic_far_zone(oof_A, time_values, event_values, far_train, 24, 1)
iso_48, gain_48 = fit_isotonic_far_zone(oof_A, time_values, event_values, far_train, 48, 2)

oof_B = oof_A.copy()
oof_B[far_train, 1] = iso_24.predict(oof_A[far_train, 1])
oof_B[far_train, 2] = iso_48.predict(oof_A[far_train, 2])

test_B = test_A.copy()
test_B[far_test, 1] = iso_24.predict(test_A[far_test, 1])
test_B[far_test, 2] = iso_48.predict(test_A[far_test, 2])

oof_B  = left_tail_push(oof_B,  far_train)
test_B = left_tail_push(test_B, far_test)

oof_B  = enforce_monotonicity(oof_B)
test_B = enforce_monotonicity(test_B)

hybrid_B, c_idx_B, wb_B = compute_hybrid_score(
    time_values, event_values, oof_B[:,1], oof_B[:,2], oof_B[:,3])
b12_B = compute_brier(time_values, event_values, oof_B[:,0], 12)
b24_B = compute_brier(time_values, event_values, oof_B[:,1], 24)
b48_B = compute_brier(time_values, event_values, oof_B[:,2], 48)

print(f"\nOOF Hybrid: {hybrid_B:.5f}  C-Index: {c_idx_B:.4f}  WBrier: {wb_B:.5f}")
print(f"  B12={b12_B:.5f}  B24={b24_B:.5f}  B48={b48_B:.5f}  B72=0.00000")
print(f"  OOF gain vs A: {hybrid_B - hybrid_A:+.5f}")

sub_B = pd.DataFrame({
    "event_id": test_df["event_id"].values,
    "prob_12h" : test_B[:,0], "prob_24h" : test_B[:,1],
    "prob_48h" : test_B[:,2], "prob_72h" : test_B[:,3],
})
sub_B = sample_sub[["event_id"]].merge(sub_B, on="event_id", how="left")
sub_B.to_csv(OUTPUT_PATH_B, index=False)
print(f"Saved: {OUTPUT_PATH_B}")

# ─────────────────────────────────────────────────────────────────────────────
# Prediction distribution diagnostics (test set)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PREDICTION DISTRIBUTION DIAGNOSTICS (test set)")
print("="*60)

buckets = [
    (0.00, 0.05), (0.05, 0.10), (0.10, 0.30), (0.30, 0.50),
    (0.50, 0.70), (0.70, 0.90), (0.90, 0.99), (0.99, 1.001),
]
bucket_labels = ["0-5%","5-10%","10-30%","30-50%","50-70%","70-90%","90-99%","99-100%"]

for sub_lbl, test_preds in [("A", test_A), ("B", test_B)]:
    print(f"\n--- Submission {sub_lbl} ---")
    for col_idx, h_lbl in enumerate(["12h","24h","48h","72h"]):
        vals_near = test_preds[near_test,  col_idx]
        vals_far  = test_preds[~near_test, col_idx]
        near_counts = [((vals_near >= lo) & (vals_near < hi)).sum() for lo, hi in buckets]
        far_counts  = [((vals_far  >= lo) & (vals_far  < hi)).sum() for lo, hi in buckets]
        print(f"  {h_lbl}  Near({near_test.sum()}): {dict(zip(bucket_labels, near_counts))}")
        print(f"       Far({(~near_test).sum()}):  {dict(zip(bucket_labels, far_counts))}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary + decision rule
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Submission A (baseline):  OOF={hybrid_A:.5f}  -> {OUTPUT_PATH_A}")
print(f"Submission B (isotonic):  OOF={hybrid_B:.5f}  -> {OUTPUT_PATH_B}")
print(f"OOF delta B-A: {hybrid_B - hybrid_A:+.5f}")
print()
print("Submit both to LB. Report back A_LB and B_LB.")
print("Decision rule for v8:")
print("  B_LB > A_LB + 0.0003  ->  isotonic confirmed, lock it in for v8")
print("  B_LB <= A_LB           ->  isotonic overfits far-zone, drop for v8")
print("  In between             ->  lean toward A (simpler is safer)")