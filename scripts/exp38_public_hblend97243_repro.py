
import numpy as np
import pandas as pd
import warnings
import os
import time as timer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

warnings.filterwarnings('ignore')
np.random.seed(777)
HORIZONS_PRED = np.array([12, 24, 48, 72], dtype=float)
def locate_datasets():
    train_path, test_path = 'train.csv', 'test.csv'
    for search_root in ['/kaggle/input', '../input']:
        if os.path.exists(search_root):
            for root, _, files in os.walk(search_root):
                if 'train.csv' in files: train_path = os.path.join(root, 'train.csv')
                if 'test.csv' in files: test_path = os.path.join(root, 'test.csv')
    return train_path, test_path

train_path, test_path = locate_datasets()
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(f'Training: {len(train_df)} samples, Test: {len(test_df)} samples')
def create_features(df):
    result = df.copy()
    dist = result['dist_min_ci_0_5h'].clip(lower=1)
    speed = result['closing_speed_m_per_h']
    perimeters = result['num_perimeters_0_5h']
    area_first = result['area_first_ha']
    result['log_distance'] = np.log1p(dist)
    result['inv_distance'] = 1 / (dist / 1000 + 0.1)
    result['inv_distance_sq'] = result['inv_distance'] ** 2
    result['sqrt_distance'] = np.sqrt(dist)
    result['dist_km'] = dist / 1000
    result['dist_km_sq'] = (dist / 1000) ** 2
    result['dist_rank'] = dist.rank(pct=True)
    fire_radius = np.sqrt(area_first * 10000 / np.pi)
    result['radius_to_dist'] = fire_radius / dist
    result['area_to_dist_ratio'] = area_first / (dist / 1000 + 0.1)
    result['log_area_dist_ratio'] = np.log1p(area_first) - np.log1p(dist)
    result['has_movement'] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    result['eta_hours'] = np.where(closing_pos > 0.01, dist / closing_pos, 9999).clip(max=9999)
    result['log_eta'] = np.log1p(result['eta_hours'].clip(0, 9999))
    radial_growth = result['radial_growth_rate_m_per_h'].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    result['effective_closing_speed'] = effective_closing
    result['eta_effective'] = np.where(effective_closing > 0.01, dist / effective_closing, 9999).clip(max=9999)
    result['threat_score'] = result['alignment_abs'] * speed / np.log1p(dist)
    result['fire_urgency'] = perimeters * speed
    result['growth_intensity'] = result['area_growth_rate_ha_per_h'] * perimeters
    result['zone_critical'] = (dist < 5000).astype(float)
    result['zone_warning'] = ((dist >= 5000) & (dist < 10000)).astype(float)
    result['zone_safe'] = (dist >= 10000).astype(float)
    result['is_summer'] = result['event_start_month'].isin([6, 7, 8]).astype(float)
    result['is_afternoon'] = ((result['event_start_hour'] >= 12) & (result['event_start_hour'] < 20)).astype(float)
    drop_cols = ['relative_growth_0_5h', 'projected_advance_m', 'centroid_displacement_m',
                 'centroid_speed_m_per_h', 'closing_speed_abs_m_per_h', 'area_growth_abs_0_5h']
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result

train_processed = create_features(train_df)
test_processed = create_features(test_df)
def get_surv_predictions(model, X):
    surv_fns = model.predict_survival_function(X)
    preds = np.empty((len(surv_fns), len(HORIZONS_PRED)), dtype=float)
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.domain
        preds[i, :] = fn(np.clip(HORIZONS_PRED, t_min, t_max))
    return 1.0 - preds

def sigmoid_pred(dist, threshold, scale):
    return 1.0 / (1.0 + np.exp((dist - threshold) / scale))

def make_binary_target(time_vals, event_vals, horizon):
    unknown = (event_vals == 0) & (time_vals < horizon)
    y = ((event_vals == 1) & (time_vals <= horizon)).astype(float)
    return y, ~unknown

def compute_ipcw_weights(times, events, horizon):
    unique_t = np.sort(np.unique(times))
    surv = np.ones(len(unique_t))
    for i, t in enumerate(unique_t):
        at_risk = (times >= t).sum()
        censored_at_t = ((times == t) & (events == 0)).sum()
        if at_risk > 0: surv[i] = 1 - censored_at_t / at_risk
        if i > 0: surv[i] *= surv[i - 1]
    def G(t):
        idx = np.searchsorted(unique_t, t, side='right') - 1
        return max(surv[idx], 0.01) if idx >= 0 else 1.0
    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon: weights[i] = 1.0 / G(times[i])
        elif times[i] >= horizon: weights[i] = 1.0 / G(horizon)
    return weights

def enforce_monotonicity(preds):
    result = np.clip(preds, 0, 1)
    for i in range(1, result.shape[1]):
        result[:, i] = np.maximum(result[:, i], result[:, i-1])
    return result
X_surv_train = train_df.drop(columns=['event_id', 'event', 'time_to_hit_hours'])
X_surv_test = test_df.drop(columns=['event_id'])
y_surv = Surv.from_arrays(event=train_df['event'].astype(bool), time=train_df['time_to_hit_hours'])
event_values = train_df['event'].values
time_values = train_df['time_to_hit_hours'].values
dist_test = test_df['dist_min_ci_0_5h'].values

gbsa_configs = [
    {'learning_rate': 0.01, 'subsample': 0.7,  'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 15, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.6,  'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 1200, 'dropout_rate': 0.0},
    {'learning_rate': 0.005,'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 12, 'min_samples_split': 3, 'n_estimators': 2000, 'dropout_rate': 0.0},
    {'learning_rate': 0.01, 'subsample': 0.85, 'max_depth': 3, 'min_samples_leaf': 20, 'min_samples_split': 3, 'n_estimators': 1400, 'dropout_rate': 0.0},
]
SEEDS = (123, 456, 789, 777, 666,
         1511, 1523, 2025, 2026, 2033,
        279, 239, 70, 77, 31,
        2024, 2077, 3077, 123456, 654321)
N_SEEDS = len(SEEDS)

test_gbsa = np.zeros((len(X_surv_test), 4))
total_models = len(gbsa_configs) * N_SEEDS * 5
model_count = 0
t_start = timer.time()

for cfg_idx, cfg in enumerate(gbsa_configs):
    cfg_test = np.zeros((len(X_surv_test), 4))
    for seed in SEEDS:
        seed_test = np.zeros((len(X_surv_test), 4))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_surv_train, event_values)):
            m = GradientBoostingSurvivalAnalysis(**{**cfg, 'random_state': seed})
            m.fit(X_surv_train.iloc[tr_idx], y_surv[tr_idx])
            seed_test += get_surv_predictions(m, X_surv_test) / 5
            model_count += 1
        cfg_test += seed_test / N_SEEDS
    test_gbsa += cfg_test / len(gbsa_configs)
    elapsed = timer.time() - t_start
    print(f'Config {cfg_idx+1}/{len(gbsa_configs)} done [{model_count}/{total_models}, {elapsed/60:.1f}m]')

# PowerCal 24h
test_gbsa[:, 1] = np.clip(test_gbsa[:, 1] ** 1.1, 0, 1)
print(f'GBSA done: {total_models} fold models')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_lgb_train = train_processed.drop(columns=['event_id', 'event', 'time_to_hit_hours'])
X_lgb_test = test_processed.drop(columns=['event_id'])

lgb_cfgs = {
    24: {'max_depth': 3, 'learning_rate': 0.03, 'n_estimators': 300,
         'subsample': 0.7, 'colsample_bytree': 0.7, 'min_child_samples': 8,
         'reg_alpha': 0.5, 'reg_lambda': 2.0, 'num_leaves': 7},
    48: {'max_depth': 2, 'learning_rate': 0.05, 'n_estimators': 200,
         'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_samples': 5,
         'reg_alpha': 0.1, 'reg_lambda': 1.0, 'num_leaves': 4},
}
LGB_SEEDS = (123, 456, 789, 777, 666,
             1511, 1523, 2025, 2026, 2033,
             279, 239, 70, 77, 31,
             2024, 2077, 3077, 123456, 654321,
             2034, 2035, 2036, 1984, 1991)
N_LGB_SEEDS = len(LGB_SEEDS)
lgb_test = {}

# --- original global 24/48 heads kept intact ---
for horizon in [24, 48]:
    y_bin, mask = make_binary_target(time_values, event_values, horizon)
    valid_idx = np.where(mask)[0]
    cfg = lgb_cfgs[horizon]
    all_test = np.zeros(len(X_lgb_test))
    for seed in LGB_SEEDS:
        seed_test = np.zeros(len(X_lgb_test))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_v, va_v in cv.split(valid_idx, y_bin[mask]):
            tr_idx = valid_idx[tr_v]
            ipcw_w = compute_ipcw_weights(time_values[tr_idx], event_values[tr_idx], horizon)
            m = lgb.LGBMClassifier(**cfg, objective='binary', random_state=seed, verbose=-1)
            m.fit(X_lgb_train.iloc[tr_idx], y_bin[tr_idx], sample_weight=ipcw_w)
            seed_test += m.predict_proba(X_lgb_test)[:, 1] / 5
        all_test += seed_test
    lgb_test[horizon] = all_test / N_LGB_SEEDS
    print(f'LGB {horizon}h IPCW CV-bagged done')

# --- new: near-only timing heads (uncensored active regime) ---
near_train = train_df['dist_min_ci_0_5h'].values < 5000
near_test = test_df['dist_min_ci_0_5h'].values < 5000

NEAR_FEATURES = [
    'low_temporal_resolution_0_5h',
    'dt_first_last_0_5h',
    'num_perimeters_0_5h',
    'eta_effective',
    'alignment_abs',
    'log1p_area_first',
    'log_area_dist_ratio',
]

X_near_train = train_processed.loc[near_train, NEAR_FEATURES].reset_index(drop=True)
X_near_test = test_processed.loc[near_test, NEAR_FEATURES].reset_index(drop=True)
time_near = train_df.loc[near_train, 'time_to_hit_hours'].values

NEAR_LR_SEEDS = (123, 456, 789, 777, 666, 1511, 1523, 2025, 2026, 2033)
near_lr_cfg = {
    12: {'C': 0.50},
    24: {'C': 0.30},
    48: {'C': 0.20},
}
near_lr_test = {}

for horizon in [12, 24, 48]:
    y_h = (time_near <= horizon).astype(int)
    class_counts = np.bincount(y_h)
    min_class = class_counts.min()
    n_splits = max(2, min(5, int(min_class)))

    all_test = np.zeros(len(X_near_test))
    for seed in NEAR_LR_SEEDS:
        seed_test = np.zeros(len(X_near_test))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        for tr_idx, va_idx in cv.split(X_near_train, y_h):
            m = Pipeline([
                ('sc', StandardScaler()),
                ('lr', LogisticRegression(
                    C=near_lr_cfg[horizon]['C'],
                    max_iter=5000,
                    solver='lbfgs',
                    class_weight='balanced'
                ))
            ])
            m.fit(X_near_train.iloc[tr_idx], y_h[tr_idx])
            seed_test += m.predict_proba(X_near_test)[:, 1] / n_splits

        m_full = Pipeline([
            ('sc', StandardScaler()),
            ('lr', LogisticRegression(
                C=near_lr_cfg[horizon]['C'],
                max_iter=5000,
                solver='lbfgs',
                class_weight='balanced'
            ))
        ])
        m_full.fit(X_near_train, y_h)
        full_pred = m_full.predict_proba(X_near_test)[:, 1]

        all_test += (0.80 * seed_test + 0.20 * full_pred) / len(NEAR_LR_SEEDS)

    near_lr_test[horizon] = np.clip(all_test, 0, 1)
    print(f'Near-only LR {horizon}h done | n={len(X_near_train)} | positives={int(y_h.sum())} | negatives={int((1-y_h).sum())}')
W24 = 0.95
W48 = 0.45

test_blend = test_gbsa.copy()

# keep the strong notebook anchor
test_blend[:, 1] = W24 * test_gbsa[:, 1] + (1 - W24) * lgb_test[24]
test_blend[:, 2] = W48 * test_gbsa[:, 2] + (1 - W48) * lgb_test[48]

dist_train = train_df['dist_min_ci_0_5h'].values
dist_test = test_df['dist_min_ci_0_5h'].values
lowtemp_train = train_df['low_temporal_resolution_0_5h'].astype(int).values
lowtemp_test = test_df['low_temporal_resolution_0_5h'].astype(int).values

near_train = dist_train < 5000
near_test = dist_test < 5000
far_test = ~near_test

stable_tr = near_train & (lowtemp_train == 0)
sparse_tr = near_train & (lowtemp_train == 1)

stable_te = near_test & (lowtemp_test == 0)
sparse_te = near_test & (lowtemp_test == 1)

def rate_within(mask, horizon):
    return (train_df.loc[mask, 'time_to_hit_hours'] <= horizon).mean()

p12_stable = rate_within(stable_tr, 12)   # 0.944444
p24_stable = rate_within(stable_tr, 24)   # 0.972222
p48_stable = rate_within(stable_tr, 48)   # 1.0

p12_sparse = rate_within(sparse_tr, 12)   # 0.454545
p24_sparse = rate_within(sparse_tr, 24)   # 0.848485
p48_sparse = rate_within(sparse_tr, 48)   # 0.909091

# new: learned near-only timing refinement
test_blend[near_test, 0] = 0.78 * test_blend[near_test, 0] + 0.22 * near_lr_test[12]
test_blend[near_test, 1] = 0.97 * test_blend[near_test, 1] + 0.03 * near_lr_test[24]
test_blend[near_test, 2] = 0.88 * test_blend[near_test, 2] + 0.12 * near_lr_test[48]

# stable near-zone fires: enforce strong early timing floors
test_blend[stable_te, 0] = np.maximum(
    0.80 * test_blend[stable_te, 0] + 0.20 * p12_stable,
    0.90
)
test_blend[stable_te, 1] = np.maximum(
    0.78 * test_blend[stable_te, 1] + 0.22 * p24_stable,
    0.965
)
test_blend[stable_te, 2] = np.maximum(
    0.82 * test_blend[stable_te, 2] + 0.18 * p48_stable,
    0.995
)

# sparse near-zone fires: softer calibration
test_blend[sparse_te, 0] = (
    0.80 * test_blend[sparse_te, 0] + 0.20 * p12_sparse
)
test_blend[sparse_te, 1] = (
    0.70 * test_blend[sparse_te, 1] + 0.30 * p24_sparse
)
test_blend[sparse_te, 2] = (
    0.65 * test_blend[sparse_te, 2] + 0.35 * p48_sparse
)

# structural support from train: 72h is deterministic under the 5km gate
test_blend[near_test, 3] = 1.0
test_blend[far_test, :] = 0.0

test_final = enforce_monotonicity(test_blend)

submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': test_final[:, 0],
    'prob_24h': test_final[:, 1],
    'prob_48h': test_final[:, 2],
    'prob_72h': test_final[:, 3],
})

output_path = '/kaggle/working/submission.csv' if os.path.isdir('/kaggle/working') else 'submission.csv'
submission.to_csv(output_path, index=False)

print("p12_stable =", round(float(p12_stable), 6))
print("p24_stable =", round(float(p24_stable), 6))
print("p48_stable =", round(float(p48_stable), 6))
print("p12_sparse =", round(float(p12_sparse), 6))
print("p24_sparse =", round(float(p24_sparse), 6))
print("p48_sparse =", round(float(p48_sparse), 6))
print("near_test_count =", int(near_test.sum()))
print(f"Saved: {output_path}")

submission.describe().round(4)
