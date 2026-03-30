"""Exp32: Reproduce loopassembly/0-964-top-1-dual-objective-xgboost-survival (LB~0.964).
3x GBSA ensemble, 10-fold CV, isotonic calibration.
Usage: .venv_sksurv22/Scripts/python scripts/exp32_loop_gbsa.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [12, 24, 48, 72]


def augment_features(df):
    df = df.copy()
    df['projected_arrival_hours'] = np.where(
        df['closing_speed_m_per_h'] > 0,
        df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 1e-4), 9999)
    df['danger_momentum'] = df['closing_speed_m_per_h'] * df['dist_accel_m_per_h2']
    df['proximity_to_area_ratio'] = df['dist_min_ci_0_5h'] / (df['area_first_ha'] + 1e-4)
    perims = np.maximum(1, df['num_perimeters_0_5h'] - 1)
    df['growth_per_perimeter_unit'] = df['area_growth_abs_0_5h'] / perims
    df['acceleration_danger_index'] = df['dist_accel_m_per_h2'] * df['area_first_ha']
    df['is_night_ignition'] = ((df['event_start_hour'] >= 18) | (df['event_start_hour'] <= 6)).astype(int)
    return df


def main():
    train_df = pd.read_csv(os.path.join(PROJECT, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PROJECT, 'test.csv'))
    test_ids = test_df['event_id'].values

    train_df = augment_features(train_df)
    test_df = augment_features(test_df)

    label_cols = ['event_id', 'time_to_hit_hours', 'event']
    X_raw = train_df.drop(columns=label_cols)
    X_test_raw = test_df.drop(columns=['event_id'])

    actual_times = train_df['time_to_hit_hours'].values
    events_occurred = train_df['event'].values
    surv_y = Surv.from_dataframe('event', 'time_to_hit_hours', train_df)

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X = scaler.fit_transform(imputer.fit_transform(X_raw))
    X_test = scaler.transform(imputer.transform(X_test_raw))

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    val_probs = {h: np.zeros(len(X)) for h in HORIZONS}
    test_probs_folds = {h: [] for h in HORIZONS}

    print("Training 10-fold GBSA ensemble...")
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X)):
        models = [
            GradientBoostingSurvivalAnalysis(n_estimators=300, learning_rate=0.01, max_depth=2, subsample=1.0, random_state=42 + fold * 5),
            GradientBoostingSurvivalAnalysis(n_estimators=250, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=43 + fold * 5),
            GradientBoostingSurvivalAnalysis(n_estimators=400, learning_rate=0.015, max_depth=3, subsample=0.9, random_state=44 + fold * 5),
        ]
        fold_val = {h: np.zeros(len(val_idx)) for h in HORIZONS}
        fold_test = {h: np.zeros(len(X_test)) for h in HORIZONS}

        for m in models:
            m.fit(X[trn_idx], surv_y[trn_idx])
            for h in HORIZONS:
                fold_val[h] += np.array([1.0 - c(min(h, c.x[-1])) for c in m.predict_survival_function(X[val_idx])])
                fold_test[h] += np.array([1.0 - c(min(h, c.x[-1])) for c in m.predict_survival_function(X_test)])

        for h in HORIZONS:
            val_probs[h][val_idx] = fold_val[h] / len(models)
            test_probs_folds[h].append(fold_test[h] / len(models))
        print(f"  Fold {fold+1}/10 done")

    print("Isotonic calibration...")
    final = {}
    for h in HORIZONS:
        is_hit = (events_occurred == 1) & (actual_times <= h)
        valid = is_hit | (actual_times > h)
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
        iso.fit(val_probs[h][valid], is_hit[valid].astype(int))
        final[f'prob_{h}h'] = iso.predict(np.mean(test_probs_folds[h], axis=0))

    sub = pd.DataFrame({'event_id': test_ids})
    sub['prob_12h'] = final['prob_12h']
    sub['prob_24h'] = np.maximum(sub['prob_12h'], final['prob_24h'])
    sub['prob_48h'] = np.maximum(sub['prob_24h'], final['prob_48h'])
    sub['prob_72h'] = np.maximum(sub['prob_48h'], final['prob_72h'])

    out = os.path.join(PROJECT, 'submissions', 'submission_exp32_loop_gbsa.csv')
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
        v = sub[col].values
        print(f"  {col}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f}")


if __name__ == '__main__':
    main()
