"""Exp33: Reproduce sigmaborov/wids-simple-lgb-baseline-0-96 (LB~0.96+).
4 LightGBM models (one per horizon), 80/20 split, hybrid score early stopping.
Usage: .venv_sksurv22/Scripts/python scripts/exp33_sigma_lgb.py
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HORIZONS = [12, 24, 48, 72]
EPS = 1e-6


def add_features(df):
    df = df.copy()
    dist_m = df['dist_min_ci_0_5h'].astype(float).values
    closing = df['closing_speed_m_per_h'].astype(float).values
    growth = df['area_growth_rate_ha_per_h'].astype(float).values
    align = df['alignment_abs'].astype(float).values
    df['log1p_dist_min_km'] = np.log1p(dist_m / 1000.0)
    ttc = dist_m / (np.maximum(closing, 0.0) + EPS)
    ttc[closing <= 0.0] = 1e6
    df['ttc_hours'] = np.clip(ttc, 0.0, 1e6)
    df['pressure_to_evac'] = (growth * align) / (dist_m / 1000.0 + 1.0)
    df['aligned_closing_speed'] = df['closing_speed_m_per_h'].astype(float) * align
    return df


def harrell_c(t, e, r):
    conc = ties = comp = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        for j in range(len(t)):
            if j == i or t[i] >= t[j]:
                continue
            comp += 1.0
            if r[i] > r[j]:
                conc += 1.0
            elif r[i] == r[j]:
                ties += 1.0
    return (conc + 0.5 * ties) / comp if comp > 0 else 0.5


def brier_h(t, e, p, H):
    valid = ~((e == 0) & (t < H))
    if not valid.any():
        return 0.25
    y = ((e == 1) & (t <= H)).astype(float)[valid]
    return float(np.mean((np.clip(p[valid], 0, 1) - y) ** 2))


def hybrid(t, e, p12, p24, p48, p72):
    risk = 0.3 * p24 + 0.4 * p48 + 0.3 * p72
    c = harrell_c(t, e, risk)
    wb = 0.3 * brier_h(t, e, p24, 24) + 0.4 * brier_h(t, e, p48, 48) + 0.3 * brier_h(t, e, p72, 72)
    return 0.3 * c + 0.7 * (1.0 - wb)


def monotone(p12, p24, p48, p72):
    p24 = np.maximum(p24, p12)
    p48 = np.maximum(p48, p24)
    p72 = np.maximum(p72, p48)
    return np.clip(p12, 0, 1), np.clip(p24, 0, 1), np.clip(p48, 0, 1), np.clip(p72, 0, 1)


def main():
    train_df = pd.read_csv(os.path.join(PROJECT, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PROJECT, 'test.csv'))
    test_ids = test_df['event_id'].values

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    exclude = {'event_id', 'time_to_hit_hours', 'event'}
    feat_cols = [c for c in train_df.columns if c not in exclude]

    t_all = train_df['time_to_hit_hours'].astype(float).values
    e_all = train_df['event'].astype(int).values

    scaler = StandardScaler()
    X_all = np.nan_to_num(scaler.fit_transform(train_df[feat_cols].astype(float).values))
    X_test = np.nan_to_num(scaler.transform(test_df[feat_cols].astype(float).values))

    idx = np.arange(len(train_df))
    idx_tr, idx_va = train_test_split(idx, test_size=0.2, random_state=42, stratify=e_all)
    X_va = X_all[idx_va]
    t_va, e_va = t_all[idx_va], e_all[idx_va]

    base_params = dict(objective='binary', learning_rate=0.043, num_leaves=42,
                       min_data_in_leaf=32, feature_fraction=0.73, bagging_fraction=0.86,
                       max_depth=2, bagging_freq=2, lambda_l2=0.0, lambda_l1=0.0,
                       verbosity=-1, seed=42, force_col_wise=True)

    def make_ds(h):
        y_all = ((e_all == 1) & (t_all <= h)).astype(float)
        valid = ~((e_all == 0) & (t_all < h))
        m_tr = valid[idx_tr]
        m_va = valid[idx_va]
        dtr = lgb.Dataset(X_all[idx_tr][m_tr], label=y_all[idx_tr][m_tr], free_raw_data=False)
        dva = lgb.Dataset(X_all[idx_va][m_va], label=y_all[idx_va][m_va], free_raw_data=False)
        return dtr, dva

    boosters = {h: lgb.Booster(params=base_params, train_set=make_ds(h)[0]) for h in HORIZONS}

    best_score, best_iter, since_best = -1e18, 0, 0
    best_strings = None
    PATIENCE, MAX_ROUNDS, PRINT_EVERY = 200, 5000, 10

    print("Training...")
    for it in range(1, MAX_ROUNDS + 1):
        for b in boosters.values():
            b.update()
        if it % PRINT_EVERY == 0 or it == 1:
            preds = {h: boosters[h].predict(X_va, num_iteration=it) for h in HORIZONS}
            p12, p24, p48, p72 = monotone(*[preds[h] for h in HORIZONS])
            score = hybrid(t_va, e_va, p12, p24, p48, p72)
            if score > best_score + 1e-12:
                best_score, best_iter, since_best = score, it, 0
                best_strings = {h: boosters[h].model_to_string(num_iteration=it) for h in HORIZONS}
            else:
                since_best += PRINT_EVERY
                if since_best >= PATIENCE:
                    break
            if it % 100 == 0:
                print(f"  iter={it} hybrid={score:.5f} best={best_score:.5f}@{best_iter}")

    print(f"Best iter={best_iter} hybrid={best_score:.5f}")

    # Retrain on full data
    def train_full(h):
        y = ((e_all == 1) & (t_all <= h)).astype(float)
        valid = ~((e_all == 0) & (t_all < h))
        dtr = lgb.Dataset(X_all[valid], label=y[valid], free_raw_data=False)
        b = lgb.Booster(params=base_params, train_set=dtr)
        for _ in range(best_iter):
            b.update()
        return b

    full_boosters = {h: train_full(h) for h in HORIZONS}
    preds_test = {h: full_boosters[h].predict(X_test) for h in HORIZONS}
    p12, p24, p48, p72 = monotone(*[preds_test[h] for h in HORIZONS])

    sub = pd.DataFrame({'event_id': test_ids, 'prob_12h': p12, 'prob_24h': p24,
                        'prob_48h': p48, 'prob_72h': p72})
    out = os.path.join(PROJECT, 'submissions', 'submission_exp33_sigma_lgb.csv')
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
        v = sub[col].values
        print(f"  {col}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f}")


if __name__ == '__main__':
    main()
