"""Exp30: Reproduce ple-stacker (LB=0.96654) base models — no PyTorch PLE, use Ridge meta."""
import os, numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
horizons = np.array([12, 24, 48, 72])

def physics_ple(df):
    df = df.copy()
    df['physics_time_to_hit'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 0.1)
    df['danger_vector'] = df['centroid_speed_m_per_h'] * df['alignment_cos']
    df['accel_interaction'] = df['physics_time_to_hit'] * df['dist_accel_m_per_h2']
    return df

def surv_preds(model, X):
    surv = model.predict_survival_function(X)
    out = np.zeros((len(X), 4))
    for i, f in enumerate(surv):
        tmin, tmax = f.domain
        t = np.clip(horizons, tmin, tmax)
        out[i] = 1 - f(t)
    return out

def main():
    print("=== Exp30: Reproduce PLE stacker (0.96654) base models ===")
    train = pd.read_csv(os.path.join(PROJECT, "train.csv"))
    test = pd.read_csv(os.path.join(PROJECT, "test.csv"))

    y_time = train["time_to_hit_hours"]
    y_event = train["event"]
    X = train.drop(columns=["event_id", "event", "time_to_hit_hours"])
    X_test = test.drop(columns=["event_id"])

    X, X_test = physics_ple(X), physics_ple(X_test)
    y = Surv.from_arrays(event=y_event.astype(bool), time=y_time)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    bins = pd.qcut(y_time, 5, labels=False)

    oof_gb = np.zeros((len(X), 4))
    oof_rsf = np.zeros((len(X), 4))
    oof_xgb = np.zeros((len(X), 4))
    test_gb = np.zeros((len(X_test), 4))
    test_rsf = np.zeros((len(X_test), 4))
    test_xgb = np.zeros((len(X_test), 4))

    times_arr = y_time.values
    event_arr = y_event.values

    for fold, (tr, va) in enumerate(skf.split(X, bins)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr = y[tr]

        # GBSA
        gb = GradientBoostingSurvivalAnalysis(
            n_estimators=400, learning_rate=0.01, max_depth=3, random_state=42)
        gb.fit(Xtr, ytr)
        oof_gb[va] = surv_preds(gb, Xva)
        test_gb += surv_preds(gb, X_test) / 5

        # RSF
        rsf = RandomSurvivalForest(
            n_estimators=500, min_samples_leaf=10, max_depth=5, random_state=42)
        rsf.fit(Xtr, ytr)
        oof_rsf[va] = surv_preds(rsf, Xva)
        test_rsf += surv_preds(rsf, X_test) / 5

        # XGBoost IPCW per horizon
        is_cens_tr = ~train.iloc[tr]['event'].astype(bool)
        km_t, km_s = kaplan_meier_estimator(is_cens_tr, times_arr[tr])

        def G_fold(t):
            idx = np.searchsorted(km_t, t, side='right') - 1
            return 1.0 if idx < 0 else max(km_s[idx], 0.05)

        for j, h in enumerate(horizons):
            y_bin = np.zeros(len(Xtr))
            w_bin = np.zeros(len(Xtr))
            for i_loc, i_glob in enumerate(tr):
                t_i = times_arr[i_glob]
                if event_arr[i_glob] and t_i <= h:
                    y_bin[i_loc], w_bin[i_loc] = 1.0, 1.0 / G_fold(t_i - 1e-5)
                elif t_i > h:
                    y_bin[i_loc], w_bin[i_loc] = 0.0, 1.0 / G_fold(h)

            clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                objective='binary:logistic', eval_metric='auc',
                random_state=42, base_score=0.5)
            clf.fit(Xtr, y_bin, sample_weight=w_bin)
            oof_xgb[va, j] = clf.predict_proba(Xva)[:, 1]
            test_xgb[:, j] += clf.predict_proba(X_test)[:, 1] / 5

        print(f"  Fold {fold+1}/5 done")

    # Ridge meta-model instead of PLE neural network
    is_cens = ~y_event.astype(bool)
    km_t, km_s = kaplan_meier_estimator(is_cens, times_arr)

    def G(t):
        idx = np.searchsorted(km_t, t, side='right') - 1
        return 1.0 if idx < 0 else max(km_s[idx], 0.05)

    meta_X = np.hstack([oof_gb, oof_rsf, oof_xgb])
    meta_test = np.hstack([test_gb, test_rsf, test_xgb])

    Y = np.zeros((len(X), 4))
    W = np.zeros((len(X), 4))
    for i in range(len(X)):
        for j, h in enumerate(horizons):
            t = times_arr[i]
            if event_arr[i] and t <= h:
                Y[i, j], W[i, j] = 1.0, 1.0 / G(t - 1e-6)
            elif t > h:
                Y[i, j], W[i, j] = 0.0, 1.0 / G(h)

    final_test = np.zeros((len(X_test), 4))
    for j in range(4):
        model = Ridge(alpha=1.0)
        model.fit(meta_X, Y[:, j], sample_weight=W[:, j])
        final_test[:, j] = model.predict(meta_test)

    # Also save simple average (no meta-model)
    avg_test = (test_gb + test_rsf + test_xgb) / 3

    for label, preds in [("ridge", final_test), ("avg", avg_test)]:
        p = np.maximum.accumulate(preds, axis=1)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        sub = pd.DataFrame({
            "event_id": test["event_id"],
            "prob_12h": p[:, 0], "prob_24h": p[:, 1],
            "prob_48h": p[:, 2], "prob_72h": p[:, 3],
        })
        out = os.path.join(PROJECT, "submissions", f"submission_exp30_ple_{label}.csv")
        sub.to_csv(out, index=False)
        print(f"\n  Saved: {out}")
        for c in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
            v = sub[c].values
            print(f"  {c}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f}")

if __name__ == "__main__":
    main()
