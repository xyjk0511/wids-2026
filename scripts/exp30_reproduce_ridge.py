"""Exp30: Reproduce rhythmghai notebook (LB=0.96536) — Ridge IPCW stacker."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
horizons = np.array([12, 24, 48, 72])

def physics(df):
    df = df.copy()
    df["speed_ratio"] = df["centroid_speed_m_per_h"] / (df["closing_speed_abs_m_per_h"] + 1)
    df["distance_pressure"] = df["closing_speed_m_per_h"] / (df["dist_min_ci_0_5h"] + 1)
    df["growth_pressure"] = df["area_growth_rate_ha_per_h"] * df["alignment_abs"]
    df["directional_force"] = df["alignment_cos"] * df["centroid_speed_m_per_h"]
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
    print("=== Exp30: Reproduce Ridge stacker (0.96536) ===")
    train = pd.read_csv(os.path.join(PROJECT, "train.csv"))
    test = pd.read_csv(os.path.join(PROJECT, "test.csv"))

    y_time = train["time_to_hit_hours"]
    y_event = train["event"]
    X = train.drop(columns=["event_id", "event", "time_to_hit_hours"])
    X_test = test.drop(columns=["event_id"])

    X, X_test = physics(X), physics(X_test)
    y = Surv.from_arrays(event=y_event.astype(bool), time=y_time)

    # 5-fold CV for OOF predictions
    folds = StratifiedKFold(5, shuffle=True, random_state=42)
    bins = pd.qcut(y_time, 5, labels=False)

    oof_gb = np.zeros((len(X), 4))
    oof_rsf = np.zeros((len(X), 4))
    test_gb = np.zeros((len(X_test), 4))
    test_rsf = np.zeros((len(X_test), 4))

    for fold, (tr, va) in enumerate(folds.split(X, bins)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr = y[tr]

        gb = GradientBoostingSurvivalAnalysis(
            n_estimators=500, learning_rate=0.02, max_depth=3, random_state=42)
        gb.fit(Xtr, ytr)
        oof_gb[va] = surv_preds(gb, Xva)
        test_gb += surv_preds(gb, X_test) / 5

        rsf = RandomSurvivalForest(
            n_estimators=600, min_samples_leaf=8, max_depth=6, random_state=42)
        rsf.fit(Xtr, ytr)
        oof_rsf[va] = surv_preds(rsf, Xva)
        test_rsf += surv_preds(rsf, X_test) / 5
        print(f"  Fold {fold+1}/5 done")

    # IPCW weights
    is_cens = ~y_event.astype(bool)
    times = y_time.values
    km_t, km_s = kaplan_meier_estimator(is_cens, times)

    def G(t):
        idx = np.searchsorted(km_t, t, side="right") - 1
        return 1.0 if idx < 0 else max(km_s[idx], 0.05)

    # Ridge meta-model
    meta_X = np.hstack([oof_gb, oof_rsf])
    meta_test = np.hstack([test_gb, test_rsf])

    Y = np.zeros((len(X), 4))
    W = np.zeros((len(X), 4))
    for i in range(len(X)):
        for j, h in enumerate(horizons):
            t = times[i]
            if y_event.iloc[i] and t <= h:
                Y[i, j], W[i, j] = 1.0, 1.0 / G(t - 1e-6)
            elif t > h:
                Y[i, j], W[i, j] = 0.0, 1.0 / G(h)

    final_test = np.zeros((len(X_test), 4))
    for j in range(4):
        model = Ridge(alpha=1.0)
        model.fit(meta_X, Y[:, j], sample_weight=W[:, j])
        final_test[:, j] = model.predict(meta_test)

    # Postprocess
    final_test = np.maximum.accumulate(final_test, axis=1)
    final_test = np.clip(final_test, 1e-6, 1 - 1e-6)

    sub = pd.DataFrame({
        "event_id": test["event_id"],
        "prob_12h": final_test[:, 0], "prob_24h": final_test[:, 1],
        "prob_48h": final_test[:, 2], "prob_72h": final_test[:, 3],
    })
    out = os.path.join(PROJECT, "submissions", "submission_exp30_ridge.csv")
    sub.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    for c in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub[c].values
        print(f"  {c}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f}")

if __name__ == "__main__":
    main()
