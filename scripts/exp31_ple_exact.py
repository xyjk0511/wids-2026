"""Exp31: Exact reproduction of suman2208/ple-stacker (LB=0.96654).

Faithful to original notebook. Run multiple times (PyTorch has no fixed seed in original).
Usage: .venv_sksurv22/Scripts/python scripts/exp31_ple_exact.py [--seed SEED] [--tag TAG]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
horizons = np.array([12, 24, 48, 72])


def create_physics_features(df):
    df = df.copy()
    df['physics_time_to_hit'] = df['dist_min_ci_0_5h'] / (df['closing_speed_m_per_h'] + 0.1)
    df['danger_vector'] = df['centroid_speed_m_per_h'] * df['alignment_cos']
    df['accel_interaction'] = df['physics_time_to_hit'] * df['dist_accel_m_per_h2']
    return df


def get_risk_preds(model, X_data):
    surv_funcs = model.predict_survival_function(X_data)
    preds = np.empty((len(surv_funcs), 4))
    for i, fn in enumerate(surv_funcs):
        t_min, t_max = fn.x[0], fn.x[-1]  # sksurv 0.21.0 compat: clip to first event, not domain(0)
        preds[i, :] = 1.0 - fn(np.clip(horizons, t_min, t_max))
    return preds


class PeriodicLinearEncoding(nn.Module):
    def __init__(self, num_bins=4, output_dim=2):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_bins, output_dim))
        self.bias = nn.Parameter(torch.randn(num_bins, output_dim))

    def forward(self, x):
        return torch.cat([torch.cos(x * self.weights[i] + self.bias[i]) for i in range(self.weights.shape[0])], dim=1)


class PLE_Stacker(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ple_layers = nn.ModuleList([PeriodicLinearEncoding(4, 2) for _ in range(num_features)])
        total_ple = num_features * 8  # 4 bins * 2 dim
        self.context_net = nn.Sequential(
            nn.Linear(total_ple, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.manager = nn.Sequential(
            nn.Linear(32 + 12, 16), nn.ReLU(),
            nn.Linear(16, 3), nn.Softmax(dim=1)
        )

    def forward(self, x_raw, preds_gb, preds_rsf, preds_xgb):
        x_ple = torch.cat([self.ple_layers[i](x_raw[:, i].unsqueeze(1)) for i in range(x_raw.shape[1])], dim=1)
        ctx = self.context_net(x_ple)
        weights = self.manager(torch.cat([ctx, preds_gb, preds_rsf, preds_xgb], dim=1))
        return (weights[:, 0:1] * preds_gb) + (weights[:, 1:2] * preds_rsf) + (weights[:, 2:3] * preds_xgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="PyTorch seed (None = no seed, like original)")
    p.add_argument("--tag", default="", help="Output file tag")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"PyTorch seed: {args.seed}")

    print("=== Exp31: PLE Stacker Exact Reproduction ===")
    train_df = pd.read_csv(os.path.join(PROJECT, "train.csv")).drop("event_id", axis=1)
    test_df = pd.read_csv(os.path.join(PROJECT, "test.csv"))
    test_ids = test_df["event_id"].values
    test_df = test_df.drop("event_id", axis=1)

    train_df = create_physics_features(train_df)
    test_df = create_physics_features(test_df)

    X = train_df.drop(columns=["event", "time_to_hit_hours"])
    y = Surv.from_arrays(event=train_df["event"].astype(bool), time=train_df["time_to_hit_hours"])
    X_test = test_df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    time_bins = pd.qcut(train_df['time_to_hit_hours'], q=5, labels=False)

    oof_gb = np.zeros((len(X), 4))
    oof_rsf = np.zeros((len(X), 4))
    oof_xgb = np.zeros((len(X), 4))
    test_gb = np.zeros((len(X_test), 4))
    test_rsf = np.zeros((len(X_test), 4))
    test_xgb = np.zeros((len(X_test), 4))

    times_array = train_df['time_to_hit_hours'].values
    event_bool = train_df['event'].values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, time_bins)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]

        gb = GradientBoostingSurvivalAnalysis(n_estimators=400, learning_rate=0.01, max_depth=3, random_state=42)
        gb.fit(X_tr, y_tr)
        oof_gb[val_idx] = get_risk_preds(gb, X_val)
        test_gb += get_risk_preds(gb, X_test) / n_folds

        rsf = RandomSurvivalForest(n_estimators=500, min_samples_leaf=10, max_depth=5, random_state=42)
        rsf.fit(X_tr, y_tr)
        oof_rsf[val_idx] = get_risk_preds(rsf, X_val)
        test_rsf += get_risk_preds(rsf, X_test) / n_folds

        is_cens_tr = ~train_df.iloc[train_idx]['event'].astype(bool)
        times_tr = times_array[train_idx]
        km_t, km_s = kaplan_meier_estimator(is_cens_tr, times_tr)

        def G_fold(t):
            idx = np.searchsorted(km_t, t, side='right') - 1
            return 1.0 if idx < 0 else max(km_s[idx], 0.05)

        ev_tr = event_bool[train_idx]
        for j, h in enumerate(horizons):
            y_bin, w_bin = np.zeros(len(X_tr)), np.zeros(len(X_tr))
            for i, t_i in enumerate(times_tr):
                if ev_tr[i] and t_i <= h:
                    y_bin[i], w_bin[i] = 1.0, 1.0 / G_fold(t_i - 1e-5)
                elif t_i > h:
                    y_bin[i], w_bin[i] = 0.0, 1.0 / G_fold(h)
            clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     objective='binary:logistic', eval_metric='auc',
                                     random_state=42, base_score=0.5)
            clf.fit(X_tr, y_bin, sample_weight=w_bin)
            oof_xgb[val_idx, j] = clf.predict_proba(X_val)[:, 1]
            test_xgb[:, j] += clf.predict_proba(X_test)[:, 1] / n_folds

        print(f"  Fold {fold+1}/{n_folds} done")

    # IPCW for PLE training
    is_cens = ~train_df['event'].astype(bool)
    km_t, km_s = kaplan_meier_estimator(is_cens, times_array)

    def G(t):
        idx = np.searchsorted(km_t, t, side='right') - 1
        return 1.0 if idx < 0 else max(km_s[idx], 0.05)

    y_targets = np.zeros((len(X), 4))
    w_targets = np.zeros((len(X), 4))
    for i in range(len(X)):
        for j, h in enumerate(horizons):
            t_i = times_array[i]
            if event_bool[i] and t_i <= h:
                y_targets[i, j], w_targets[i, j] = 1.0, 1.0 / G(t_i - 1e-5)
            elif t_i > h:
                y_targets[i, j], w_targets[i, j] = 0.0, 1.0 / G(h)

    X_tensor = torch.FloatTensor(X_scaled)
    gb_tensor = torch.FloatTensor(oof_gb)
    rsf_tensor = torch.FloatTensor(oof_rsf)
    xgb_tensor = torch.FloatTensor(oof_xgb)
    y_tensor = torch.FloatTensor(y_targets)
    w_tensor = torch.FloatTensor(w_targets)

    model = PLE_Stacker(num_features=X_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("Training PLE Stacker (300 epochs)...")
    for epoch in range(300):
        optimizer.zero_grad()
        outputs = model(X_tensor, gb_tensor, rsf_tensor, xgb_tensor)
        loss = torch.mean(w_tensor * (outputs - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"  Epoch {epoch} Loss: {loss.item():.5f}")

    X_test_tensor = torch.FloatTensor(X_test_scaled)
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_tensor, torch.FloatTensor(test_gb),
                            torch.FloatTensor(test_rsf), torch.FloatTensor(test_xgb)).numpy()

    # Original postprocessing: monotonic accumulate + clip(0,1), NO p72=1.0
    final_preds = np.maximum.accumulate(final_preds, axis=1)
    final_preds = np.clip(final_preds, 0.0, 1.0)

    tag = f"_seed{args.seed}" if args.seed is not None else ""
    if args.tag:
        tag += f"_{args.tag}"
    out = os.path.join(PROJECT, "submissions", f"submission_exp31_ple{tag}.csv")
    sub = pd.DataFrame({"event_id": test_ids,
                        "prob_12h": final_preds[:, 0], "prob_24h": final_preds[:, 1],
                        "prob_48h": final_preds[:, 2], "prob_72h": final_preds[:, 3]})
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    for c in ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]:
        v = sub[c].values
        print(f"  {c}: min={v.min():.4f} med={np.median(v):.4f} max={v.max():.4f} unique={len(np.unique(v))}")


if __name__ == "__main__":
    main()
