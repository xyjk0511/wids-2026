"""Stacking: per-horizon heads on base survival model predictions.

Anti-leakage design: fold-internal base feature reconstruction.
Each outer fold retrains base models; inner CV produces base_train_oof
so the head never sees leaked base predictions.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.labels import build_horizon_labels
from src.models import RSF, EST, XGBoostAFT, CoxPH, WeibullAFT, RankXGB

HEAD_HORIZONS = [12, 24, 48]
BASE_NAMES = ["RSF", "EST", "XGBCox"]
EXTENDED_BASE_NAMES = ["RSF", "EST", "XGBCox", "CoxPH", "WeibullAFT"]


def _make_base(name, seed):
    if name == "RSF":
        return RSF(n_estimators=200, min_samples_leaf=5, random_state=seed)
    if name == "EST":
        return EST(n_estimators=200, min_samples_leaf=5, random_state=seed)
    if name == "CoxPH":
        return CoxPH(penalizer=0.1)
    if name == "WeibullAFT":
        return WeibullAFT(penalizer=0.05)
    if name == "XGBCox":
        return XGBoostAFT(n_estimators=200, random_state=seed)
    raise ValueError(f"Unknown base model: {name}")


def _train_predict_base(X_tr, yt_tr, ye_tr, X_pred, seed=42, base_names=None):
    """Train base models on scaled data, predict on X_pred.

    Returns {model_name: {horizon: np.array}}.
    """
    base_names = base_names or BASE_NAMES
    scaler = StandardScaler()
    cols = list(X_tr.columns)
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=cols, index=X_tr.index)
    X_pred_s = pd.DataFrame(scaler.transform(X_pred), columns=cols, index=X_pred.index)

    preds = {}
    for name in base_names:
        m = _make_base(name, seed)
        m.fit(X_tr_s, yt_tr, ye_tr)
        preds[name] = m.predict_proba(X_pred_s, horizons=HEAD_HORIZONS)
    return preds


def _to_rank01(x):
    """Stable rank transform to (0, 1)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    return ranks / (len(x) + 1.0)


def _to_logit(x, eps=1e-6):
    """Numerically-stable logit transform."""
    p = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _concat_head_features(
    X_orig,
    base_preds,
    base_feature_mode="raw",
    use_orig_features=True,
    base_names=None,
):
    """Build head feature matrix from original features + base predictions."""
    base_names = base_names or BASE_NAMES
    df = X_orig.reset_index(drop=True).copy() if use_orig_features else pd.DataFrame(index=np.arange(len(X_orig)))

    for name in base_names:
        if name not in base_preds:
            continue
        for h in HEAD_HORIZONS:
            if h not in base_preds[name]:
                continue
            col_base = np.asarray(base_preds[name][h], dtype=float)
            if base_feature_mode == "raw":
                df[f"{name}_{h}h_raw"] = col_base
            elif base_feature_mode == "rank_logit":
                df[f"{name}_{h}h_rank"] = _to_rank01(col_base)
                df[f"{name}_{h}h_logit"] = _to_logit(col_base)
            elif base_feature_mode == "raw_rank_logit":
                df[f"{name}_{h}h_raw"] = col_base
                df[f"{name}_{h}h_rank"] = _to_rank01(col_base)
                df[f"{name}_{h}h_logit"] = _to_logit(col_base)
            else:
                raise ValueError(f"Unknown base_feature_mode: {base_feature_mode}")
    return df


def _inner_cv_base_oof(X_tr, yt_tr, ye_tr, n_inner, seed, base_names=None):
    """Inner CV to produce base_train_oof (anti-leak base features for head training)."""
    base_names = base_names or BASE_NAMES
    n = len(X_tr)
    base_oof = {name: {h: np.zeros(n) for h in HEAD_HORIZONS} for name in base_names}

    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    strat = ye_tr.astype(int)

    for in_tr, in_va in skf.split(X_tr, strat):
        X_in_tr = X_tr.iloc[in_tr]
        X_in_va = X_tr.iloc[in_va]
        preds = _train_predict_base(
            X_in_tr, yt_tr[in_tr], ye_tr[in_tr], X_in_va, seed=seed,
            base_names=base_names,
        )
        for name in base_names:
            for h in HEAD_HORIZONS:
                base_oof[name][h][in_va] = preds[name][h]

    return base_oof


def _apply_calibrator(cal, probs):
    if cal is None:
        return probs
    if isinstance(cal, LogisticRegression):
        return cal.predict_proba(probs.reshape(-1, 1))[:, 1]
    return cal.predict(probs)


def _fit_calibrator(raw_probs, labels, horizon, yt_cal, ye_cal, calibration_mode="auto"):
    """Fit Platt + Isotonic, pick best per horizon rules."""
    from src.evaluation import c_index, horizon_brier_score

    if calibration_mode == "none":
        return None, "none"

    if calibration_mode == "iso24_48" and horizon == 12:
        # Keep 12h ranking untouched in this mode.
        return None, "none"

    if len(raw_probs) < 10 or labels.sum() < 2:
        return None, "none"

    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(raw_probs.reshape(-1, 1), labels)
    p_platt = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, labels)
    p_iso = iso.predict(raw_probs)

    if calibration_mode == "iso24_48":
        return iso, "isotonic"

    if horizon == 12:
        ci_raw = c_index(yt_cal, ye_cal, raw_probs)
        ci_platt = c_index(yt_cal, ye_cal, p_platt)
        ci_iso = c_index(yt_cal, ye_cal, p_iso)
        if ci_platt >= ci_raw - 0.002:
            return platt, "platt"
        if ci_iso >= ci_raw - 0.002:
            return iso, "isotonic"
        return None, "none"

    b_raw = horizon_brier_score(yt_cal, ye_cal, raw_probs, horizon)
    b_platt = horizon_brier_score(yt_cal, ye_cal, p_platt, horizon)
    b_iso = horizon_brier_score(yt_cal, ye_cal, p_iso, horizon)
    best = min(
        [("none", b_raw, None), ("platt", b_platt, platt), ("isotonic", b_iso, iso)],
        key=lambda x: x[1],
    )
    return best[2], best[0]


def train_horizon_heads(
    X_features,
    base_oof,
    y_time,
    y_event,
    horizons=None,
    n_splits=5,
    n_repeats=3,
    n_inner_splits=3,
    random_state=1042,
    head_model="xgb",
    base_feature_mode="raw",
    use_orig_features=True,
    calibration_mode="auto",
    base_names=None,
):
    """Train per-horizon heads with fold-internal base feature reconstruction.

    Returns (head_oof, heads).
      head_oof: {horizon: np.array} calibrated OOF probabilities
      heads: dict with trained models for test prediction
    """
    horizons = horizons or HEAD_HORIZONS
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    n = len(y_time)

    head_oof = {h: np.zeros(n) for h in horizons}
    oof_counts = np.zeros(n)
    fold_heads = {h: [] for h in horizons}

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,
    )

    fold_idx = 0
    for tr_idx, va_idx in rskf.split(X_features, y_event):
        fold_idx += 1
        X_tr = X_features.iloc[tr_idx].reset_index(drop=True)
        X_va = X_features.iloc[va_idx].reset_index(drop=True)
        yt_tr, ye_tr = y_time[tr_idx], y_event[tr_idx]
        yt_va, ye_va = y_time[va_idx], y_event[va_idx]

        # Step 1: base models predict val (outer OOF)
        base_val = _train_predict_base(X_tr, yt_tr, ye_tr, X_va, seed=random_state, base_names=base_names)

        # Step 2: inner CV for base_train_oof
        base_train_oof = _inner_cv_base_oof(
            X_tr, yt_tr, ye_tr, n_inner_splits, seed=random_state + fold_idx,
            base_names=base_names,
        )

        # Step 3: build head features
        X_head_tr = _concat_head_features(
            X_tr,
            base_train_oof,
            base_feature_mode=base_feature_mode,
            use_orig_features=use_orig_features,
            base_names=base_names,
        )
        X_head_va = _concat_head_features(
            X_va,
            base_val,
            base_feature_mode=base_feature_mode,
            use_orig_features=use_orig_features,
            base_names=base_names,
        )

        # Step 4: split train 80/20 for head training vs calibration
        n_tr = len(X_head_tr)
        rng = np.random.RandomState(random_state + fold_idx)
        perm = rng.permutation(n_tr)
        n_cal = max(int(n_tr * 0.2), 5)
        cal_mask = np.zeros(n_tr, dtype=bool)
        cal_mask[perm[:n_cal]] = True
        fit_mask = ~cal_mask

        # Step 5: train head per horizon
        for h in horizons:
            labels_full, elig_full = build_horizon_labels(yt_tr, ye_tr, h)
            fit_elig = fit_mask & elig_full
            cal_elig = cal_mask & elig_full

            if fit_elig.sum() < 5:
                head_oof[h][va_idx] += 0.5
                oof_counts[va_idx] += 1
                fold_heads[h].append({"model": None, "calibrator": None})
                continue

            X_fit = X_head_tr.values[fit_elig]
            y_fit = labels_full[fit_elig]
            n_pos = y_fit.sum()
            n_neg = len(y_fit) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

            if head_model == "logit":
                clf = LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                )
                clf.fit(X_fit, y_fit)
            elif head_model == "xgb":
                import xgboost as xgb
                clf = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    scale_pos_weight=spw,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=5,
                    eval_metric="auc" if h == 12 else "logloss",
                    random_state=random_state,
                    verbosity=0,
                )
                clf.fit(X_fit, y_fit)
            else:
                raise ValueError(f"Unknown head_model: {head_model}")

            # Calibration
            cal_obj = None
            if cal_elig.sum() >= 5:
                X_cal = X_head_tr.values[cal_elig]
                y_cal = labels_full[cal_elig]
                raw_cal = clf.predict_proba(X_cal)[:, 1]
                cal_obj, cal_name = _fit_calibrator(
                    raw_cal, y_cal, h, yt_tr[cal_elig], ye_tr[cal_elig], calibration_mode=calibration_mode,
                )

            # Predict val
            raw_va = clf.predict_proba(X_head_va.values)[:, 1]
            calibrated_va = _apply_calibrator(cal_obj, raw_va)
            head_oof[h][va_idx] += calibrated_va
            fold_heads[h].append({"model": clf, "calibrator": cal_obj})

        oof_counts[va_idx] += 1
        if fold_idx % n_splits == 0:
            print(f"  Head repeat {fold_idx // n_splits}/{n_repeats} done")

    # Average over repeats
    mask = oof_counts > 0
    for h in horizons:
        head_oof[h][mask] /= oof_counts[mask]

    heads = {
        "meta": {
            "base_names": list(base_names or BASE_NAMES),
            "orig_cols": list(X_features.columns),
            "horizons": horizons,
            "head_feature_cols": list(X_head_tr.columns),
            "head_model": head_model,
            "base_feature_mode": base_feature_mode,
            "use_orig_features": use_orig_features,
            "calibration_mode": calibration_mode,
        },
    }
    for h in horizons:
        heads[h] = fold_heads[h]

    return head_oof, heads


def predict_horizon_heads(heads, X_features, base_test_preds):
    """Predict test probabilities using trained fold heads.

    Args:
        heads: output from train_horizon_heads
        X_features: test original features
        base_test_preds: {model_name: {horizon: np.array}} from full retrain

    Returns: {horizon: np.array}
    """
    meta = heads["meta"]
    X_test = _concat_head_features(
        X_features,
        base_test_preds,
        base_feature_mode=meta["base_feature_mode"],
        use_orig_features=meta["use_orig_features"],
        base_names=meta.get("base_names"),
    )
    X_test = X_test.reindex(columns=meta["head_feature_cols"], fill_value=0.0)
    Xv = X_test.values

    result = {}
    for h in meta["horizons"]:
        fold_preds = []
        for entry in heads[h]:
            if entry["model"] is None:
                fold_preds.append(np.full(len(X_features), 0.5))
                continue
            raw = entry["model"].predict_proba(Xv)[:, 1]
            cal = _apply_calibrator(entry["calibrator"], raw)
            fold_preds.append(cal)
        result[h] = np.mean(fold_preds, axis=0)

    return result
