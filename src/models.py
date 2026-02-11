"""Base model wrappers with unified fit / predict_proba interface."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.config import HORIZONS, FEATURES_MEDIUM

SEED_AVG_SEEDS = [42, 123, 456]


class BaseSurvivalModel(ABC):
    """Common interface for all survival models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y_time: np.ndarray,
        y_event: np.ndarray,
    ) -> "BaseSurvivalModel":
        ...

    @abstractmethod
    def predict_proba(
        self,
        X: pd.DataFrame,
        horizons: list[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Return {horizon: prob_array} where prob = P(event <= horizon)."""
        ...


# ---- 1. Cox Proportional Hazards ----

class CoxPH(BaseSurvivalModel):
    """Lifelines CoxPH with L2 penalty."""

    def __init__(self, penalizer: float = 0.1, features: list[str] | None = None):
        self.penalizer = penalizer
        self.features = features or FEATURES_MEDIUM
        self.model = CoxPHFitter(penalizer=penalizer)

    def fit(self, X, y_time, y_event):
        cols = [c for c in self.features if c in X.columns]
        df = X[cols].copy()
        # standardize features for CoxPH stability
        self._means = df.mean()
        self._stds = df.std().clip(lower=1e-8)
        df = (df - self._means) / self._stds
        df["T"] = np.asarray(y_time, dtype=float)
        df["E"] = np.asarray(y_event, dtype=int)
        df["T"] = df["T"].clip(lower=0.01)
        self.model.fit(df, duration_col="T", event_col="E", show_progress=False)
        self._cols = cols
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        cols = [c for c in self._cols if c in X.columns]
        df = (X[cols] - self._means) / self._stds
        sf = self.model.predict_survival_function(df)
        # sf: DataFrame, index=time, columns=samples
        idx = sf.index.values.astype(float)
        result = {}
        for h in horizons:
            if h <= idx.min():
                surv = sf.iloc[0].values
            elif h >= idx.max():
                surv = sf.iloc[-1].values
            else:
                # per-sample 1D interpolation
                surv = np.array([
                    np.interp(h, idx, sf.iloc[:, i].values)
                    for i in range(sf.shape[1])
                ])
            result[h] = np.clip(1.0 - surv, 0.0, 1.0)
        return result


# ---- 1b. Weibull AFT ----

class WeibullAFT(BaseSurvivalModel):
    """Weibull Accelerated Failure Time model via lifelines.

    Assumes monotonically increasing hazard -- matches fire approach physics.
    """

    def __init__(self, penalizer: float = 0.05, features: list[str] | None = None):
        self.penalizer = penalizer
        self.features = features or FEATURES_MEDIUM

    def fit(self, X, y_time, y_event):
        from lifelines import WeibullAFTFitter
        cols = [c for c in self.features if c in X.columns]
        df = X[cols].copy()
        self._means = df.mean()
        self._stds = df.std().clip(lower=1e-8)
        df = (df - self._means) / self._stds
        df["T"] = np.asarray(y_time, dtype=float).clip(min=0.01)
        df["E"] = np.asarray(y_event, dtype=int)
        self.model = WeibullAFTFitter(penalizer=self.penalizer)
        self.model.fit(df, duration_col="T", event_col="E", show_progress=False)
        self._cols = cols
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        df = (X[self._cols] - self._means) / self._stds
        sf = self.model.predict_survival_function(df)
        idx = sf.index.values.astype(float)
        result = {}
        for h in horizons:
            if h <= idx.min():
                surv = sf.iloc[0].values
            elif h >= idx.max():
                surv = sf.iloc[-1].values
            else:
                surv = np.array([
                    np.interp(h, idx, sf.iloc[:, i].values)
                    for i in range(sf.shape[1])
                ])
            result[h] = np.clip(1.0 - surv, 0.0, 1.0)
        return result


# ---- 1c. LogNormal AFT ----

class LogNormalAFT(BaseSurvivalModel):
    """LogNormal AFT model via lifelines.

    Assumes non-monotonic hazard (rises then falls) -- captures
    'if not hit after long time, probably won't be hit' pattern.
    """

    def __init__(self, penalizer: float = 0.05, features: list[str] | None = None):
        self.penalizer = penalizer
        self.features = features or FEATURES_MEDIUM

    def fit(self, X, y_time, y_event):
        from lifelines import LogNormalAFTFitter
        cols = [c for c in self.features if c in X.columns]
        df = X[cols].copy()
        self._means = df.mean()
        self._stds = df.std().clip(lower=1e-8)
        df = (df - self._means) / self._stds
        df["T"] = np.asarray(y_time, dtype=float).clip(min=0.01)
        df["E"] = np.asarray(y_event, dtype=int)
        self.model = LogNormalAFTFitter(penalizer=self.penalizer)
        self.model.fit(df, duration_col="T", event_col="E", show_progress=False)
        self._cols = cols
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        df = (X[self._cols] - self._means) / self._stds
        sf = self.model.predict_survival_function(df)
        idx = sf.index.values.astype(float)
        result = {}
        for h in horizons:
            if h <= idx.min():
                surv = sf.iloc[0].values
            elif h >= idx.max():
                surv = sf.iloc[-1].values
            else:
                surv = np.array([
                    np.interp(h, idx, sf.iloc[:, i].values)
                    for i in range(sf.shape[1])
                ])
            result[h] = np.clip(1.0 - surv, 0.0, 1.0)
        return result


# ---- 2. Random Survival Forest ----

def _sksurv_y(y_time, y_event):
    """Build scikit-survival structured array."""
    y_event = np.asarray(y_event, dtype=bool)
    y_time = np.asarray(y_time, dtype=float)
    return np.array(
        list(zip(y_event, y_time)),
        dtype=[("event", bool), ("time", float)],
    )


def _sf_to_probs(surv_fns, horizons):
    """Convert sksurv step survival functions to horizon probabilities.

    Matches original 0.96624 code: explicit boundary check via fn.x[-1].
    """
    result = {}
    for h in horizons:
        p_arr = np.zeros(len(surv_fns))
        for i, fn in enumerate(surv_fns):
            s_val = fn(h) if h <= fn.x[-1] else fn(fn.x[-1])
            p_arr[i] = 1.0 - s_val
        result[h] = np.clip(p_arr, 0.0, 1.0)
    return result


class RSF(BaseSurvivalModel):
    """Random Survival Forest via scikit-survival."""

    def __init__(self, n_estimators=1000, max_depth=5, min_samples_leaf=5,
                 min_samples_split=10, max_features="sqrt", random_state=42):
        from sksurv.ensemble import RandomSurvivalForest
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X, y_time, y_event):
        y = _sksurv_y(y_time, y_event)
        self.model.fit(X.values, y)
        self._cols = list(X.columns)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        surv_fns = self.model.predict_survival_function(X[self._cols].values)
        return _sf_to_probs(surv_fns, horizons)


# ---- 3. Gradient Boosting Survival Analysis ----

class GBSA(BaseSurvivalModel):
    """Gradient Boosting Survival Analysis via scikit-survival."""

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.02,
                 subsample=0.8, dropout_rate=0.1, random_state=42):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        self.model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            dropout_rate=dropout_rate,
            min_samples_leaf=8,
            min_samples_split=16,
            random_state=random_state,
        )

    def fit(self, X, y_time, y_event):
        y = _sksurv_y(y_time, y_event)
        self.model.fit(X.values, y)
        self._cols = list(X.columns)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        surv_fns = self.model.predict_survival_function(X[self._cols].values)
        return _sf_to_probs(surv_fns, horizons)


# ---- 4. Multi-Horizon XGBoost ----

def _build_horizon_labels(y_time, y_event, horizon):
    """Build binary labels for a single horizon."""
    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    n = len(y_time)
    labels = np.full(n, np.nan)

    labels[(y_event == 1) & (y_time <= horizon)] = 1.0
    labels[(y_event == 1) & (y_time > horizon)] = 0.0
    labels[(y_event == 0) & (y_time >= horizon)] = 0.0

    eligible = ~np.isnan(labels)
    return labels, eligible


class MultiHorizonXGB(BaseSurvivalModel):
    """Per-horizon XGBoost binary classifiers with seed averaging."""

    CONST_72H = 0.98

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clfs = {}

    def fit(self, X, y_time, y_event):
        import xgboost as xgb

        self._cols = list(X.columns)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        for h in HORIZONS:
            labels, eligible = _build_horizon_labels(y_time, y_event, h)
            if h == 72:
                self.clfs[h] = None
                continue

            X_h = X.values[eligible]
            y_h = labels[eligible]
            n_pos = y_h.sum()
            n_neg = len(y_h) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

            seed_clfs = []
            for seed in SEED_AVG_SEEDS:
                clf = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=2,
                    learning_rate=0.03,
                    scale_pos_weight=spw,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=5,
                    random_state=seed,
                    eval_metric="logloss",
                    verbosity=0,
                )
                clf.fit(X_h, y_h)
                seed_clfs.append(clf)
            self.clfs[h] = seed_clfs
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        result = {}
        for h in horizons:
            if h == 72 or self.clfs.get(h) is None:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                preds = np.column_stack([
                    clf.predict_proba(Xv)[:, 1] for clf in self.clfs[h]
                ])
                result[h] = preds.mean(axis=1)
        return result


# ---- 5. Rank-optimized XGBoost for 12h C-index ----

class RankXGB(BaseSurvivalModel):
    """XGBoost optimized for AUC on 12h only, with heuristic scaling to other horizons.

    Uses a different prediction distribution than MultiHorizonXGB to provide
    ensemble diversity: trains only on 12h labels, then scales via power law.
    """

    CONST_72H = 0.98

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clf = None

    def fit(self, X, y_time, y_event):
        import xgboost as xgb

        self._cols = list(X.columns)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        labels, eligible = _build_horizon_labels(y_time, y_event, 12)
        X_h = X.values[eligible]
        y_h = labels[eligible]

        n_pos = y_h.sum()
        n_neg = len(y_h) - n_pos
        spw = max(n_neg / max(n_pos, 1), 1.0)

        self.clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            scale_pos_weight=spw,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            min_child_weight=5,
            eval_metric="auc",
            random_state=self.random_state,
            verbosity=0,
        )
        self.clf.fit(X_h, y_h)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        scores_12h = self.clf.predict_proba(Xv)[:, 1]
        result = {}
        for h in horizons:
            if h == 72:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                result[h] = np.clip(scores_12h * (h / 12) ** 0.3, 0.0, 1.0)
        return result


# ---- 6. Multi-Horizon LightGBM ----

class MultiHorizonLGBM(BaseSurvivalModel):
    """Per-horizon LightGBM binary classifiers with seed averaging."""

    CONST_72H = 0.98

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clfs = {}

    def fit(self, X, y_time, y_event):
        import lightgbm as lgb

        self._cols = list(X.columns)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        for h in HORIZONS:
            labels, eligible = _build_horizon_labels(y_time, y_event, h)
            if h == 72:
                self.clfs[h] = None
                continue

            X_h = X.values[eligible]
            y_h = labels[eligible]
            n_pos = y_h.sum()
            n_neg = len(y_h) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

            seed_clfs = []
            for seed in SEED_AVG_SEEDS:
                clf = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.03,
                    num_leaves=8,
                    min_child_samples=10,
                    scale_pos_weight=spw,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=seed,
                    verbosity=-1,
                )
                clf.fit(X_h, y_h)
                seed_clfs.append(clf)
            self.clfs[h] = seed_clfs
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        result = {}
        for h in horizons:
            if h == 72 or self.clfs.get(h) is None:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                preds = np.column_stack([
                    clf.predict_proba(Xv)[:, 1] for clf in self.clfs[h]
                ])
                result[h] = preds.mean(axis=1)
        return result


# ---- 7. Multi-Horizon CatBoost ----

class MultiHorizonCatBoost(BaseSurvivalModel):
    """Per-horizon CatBoost binary classifiers with seed averaging."""

    CONST_72H = 0.98

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.clfs = {}

    def fit(self, X, y_time, y_event):
        from catboost import CatBoostClassifier

        self._cols = list(X.columns)
        y_time = np.asarray(y_time, dtype=float)
        y_event = np.asarray(y_event, dtype=int)

        for h in HORIZONS:
            labels, eligible = _build_horizon_labels(y_time, y_event, h)
            if h == 72:
                self.clfs[h] = None
                continue

            X_h = X.values[eligible]
            y_h = labels[eligible]
            n_pos = y_h.sum()
            n_neg = len(y_h) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

            seed_clfs = []
            for seed in SEED_AVG_SEEDS:
                clf = CatBoostClassifier(
                    iterations=200,
                    depth=3,
                    learning_rate=0.03,
                    l2_leaf_reg=3.0,
                    scale_pos_weight=spw,
                    subsample=0.8,
                    random_seed=seed,
                    verbose=0,
                )
                clf.fit(X_h, y_h)
                seed_clfs.append(clf)
            self.clfs[h] = seed_clfs
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        result = {}
        for h in horizons:
            if h == 72 or self.clfs.get(h) is None:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                preds = np.column_stack([
                    clf.predict_proba(Xv)[:, 1] for clf in self.clfs[h]
                ])
                result[h] = preds.mean(axis=1)
        return result
