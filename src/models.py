"""Base model wrappers with unified fit / predict_proba interface."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.config import HORIZONS, FEATURES_MEDIUM


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
    """Convert sksurv step survival functions to horizon probabilities."""
    result = {}
    for h in horizons:
        probs = []
        for sf in surv_fns:
            try:
                s_val = sf(h)
            except ValueError:
                # horizon exceeds max observed time; use last known S(t)
                s_val = sf.y[-1]
            probs.append(1.0 - s_val)
        result[h] = np.clip(np.array(probs), 0.0, 1.0)
    return result


class RSF(BaseSurvivalModel):
    """Random Survival Forest via scikit-survival."""

    def __init__(self, n_estimators=500, max_depth=4, min_samples_leaf=15,
                 random_state=42):
        from sksurv.ensemble import RandomSurvivalForest
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            min_samples_split=20,
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

    def __init__(self, n_estimators=300, max_depth=2, learning_rate=0.02,
                 subsample=0.8, random_state=42):
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis
        self.model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=10,
            min_samples_split=20,
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
    """Per-horizon XGBoost binary classifiers."""

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
                # 72h: almost all eligible are positive, skip training
                self.clfs[h] = None
                continue

            X_h = X.values[eligible]
            y_h = labels[eligible]
            n_pos = y_h.sum()
            n_neg = len(y_h) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

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
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
            )
            clf.fit(X_h, y_h)
            self.clfs[h] = clf
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        result = {}
        for h in horizons:
            if h == 72 or self.clfs.get(h) is None:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                result[h] = self.clfs[h].predict_proba(Xv)[:, 1]
        return result


# ---- 5. Rank-optimized XGBoost for 12h C-index ----

class RankXGB(BaseSurvivalModel):
    """XGBoost optimized for AUC on each horizon independently."""

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
            if h == 72:
                self.clfs[h] = None
                continue

            labels, eligible = _build_horizon_labels(y_time, y_event, h)
            X_h = X.values[eligible]
            y_h = labels[eligible]

            n_pos = y_h.sum()
            n_neg = len(y_h) - n_pos
            spw = max(n_neg / max(n_pos, 1), 1.0)

            clf = xgb.XGBClassifier(
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
            clf.fit(X_h, y_h)
            self.clfs[h] = clf
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        Xv = X[self._cols].values
        result = {}
        for h in horizons:
            if h == 72 or self.clfs.get(h) is None:
                result[h] = np.full(len(X), self.CONST_72H)
            else:
                result[h] = self.clfs[h].predict_proba(Xv)[:, 1]
        return result
