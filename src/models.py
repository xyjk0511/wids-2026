"""Base model wrappers with unified fit / predict_proba interface."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from src.config import HORIZONS, FEATURES_MEDIUM
from src.labels import build_horizon_labels
from src.surv_post import surv_fns_to_probs

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


def _sf_to_probs(surv_fns, horizons, policy="clip"):
    """Convert sksurv step survival functions to horizon probabilities.

    Uses a centralized conversion layer for reproducible boundary behavior.
    """
    return surv_fns_to_probs(surv_fns, horizons, policy=policy)


class RSF(BaseSurvivalModel):
    """Random Survival Forest via scikit-survival."""

    def __init__(self, n_estimators=1000, max_depth=5, min_samples_leaf=5,
                 min_samples_split=10, max_features="sqrt", random_state=42,
                 sf_policy="clip"):
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
        self.sf_policy = sf_policy

    def fit(self, X, y_time, y_event):
        y = _sksurv_y(y_time, y_event)
        self.model.fit(X.values, y)
        self._cols = list(X.columns)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        surv_fns = self.model.predict_survival_function(X[self._cols].values)
        return _sf_to_probs(surv_fns, horizons, policy=self.sf_policy)

    def predict_risk(self, X):
        """Return cumulative hazard sum (continuous risk score, higher = more risk)."""
        return self.model.predict(X[self._cols].values)


# ---- 2b. Extra Survival Trees ----

class EST(BaseSurvivalModel):
    """Extra Survival Trees via scikit-survival (random splits, lower variance)."""

    def __init__(self, n_estimators=1000, max_depth=5, min_samples_leaf=5,
                 min_samples_split=10, max_features="sqrt", random_state=42,
                 sf_policy="clip"):
        from sksurv.ensemble import ExtraSurvivalTrees
        self.model = ExtraSurvivalTrees(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
        )
        self.sf_policy = sf_policy

    def fit(self, X, y_time, y_event):
        y = _sksurv_y(y_time, y_event)
        self.model.fit(X.values, y)
        self._cols = list(X.columns)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        surv_fns = self.model.predict_survival_function(X[self._cols].values)
        return _sf_to_probs(surv_fns, horizons, policy=self.sf_policy)

    def predict_risk(self, X):
        """Return cumulative hazard sum (continuous risk score, higher = more risk)."""
        return self.model.predict(X[self._cols].values)


# ---- 3. Gradient Boosting Survival Analysis ----

class GBSA(BaseSurvivalModel):
    """Gradient Boosting Survival Analysis via scikit-survival."""

    def __init__(self, n_estimators=300, max_depth=3, learning_rate=0.02,
                 subsample=0.8, dropout_rate=0.1, random_state=42,
                 sf_policy="clip"):
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
        self.sf_policy = sf_policy

    def fit(self, X, y_time, y_event):
        y = _sksurv_y(y_time, y_event)
        self.model.fit(X.values, y)
        self._cols = list(X.columns)
        return self

    def predict_proba(self, X, horizons=None):
        horizons = horizons or HORIZONS
        surv_fns = self.model.predict_survival_function(X[self._cols].values)
        return _sf_to_probs(surv_fns, horizons, policy=self.sf_policy)


# ---- 4. Multi-Horizon XGBoost ----


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
            labels, eligible = build_horizon_labels(y_time, y_event, h)
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

        labels, eligible = build_horizon_labels(y_time, y_event, 12)
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
            labels, eligible = build_horizon_labels(y_time, y_event, h)
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
            labels, eligible = build_horizon_labels(y_time, y_event, h)
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


# ---- 8. XGBoost Cox Survival ----

class XGBoostAFT(BaseSurvivalModel):
    """XGBoost with survival:cox objective + Breslow baseline estimator.

    Outputs hazard ratios, then converts to survival probabilities via
    S(t|x) = S0(t)^exp(margin), where S0(t) is the Breslow baseline.
    Named XGBoostAFT for backward compatibility with train.py imports.
    """

    def __init__(self, n_estimators=200, max_depth=3, learning_rate=0.05,
                 subsample=0.8, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.model = None
        self._baseline_times = None
        self._baseline_survival = None

    def fit(self, X, y_time, y_event):
        import xgboost as xgb

        self._cols = list(X.columns)
        y_time = np.asarray(y_time, dtype=float).clip(min=0.01)
        y_event = np.asarray(y_event, dtype=int)

        # Cox label: +time if event, -time if censored
        y_cox = np.where(y_event == 1, y_time, -y_time)

        dtrain = xgb.DMatrix(X.values, label=y_cox)

        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "tree_method": "hist",
            "seed": self.random_state,
            "verbosity": 0,
        }

        self.model = xgb.train(
            params, dtrain, num_boost_round=self.n_estimators,
        )

        # Breslow baseline hazard estimation
        margins = self.model.predict(dtrain, output_margin=True)
        self._estimate_breslow(y_time, y_event, margins)
        return self

    def _estimate_breslow(self, y_time, y_event, margins):
        """Estimate Breslow baseline cumulative hazard and survival."""
        order = np.argsort(y_time)
        times_sorted = y_time[order]
        events_sorted = y_event[order]
        exp_margins = np.exp(margins[order])

        unique_times = np.unique(times_sorted[events_sorted == 1])
        if len(unique_times) == 0:
            self._baseline_times = np.array([0.0])
            self._baseline_cum_hazard = np.array([0.0])
            self._baseline_survival = np.array([1.0])
            return
        cum_hazard = np.zeros(len(unique_times))

        for j, t in enumerate(unique_times):
            at_risk = exp_margins[times_sorted >= t].sum()
            n_events = events_sorted[times_sorted == t].sum()
            h = n_events / max(at_risk, 1e-8)
            cum_hazard[j] = (cum_hazard[j - 1] if j > 0 else 0.0) + h

        self._baseline_times = unique_times
        self._baseline_cum_hazard = cum_hazard
        self._baseline_survival = np.exp(-cum_hazard)

    def predict_proba(self, X, horizons=None):
        import xgboost as xgb

        horizons = horizons or HORIZONS
        dmat = xgb.DMatrix(X[self._cols].values)
        margins = self.model.predict(dmat, output_margin=True)
        exp_margins = np.exp(margins)

        result = {}
        for h in horizons:
            # Interpolate baseline cumulative hazard at horizon h
            H0_h = np.interp(h, self._baseline_times, self._baseline_cum_hazard,
                             left=0.0, right=self._baseline_cum_hazard[-1])
            # S(h|x) = exp(-H0(h) * exp(margin))
            surv = np.exp(-H0_h * exp_margins)
            result[h] = np.clip(1.0 - surv, 0.0, 1.0)
        return result
