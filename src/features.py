import numpy as np
import pandas as pd

from src.config import (
    REDUNDANT_DROP, FEATURES_MINIMAL, FEATURES_MEDIUM, FEATURES_FULL,
    FEATURES_V96624_BASE, FEATURES_V96624_ENGINEERED, FEATURES_V96624_PLUS,
)


def remove_redundant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop highly correlated redundant features."""
    cols_to_drop = [c for c in REDUNDANT_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features with physical meaning."""
    df = df.copy()

    # log-distance: compress long-tail distribution
    df["log_dist"] = np.log1p(df["dist_min_ci_0_5h"])

    # distance-area ratio: closer + larger fire = higher threat
    area_sqrt = np.sqrt(df["area_first_ha"].clip(lower=1e-6))
    df["dist_area_ratio"] = df["dist_min_ci_0_5h"] / area_sqrt

    # binary growth flag
    df["has_growth"] = (df["area_growth_abs_0_5h"] > 0).astype(int)

    # closing speed relative to distance
    dist_safe = df["dist_min_ci_0_5h"].clip(lower=1.0)
    df["speed_dist_ratio"] = df["closing_speed_m_per_h"] / dist_safe

    # --- new engineered features for C-index boost ---

    # ETA: estimated time of arrival (distance / closing speed)
    speed_safe = df["closing_speed_m_per_h"].clip(lower=0.1)
    df["eta_hours"] = dist_safe / speed_safe

    # inverse distance: amplifies near-range discrimination
    df["inv_dist"] = 1.0 / dist_safe

    # binary: fire already within 5km threshold
    df["dist_close"] = (df["dist_min_ci_0_5h"] < 5000).astype(int)

    # distance x alignment interaction
    df["dist_alignment"] = df["log_dist"] * df["alignment_abs"]

    # area growth rate relative to distance
    df["growth_dist_ratio"] = df["area_growth_rate_ha_per_h"] / dist_safe

    # --- round 2: features targeting [657m, 4545m] overlap zone ---

    # sigmoid soft indicator for the critical overlap zone
    center, scale = 2600, 1000
    df["dist_zone_score"] = 1.0 / (1.0 + np.exp((df["dist_min_ci_0_5h"] - center) / scale))

    # static threat index = sqrt(area) / distance
    df["threat_static"] = np.sqrt(df["area_first_ha"].clip(lower=1)) / df["dist_min_ci_0_5h"].clip(lower=100)

    # multi-observation flag (addresses 88.7% zero-inflation)
    df["has_dynamics"] = (df["num_perimeters_0_5h"] > 1).astype(int)

    # alignment-weighted closing speed
    df["effective_closing"] = df["closing_speed_m_per_h"] * df["alignment_abs"]

    # exponential distance decay (5km scale)
    df["dist_neg_exp"] = np.exp(-df["dist_min_ci_0_5h"] / 5000.0)

    # hour cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["event_start_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["event_start_hour"] / 24)

    # log-scale threat = log(area) - log(distance)
    df["log_area_over_dist"] = np.log1p(df["area_first_ha"]) - df["log_dist"]

    # --- v96624 engineered features ---
    if "dist_slope_ci_0_5h" in df.columns:
        df["is_approaching"] = (df["dist_slope_ci_0_5h"] < 0).astype(int)
    df["log_dist_min"] = np.log1p(df["dist_min_ci_0_5h"])

    # --- round 3: interaction features ---

    # threat urgency: close distance + fast approach
    df["threat_urgency"] = df["dist_min_ci_0_5h"] * df["closing_speed_m_per_h"]

    # directional threat: large fire aligned toward community
    df["directional_threat"] = df["area_first_ha"] * df["alignment_abs"]

    # expansion threat: fast-growing fire at close range
    df["expansion_threat"] = df["radial_growth_m"] * df["inv_dist"]

    return df


def add_engineered_0_97092(df: pd.DataFrame) -> pd.DataFrame:
    """Add 0.97092 distance-oriented features (30+ features).

    精确复现 0.97092 notebook 的 create_features()，
    不做 remove_redundant()，直接在原始列上追加距离交互特征。
    """
    df = df.copy()

    dist_safe = df["dist_min_ci_0_5h"].clip(lower=1.0)
    area_sqrt = np.sqrt(df["area_first_ha"].clip(lower=1e-6))
    speed_safe = df["closing_speed_m_per_h"].clip(lower=0.1)
    wind_safe = df["wind_speed_m_per_s"].clip(lower=0.1) if "wind_speed_m_per_s" in df.columns else pd.Series(1.0, index=df.index)
    temp_safe = df["temp_c"].clip(lower=0.1) if "temp_c" in df.columns else pd.Series(20.0, index=df.index)
    humidity_safe = df["humidity_percent"].clip(lower=1.0) if "humidity_percent" in df.columns else pd.Series(50.0, index=df.index)

    # === Core distance features ===
    df["log_dist"] = np.log1p(df["dist_min_ci_0_5h"])
    df["inv_dist"] = 1.0 / dist_safe
    df["dist_close"] = (df["dist_min_ci_0_5h"] < 5000).astype(int)

    # === Distance-area interactions ===
    df["dist_area_ratio"] = dist_safe / area_sqrt
    df["inv_dist_area"] = df["inv_dist"] * area_sqrt

    # === Distance-speed interactions ===
    df["speed_dist_ratio"] = speed_safe / dist_safe
    df["eta_hours"] = dist_safe / speed_safe
    df["log_eta"] = np.log1p(df["eta_hours"])

    # === Distance-alignment interactions ===
    df["dist_alignment"] = df["log_dist"] * df["alignment_abs"]
    df["inv_dist_alignment"] = df["inv_dist"] * df["alignment_abs"]

    # === Distance-growth interactions ===
    df["growth_dist_ratio"] = df["area_growth_rate_ha_per_h"] / dist_safe
    df["has_growth"] = (df["area_growth_abs_0_5h"] > 0).astype(int)
    df["inv_dist_growth"] = df["inv_dist"] * (1 + df["area_growth_rate_ha_per_h"])

    # === Distance-wind interactions (fallback if no wind column) ===
    df["wind_dist_ratio"] = wind_safe / dist_safe
    df["inv_dist_wind"] = df["inv_dist"] * wind_safe

    # === Distance-temperature interactions (fallback if no temp column) ===
    df["temp_dist_ratio"] = temp_safe / dist_safe

    # === Distance-humidity interactions (fallback if no humidity column) ===
    df["humidity_dist_ratio"] = humidity_safe / dist_safe
    df["inv_dist_dryness"] = df["inv_dist"] * (100.0 / humidity_safe)

    # === Composite threat scores ===
    df["threat_score_1"] = df["inv_dist"] * df["speed_dist_ratio"] * area_sqrt
    df["threat_score_2"] = df["inv_dist"] * df["growth_dist_ratio"] * df["alignment_abs"]
    df["threat_score_3"] = df["inv_dist"] * df["wind_dist_ratio"] * (100.0 / humidity_safe)

    # === Distance bins ===
    df["dist_bin"] = pd.cut(
        df["dist_min_ci_0_5h"],
        bins=[0, 1000, 2000, 5000, 10000, np.inf],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # === Log-scale features ===
    df["log_area"] = np.log1p(df["area_first_ha"])
    df["log_speed"] = np.log1p(df["closing_speed_m_per_h"].clip(lower=0.0))

    return df


def get_feature_cols_0_97092(df: pd.DataFrame) -> list[str]:
    """Return feature columns for the 0.97092 feature set.

    Excludes ID, time, and event columns from the 0.97092-engineered DataFrame.
    """
    exclude = {"event_id", "time_to_hit_hours", "event"}
    return [c for c in df.columns if c not in exclude]


def get_feature_set(df: pd.DataFrame, level: str = "medium") -> list[str]:
    """Return feature column names for the given level.

    Args:
        df: DataFrame (used to verify columns exist).
        level: One of 'minimal', 'medium', 'full', 'v96624'.

    Returns:
        List of feature column names present in df.
    """
    if level == "v96624":
        candidates = list(FEATURES_V96624_BASE) + list(FEATURES_V96624_ENGINEERED)
        return [c for c in candidates if c in df.columns]

    if level == "v96624_plus":
        return [c for c in FEATURES_V96624_PLUS if c in df.columns]

    if level == "v96624_elim":
        from src.config import FEATURES_V96624_ELIM
        return [c for c in FEATURES_V96624_ELIM if c in df.columns]

    mapping = {
        "minimal": FEATURES_MINIMAL,
        "medium": FEATURES_MEDIUM,
        "full": FEATURES_FULL,
    }
    candidates = mapping[level]
    engineered = [
        "log_dist", "dist_area_ratio", "has_growth", "speed_dist_ratio",
        "eta_hours", "inv_dist", "dist_close", "dist_alignment", "growth_dist_ratio",
        "dist_zone_score", "threat_static", "has_dynamics", "effective_closing",
        "dist_neg_exp", "hour_sin", "hour_cos", "log_area_over_dist",
        "threat_urgency", "directional_threat", "expansion_threat",
    ]
    if level in ("medium", "full"):
        candidates = list(candidates) + engineered
    return [c for c in candidates if c in df.columns]
