from pathlib import Path

# --- Paths ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = PROJECT_DIR / "submission.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

# --- Target ---
ID_COL = "event_id"
TIME_COL = "time_to_hit_hours"
EVENT_COL = "event"
HORIZONS = [12, 24, 48, 72]
PROB_COLS = [f"prob_{h}h" for h in HORIZONS]

# --- CV ---
N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42

# --- All 34 features ---
ALL_FEATURES = [
    # temporal_coverage (3)
    "num_perimeters_0_5h",
    "dt_first_last_0_5h",
    "low_temporal_resolution_0_5h",
    # growth (10)
    "area_first_ha",
    "area_growth_abs_0_5h",
    "area_growth_rel_0_5h",
    "area_growth_rate_ha_per_h",
    "log1p_area_first",
    "log1p_growth",
    "log_area_ratio_0_5h",
    "relative_growth_0_5h",
    "radial_growth_m",
    "radial_growth_rate_m_per_h",
    # centroid_kinematics (5)
    "centroid_displacement_m",
    "centroid_speed_m_per_h",
    "spread_bearing_deg",
    "spread_bearing_sin",
    "spread_bearing_cos",
    # distance (9)
    "dist_min_ci_0_5h",
    "dist_std_ci_0_5h",
    "dist_change_ci_0_5h",
    "dist_slope_ci_0_5h",
    "closing_speed_m_per_h",
    "closing_speed_abs_m_per_h",
    "projected_advance_m",
    "dist_accel_m_per_h2",
    "dist_fit_r2_0_5h",
    # directionality (4)
    "alignment_cos",
    "alignment_abs",
    "cross_track_component",
    "along_track_speed",
    # temporal_metadata (3)
    "event_start_hour",
    "event_start_dayofweek",
    "event_start_month",
]

# --- Redundant pairs (|corr| > 0.95): drop second, keep first ---
REDUNDANT_DROP = [
    "relative_growth_0_5h",      # == area_growth_rel_0_5h (r=1.0)
    "projected_advance_m",       # == -dist_change_ci_0_5h (r=-1.0)
    "closing_speed_abs_m_per_h", # ~= closing_speed_m_per_h (r~0.99)
    "centroid_displacement_m",   # ~= radial_growth_m (r~0.99)
    "centroid_speed_m_per_h",    # ~= radial_growth_rate_m_per_h (r~0.99)
    "log_area_ratio_0_5h",       # ~= area_growth_rel_0_5h (r~0.99)
    "log1p_growth",              # ~= area_growth_abs_0_5h (r~0.97)
    "alignment_cos",             # ~= alignment_abs (sign flip)
    "dist_slope_ci_0_5h",        # ~= closing_speed_m_per_h (r~0.99)
    "along_track_speed",         # ~= closing_speed_m_per_h (r~0.96)
]

# --- Feature sets (after redundancy removal) ---
FEATURES_MINIMAL = [
    "dist_min_ci_0_5h",
    "low_temporal_resolution_0_5h",
    "alignment_abs",
    "closing_speed_m_per_h",
    "area_first_ha",
    "dt_first_last_0_5h",
]

FEATURES_MEDIUM = [
    "dist_min_ci_0_5h",
    "low_temporal_resolution_0_5h",
    "alignment_abs",
    "closing_speed_m_per_h",
    "area_first_ha",
    "dt_first_last_0_5h",
    "num_perimeters_0_5h",
    "area_growth_abs_0_5h",
    "radial_growth_m",
    "dist_change_ci_0_5h",
    "dist_std_ci_0_5h",
    "cross_track_component",
    "event_start_month",
    "dist_accel_m_per_h2",
    "dist_fit_r2_0_5h",
    "area_growth_rate_ha_per_h",
]

FEATURES_FULL = [
    f for f in ALL_FEATURES if f not in REDUNDANT_DROP
]
