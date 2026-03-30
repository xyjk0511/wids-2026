"""OOF evaluation that mirrors submission policy decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import EST, RSF  # noqa: E402

from cv_protocol import (  # noqa: E402
    blend_two_models,
    load_prepared_train,
    run_oof_cv,
    score_probs,
    search_global_weight,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default="medium")
    p.add_argument("--strat-mode", choices=["event", "event_time"], default="event_time")
    p.add_argument("--compare-strat", action="store_true")
    p.add_argument("--cv-seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--max-rsf-weight", type=float, default=0.90)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--quiet-cv", action="store_true")
    return p


def evaluate_once(
    train: pd.DataFrame,
    feature_cols: list[str],
    strat_mode: str,
    args: argparse.Namespace,
) -> dict:
    model_builders = {
        "RSF": lambda seed: RSF(n_estimators=args.n_estimators, random_state=seed),
        "EST": lambda seed: EST(n_estimators=args.n_estimators, random_state=seed),
    }

    oof, y_time, y_event = run_oof_cv(
        train=train,
        feature_cols=feature_cols,
        model_builders=model_builders,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.cv_seed,
        strat_mode=strat_mode,
        verbose=not args.quiet_cv,
    )

    # Raw single-model
    rsf_raw, rsf_raw_det = score_probs(oof["RSF"], y_time, y_event, apply_postprocess=False)
    est_raw, est_raw_det = score_probs(oof["EST"], y_time, y_event, apply_postprocess=False)

    # Postprocessed single-model
    rsf_pp, rsf_pp_det = score_probs(oof["RSF"], y_time, y_event, apply_postprocess=True)
    est_pp, est_pp_det = score_probs(oof["EST"], y_time, y_event, apply_postprocess=True)

    # OOF-optimal global blend, then capped policy like train.py retrain stage.
    opt_w, opt_raw = search_global_weight(
        oof["RSF"], oof["EST"], y_time, y_event, step=args.weight_step
    )
    uncapped_blend = blend_two_models(oof["RSF"], oof["EST"], opt_w)
    uncapped_pp, uncapped_pp_det = score_probs(
        uncapped_blend, y_time, y_event, apply_postprocess=True
    )

    cap_w = min(opt_w, args.max_rsf_weight)
    capped_blend = blend_two_models(oof["RSF"], oof["EST"], cap_w)
    capped_raw, capped_raw_det = score_probs(
        capped_blend, y_time, y_event, apply_postprocess=False
    )
    capped_pp, capped_pp_det = score_probs(
        capped_blend, y_time, y_event, apply_postprocess=True
    )

    fixed90_blend = blend_two_models(oof["RSF"], oof["EST"], 0.90)
    fixed90_pp, fixed90_pp_det = score_probs(
        fixed90_blend, y_time, y_event, apply_postprocess=True
    )

    return {
        "strat_mode": strat_mode,
        "rsf_raw": rsf_raw,
        "rsf_raw_ci": rsf_raw_det["c_index"],
        "rsf_raw_wb": rsf_raw_det["weighted_brier"],
        "est_raw": est_raw,
        "est_raw_ci": est_raw_det["c_index"],
        "est_raw_wb": est_raw_det["weighted_brier"],
        "rsf_pp": rsf_pp,
        "rsf_pp_ci": rsf_pp_det["c_index"],
        "rsf_pp_wb": rsf_pp_det["weighted_brier"],
        "est_pp": est_pp,
        "est_pp_ci": est_pp_det["c_index"],
        "est_pp_wb": est_pp_det["weighted_brier"],
        "opt_w": opt_w,
        "opt_raw": opt_raw,
        "uncapped_pp": uncapped_pp,
        "uncapped_pp_ci": uncapped_pp_det["c_index"],
        "uncapped_pp_wb": uncapped_pp_det["weighted_brier"],
        "cap_w": cap_w,
        "capped_raw": capped_raw,
        "capped_raw_ci": capped_raw_det["c_index"],
        "capped_raw_wb": capped_raw_det["weighted_brier"],
        "capped_pp": capped_pp,
        "capped_pp_ci": capped_pp_det["c_index"],
        "capped_pp_wb": capped_pp_det["weighted_brier"],
        "fixed90_pp": fixed90_pp,
        "fixed90_pp_ci": fixed90_pp_det["c_index"],
        "fixed90_pp_wb": fixed90_pp_det["weighted_brier"],
    }


def main() -> None:
    args = _build_argparser().parse_args()
    modes = ["event", "event_time"] if args.compare_strat else [args.strat_mode]

    print("=== Isomorphic OOF evaluation ===")
    print(
        f"feature={args.feature_level} cv_seed={args.cv_seed} "
        f"cv={args.n_splits}x{args.n_repeats} n_estimators={args.n_estimators}"
    )

    train, feature_cols = load_prepared_train(args.feature_level)
    print(f"n={len(train)} features={len(feature_cols)}")

    rows = []
    for mode in modes:
        print(f"\n--- strat_mode={mode} ---")
        row = evaluate_once(train, feature_cols, mode, args)
        rows.append(row)
        print(
            f"RSF(pp)={row['rsf_pp']:.4f} EST(pp)={row['est_pp']:.4f} "
            f"OOF-opt-w={row['opt_w']:.2f} capped={row['cap_w']:.2f}"
        )
        print(
            f"uncapped(pp)={row['uncapped_pp']:.4f} "
            f"capped(pp)={row['capped_pp']:.4f} fixed90(pp)={row['fixed90_pp']:.4f}"
        )

    out = pd.DataFrame(rows)
    keep_cols = [
        "strat_mode",
        "rsf_pp",
        "est_pp",
        "opt_w",
        "cap_w",
        "uncapped_pp",
        "capped_pp",
        "fixed90_pp",
        "rsf_pp_ci",
        "rsf_pp_wb",
        "capped_pp_ci",
        "capped_pp_wb",
    ]
    print("\n=== Summary table ===")
    print(out[keep_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

