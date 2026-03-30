"""Multi-seed CV stability benchmark with noise-aware decision rules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
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


@dataclass
class SeedResult:
    cv_seed: int
    rsf_pp: float
    fixed_pp: float
    capped_pp: float
    opt_pp: float
    opt_weight: float
    capped_weight: float


def _parse_seed_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-level", default="medium")
    p.add_argument("--strat-mode", choices=["event", "event_time"], default="event_time")
    p.add_argument("--cv-seeds", default="1,7,21,42,2026")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--n-estimators", type=int, default=1000)
    p.add_argument("--fixed-rsf-weight", type=float, default=0.90)
    p.add_argument("--max-rsf-weight", type=float, default=0.90)
    p.add_argument("--weight-step", type=float, default=0.05)
    p.add_argument("--quiet-cv", action="store_true")
    return p


def run_one_seed(
    train: pd.DataFrame,
    feature_cols: list[str],
    cv_seed: int,
    args: argparse.Namespace,
) -> SeedResult:
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
        random_state=cv_seed,
        strat_mode=args.strat_mode,
        verbose=not args.quiet_cv,
    )

    rsf_pp, _ = score_probs(oof["RSF"], y_time, y_event, apply_postprocess=True)

    fixed_blend = blend_two_models(oof["RSF"], oof["EST"], args.fixed_rsf_weight)
    fixed_pp, _ = score_probs(fixed_blend, y_time, y_event, apply_postprocess=True)

    opt_w, _ = search_global_weight(
        oof["RSF"],
        oof["EST"],
        y_time,
        y_event,
        step=args.weight_step,
    )
    opt_blend = blend_two_models(oof["RSF"], oof["EST"], opt_w)
    opt_pp, _ = score_probs(opt_blend, y_time, y_event, apply_postprocess=True)

    cap_w = min(opt_w, args.max_rsf_weight)
    cap_blend = blend_two_models(oof["RSF"], oof["EST"], cap_w)
    cap_pp, _ = score_probs(cap_blend, y_time, y_event, apply_postprocess=True)

    return SeedResult(
        cv_seed=cv_seed,
        rsf_pp=rsf_pp,
        fixed_pp=fixed_pp,
        capped_pp=cap_pp,
        opt_pp=opt_pp,
        opt_weight=opt_w,
        capped_weight=cap_w,
    )


def summarize(results: list[SeedResult]) -> None:
    df = pd.DataFrame([r.__dict__ for r in results]).sort_values("cv_seed")
    print("\n=== Per-seed results (postprocessed OOF Hybrid) ===")
    print(
        df[
            [
                "cv_seed",
                "rsf_pp",
                "fixed_pp",
                "capped_pp",
                "opt_pp",
                "opt_weight",
                "capped_weight",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    noise_std = df["rsf_pp"].std(ddof=1) if len(df) > 1 else 0.0
    print("\n=== Multi-seed summary ===")
    print(
        f"RSF-only baseline: mean={df['rsf_pp'].mean():.5f} std={noise_std:.5f} "
        f"(noise gate ~ 1 std)"
    )

    for col, name in [
        ("fixed_pp", "Fixed blend"),
        ("capped_pp", "Capped policy"),
        ("opt_pp", "Uncapped OOF-opt"),
    ]:
        delta = df[col] - df["rsf_pp"]
        delta_mean = delta.mean()
        delta_std = delta.std(ddof=1) if len(delta) > 1 else 0.0
        z_like = (delta_mean / noise_std) if noise_std > 0 else np.nan
        z_like_str = f"{z_like:+.2f}" if np.isfinite(z_like) else "n/a"
        print(
            f"{name:18s}: mean={df[col].mean():.5f} "
            f"delta_mean={delta_mean:+.5f} delta_std={delta_std:.5f} "
            f"delta/noise={z_like_str}"
        )

    if "opt_weight" in df:
        print(
            f"\nOOF-optimal RSF weight: mean={df['opt_weight'].mean():.3f} "
            f"min={df['opt_weight'].min():.3f} max={df['opt_weight'].max():.3f}"
        )


def main() -> None:
    args = _build_argparser().parse_args()
    cv_seeds = _parse_seed_list(args.cv_seeds)

    print("=== Stability benchmark ===")
    print(
        f"feature={args.feature_level} strat={args.strat_mode} "
        f"cv={args.n_splits}x{args.n_repeats} seeds={cv_seeds}"
    )

    train, feature_cols = load_prepared_train(args.feature_level)
    print(f"n={len(train)} features={len(feature_cols)}")

    results: list[SeedResult] = []
    for i, cv_seed in enumerate(cv_seeds, 1):
        print(f"\n--- [{i}/{len(cv_seeds)}] cv_seed={cv_seed} ---")
        res = run_one_seed(train, feature_cols, cv_seed, args)
        results.append(res)
        print(
            f"  RSF={res.rsf_pp:.4f} fixed={res.fixed_pp:.4f} "
            f"capped={res.capped_pp:.4f} opt={res.opt_pp:.4f} "
            f"(w_opt={res.opt_weight:.2f} -> w_cap={res.capped_weight:.2f})"
        )

    summarize(results)


if __name__ == "__main__":
    main()
