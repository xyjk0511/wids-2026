"""
Exp30: Multi-anchor blend with admission gates + OOF-driven weight search.

Usage:
  python scripts/exp30_blend_anchors.py \
    --sub_a submissions/submission_0.96624.csv \
    --sub_b submissions/anchor_suman2208/submission.csv \
    --oof_a <oof_a.csv> --oof_b <oof_b.csv> \
    --lb_b 0.96086

Gates (all must pass):
  Gate 1: lb_b > 0.96624
  Gate 2: rho48 in [0.90, 0.99]
  Gate 3: p48 std > 0.01 and no >80% values at boundary
"""
import argparse, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
EVAL_TIMES = [12, 24, 48, 72]
REF_LB = 0.96624


def submission_postprocess(pred_probs):
    r = {}
    r[12] = np.clip(pred_probs[12], 0.01, 0.99)
    r[24] = pred_probs[24].copy()
    for t in [48, 72]:
        r[t] = np.maximum(pred_probs[t], r[t - 24] if t == 48 else r[48])
    r[72] = np.ones(len(r[72]))
    for t in [24, 48]:
        r[t] = np.clip(r[t], 0.01, 0.99)
    n = len(r[12])
    for i in range(n):
        prev = 0.0
        for t in EVAL_TIMES:
            r[t][i] = max(r[t][i], prev)
            prev = r[t][i]
    for t in EVAL_TIMES:
        r[t] = np.clip(r[t], 0.01, 1.0 if t == 72 else 0.99)
    return r


def oof_score(oof_a, oof_b, w):
    """Proxy: mean Spearman of blended OOF vs oof_a across shared columns."""
    blended = w * oof_a.values + (1 - w) * oof_b.values
    scores = [abs(spearmanr(blended[:, j], oof_a.values[:, j])[0])
              for j in range(blended.shape[1])
              if oof_a.values[:, j].std() > 1e-9]
    return np.mean(scores) if scores else 0.0


def blend_submission(sub_a, sub_b, w):
    probs = {t: w * sub_a[c].values + (1 - w) * sub_b[c].values
             for t, c in zip(EVAL_TIMES, PROB_COLS)}
    pp = submission_postprocess(probs)
    return pd.DataFrame({
        "event_id": sub_a["event_id"].values,
        "prob_12h": pp[12], "prob_24h": pp[24],
        "prob_48h": pp[48], "prob_72h": pp[72],
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_a", required=True)
    p.add_argument("--sub_b", required=True)
    p.add_argument("--oof_a", required=True)
    p.add_argument("--oof_b", required=True)
    p.add_argument("--lb_b", type=float, required=True)
    args = p.parse_args()

    sub_a = pd.read_csv(args.sub_a).sort_values("event_id").reset_index(drop=True)
    sub_b = pd.read_csv(args.sub_b).sort_values("event_id").reset_index(drop=True)

    # Gate 1
    if args.lb_b <= REF_LB:
        print(f"[GATE 1 FAILED] lb_b={args.lb_b:.5f} <= ref={REF_LB:.5f}. No blend produced.")
        sys.exit(0)

    # Gate 2
    rho48, _ = spearmanr(sub_a["prob_48h"].values, sub_b["prob_48h"].values)
    if not (0.90 <= rho48 <= 0.99):
        print(f"[GATE 2 FAILED] rho48={rho48:.4f} not in [0.90, 0.99]. No blend produced.")
        sys.exit(0)

    # Gate 3
    p48 = sub_b["prob_48h"].values
    boundary_frac = np.mean((p48 <= 0.011) | (p48 >= 0.989))
    if p48.std() <= 0.01 or boundary_frac > 0.80:
        print(f"[GATE 3 FAILED] p48 std={p48.std():.4f}, boundary_frac={boundary_frac:.2f}. No blend produced.")
        sys.exit(0)

    print(f"[ALL GATES PASSED] rho48={rho48:.4f}, lb_b={args.lb_b:.5f}")

    oof_a = pd.read_csv(args.oof_a)
    oof_b = pd.read_csv(args.oof_b)
    common = [c for c in oof_a.columns if c in oof_b.columns]
    oa, ob = oof_a[common], oof_b[common]

    # Coarse search
    coarse = {w: oof_score(oa, ob, w) for w in [0.3, 0.4, 0.5, 0.6, 0.7]}
    best_c = max(coarse, key=coarse.get)
    print(f"Coarse best: w={best_c:.2f} score={coarse[best_c]:.6f}")

    # Fine search
    fine_ws = np.round(np.arange(best_c - 0.05, best_c + 0.051, 0.01), 3)
    fine = {w: oof_score(oa, ob, w) for w in fine_ws if 0.0 <= w <= 1.0}
    all_scores = {**coarse, **fine}
    top3 = sorted(all_scores, key=all_scores.get, reverse=True)[:3]

    print(f"\n{'w_a':>6}  {'OOF_score':>10}  {'rho_p48':>8}  path")
    for w in top3:
        blend = blend_submission(sub_a, sub_b, w)
        rho, _ = spearmanr(blend["prob_48h"].values, sub_a["prob_48h"].values)
        out = f"submissions/exp30_blend_w{w:.2f}.csv"
        blend.to_csv(out, index=False)
        print(f"  {w:.2f}  {all_scores[w]:.6f}  {rho:.4f}  {out}")


if __name__ == "__main__":
    main()
