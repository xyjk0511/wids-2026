"""Exp30: Rank-average blend of two anchors with weight search."""
import argparse, os, numpy as np, pandas as pd
from scipy.stats import spearmanr

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def postprocess(df):
    """Force p72=1.0, monotonicity, clip — match 0.96624 postprocessing."""
    df = df.copy()
    df["prob_72h"] = 1.0
    for i in range(len(df)):
        prev = 0.0
        for c in PROB_COLS:
            df.loc[df.index[i], c] = max(df.iloc[i][c], prev)
            prev = df.iloc[i][c]
    for c in PROB_COLS[:-1]:
        df[c] = df[c].clip(0.01, 0.99)
    df["prob_72h"] = df["prob_72h"].clip(0.01, 1.0)
    return df

def rank_blend(a, b, w, col):
    """Rank-average blend: w * rank(a) + (1-w) * rank(b), then map back to values."""
    ra = a[col].rank(pct=True)
    rb = b[col].rank(pct=True)
    blended_rank = w * ra + (1 - w) * rb
    return blended_rank.rank(pct=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sub_a", required=True, help="Primary anchor (e.g. 0.96624)")
    p.add_argument("--sub_b", required=True, help="New anchor to blend")
    p.add_argument("--weights", default="0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                   help="Comma-separated weights for sub_a")
    p.add_argument("--postprocess_b", action="store_true",
                   help="Apply postprocessing to sub_b before blending")
    args = p.parse_args()

    a = pd.read_csv(args.sub_a).sort_values("event_id").reset_index(drop=True)
    b = pd.read_csv(args.sub_b).sort_values("event_id").reset_index(drop=True)

    if args.postprocess_b:
        print("Applying postprocessing to sub_b...")
        b = postprocess(b)

    # Pre-check: Spearman on p48
    rho48, _ = spearmanr(a["prob_48h"], b["prob_48h"])
    print(f"\np48 Spearman(A, B) = {rho48:.6f}")
    if rho48 >= 0.99:
        print("SKIP: p48 rho >= 0.99, blend will not change LB ranking")
        return

    weights = [float(x) for x in args.weights.split(",")]
    ref_p48 = a["prob_48h"].values

    for w in weights:
        sub = pd.DataFrame({"event_id": a["event_id"]})
        for col in PROB_COLS:
            # Use value-space blend (not rank-space) to preserve calibration
            sub[col] = w * a[col].values + (1 - w) * b[col].values

        # Apply postprocessing
        sub = postprocess(sub)

        rho, _ = spearmanr(sub["prob_48h"], ref_p48)
        out = os.path.join(PROJECT, "submissions",
                           f"submission_exp30_blend_w{w:.1f}.csv")
        sub.to_csv(out, index=False)
        p48 = sub["prob_48h"].values
        print(f"  w={w:.1f}: p48 rho_vs_ref={rho:.4f}  "
              f"min={p48.min():.4f} med={np.median(p48):.4f} max={p48.max():.4f}  "
              f"-> {out}")

if __name__ == "__main__":
    main()
