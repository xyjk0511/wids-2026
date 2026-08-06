"""Exp30: Compare two submission CSVs — Spearman per horizon + blend eligibility."""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROB_COLS = ["prob_12h", "prob_24h", "prob_48h", "prob_72h"]

def compare(a_path, b_path):
    a = pd.read_csv(a_path).sort_values("event_id").reset_index(drop=True)
    b = pd.read_csv(b_path).sort_values("event_id").reset_index(drop=True)
    assert list(a["event_id"]) == list(b["event_id"]), "event_id mismatch"

    eligible = False
    print(f"\n{'='*60}")
    print(f"A: {a_path}")
    print(f"B: {b_path}")
    print(f"{'='*60}")

    for col in PROB_COLS:
        va, vb = a[col].values, b[col].values
        rho, _ = spearmanr(va, vb)
        diff = np.abs(va - vb)
        if rho < 0.99:
            eligible = True
        print(f"\n  {col}:")
        print(f"    Spearman = {rho:.6f}  max_diff = {diff.max():.6e}  mean_diff = {diff.mean():.6e}")
        print(f"    A: min={va.min():.4f} med={np.median(va):.4f} max={va.max():.4f} unique={len(np.unique(va))}")
        print(f"    B: min={vb.min():.4f} med={np.median(vb):.4f} max={vb.max():.4f} unique={len(np.unique(vb))}")

    verdict = "ELIGIBLE (rho < 0.99 on at least one horizon)" if eligible else "SKIP (all horizons rho >= 0.99)"
    print(f"\n  Blend verdict: {verdict}")
    return eligible

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sub_a", required=True)
    p.add_argument("--sub_b", required=True)
    args = p.parse_args()
    compare(args.sub_a, args.sub_b)
