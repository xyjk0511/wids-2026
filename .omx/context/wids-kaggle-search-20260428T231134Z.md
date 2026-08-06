# Ralph Context Snapshot

Task statement: User pasted WiDS Global Datathon 2026 Kaggle competition overview, discussion about being stuck around 0.9545, leaderboard with current local best/public scores, and asks via "Search" / Kaggle context. Treat as: investigate current public guidance and local repo state, then produce actionable improvement path / candidate submission if safe.

Desired outcome: Improve or at least diagnose the current D:\wids solution against Kaggle public leaderboard constraints; produce a validated submission artifact locally, not externally submitted unless explicitly authorized.

Known facts/evidence:
- Competition: survival probabilities prob_12h <= prob_24h <= prob_48h <= prob_72h; metric 0.3 C-index + 0.7 (1 - weighted Brier at 24/48/72).
- Dataset tiny: train 221, test 95.
- Local repo has many submissions; latest exp34 logs indicate 0.97092 public score.
- Public discussion says GBSA/CoxPH from scikit-survival likely better than independent LightGBM horizon classifiers; depth=2 GBSA mentioned.

Constraints:
- Do not submit to Kaggle without explicit user authorization.
- Preserve monotonic probability schema.
- Verify generated CSV schema locally.

Unknowns/open questions:
- Whether public notebooks/discussions expose stronger anchors above 0.98.
- Whether local code can reproduce and blend stronger candidates without overfitting public LB.

Likely codebase touchpoints:
- scripts/exp34_gbsa_ensemble.py
- src/train.py, src/models.py, src/ensemble.py, src/calibration.py, src/evaluation.py, src/features.py
- submissions/*.csv, logs/*.log, .planning/*.md
