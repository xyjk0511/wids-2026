# Project State — WiDS 2026

## Current Phase: 3 (Conformal Calibration) — CLOSED (止损)
## Milestone: 1 (先破 0.9685，再谈 0.975)

## Progress
- [x] Codebase mapping complete (7 documents)
- [x] PROJECT.md created
- [x] REQUIREMENTS.md created
- [x] ROADMAP.md created
- [x] Phase 1: Stacking & Feature Baseline Fix — CLOSED (head overfits, both plans failed)
- [x] Phase 2 Plan 01: IPCW stacking — CLOSED (no signal, OOF hybrid=0.96108 < 0.9697 gate)
- [x] Phase 2 Plan 02: Calibration — CLOSED (LB=0.96338, OOF +0.0059 but LB -0.00445)
- [x] Phase 3: Conformal Calibration — CLOSED (止损，OOF+0.0071但rho=1.0，不提交)
- [ ] Phase 4: 更强锚点获取/复现

## Key Metrics
- PB: 0.96783 (Exp23, 2026-02-20)
- Plan 01 LR-head: LB=0.96274 <- FAILED
- Plan 02 XGB-head 19feat: LB=0.95511 <- FAILED (catastrophic)
- Exp31 IPCW stacking: OOF hybrid=0.96108 < gate 0.9697 — no submission (cross-fit后leak修正)
- Exp32 calibration: LB=0.96338 — OOF过拟合不迁移
- Target: 0.975+
- Submissions used: ~40
- Submissions remaining: >18

## Decisions
- Use .venv_sksurv22 Python env for all pipeline runs (has lifelines + sksurv)
- Stacking heads abandoned: 221 samples too small, XGB/LR heads overfit severely
- RSF+EST baseline + logit post-processing remains best path (PB=0.96783)
- IPCW stacking no signal: cross-fit后OOF从0.96610降到0.96108(leak膨胀+0.005); meta-learning on 221 samples overfits
- Calibration迁移失败: Platt/B/48h OOF +0.0059 但LB -0.00445, N=221校准过拟合, RSF+EST test分布与anchor差异导致迁移失败
- Exp33 conformal止损: OOF+0.0071但rho=1.0排序不变, 与Exp32同模式, 不提交
- 校准天花板: logit线性/Platt参数/非参数分位数三类校准均无法在LB超越锚点后处理
- 新提交门槛: 仅允许改变排序且有稳定性证据的方案

## Stopped At: Phase 3 CLOSED (止损), ready for Phase 4 (更强锚点)
## Last Updated: 2026-02-22
