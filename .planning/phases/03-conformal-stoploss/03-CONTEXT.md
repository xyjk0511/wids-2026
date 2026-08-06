# Phase 3: Conformal Calibration (止损式)

## 锁定决策
1. 仅作用 24h/48h，12h/72h 锁定锚点值
2. Anchor-based增量调整，不独立替换
3. 最多2条实验，提交上限3次
4. 硬门槛：OOF Hybrid >= +0.0015, rho24/48 >= 0.90, CI不下降
5. 若3次提交后LB < 0.9685，立即关闭转Phase 4

## 与Exp22/32的区别
- Exp22: logit空间线性(A,B)，Exp32: Platt/isotonic参数校准
- Conformal: 非参数，基于nonconformity score分位数调整
- 不假设线性关系，可能捕捉非线性偏差
