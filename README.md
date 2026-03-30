# WiDS 2026 Clinical Prediction Pipeline / WiDS 2026 临床预测项目

This repository contains my competition workflow for **WiDS Datathon 2026**, focused on clinical risk prediction with gradient boosting ensembles, calibration, and leaderboard-oriented validation.

本仓库记录了我在 **WiDS Datathon 2026** 中的主要建模与实验流程，重点是临床预测任务中的集成学习、校准、特征工程和离线验证策略。

---

## Overview / 项目概述

**Goal / 目标**
- Build a high-performing clinical prediction pipeline for the WiDS 2026 challenge.
- 构建一个高性能的临床预测系统，并尽可能缩小本地验证与 Kaggle leaderboard 之间的差距。

**What makes this project interesting / 为什么这个项目值得看**
- Not just a single model: this repo captures iterative experimentation across multiple ensemble strategies.
- 不只是一个模型，而是一整套从 baseline 到 blend / calibration / risk modeling 的实验演进。
- It reflects how I think about competition ML in practice: validation design, ablation, robustness, and leaderboard trade-offs.
- 它更像一个完整的竞赛研究工作区，而不是单次提交脚本。

---

## Results / 核心结果

- **Public leaderboard score:** **0.97089**
- **Primary modeling direction:** gradient boosting ensembles + calibration
- **Experiment themes:** GBSA blend, subgroup odds scaling, rank calibration, piecewise gating, PLE-style exact blending

### Highlights / 亮点
- Built and compared multiple post-ensemble calibration strategies instead of relying on a single leaderboard submission.
- 将集成之后的校准单独作为研究方向，而不是只做简单模型融合。
- Maintained a broad experiment log across multiple versions, including ablation, risk scaling, and blending studies.
- 保留了较完整的实验记录，能反映真实竞赛中的迭代过程。

---

## Repository Structure / 仓库结构

```text
src/                    # Core modeling code
scripts/                # Experiment scripts and submission generation
experiments/            # Structured experiment subfolders
notebooks/              # Analysis notebooks
competition_analysis.md # Competition framing and observations
experiments.md          # Experiment registry / notes
```

### Notable areas / 重点目录
- `scripts/`: experimental scripts for blending, calibration, and submission generation
- `experiments/`: grouped experiment folders with results and notes
- `notebooks/`: exploratory and validation notebooks

---

## Modeling Themes / 主要技术路线

### 1. Ensemble learning / 集成学习
This project explores ensemble combinations built around strong gradient boosting baselines rather than treating a single model as final.

本项目核心不是“找一个最好模型”，而是围绕强基线构建更稳健的集成组合。

### 2. Calibration / 校准
Several experiments focus on calibration-oriented techniques, including rank-based and quantile-based variants, to better align offline validation with leaderboard behavior.

多个实验围绕校准展开，目标是减少离线验证与 leaderboard 表现之间的偏移。

### 3. Subgroup and risk scaling / 子群体与风险尺度
The project also explores subgroup-aware scaling and risk-sensitive adjustments, reflecting a more competition-realistic approach than pure average blending.

项目中还包含针对子群体与风险尺度的实验，这比单纯平均融合更贴近真实竞赛优化思路。

---

## Practical Engineering Lessons / 工程与竞赛经验

This repository reflects several practical lessons from competition work:

- Validation design matters as much as model architecture.
- 验证设计往往和模型本身一样重要。
- Small leaderboard gains often come from calibration and postprocessing, not just bigger models.
- 很多分数提升来自校准与后处理，而不只是更复杂的模型。
- A strong experiment log is essential for avoiding repeated mistakes.
- 清晰的实验记录能显著减少重复踩坑。

---

## Reproducibility / 复现说明

This repository is best viewed as a **competition research workspace** rather than a fully packaged benchmark repo.

这个仓库更适合被理解为一个**竞赛研究工作区**，而不是完全产品化、一步运行的标准 benchmark 仓库。

If I productionize it further, the next steps would be:
- add a cleaner entrypoint for training / inference
- normalize configs and paths
- add lightweight documentation for the main experiment families
- separate reusable utilities from one-off competition scripts

如果后续进一步整理，我会优先做：
- 统一训练 / 推理入口
- 整理配置与路径
- 为主要实验方向补充更清晰的文档
- 把可复用模块和一次性竞赛脚本彻底拆开

---

## Why this repository matters / 这个仓库体现了什么

This project is one of the best examples of how I work in machine learning competitions:
- iterative modeling
- systematic ablation
- leaderboard-aware validation
- practical engineering under uncertainty

这个仓库最能体现我在 ML 竞赛里的工作方式：
- 持续迭代
- 系统化实验
- 面向 leaderboard 的验证设计
- 在不确定条件下推进工程实现
