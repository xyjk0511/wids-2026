# WiDS Global Datathon 2026 - 竞赛分析

## 1. 竞赛概述

- **名称**: WiDS Global Datathon 2026: Predicting Time-to-Threat for Evacuation Zones Using Survival Analysis
- **主办方**: WiDS Worldwide + WatchDuty
- **总奖金**: $25,000 (前5名各$3,000, 学生/高中/新手特别奖各$2,500)
- **许可证**: Open Source (获奖方案需开源)

## 2. 问题定义

- **任务类型**: 生存分析 (Survival Analysis)
- **目标**: 预测野火从 t0+5h 时刻起, 在不同时间窗口内到达疏散区域(5km范围)的概率
- **训练目标列**:
  - `time_to_hit_hours`: 从 t0+5h 到火灾到达疏散区的时间(小时), 范围 [0, 72]。对于删失事件(event=0), 该值为72h窗口内的最后观测时间(<=72), 而非固定72
  - `event`: 事件指示器, 1=72h内到达, 0=删失(censored, 72h内未到达)
- **提交格式**: 4个概率列
  - `prob_12h`: 12小时内到达的概率
  - `prob_24h`: 24小时内到达的概率
  - `prob_48h`: 48小时内到达的概率
  - `prob_72h`: 72小时内到达的概率

## 3. 数据规模

- **总火灾事件数**: 316个(均有早期perimeter观测和确认结果)

| 数据集                | 样本数 | 特征数                     |
| --------------------- | ------ | -------------------------- |
| train.csv             | 221    | 34个特征 + 2个目标 + 1个ID |
| test.csv              | 95     | 34个特征 + 1个ID           |
| sample_submission.csv | 95     | 4个概率列 + 1个ID          |

- **训练集标签分布**: 69个hits(event=1) + 152个censored(event=0)
- **注意**: 数据集非常小, 需要特别注意过拟合问题

**为什么只有316个火灾**: 大多数野火不同时满足两个条件: (1) 前5小时内有perimeter更新用于特征提取; (2) 有足够的后续perimeter观测来确认火灾是否以及何时到达阈值。这是数据采集的结构性约束。

## 4. 特征分类 (共34个)

所有特征均基于火灾检测后前5小时(t0~t0+5h)的数据计算。

### temporal_coverage (3个) -- 时间覆盖

> **主办方澄清**: "perimeter" 指火灾边界的时间戳快照(fire perimeter snapshot), 不是疏散区或管辖区域。t0 定义为该火灾首次 perimeter 观测的时间戳。

- `num_perimeters_0_5h`: 前5小时内的 perimeter 快照数量(即火灾边界被观测/绘制了几次)
- `dt_first_last_0_5h`: 首末 perimeter 快照的时间跨度(小时), 仅1个快照时为0
- `low_temporal_resolution_0_5h`: 标志位, dt<0.5h或仅1个perimeter时为1

### growth (10个) -- 火灾增长

- `area_first_ha`: 初始火灾面积(公顷)
- `area_growth_abs_0_5h`: 绝对面积增长(公顷)
- `area_growth_rel_0_5h`: 相对面积增长(比例)
- `area_growth_rate_ha_per_h`: 面积增长率(公顷/小时)
- `log1p_area_first`: log(1+初始面积)
- `log1p_growth`: log(1+绝对增长)
- `log_area_ratio_0_5h`: 终末/初始面积的对数比
- `relative_growth_0_5h`: 相对增长(同area_growth_rel_0_5h)
- `radial_growth_m`: 有效半径变化(米)
- `radial_growth_rate_m_per_h`: 半径增长率(米/小时)

### centroid_kinematics (5个) -- 质心运动学

- `centroid_displacement_m`: 火灾质心总位移(米)
- `centroid_speed_m_per_h`: 质心移动速度(米/小时)
- `spread_bearing_deg`: 火灾蔓延方向(度)
- `spread_bearing_sin`: 蔓延方向正弦(圆形编码)
- `spread_bearing_cos`: 蔓延方向余弦(圆形编码)

### distance (9个) -- 与疏散区距离

- `dist_min_ci_0_5h`: 到最近疏散区质心的最小距离(米)
- `dist_std_ci_0_5h`: 距离标准差
- `dist_change_ci_0_5h`: 距离变化(d5-d0, 负值=接近)
- `dist_slope_ci_0_5h`: 距离vs时间的线性斜率(米/小时)
- `closing_speed_m_per_h`: 逼近速度(米/小时, 正值=接近)
- `closing_speed_abs_m_per_h`: 逼近速度绝对值
- `projected_advance_m`: 向疏散区的投影推进(d0-d5)
- `dist_accel_m_per_h2`: 距离变化加速度(米/小时^2)
- `dist_fit_r2_0_5h`: 距离vs时间线性拟合的R^2

### directionality (4个) -- 方向性

- `alignment_cos`: 火灾运动方向与疏散区方向的夹角余弦
- `alignment_abs`: 绝对对齐度(0-1, 越高越对齐)
- `cross_track_component`: 横向漂移分量
- `along_track_speed`: 沿疏散区方向的速度分量

### temporal_metadata (3个) -- 时间元数据

- `event_start_hour`: 火灾开始的小时(0-23)
- `event_start_dayofweek`: 星期几(0=周一, 6=周日)
- `event_start_month`: 月份(1-12)

## 5. 数据特点观察

- **5km 距离阈值完美分离** (主办方确认非 data leakage):
  - 训练集: 所有 69 个 `dist_min_ci_0_5h < 5km` 的火灾 event=1 (100% hit), 所有 152 个 >= 5km 的火灾 event=0 (0% hit), 无一例外
  - 测试集分布类似: ~28 个 < 5km (29.5%) vs 训练集 69 个 (31.2%)
  - 原因: "hit" 的物理定义本身基于距离(到疏散区 5km 范围), 距离特征在前5小时计算, 事件标签在后续预测窗口定义, 时间上严格分离
  - 建模含义: 二分类几乎 trivial, 真正的竞赛难点在于多 horizon 概率校准和组内紧急程度排序
- **大量零值**: 很多样本只有1个perimeter观测(low_temporal_resolution=1), 导致增长/运动相关特征全为0 (closing_speed: 91.86% 零值, dist_change: 91.86% 零值)
- **删失数据**: event=0 表示72h内火灾未到达疏散区, 这是典型的右删失(right-censored)数据。删失样本的time_to_hit_hours是最后观测时间, 不一定等于72
- **Informative Censoring(信息性删失)**:
  - 删失不是纯随机的, 而是操作驱动的(覆盖范围、观测可用性等)
  - 删失行的 time_to_hit_hours 仍有信息量: 更长的观测时间提供更强的"未命中"证据
  - 这意味着删失机制与事件风险相关, 建模时需注意
- **72h horizon 边界情况**:
  - 大多数 censored 行的 last_observed_time < 72h, 在 72h horizon 评估时被排除
  - 剩余 eligible 行几乎全是 positive(event=1)
  - 因此预测 prob_72h=1.0 是合理策略(eligible 集全为正例时 Brier loss 最小化就是预测 1.0)
- **关键特征**:
  - `dist_min_ci_0_5h`: 到最近疏散区的最小距离(米), 值域跨度很大(几百到几十万)
  - `area_first_ha`: 初始火灾面积, 变化范围大
  - `closing_speed_m_per_h`: 火灾逼近疏散区的速度

## 6. 评估指标 (竞赛实际公式)

### 6.1 综合分数 (Hybrid Score)

```
HybridScore = 0.3 * CI + 0.7 * (1 - WBrier)
```

- 越高越好, 满分 1.0
- CI 占 30%, Brier 占 70%
- 代码实现: `hybrid_score()` in `src/evaluation.py`

### 6.2 加权 Brier Score (WBrier)

```
WBrier = 0.3 * Brier@24h + 0.4 * Brier@48h + 0.3 * Brier@72h
```

- 不含 12h (12h 仅影响 C-index)
- 48h 权重最高 (0.4)
- 代码实现: `weighted_brier_score()` in `src/evaluation.py`

每个 horizon H 的 Brier Score 按以下规则确定 eligible 样本:

| 条件                               | 标签     | 是否纳入计算 |
| ---------------------------------- | -------- | ------------ |
| event=1 且 time_to_hit_hours <= H  | label=1  | 纳入         |
| event=1 且 time_to_hit_hours > H   | label=0  | 纳入         |
| event=0 且 last_observed_time >= H | label=0  | 纳入         |
| event=0 且 last_observed_time < H  | 信息不足 | 排除         |

Brier Score = mean((prob_H - label)^2)

### 6.3 C-index

- 仅使用 prob_12h 作为风险分数, 计算一次
- 使用 (time_to_hit_hours, event) 构建可比较对(comparable pairs)
- 已通过可比较对机制自然处理删失
- 代码实现: `c_index()` in `src/evaluation.py`

### 6.4 竞赛后处理策略

提交前对预测值做后处理, 可显著提升分数:

| Horizon | 处理方式                   | 原因                                   |
| ------- | -------------------------- | -------------------------------------- |
| 12h     | 独立 clip [0.01, 0.99]     | 仅影响 CI, 不参与 Brier                |
| 24h     | 单调链 + clip [0.01, 0.99] | 确保 24h <= 48h                        |
| 48h     | 单调链 + clip [0.01, 0.99] | 确保 48h <= 72h                        |
| 72h     | 硬编码 1.0                 | eligible 样本几乎全为正例, Brier@72h=0 |

- 72h 硬编码 1.0 直接拿满 WBrier 中 30% 权重的分数
- 12h 与 24/48/72h 的单调链独立, 因为 12h 不参与 Brier 计算
- 代码实现: `submission_postprocess()` in `src/monotonic.py`

### 6.5 建模重点

- 优化目标必须对齐竞赛公式 (HybridScore), 不能用自定义公式
- CI 仅依赖 prob_12h: 重点优化 12h 的风险排序能力
- Brier 不含 12h: 24h/48h 的概率校准更重要
- 概率校准优于 raw accuracy
- ensemble 权重优化直接最大化 HybridScore

## 7. 建模思路

**适合的方法**:

- Cox比例风险模型 (Cox PH)
- 加速失效时间模型 (AFT)
- 随机生存森林 (Random Survival Forest)
- 基于梯度提升的生存模型 (如 XGBoost + 自定义生存损失)
- 深度学习生存模型 (如 DeepSurv, 但数据量太小可能不适合)

**关键注意事项**:

- 数据量极小(221条), 交叉验证策略很重要
- 需要正确处理删失数据
- 输出是累积分布函数 F(t) 在 t=12,24,48,72 处的值
- 概率应满足单调性: prob_12h <= prob_24h <= prob_48h <= prob_72h
- prob_72h=1.0 是合理策略: 72h horizon的eligible集几乎全为正例, 预测1.0可最小化Brier loss
- C-index仅用prob_12h计算: 需要重点优化12h预测的风险排序能力, 这直接影响C-index得分
- 数据语义 vs 评估规则是不同层面: time_to_hit_hours和event描述数据含义, 评估时按horizon规则转换为标签

## 8. 竞赛规则要点

- 每天最多提交3次
- 最多选择2个最终提交
- 团队最多4人, 至少一半为女性
- 允许使用外部数据(需公开可用)
- 允许使用AutoML工具
- 获奖方案需开源
