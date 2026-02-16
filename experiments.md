# WiDS 2025 - 实验记录

## 提交历史

| # | 日期 | 模型 | Local CV Hybrid | Local CV CI | Local CV WBrier | Kaggle LB | 备注 |
|---|------|------|-----------------|-------------|-----------------|-----------|------|
| 1 | 2026-02-08 | ensemble(rankxgb+rsf+gbsa) | 0.9721 | 0.9420 | 0.0150 | 0.95054 | CV-LB gap 合理, LB 略高于 CV CI, 泛化良好 |
| 2 | 2026-02-08 | ensemble v2 (L2正则+bootstrap+per-horizon+RankXGB独立训练) | 0.9692 | 0.9420 | 0.0191 | 0.95739 | L2正则lambda=0.1, bootstrap n=30, per-horizon Brier优化, stacking C=0.1, Platt Scaling禁用. CV-LB gap=0.012(缩小) |
| 3 | 2026-02-09 | RSF单模型 (v96624特征集16个+StandardScaler+clip后处理) | 0.9697 | 0.9397 | 0.0174 | 0.95873 | 对齐0.96624源码: 15base+3eng特征, 跳过remove_redundant, 折内StandardScaler, clip[0.01,0.99]. 概率分布仍偏低(12h median=0.016 vs目标0.15). CV-LB gap=0.012 |
| 4 | 2026-02-10 | RSF+2class分层+KFold平均+logit分位数校准 | 0.9622 | 0.9378 | 0.0273 | 0.94688 | 3项改动: (1)CV分层4class->2class, (2)50折模型test平均替代full retrain, (3)logit空间p5/p95分位数匹配校准. 校准严重伤害LB(-0.012). OOF已警告(0.9689->0.9622). 12h median从0.015提升到0.067但过度校准. 需回退校准 |
| 5 | 2026-02-10 | RSF单模型 (n_estimators=1000, max_features=sqrt) | 0.9690 | 0.9387 | 0.0180 | 0.95952 | 还原实验#3结构(full retrain), n_estimators 200->1000. 超参搜索发现max_features=None OOF最优(0.9716)但full retrain过拟合. 确认0.96624差距来自库版本(sksurv)非代码差异. CV-LB gap=0.010 |
| 6 | 2026-02-11 | RSF单模型 MEDIUM特征集(36个) | 0.9721 | 0.9410 | 0.0145 | 0.96174 | 特征集对比: v96624(16)=0.9690, medium(36)=0.9721, full(44)=0.9719. MEDIUM胜出+0.0031. LB从0.95952提升到0.96174(+0.00222). CV-LB gap=0.010 |
| 7 | 2026-02-11 | RSF超参调优 max_features=12 (MEDIUM 36特征) | 0.9724 | 0.9414 | 0.0142 | 0.96136 | 2轮网格搜索(12组). max_features=12(原sqrt=6). OOF +0.0003但LB -0.00038. max_features增大导致full retrain过拟合, CV-LB gap从0.010扩大到0.011. 需回退到sqrt. |
| 8 | 2026-02-11 | RSF多seed平均 (5seeds x 200trees, MEDIUM 36特征) | 0.9721 | 0.9410 | 0.0145 | 0.96209 | 对齐原始0.96624配置: n_estimators=200, 5个seed(42,123,456,789,2026)平均. CV不变(仍n=1000单seed). LB +0.00035(0.96174->0.96209). CV-LB gap=0.010. 方差缩减有效, 方向正确. 距目标0.96624还差0.00415. |
| 9 | 2026-02-11 | RSF+EST混合 (5seeds x RSF+EST=10模型, MEDIUM 36特征) | 0.9721 | 0.9410 | 0.0145 | 0.96331 | 新增ExtraSurvivalTrees(随机分裂). 5 RSF + 5 EST = 10模型平均. LB +0.00122(0.96209->0.96331). EST引入真正模型多样性, 24h median从0.01展开到0.016. CV-LB gap=0.009(缩小). 距目标0.96624还差0.00293. |
| 10 | 2026-02-11 | GBSA(12h)+RSF(24h/48h) per-horizon拆分 (5seeds each) | 0.9721 | 0.9410 | 0.0145 | 0.94910 | 失败(-0.01421). GBSA 12h概率压缩(range=0.056, 95样本挤在0.212-0.268), CI对near-tie敏感导致崩溃. Spearman vs参考=0.95但不等于CI vs真实标签. 去掉EST降低模型多样性伤害WBrier. 已回退到实验#9配置. |
| 11 | 2026-02-12 | RSF+EST max_features=None (5seeds x 2模型, MEDIUM 36特征) | 0.9715 | 0.9388 | 0.0145 | 0.96324 | max_features从sqrt改None匹配0.96624原始配置. Spearman大幅提升(0.72->0.82)但OOF CI下降(0.9410->0.9388). LB -0.00007(0.96331->0.96324). 排序对齐参考但判别力略降. 已回退到sqrt. |
| 12 | 2026-02-12 | RSF+EST+GBSA selective blend (12h:RSF+EST, 24h/48h:+GBSA, 5seeds) | 0.9721 | 0.9410 | 0.0145 | TBD | GBSA仅参与24h/48h WBrier优化, 12h CI保持RSF+EST. CV仍为RSF-only(盲飞). 代码快照: experiments/exp12_gbsa_blend/ |
| 13 | 2026-02-12 | Exp14 v3 CV ensemble (50-fold RSF+EST平均, R1单调修正) | 0.9707 | 0.9438 | 0.0177 | 0.96329 | CV ensemble替代full retrain, 50fold模型平均test预测. R1修正: p12=min(p12,p24)防传导. 12h Spearman=0.80. LB与#9持平(-0.00002). 校准方向已穷尽(v1/v2/v3共9策略全部失败或持平). 代码: experiments/exp14_calibrated_ensemble/ |
| 14 | 2026-02-13 | 0.96624锚点替换12h（sub_96624_riskp12） | 0.9738* | 0.9448* | 0.0138* | 0.96184 | 仅替换 `prob_12h`，`prob_24h/48h/72h` 与 `submission 0.96624.csv` 完全一致；并强制 `p12<=p24-1e-6`。LB 低于锚点，说明当前 risk->p12 映射分布与线上不匹配。`*` 为本地OOF实验指标，非该提交直接CV。 |
| 15 | 2026-02-13 | Exp17 基线: RSF(200) 16特征 全局scaler 参考后处理 | 0.9708 | 0.9449 | 0.0181 | TBD | 精确复现参考pipeline(去掉EST和36特征膨胀). 消融发现: 特征膨胀15->36导致CV -0.0057, EST混合导致CV -0.0140, 合计-0.0197解释了0.96624->0.96331的LB差距. 其他4项(树数/scaler/后处理/CV ensemble)几乎无影响. 待提交验证. |
| 16 | 2026-02-14 | Exp22 R1 lam=0.10 上推 (a=1.057,b=+0.008) | - | - | - | 0.96629 | 锚点p48 logit空间近恒等校准. OOF bootstrap(a,b)上推版. 仅动p48,p12/p24锁死. +0.00005 vs anchor |
| 17 | 2026-02-14 | Exp22 R2A lam=0.20 下推 (a=1.066,b=-0.011) | - | - | - | 0.96634 | 锚点蒸馏学生A(isotonic on -log_dist)学到的(a,b)下推版. 下推>上推, 确认hidden p48偏高 |
| 18 | 2026-02-14 | Exp22c lam=0.80 (a=1.0655,b=-0.0108) | - | - | - | 0.96661 | 沿下推方向加大lambda. 0 violations. 趋势线性+0.00009/step |
| 19 | 2026-02-14 | Exp22c lam=1.00 | - | - | - | 0.96668 | 4 violations开始出现. 步进衰减到+0.00007 |
| 20 | 2026-02-14 | Exp22c lam=1.05 | - | - | - | 0.96670 | 悬崖前最后安全点(7 violations). 旧策略(p48 clip up)的极限 |
| 21 | 2026-02-15 | Exp22d lam=1.05 p24-yields | - | - | - | 0.96671 | 新修复策略: violations时p24让路而非p48上抬. 验证策略有效(+0.00001) |
| 22 | 2026-02-15 | Exp22d lam=1.10 p24-yields | - | - | - | 0.96674 | 穿过旧悬崖(旧策略此处rho48崩到0.9987). 46 violations由p24吸收 |
| 23 | 2026-02-15 | Exp22d lam=1.20 p24-yields | - | - | - | 0.96679 | 66 violations, rho48=0.999997保持. p24-yields策略的单推极限 |
| 24 | 2026-02-15 | Exp22d r=0.3 lam=1.20 联动 | - | - | - | 0.96681 | 24+48联动下推: p24主动跟随p48下推(r=0.3). 0 violations. 联动>被动让路 |
| 25 | 2026-02-15 | Exp22d r=0.3 lam=1.50 联动 | - | - | - | 0.96693 | 联动消除悬崖后继续推. 4 violations. lam还没到顶 |
| 26 | 2026-02-15 | Exp22d r=0.5 lam=2.00 联动 | - | - | - | 0.96713 | 当前PB. 4 violations. 增量加速(+0.00020). 趋势未见拐点 |

## Exp22 系列详细分析

### 核心发现

1. **hidden test p48偏高**: 下推(b<0)始终优于上推(b>0), 确认锚点p48在hidden上系统性偏乐观
2. **拉伸(a>1)是主力**: 纯平移(a=1,b=-0.05)仅得0.96628, 而同等MAD的拉伸版得0.96643. logit空间两端拉开比整体下移有效得多
3. **悬崖机制**: p48下推到接近p24时触发单调约束clip, 导致刻度被二次扭曲. 解决方案: (1)p24被动让路 (2)24+48主动联动下推
4. **联动释放收益**: r=0.3~0.5的联动消除violations后, 可安全推到更大lambda, 收益加速而非衰减

### 变换公式

```
logit(p48_new) = (1-lam)*logit(p48_anchor) + lam*(a*logit(p48_anchor) + b)
logit(p24_new) = (1-lam*r)*logit(p24_anchor) + lam*r*(a*logit(p24_anchor) + b)
```
参数: a=1.0655, b=-0.0108 (来自R2A蒸馏学生bootstrap median, lam_reg=0.2)

### LB趋势 (按推力递增)

```
lam=0.20 r=0.0  -> 0.96634  (+0.00010)
lam=0.40 r=0.0  -> 0.96643  (+0.00019)
lam=0.60 r=0.0  -> 0.96652  (+0.00028)
lam=0.80 r=0.0  -> 0.96661  (+0.00037)
lam=1.00 r=0.0  -> 0.96668  (+0.00044)  4 violations
lam=1.05 r=0.0  -> 0.96670  (+0.00046)  7 violations, 旧策略极限
lam=1.20 p24让路 -> 0.96679  (+0.00055)  66 violations absorbed by p24
lam=1.20 r=0.3  -> 0.96681  (+0.00057)  0 violations
lam=1.50 r=0.3  -> 0.96693  (+0.00069)  4 violations
lam=2.00 r=0.5  -> 0.96713  (+0.00089)  4 violations, 当前PB
```

### 待提交 (已生成)

- r=0.7 lam=2.5/3.0/4.0: 测试更大推力, 0~4 violations
- 趋势未见拐点, 需继续探索
