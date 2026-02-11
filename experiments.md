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
