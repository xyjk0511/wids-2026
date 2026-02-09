# WiDS 2026 Kaggle Competition

## 实验管理规则

1. 每次代码改动后运行 `python -m src.train` 验证 CV 指标
2. 每次 Kaggle 提交后，用户提供 LB 分数时，必须在 experiments.md 详细记录：
   - 提交编号、日期
   - 模型配置和关键参数
   - Local CV Hybrid / CI / WBrier
   - Kaggle LB 分数
   - 改动摘要和分析
3. 每次重大改动创建 git commit
4. 保持工作目录整洁，不创建临时文件
