# 基于时空特征与XGBoost的城市路段旅行时间预测

阿里天池经典赛题复现 | 个人项目

## 项目概述

基于贵阳市132条路段2016-2017年的真实交通数据（约2500万条记录，2GB+），构建多维度时空特征体系，使用XGBoost与RandomForest进行路段旅行时间预测。

## 数据来源

阿里天池城市路段旅行时间预测赛题，包含132条路段2016年3月-2017年3月约2500万条记录。由于数据量较大（2GB+），本仓库不包含原始数据。

## 技术栈

- **数据处理**: DuckDB（加载）、Polars（特征工程）
- **模型**: XGBoost、RandomForest
- **可解释性**: SHAP
- **可视化**: matplotlib、seaborn
- **开发环境**: VS Code + DeepSeek V4 Pro

## 特征体系

| 类别 | 特征 |
|------|------|
| 时间特征 | 小时、星期、是否高峰时段 |
| 滞后特征 | lag1、lag2、lag4 |
| 空间特征 | 上游路段旅行时间 |
| 静态特征 | 路段长度、车道数等 |

## 模型效果

| 模型 | MAE（秒） | R² |
|------|-----------|-----|
| XGBoost | 5.24 | 0.79 |
| RandomForest | — | — |

## 可视化输出

| 图表 | 说明 |
|------|------|
| 01_line_actual_vs_pred | 不同时段预测值 vs 真实值对比 |
| 02_scatter_pred_vs_true | 预测值 vs 真实值散点图 |
| 03_heatmap_5links | 5条路段预测误差热力图 |
| 04_feature_importance | 特征重要性排序 |
| 05_model_comparison | XGBoost vs RandomForest 效果对比 |
| 06_shap_importance | SHAP 特征重要性 |
| 07_shap_summary | SHAP 摘要图 |

## 项目结构

```
├── figures/                # 输出图表
├── traffic_prediction.py   # 主程序
├── traffic_prediction.ipynb # Jupyter Notebook
├── test_pipeline.py        # 测试脚本
├── requirements.txt        # 依赖包
└── README.md
```

## 运行方式

```bash
pip install -r requirements.txt
jupyter notebook traffic_prediction.ipynb
```

## 致谢

本项目在开发过程中借助了 DeepSeek V4 Pro 进行代码辅助与调试。
