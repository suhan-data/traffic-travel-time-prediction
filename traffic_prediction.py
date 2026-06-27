#!/usr/bin/env python3
"""
基于时空特征与XGBoost的城市路段旅行时间预测
============================================
端到端方案：DuckDB 加载 → Polars 特征工程 → XGBoost 回归 → 可视化

运行方式：
  TEST_MODE=True  → 仅读取前 N 行快速验证流程
  TEST_MODE=False → 全量数据运行

依赖：duckdb, polars, xgboost, scikit-learn, matplotlib, seaborn, numpy
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

from datetime import datetime

import numpy as np
import polars as pl
import duckdb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import shap

import matplotlib
matplotlib.use("Agg")  # 非交互后端，防止弹窗
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════════

TEST_MODE = False          # 快速验证用；正式运行改为 False
TEST_NROWS = 1000          # 测试模式下读取的行数

BASE_DIR = Path(r"C:\Users\苏菡\Desktop\智慧交通预测")
OUTPUT_DIR = BASE_DIR / "processed"
FIGURE_DIR = BASE_DIR / "figures"

STATIC_INFO_FILE = BASE_DIR / "gy_link_info.txt"
TOPOLOGY_FILE   = BASE_DIR / "gy_link_top.txt"
TRAVEL_TIME_FILES = {
    "part1": BASE_DIR / "gy_link_travel_time_part1.txt",
    "part2": BASE_DIR / "gy_link_travel_time_part2.txt",
    "part3": BASE_DIR / "gy_link_travel_time_part3.txt",
}

# XGBoost 超参
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# ══════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════

def ensure_dirs() -> None:
    """创建输出目录。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def print_progress(msg: str) -> None:
    """带时间戳的进度打印。"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# ══════════════════════════════════════════════════════════════════════
# 1. 加载静态数据
# ══════════════════════════════════════════════════════════════════════

def load_static_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    加载路段静态信息和拓扑关系。

    Returns
    -------
    static_df : 列 [link_id, length, lane_num, link_class]
    topo_df   : 列 [link_id, in_link_ids, out_link_ids]
    """
    static = pl.read_csv(
        STATIC_INFO_FILE,
        separator=";",
        schema_overrides={
            "link_ID": pl.Utf8,       # 19位ID，超i64范围
            "length": pl.Float64,
            "width": pl.Float64,      # 原始列名是width
            "link_class": pl.Int32,
        },
    )
    static = static.rename({
        "link_ID": "link_id",
        "width": "lane_num",
    })

    topo = pl.read_csv(
        TOPOLOGY_FILE,
        separator=";",
        schema_overrides={
            "link_ID": pl.Utf8,
            "in_links": pl.Utf8,
            "out_links": pl.Utf8,
        },
    )
    topo = topo.rename({"link_ID": "link_id"})

    print_progress(f"加载静态表: {static.shape}  拓扑表: {topo.shape}")
    return static, topo


# ══════════════════════════════════════════════════════════════════════
# 2. 加载旅行时间数据 (DuckDB)
# ══════════════════════════════════════════════════════════════════════

def load_travel_time(
    file_path: Path,
    col_link: str = "link_ID",
    test_nrows: int | None = None,
) -> pl.DataFrame:
    """
    用 DuckDB 高效读取旅行时间 CSV，解析 time_interval 字段，返回 Polars DataFrame。

    参数
    ----
    file_path  : CSV 文件路径
    col_link   : CSV 中 link_id 列的列名（part2 为 'linkID'，其他为 'link_ID'）
    test_nrows : 测试模式下限制行数
    """
    con = duckdb.connect(":memory:")
    limit_clause = f"LIMIT {test_nrows}" if test_nrows else ""

    # 显式指定列类型，避免19位 link_id 被当成 DOUBLE 丢失精度
    fp = file_path.as_posix()
    sql = (
        "WITH raw AS ( "
        "  SELECT * FROM read_csv('" + fp + "', sep=';', header=true, "
        "    columns={'" + col_link + "': 'VARCHAR', 'date': 'VARCHAR', "
        "             'time_interval': 'VARCHAR', 'travel_time': 'DOUBLE'}) "
        ") "
        "SELECT "
        "  \"" + col_link + "\" AS link_id, "
        "  CAST(\"date\" AS DATE) AS date, "
        # 用字符串分割代替正则：time_interval = [start, end)
        "  CAST(str_split(str_split(\"time_interval\", '[')[2], ',')[1] AS TIMESTAMP) AS time_start, "
        "  CAST(trim(str_split(str_split(\"time_interval\", ')')[1], ',')[2], ' ') AS TIMESTAMP) AS time_end, "
        "  \"travel_time\" "
        "FROM raw "
        "WHERE \"travel_time\" IS NOT NULL "
        "  AND \"time_interval\" IS NOT NULL "
        + limit_clause
    )
    df = con.execute(sql).pl()
    con.close()

    print_progress(f"加载 {file_path.name}: {df.shape[0]:,} 行")
    return df


# ══════════════════════════════════════════════════════════════════════
# 3. 特征工程 (Polars)
# ══════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pl.DataFrame,
    static_df: pl.DataFrame,
    topo_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    从原始旅行时间数据构建时空特征矩阵。

    步骤:
      1. 时间特征：hour, minute, day_of_week, is_peak
      2. 滞后特征：按 link 分组、时间排序，构造 lag1/lag2/lag4
      3. 空间特征：上游 link 同一时间片的 travel_time
      4. 静态特征：link_class one-hot, length, lane_num
      5. 目标变量：下一时间片的 travel_time (shift(-1))
      6. 剔除含空值的行
    """
    n_before = df.shape[0]

    # --- 3a. 时间特征 ---
    df = df.with_columns([
        pl.col("time_start").dt.hour().alias("hour"),
        pl.col("time_start").dt.minute().alias("minute"),
        pl.col("time_start").dt.weekday().alias("day_of_week"),
    ])
    df = df.with_columns(
        ((pl.col("hour") >= 6) & (pl.col("hour") < 9)).cast(pl.Int32).alias("is_morning_peak"),
        ((pl.col("hour") >= 17) & (pl.col("hour") < 19)).cast(pl.Int32).alias("is_evening_peak"),
    )

    # --- 3b. 滞后特征 ---
    # 按 link_id 分组，按 time_start 排序，计算前 1/2/4 个时间片的旅行时间
    df = df.sort(["link_id", "time_start"])
    df = df.with_columns([
        pl.col("travel_time").shift(1).over("link_id").alias("lag1"),
        pl.col("travel_time").shift(2).over("link_id").alias("lag2"),
        pl.col("travel_time").shift(4).over("link_id").alias("lag4"),
    ])

    # --- 3c. 空间特征 ---
    # 从拓扑表构建 link → 第一个上游 link 的映射
    upstream_map = topo_df.select([
        pl.col("link_id"),
        pl.col("in_links")
        .str.split(",")
        .list.first()
        .alias("upstream_link_id"),
    ]).filter(pl.col("upstream_link_id").is_not_null())

    # 自连接：用上游 link 在同一时间片的 travel_time 作为特征
    upstream_tt = df.select([
        pl.col("link_id").alias("upstream_link_id"),
        pl.col("time_start"),
        pl.col("travel_time").alias("upstream_travel_time"),
    ])

    df = df.join(upstream_map, on="link_id", how="left")
    df = df.join(
        upstream_tt,
        on=["upstream_link_id", "time_start"],
        how="left",
    )
    df = df.with_columns(
        pl.col("upstream_travel_time").fill_null(-1.0)
    )

    # --- 3d. 静态特征 ---
    static_sel = static_df.select([
        pl.col("link_id"),
        pl.col("length"),
        pl.col("lane_num"),
        pl.col("link_class"),
    ])
    df = df.join(static_sel, on="link_id", how="left")

    # 填充缺失的静态特征
    df = df.with_columns([
        pl.col("length").fill_null(pl.col("length").mean()),
        pl.col("lane_num").fill_null(pl.col("lane_num").mean()),
        pl.col("link_class").fill_null(-1),
    ])

    # link_class one-hot 编码
    link_class_dummies = df.select("link_class").to_dummies("link_class")
    # 重命名 one-hot 列
    rename_map = {
        c: f"link_class_{c.replace('link_class_', '')}"
        for c in link_class_dummies.columns
    }
    link_class_dummies = link_class_dummies.rename(rename_map)
    df = pl.concat([df, link_class_dummies], how="horizontal")

    # --- 3e. 目标变量 ---
    df = df.with_columns(
        pl.col("travel_time").shift(-1).over("link_id").alias("target")
    )

    # --- 3f. 清理 ---
    df = df.drop_nulls(subset=["target"])
    lag_cols = ["lag1", "lag2", "lag4"]
    df = df.drop_nulls(subset=lag_cols)

    n_after = df.shape[0]
    print_progress(
        f"特征工程完成: {n_before:,} → {n_after:,} 行 "
        f"(移除 {n_before - n_after:,} 行含空值)"
    )
    return df


# ══════════════════════════════════════════════════════════════════════
# 3.5 保存处理后的数据为 Parquet（按日期分区）
# ══════════════════════════════════════════════════════════════════════

def save_as_parquet(df: pl.DataFrame, output_dir: Path) -> None:
    """
    将特征工程后的 DataFrame 按 date 分区保存为 Parquet 文件。
    同时保存一份全量合并文件供快速加载。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按日期分区写入
    dates = df["date"].unique().sort().to_list()
    for d in dates:
        partition = df.filter(pl.col("date") == d)
        fname = output_dir / f"date={d}" / "data.parquet"
        fname.parent.mkdir(parents=True, exist_ok=True)
        partition.write_parquet(fname)

    # 同时保存全量合并文件
    df.write_parquet(output_dir / "all_features.parquet")

    print_progress(f"Parquet 已保存: {output_dir}  ({len(dates)} 个日期分区)")


# ══════════════════════════════════════════════════════════════════════
# 4. 数据划分
# ══════════════════════════════════════════════════════════════════════

def split_train_val(
    df: pl.DataFrame,
    train_end: str = "2017-05-31",
    val_start: str = "2017-06-01",
    val_end: str = "2017-06-07",
    hour_range: Tuple[int, int] = (8, 9),
):
    """
    按日期切分: train_end 之前 = 训练集, [val_start, val_end] 的 [8:00-9:00) = 验证集。
    默认：2016-03 ~ 2017-05 训练，2017-06 第一周 8-9 点验证。
    若切分后数据不足，自动回退到时序 80/20 切分。
    """
    exclude = {"link_id", "travel_time", "target", "time_start", "time_end",
               "date", "upstream_link_id", "link_class"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # 训练集: 2016-03-01 至 train_end
    train_df = df.filter(
        (pl.col("date") >= pl.lit("2016-03-01").cast(pl.Date))
        & (pl.col("date") <= pl.lit(train_end).cast(pl.Date))
    )
    # 验证集: [val_start, val_end] 内 [8:00-9:00) 时段
    val_df = df.filter(
        (pl.col("date") >= pl.lit(val_start).cast(pl.Date))
        & (pl.col("date") <= pl.lit(val_end).cast(pl.Date))
        & (pl.col("hour") >= hour_range[0])
        & (pl.col("hour") < hour_range[1])
    )

    # 若训练集或验证集太小，回退到时序切分
    if train_df.shape[0] < 50 or val_df.shape[0] < 20:
        print_progress("日期切分数据不足，回退到时序切分（前80%训练，后20%验证）")
        n = df.shape[0]
        split_idx = int(n * 0.8)
        df_sorted = df.sort("time_start")
        train_df = df_sorted[:split_idx]
        val_df = df_sorted[split_idx:]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("target").to_numpy().ravel()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("target").to_numpy().ravel()

    print_progress(
        f"训练集: {X_train.shape[0]:,} 样本  "
        f"验证集: {X_val.shape[0]:,} 样本  "
        f"特征数: {X_train.shape[1]}"
    )
    return X_train, y_train, X_val, y_val, train_df, val_df, feature_cols


# ══════════════════════════════════════════════════════════════════════
# 5. 模型训练与评估
# ══════════════════════════════════════════════════════════════════════

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> xgb.XGBRegressor:
    """训练 XGBoost 回归模型。"""
    print_progress(f"训练 XGBoost: {XGB_PARAMS}")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print_progress("XGBoost 训练完成")
    return model


def train_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> RandomForestRegressor:
    """训练 RandomForest 作为基线模型。"""
    rf_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    }
    print_progress(f"训练 RandomForest: {rf_params}")
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    print_progress("RandomForest 训练完成")
    return model


def evaluate(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    """计算 MAE、RMSE、MAPE、R2。"""
    y_pred = model.predict(X_val)

    mask = np.abs(y_val) > 0.1
    mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100

    metrics = {
        "MAE": mean_absolute_error(y_val, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_val, y_pred)),
        "MAPE (%)": mape,
        "R2": r2_score(y_val, y_pred),
    }
    return metrics, y_pred


def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
):
    """
    训练 XGBoost 和 RandomForest，对比评估指标，
    生成模型对比柱状图 (Fig 5).
    """
    # 训练两个模型
    xgb_model = train_model(X_train, y_train, X_val, y_val)
    rf_model = train_rf_baseline(X_train, y_train)

    # 评估
    xgb_metrics, xgb_pred = evaluate(xgb_model, X_val, y_val)
    rf_metrics, rf_pred = evaluate(rf_model, X_val, y_val)

    # 打印对比
    print("\n" + "=" * 55)
    print("  Model Comparison")
    print("=" * 55)
    print(f"  {'Metric':<12} {'XGBoost':>12} {'RandomForest':>14}")
    print("-" * 55)
    for key in xgb_metrics:
        print(f"  {key:<12} {xgb_metrics[key]:>12.4f} {rf_metrics[key]:>14.4f}")
    print("=" * 55 + "\n")

    # 模型对比柱状图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metric_names = ["MAE", "RMSE", "R2"]
    for i, m in enumerate(metric_names):
        axes[i].bar(["XGBoost", "RF"], [xgb_metrics[m], rf_metrics[m]],
                    color=["#2196F3", "#FF9800"], width=0.4)
        axes[i].set_title(m)
    fig.suptitle("XGBoost vs RandomForest", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_model_comparison.png", dpi=150)
    plt.close(fig)
    print_progress("图5/7 已保存: 模型对比柱状图")

    return xgb_model, xgb_pred, xgb_metrics


def shap_analysis(
    model: xgb.XGBRegressor,
    X_val: np.ndarray,
    feature_cols: List[str],
    val_df: pl.DataFrame,
) -> None:
    """
    SHAP 特征分析：summary plot + dependence plot (Fig 6 & 7).
    """
    # 用验证集的一小部分做 SHAP（避免过慢）
    n_shap = min(200, X_val.shape[0])
    X_sample = X_val[:n_shap]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Fig 6: SHAP summary bar
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_cols,
        plot_type="bar", show=False, max_display=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_progress("图6/7 已保存: SHAP 特征重要性")

    # Fig 7: SHAP summary dot
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_cols,
        show=False, max_display=15,
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_progress("图7/7 已保存: SHAP 特征分布")


# ══════════════════════════════════════════════════════════════════════
# 6. 可视化
# ══════════════════════════════════════════════════════════════════════

def make_visualizations(
    model: xgb.XGBRegressor,
    val_df: pl.DataFrame,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    feature_cols: List[str],
) -> None:
    """
    生成 4 张图：
      1. 单 link 单日真实 vs 预测折线
      2. 预测 vs 真实散点 + R2 标注
      3. 5 个连续 link 的旅行时间热力图
      4. 特征重要性条形图
    """
    val_plot = val_df.with_columns(pl.Series("y_pred", y_pred))

    # --- 图 1：单 link 单日折线图 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    link_id = val_plot["link_id"][0]
    one_day = val_plot.filter(
        (pl.col("link_id") == link_id) & (pl.col("date") == val_plot["date"].min())
    ).sort("time_start")
    if one_day.shape[0] > 0:
        ax.plot(
            range(one_day.shape[0]),
            one_day["target"].to_numpy(),
            "o-", markersize=4, linewidth=1, label="真实值", alpha=0.8,
        )
        ax.plot(
            range(one_day.shape[0]),
            one_day["y_pred"].to_numpy(),
            "s--", markersize=4, linewidth=1, label="预测值", alpha=0.8,
        )
        ax.set_title(f"路段 {link_id} — 单日旅行时间 (真实 vs 预测)", fontsize=13)
        ax.set_xlabel("时间片序号")
        ax.set_ylabel("旅行时间 (秒)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_line_actual_vs_pred.png", dpi=150)
    plt.close(fig)
    print_progress("图1/4 已保存: 单路段折线图")

    # --- 图 2：散点图 ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_val, y_pred, alpha=0.3, s=8, edgecolors="none")
    lims = [min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="y = x")
    r2 = r2_score(y_val, y_pred)
    ax.text(
        0.05, 0.95, f"R2 = {r2:.4f}", transform=ax.transAxes,
        fontsize=13, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_xlabel("真实 travel_time")
    ax.set_ylabel("预测 travel_time")
    ax.set_title("验证集: 预测值 vs 真实值")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_scatter_pred_vs_true.png", dpi=150)
    plt.close(fig)
    print_progress("图2/4 已保存: 预测 vs 真实散点图")

    # --- 图 3：热力图（5 个 link） ---
    fig, ax = plt.subplots(figsize=(14, 6))
    # 选取 5 个不同的 link
    top_links = val_plot["link_id"].unique(maintain_order=True).to_list()[:5]
    heat_data = val_plot.filter(
        pl.col("link_id").is_in(top_links)
    ).sort("time_start")
    if heat_data.shape[0] > 0:
        # 透视：行=link_id, 列=时段
        heat_data = heat_data.with_columns(
            pl.col("time_start").dt.strftime("%H:%M").alias("slot")
        )
        pivot = heat_data.pivot(
            values="y_pred", index="link_id", columns="slot", aggregate_function="mean"
        )
        pivot_np = pivot.drop("link_id").to_numpy()
        labels = [c for c in pivot.columns if c != "link_id"]
        im = sns.heatmap(
            pivot_np, ax=ax, cmap="YlOrRd",
            xticklabels=labels[::5], yticklabels=top_links[:pivot_np.shape[0]],
            cbar_kws={"label": "预测旅行时间 (秒)"},
        )
        ax.set_title("5 个路段 — 8:00-9:00 时段预测旅行时间热力图", fontsize=13)
        ax.set_xlabel("时间片")
        ax.set_ylabel("路段 ID")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_heatmap_5links.png", dpi=150)
    plt.close(fig)
    print_progress("图3/4 已保存: 5路段热力图")

    # --- 图 4：特征重要性 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # top 20
    ax.barh(range(len(indices)), importances[indices], align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
    ax.set_xlabel("重要性")
    ax.set_title("XGBoost 特征重要性 (Top 20)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_feature_importance.png", dpi=150)
    plt.close(fig)
    print_progress("图4/4 已保存: 特征重要性")


# ══════════════════════════════════════════════════════════════════════
# 7. 路径策略模拟
# ══════════════════════════════════════════════════════════════════════

def simulate_path(
    model: xgb.XGBRegressor,
    val_df: pl.DataFrame,
    topo_df: pl.DataFrame,
    feature_cols: List[str],
) -> None:
    """
    基于拓扑自动找出一条由 5 个 link 组成的路径，
    比较"历史均值"与"模型预测"的总旅行时间。
    """
    # 从拓扑中构建邻接表
    edges = {}
    for row in topo_df.iter_rows(named=True):
        lid = row["link_id"]
        out_str = row["out_links"]
        if out_str:
            edges[lid] = [x.strip() for x in out_str.split(",") if x.strip()]
        else:
            edges[lid] = []

    # BFS 找一条长度为 5 的路径
    def find_path(start: str, length: int = 5):
        path = [start]
        for _ in range(length - 1):
            cur = path[-1]
            if cur not in edges or not edges[cur]:
                return None
            path.append(edges[cur][0])
        return path

    path = None
    for lid in edges:
        path = find_path(lid, 5)
        if path:
            break

    if not path:
        print_progress("策略模拟: 无法找到5段连续路径，跳过")
        return

    print_progress(f"选取路径: {' → '.join(path)}")

    # 选 3 天做模拟
    dates = val_df["date"].unique().sort().to_list()[:3]

    print("\n" + "=" * 60)
    print("  路径策略模拟: 历史均值 vs 模型预测")
    print("=" * 60)

    for d in dates:
        day_data = val_df.filter(pl.col("date") == d)
        total_mean = 0.0
        total_pred = 0.0
        for lid in path:
            link_data = day_data.filter(pl.col("link_id") == lid)
            if link_data.shape[0] == 0:
                continue
            total_mean += link_data["travel_time"].mean()  # 使用当日该路段均值
            X_path = link_data.select(feature_cols).to_numpy()
            if X_path.shape[0] > 0:
                total_pred += model.predict(X_path).mean()

        if total_mean > 0:
            saving_pct = (total_mean - total_pred) / total_mean * 100
            print(
                f"  日期 {d}: "
                f"历史均值 {total_mean:.1f}s → 模型预测 {total_pred:.1f}s "
                f"({saving_pct:+.1f}%)"
            )

    print("=" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════
# 8. 主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  基于时空特征与XGBoost的城市路段旅行时间预测")
    print("=" * 60)
    if TEST_MODE:
        print(f"  [TEST MODE] reading first {TEST_NROWS:,} rows per file")
    print()

    ensure_dirs()

    # ---------- 1. 加载静态数据 ----------
    print_progress("Step 1/7: 加载静态数据")
    static_df, topo_df = load_static_data()

    # ---------- 2. 加载并合并三份旅行时间数据 ----------
    print_progress("Step 2/7: 加载旅行时间数据 (DuckDB, 3 files)")
    nrows = TEST_NROWS if TEST_MODE else None

    tt_p1 = load_travel_time(
        TRAVEL_TIME_FILES["part1"], col_link="link_ID", test_nrows=nrows
    )
    tt_p2 = load_travel_time(
        TRAVEL_TIME_FILES["part2"], col_link="linkID", test_nrows=nrows
    )
    tt_p3 = load_travel_time(
        TRAVEL_TIME_FILES["part3"], col_link="link_ID", test_nrows=nrows
    )

    tt_all = pl.concat([tt_p1, tt_p2, tt_p3], how="vertical")
    print_progress(
        f"Merged: {tt_all.shape[0]:,} rows "
        f"({tt_p1.shape[0]:,} + {tt_p2.shape[0]:,} + {tt_p3.shape[0]:,})"
    )
    print_progress(
        f"Date range: {tt_all['date'].min()} ~ {tt_all['date'].max()}, "
        f"links: {tt_all['link_id'].n_unique()}"
    )

    # ---------- 3. 特征工程 ----------
    print_progress("Step 3/7: 特征工程 (Polars)")
    df = engineer_features(tt_all, static_df, topo_df)

    # ---------- 3.5 保存 Parquet ----------
    print_progress("Step 3.5/7: 保存处理后的数据为 Parquet")
    save_as_parquet(df, OUTPUT_DIR)

    # ---------- 4. 划分训练集/验证集 ----------
    print_progress("Step 4/7: 数据划分")
    X_train, y_train, X_val, y_val, train_df, val_df, feature_cols = split_train_val(df)

    # ---------- 5. 模型对比与评估 ----------
    print_progress("Step 5/7: XGBoost vs RandomForest 模型对比")
    model, y_pred, metrics = compare_models(
        X_train, y_train, X_val, y_val, feature_cols
    )

    # ---------- 6. SHAP 特征分析 ----------
    print_progress("Step 6/7: SHAP 特征分析")
    shap_analysis(model, X_val, feature_cols, val_df)

    # ---------- 7. 可视化 ----------
    print_progress("Step 7/7: 生成可视化")
    make_visualizations(model, val_df, y_val, y_pred, feature_cols)

    # ---------- 附加: 路径策略模拟 ----------
    print_progress("附加: 路径策略模拟")
    simulate_path(model, val_df, topo_df, feature_cols)

    print("\n" + "=" * 60)
    print(f"  全部完成！")
    print(f"  Parquet 目录: {OUTPUT_DIR}")
    print(f"  图表目录:     {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()