"""
端到端快速测试 — 仅 1000 行，验证所有步骤可跑通
"""
from pathlib import Path
import numpy as np
import polars as pl
import duckdb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path(r"C:\Users\苏菡\Desktop\智慧交通预测")
NROWS = 1000

print("=" * 50)
print("Step 1: 加载静态数据")
static = pl.read_csv(
    BASE / "gy_link_info.txt", separator=";",
    schema_overrides={"link_ID": pl.Utf8, "length": pl.Float64, "width": pl.Float64, "link_class": pl.Int32},
)
static = static.rename({"link_ID": "link_id", "width": "lane_num"})

topo = pl.read_csv(
    BASE / "gy_link_top.txt", separator=";",
    schema_overrides={"link_ID": pl.Utf8, "in_links": pl.Utf8, "out_links": pl.Utf8},
)
topo = topo.rename({"link_ID": "link_id"})
print(f"  static: {static.shape}  topo: {topo.shape}")

print("\nStep 2: 加载旅行时间 (DuckDB)")
con = duckdb.connect(":memory:")
# 注意: f-string 中需要四个反斜杠 \\\\ 才能在 DuckDB regex 中得到转义后的 \[
file_path = (BASE / "gy_link_travel_time_part1.txt").as_posix()
sql = (
    "WITH raw AS ( "
    "  SELECT * FROM read_csv('" + file_path + "', sep=';', header=true, "
    "    columns={'link_ID': 'VARCHAR', 'date': 'VARCHAR', "
    "             'time_interval': 'VARCHAR', 'travel_time': 'DOUBLE'}) "
    ") "
    "SELECT "
    "  link_ID AS link_id, "
    "  CAST(date AS DATE) AS date, "
    "  CAST(regexp_extract(time_interval, '\\[(.*?),', 1) AS TIMESTAMP) AS time_start, "
    "  CAST(regexp_extract(time_interval, ',(.*?)\\)', 1) AS TIMESTAMP) AS time_end, "
    "  travel_time "
    "FROM raw "
    "WHERE travel_time IS NOT NULL AND time_interval IS NOT NULL "
    "LIMIT " + str(NROWS)
)
df = con.execute(sql).pl()
con.close()
print(f"  rows: {df.shape[0]}  links: {df['link_id'].n_unique()}")

print("\nStep 3: 特征工程")

# 时间特征
df = df.with_columns([
    pl.col("time_start").dt.hour().alias("hour"),
    pl.col("time_start").dt.minute().alias("minute"),
    pl.col("time_start").dt.weekday().alias("day_of_week"),
])
df = df.with_columns([
    ((pl.col("hour") >= 6) & (pl.col("hour") < 9)).cast(pl.Int32).alias("is_morning_peak"),
    ((pl.col("hour") >= 17) & (pl.col("hour") < 19)).cast(pl.Int32).alias("is_evening_peak"),
])

# 滞后特征（按 link 分组，时间排序）
df = df.sort(["link_id", "time_start"])
df = df.with_columns([
    pl.col("travel_time").shift(1).over("link_id").alias("lag1"),
    pl.col("travel_time").shift(2).over("link_id").alias("lag2"),
    pl.col("travel_time").shift(4).over("link_id").alias("lag4"),
])

# 空间特征：上游 link 同一时间片的 travel_time
upstream_map = topo.select([
    pl.col("link_id"),
    pl.col("in_links").str.split(",").list.first().alias("upstream_link_id"),
]).filter(pl.col("upstream_link_id").is_not_null())

upstream_tt = df.select([
    pl.col("link_id").alias("upstream_link_id"),
    pl.col("time_start"),
    pl.col("travel_time").alias("upstream_travel_time"),
])

df = df.join(upstream_map, on="link_id", how="left")
df = df.join(upstream_tt, on=["upstream_link_id", "time_start"], how="left")
df = df.with_columns(pl.col("upstream_travel_time").fill_null(-1.0))

# 静态特征
static_sel = static.select(["link_id", "length", "lane_num", "link_class"])
df = df.join(static_sel, on="link_id", how="left")
df = df.with_columns([
    pl.col("length").fill_null(pl.col("length").mean()),
    pl.col("lane_num").fill_null(pl.col("lane_num").mean()),
    pl.col("link_class").fill_null(-1).cast(pl.Int32),
])

# link_class one-hot
class_dummies = df.select("link_class").to_dummies("link_class")
class_dummies = class_dummies.rename({
    c: f"link_class_{c.replace('link_class_', '')}" for c in class_dummies.columns
})
df = pl.concat([df, class_dummies], how="horizontal")

# 目标变量
df = df.with_columns(pl.col("travel_time").shift(-1).over("link_id").alias("target"))

# 清理
df = df.drop_nulls(subset=["target", "lag1", "lag2", "lag4"])
print(f"  特征工程后: {df.shape[0]} 行, {df.shape[1]} 列")

print("\nStep 4: 训练/验证划分")
exclude = {"link_id", "travel_time", "target", "time_start", "time_end",
           "date", "upstream_link_id", "link_class"}
feature_cols = [c for c in df.columns if c not in exclude]

# 用最后20%的数据做验证
n_total = df.shape[0]
n_train = int(n_total * 0.8)
train_df = df[:n_train]
val_df = df[n_train:]

X_train = train_df.select(feature_cols).to_numpy()
y_train = train_df.select("target").to_numpy().ravel()
X_val = val_df.select(feature_cols).to_numpy()
y_val = val_df.select("target").to_numpy().ravel()
print(f"  train: {X_train.shape[0]}  val: {X_val.shape[0]}  features: {X_train.shape[1]}")

print("\nStep 5: XGBoost 训练")
model = xgb.XGBRegressor(
    n_estimators=50, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mask = np.abs(y_val) > 0.1
mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
r2 = r2_score(y_val, y_pred)
print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%  R2={r2:.4f}")

print(f"\n  Feature cols: {feature_cols}")
print("\n===== ALL STEPS PASSED =====")