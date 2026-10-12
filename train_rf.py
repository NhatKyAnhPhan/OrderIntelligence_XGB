"""Random Forest Regressor (Spark ML) - du bao total_price_usd.

Chay thu tren mau nho:
  python train_rf.py --input D:/data/ecommerce_100m --output results/rf_sample --sample 0.02
Chay day du:
  python train_rf.py --input D:/data/ecommerce_100m --output results/rf_full
"""
import argparse
import json
import os
import platform
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql import SparkSession

# 28 feature theo quy uoc nhom: 16 so + 12 categorical
NUM = [
    "order_year", "order_month", "order_day", "order_hour", "order_minute", "order_second",
    "age", "customer_loyalty_score", "total_orders_by_customer",
    "product_rating_avg", "product_reviews_count", "stock_quantity",
    "unit_price_usd", "quantity", "discount_percent", "delivery_days",
]
CAT = [
    "is_weekend", "gender", "customer_segment",
    "category", "sub_category", "brand",
    "shipping_method", "payment_method", "installment_plan",
    "campaign_source", "device_type", "traffic_source",
]
TARGET = "total_price_usd"
SEED = 42

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="thu muc chua cac file parquet")
p.add_argument("--output", required=True, help="thu muc luu ket qua")
p.add_argument("--sample", type=float, default=1.0, help="ti le lay mau de tune nhanh (1.0 = toan bo)")
p.add_argument("--num-trees", type=int, default=20)
p.add_argument("--max-depth", type=int, default=10)
p.add_argument("--driver-memory", default="10g")
p.add_argument("--cores", default="*", help="so nhan CPU Spark dung (vd 8); giam de bot ton RAM")
p.add_argument("--tmp-dir", default="spark_tmp", help="thu muc spark ghi tam khi tran RAM")
a = p.parse_args()

spark = (
    SparkSession.builder.master(f"local[{a.cores}]").appName("RandomForestRegressor")
    .config("spark.driver.memory", a.driver_memory)
    .config("spark.local.dir", a.tmp_dir)
    .config("spark.sql.files.maxPartitionBytes", "16m")  # ~1 file/partition, moi task nhe hon
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# 1. Doc du lieu, chon 28 feature + target
df = spark.read.parquet(a.input).select(*NUM, *CAT, TARGET)
if a.sample < 1.0:
    df = df.sample(fraction=a.sample, seed=SEED)
n_rows = df.count()

# 2. Chia train/test 80/20, seed 42
train, test = df.randomSplit([0.8, 0.2], seed=SEED)

# 3. Tien xu ly: chi fit tren tap train
idx_cols = [c + "_idx" for c in CAT]
prep = Pipeline(stages=[
    StringIndexer(inputCols=CAT, outputCols=idx_cols, handleInvalid="keep"),
    VectorAssembler(inputCols=NUM + idx_cols, outputCol="features"),
]).fit(train)

# maxBins phai >= so gia tri khac nhau lon nhat cua cot categorical (+1 cho nhom "keep")
max_bins = max(32, max(len(labels) for labels in prep.stages[0].labelsArray) + 1)

train_p = prep.transform(train).select("features", TARGET)
test_p = prep.transform(test).select("features", TARGET)

# 4. Train
rf = RandomForestRegressor(
    featuresCol="features", labelCol=TARGET,
    numTrees=a.num_trees, maxDepth=a.max_depth, maxBins=max_bins, seed=SEED,
)
t0 = time.time()
model = rf.fit(train_p)
train_time = time.time() - t0

# 5. Danh gia
t0 = time.time()
pred = model.transform(test_p).select(TARGET, "prediction").cache()
n_test = pred.count()
metrics = {
    m: RegressionEvaluator(labelCol=TARGET, metricName=m).evaluate(pred)
    for m in ("rmse", "mae", "r2")
}
eval_time = time.time() - t0

# 6. Luu ket qua
os.makedirs(a.output, exist_ok=True)


def dump(name, obj):
    with open(os.path.join(a.output, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


result = {
    "rows": n_rows, "train_rows": n_rows - n_test, "test_rows": n_test,
    **metrics, "train_time_sec": round(train_time, 2), "eval_time_sec": round(eval_time, 2),
}
params = {
    "algorithm": "RandomForestRegressor (Spark ML)", "numTrees": a.num_trees,
    "maxDepth": a.max_depth, "maxBins": max_bins, "seed": SEED,
    "train_test_split": "80/20", "sample_fraction": a.sample,
}
dump("metrics.json", result)
dump("model_parameters.json", params)

names = NUM + idx_cols
importance = dict(zip(names, map(float, model.featureImportances.toArray())))
dump("feature_importance.json", dict(sorted(importance.items(), key=lambda kv: -kv[1])))

# mau Actual vs Predicted (~50k dong) de ve bieu do
frac = min(1.0, 50_000 / n_test)
pred.sample(fraction=frac, seed=SEED).toPandas().to_csv(
    os.path.join(a.output, "actual_vs_predicted_sample.csv"), index=False
)

env = {
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "java": spark._jvm.System.getProperty("java.version"),
    "spark": spark.version,
    "driver_memory": a.driver_memory,
    "default_parallelism": spark.sparkContext.defaultParallelism,
}
with open(os.path.join(a.output, "run_summary.txt"), "w", encoding="utf-8") as f:
    for section in (result, params, env):
        for k, v in section.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

print(json.dumps(result, indent=2))
spark.stop()