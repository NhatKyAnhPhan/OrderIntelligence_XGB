# -*- coding: utf-8 -*-
"""
PHAN TICH VA VE BIEU DO DU LIEU E-COMMERCE TRUOC/SAU XU LY (ban chay local, 500M dong)

Chay thu nhanh (40 file ~ 10M dong):
  python phan_tich_bieu_do_ecommerce.py --input D:\\doan\\BigData\\Dataset\\ecommerce_500m --output results\\eda_test --max-files 40
Chay day du:
  python phan_tich_bieu_do_ecommerce.py --input D:\\doan\\BigData\\Dataset\\ecommerce_500m --output results\\eda_500m \\
      --driver-memory 20g --cores 12 --tmp-dir D:\\spark_tmp

So voi ban Colab:
  * Khong cache 500M dong, chi quet du lieu 4 luot (truoc: 1 luot thong ke + 1 luot lay mau;
    sau: 1 luot tinh median + 1 luot thong ke), thay vi ~15 luot.
  * Chia train/validation bang hash(order_id) giong luc train (khong dung randomSplit gay sort 29 cot).
  * Mau ve bieu do lay theo hash(order_id) -> bieu do "truoc" va "sau" dung cung mot tap dong.
  * Dung matplotlib backend Agg va bo plt.show() (khong thi script dung lai cho moi bieu do).
  * File processed_parquet (~55GB) chi ghi khi them --write-processed.
"""
import argparse
import glob
import json
import math
import os
import sys
import threading
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # luu file PNG, khong mo cua so
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--input", required=True, help="thu muc chua cac file parquet (co the co folder con)")
p.add_argument("--output", required=True, help="thu muc luu bieu do va thong ke")
p.add_argument("--max-files", type=int, default=0, help="chi dung N file rai deu de chay nhanh (0 = tat ca)")
p.add_argument("--sample-size", type=int, default=50_000, help="so dong lay mau de ve bieu do")
p.add_argument("--write-processed", action="store_true", help="ghi du lieu sau xu ly ra parquet (rat nang)")
p.add_argument("--driver-memory", default="20g")
p.add_argument("--cores", default="8")
p.add_argument("--tmp-dir", default="spark_tmp")
a = p.parse_args()

SEED, TOP_N = 42, 15
RATING_MIN, RATING_MAX = 0.0, 5.0
BUCKETS, TEST_MOD = 10_000, 5      # giong train_all_500m.py: bucket % 5 == 0 -> tap test/validation
ID = "order_id"
MED_BUCKETS = 100   # median chi tinh tren ~1% dong (hash order_id): nhanh hon rat nhieu, sai so khong dang ke

NUM = [
    "order_year", "order_month", "order_day", "order_hour", "order_minute",
    "order_second", "age", "customer_loyalty_score", "total_orders_by_customer",
    "product_rating_avg", "product_reviews_count", "stock_quantity",
    "unit_price_usd", "quantity", "discount_percent", "delivery_days",
]
CAT = [
    "is_weekend", "gender", "customer_segment", "category", "sub_category",
    "brand", "shipping_method", "payment_method", "installment_plan",
    "campaign_source", "device_type", "traffic_source",
]
TARGET = "total_price_usd"
ALL_COLS = NUM + CAT + [TARGET]
COUNT_COLS = ["product_reviews_count", "stock_quantity", "quantity",
              "total_orders_by_customer", "delivery_days"]

OUT = Path(a.output)
BEFORE_DIR, AFTER_DIR, COMPARE_DIR = (OUT / "01_truoc_tien_xu_ly", OUT / "02_sau_tien_xu_ly", OUT / "03_so_sanh")
for folder in (OUT, BEFORE_DIR, AFTER_DIR, COMPARE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

T0 = time.time()


def log(msg):
    print(time.strftime("%H:%M:%S"), f"| [{(time.time() - T0) / 60:5.1f} phut]", msg, flush=True)


# ------------------------------------------------------------------ Spark
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if os.name == "nt":   # Windows can winutils.exe + hadoop.dll
    hh = os.environ.get("HADOOP_HOME") or "C:\\hadoop"
    if os.path.isdir(os.path.join(hh, "bin")):
        os.environ["HADOOP_HOME"] = hh
        os.environ["PATH"] = os.path.join(hh, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, functions as F  # noqa: E402
from pyspark.sql.types import NumericType  # noqa: E402

Path(a.tmp_dir).mkdir(parents=True, exist_ok=True)
spark = (
    SparkSession.builder.appName("Ecommerce_EDA_Charts").master(f"local[{a.cores}]")
    .config("spark.driver.memory", a.driver_memory)
    .config("spark.local.dir", str(Path(a.tmp_dir).resolve()))
    .config("spark.sql.files.maxPartitionBytes", "64m")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.cleaner.periodicGC.interval", "5min")
    .config("spark.network.timeout", "3600s")
    .config("spark.executor.heartbeatInterval", "600s")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


class Heartbeat(threading.Thread):
    """Moi 60 giay in stage/task Spark dang chay (de biet tien do, khong phai doi thanh tien do bi cat)."""

    def __init__(self, every=60):
        super().__init__(daemon=True)
        self.every, self._halt = every, threading.Event()

    def stop(self):
        self._halt.set()

    def run(self):
        while not self._halt.wait(self.every):
            try:
                tr = spark.sparkContext.statusTracker()
                parts = []
                for sid in list(tr.getActiveStageIds())[:3]:
                    si = tr.getStageInfo(sid)
                    if si is not None and si.numTasks:
                        parts.append(f"stage {sid}: {si.numCompletedTasks}/{si.numTasks} task "
                                     f"({100 * si.numCompletedTasks / si.numTasks:.0f}%)")
                log("  ... dang chay | " + (", ".join(parts) or "chuyen stage"))
            except Exception:
                pass


hb = Heartbeat()
hb.start()


# ------------------------------------------------------------------ Ham xu ly
def clean_columns(df):
    """Buoc lam sach giong ban Colab: ep kieu so, NaN->null, gia tri khong hop le->null, chuan hoa chuoi."""
    for col in NUM + [TARGET]:
        df = df.withColumn(col, F.col(col).cast("double"))
        df = df.withColumn(col, F.when(F.isnan(F.col(col)), F.lit(None).cast("double")).otherwise(F.col(col)))
    df = df.withColumn(
        "product_rating_avg",
        F.when(F.col("product_rating_avg").between(RATING_MIN, RATING_MAX), F.col("product_rating_avg"))
        .otherwise(F.lit(None).cast("double")))
    for col in ["product_reviews_count", "stock_quantity"]:
        df = df.withColumn(col, F.when(F.col(col) >= 0, F.col(col)).otherwise(F.lit(None).cast("double")))
    for col in COUNT_COLS:
        df = df.withColumn(col, F.when((F.col(col) >= 0) & (F.col(col) % 1 == 0), F.col(col))
                           .otherwise(F.lit(None).cast("double")))
    df = df.filter(F.col(TARGET).isNotNull())
    for col in CAT:
        clean = F.trim(F.col(col).cast("string"))
        df = df.withColumn(col, F.when(clean.isNull() | (clean == ""), F.lit(None).cast("string")).otherwise(clean))
    return df


def impute(df, medians):
    for col in NUM:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(medians[col])))
    for col in CAT:
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit("__MISSING__")))
    return df


def full_report(df):
    """1 luot quet duy nhat: so dong + ty le thieu cua moi cot + thong ke cac cot so."""
    numeric_names = {f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)}
    exprs = [F.count(F.lit(1)).alias("__rows")]
    for c in ALL_COLS:
        miss = F.col(c).isNull()
        if c in numeric_names:
            miss = miss | F.isnan(F.col(c).cast("double"))
        exprs.append(F.sum(F.when(miss, 1).otherwise(0)).alias(f"{c}__miss"))
    for c in NUM + [TARGET]:
        x = F.col(c).cast("double")
        d = F.when(F.isnan(x), F.lit(None).cast("double")).otherwise(x)   # bo NaN khi tinh thong ke
        exprs += [F.count(d).alias(f"{c}__count"), F.avg(d).alias(f"{c}__mean"), F.min(d).alias(f"{c}__min"),
                  F.percentile_approx(F.when(F.col("_h") < MED_BUCKETS, d), 0.5).alias(f"{c}__median"),
                  F.max(d).alias(f"{c}__max")]
    v = df.agg(*exprs).first().asDict()
    total = int(v["__rows"])
    missing = pd.DataFrame([
        {"column": c, "missing_count": int(v.get(f"{c}__miss") or 0),
         "missing_pct": int(v.get(f"{c}__miss") or 0) * 100 / max(total, 1)} for c in ALL_COLS
    ]).sort_values("missing_pct", ascending=False)
    stats = pd.DataFrame([{
        "column": c, "count_non_null": v.get(f"{c}__count"), "mean": v.get(f"{c}__mean"),
        "median": v.get(f"{c}__median"), "min": v.get(f"{c}__min"), "max": v.get(f"{c}__max")}
        for c in NUM + [TARGET]])
    return total, missing, stats


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_missing(report, title, path):
    q = report.sort_values("missing_pct", ascending=True)
    plt.figure(figsize=(12, max(5, len(q) * 0.28)))
    plt.barh(q["column"], q["missing_pct"])
    plt.xlabel("Ty le thieu (%)")
    plt.ylabel("Thuoc tinh")
    plt.title(title)
    save_fig(path)


def plot_numeric_histograms(pdf, columns, stage, out_dir):
    for col in columns:
        values = pd.to_numeric(pdf[col], errors="coerce").dropna()
        if values.empty:
            continue
        plt.figure(figsize=(8, 4.5))
        plt.hist(values, bins=35, edgecolor="black")
        plt.title(f"{stage}: Phan bo {col}")
        plt.xlabel(col)
        plt.ylabel("Tan suat trong mau")
        save_fig(Path(out_dir) / f"hist_{col}.png")


def plot_categorical(pdf, columns, stage, out_dir):
    for col in columns:
        counts = pdf[col].fillna("(THIEU)").astype(str).value_counts().head(TOP_N).sort_values()
        if counts.empty:
            continue
        plt.figure(figsize=(10, 5))
        plt.barh(counts.index, counts.values)
        plt.title(f"{stage}: Top {len(counts)} gia tri - {col}")
        plt.xlabel("So ban ghi trong mau")
        plt.ylabel(col)
        save_fig(Path(out_dir) / f"category_{col}.png")


def plot_target(pdf, stage, out_dir):
    vals = pd.to_numeric(pdf[TARGET], errors="coerce").dropna()
    if vals.empty:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(vals, bins=40, edgecolor="black")
    plt.title(f"{stage}: Phan bo {TARGET}")
    plt.xlabel(TARGET)
    plt.ylabel("Tan suat trong mau")
    save_fig(Path(out_dir) / "target_distribution.png")


def plot_overview(total_rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(["Tong so ban ghi"], [total_rows])
    axes[0].set_title(f"Quy mo du lieu: {total_rows:,} ban ghi")
    axes[0].set_ylabel("So luong")
    axes[0].ticklabel_format(axis="y", style="plain")
    axes[1].bar(["Thuoc tinh so", "Thuoc tinh phan loai", "Bien muc tieu"], [len(NUM), len(CAT), 1])
    axes[1].set_title(f"Co cau thuoc tinh ({len(ALL_COLS)} cot phan tich)")
    axes[1].set_ylabel("So cot")
    save_fig(Path(out_dir) / "01_quy_mo_va_co_cau_du_lieu.png")


def plot_row_status(before_n, after_n, out_dir):
    removed = before_n - after_n
    labels, vals = ["Truoc xu ly", "Sau xu ly", "Chenh lech so dong"], [before_n, after_n, removed]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, vals)
    plt.ylabel("So ban ghi")
    plt.title("So luong ban ghi truoc va sau tien xu ly")
    plt.ticklabel_format(axis="y", style="plain")
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:,}", ha="center", va="bottom")
    save_fig(Path(out_dir) / "02_trang_thai_ban_ghi.png")


def plot_missing_comparison(before, after, out_dir):
    m = before[["column", "missing_pct"]].merge(after[["column", "missing_pct"]], on="column",
                                                suffixes=("_before", "_after")).sort_values("missing_pct_before")
    y, h = range(len(m)), 0.38
    plt.figure(figsize=(12, max(5, len(m) * 0.30)))
    plt.barh([i - h / 2 for i in y], m["missing_pct_before"], height=h, label="Truoc")
    plt.barh([i + h / 2 for i in y], m["missing_pct_after"], height=h, label="Sau")
    plt.yticks(list(y), m["column"])
    plt.xlabel("Ty le thieu (%)")
    plt.title("So sanh du lieu thieu truoc va sau tien xu ly")
    plt.legend()
    save_fig(Path(out_dir) / "03_so_sanh_du_lieu_thieu.png")


def plot_distribution_comparison(before_pdf, after_pdf, columns, out_dir):
    for col in columns:
        b = pd.to_numeric(before_pdf[col], errors="coerce").dropna()
        af = pd.to_numeric(after_pdf[col], errors="coerce").dropna()
        if b.empty or af.empty:
            continue
        low, high = min(b.min(), af.min()), max(b.max(), af.max())
        if not math.isfinite(low) or not math.isfinite(high):
            continue
        if low == high:
            high = low + 1
        plt.figure(figsize=(8, 4.5))
        plt.hist(b, bins=35, range=(low, high), alpha=0.55, label="Truoc")
        plt.hist(af, bins=35, range=(low, high), alpha=0.55, label="Sau")
        plt.title(f"So sanh phan bo: {col}")
        plt.xlabel(col)
        plt.ylabel("Tan suat trong mau")
        plt.legend()
        save_fig(Path(out_dir) / f"compare_hist_{col}.png")


# ------------------------------------------------------------------ 1) Doc du lieu
if a.max_files > 0:
    all_files = sorted(glob.glob(os.path.join(a.input, "**", "*.parquet"), recursive=True))
    if not all_files:
        raise FileNotFoundError(f"Khong tim thay file parquet trong {a.input}")
    if a.max_files < len(all_files):    # lay RAI DEU qua tat ca folder, khong chi lay N file dau
        step = len(all_files) / a.max_files
        files = [all_files[int(i * step)] for i in range(a.max_files)]
    else:
        files = all_files
    raw_all = spark.read.parquet(*[Path(f).resolve().as_uri() for f in files])
    log(f"Dung {len(files)}/{len(all_files)} file, chon rai deu tren toan bo du lieu")
else:
    if not os.path.exists(a.input):
        raise FileNotFoundError(f"Khong tim thay duong dan: {a.input}")
    raw_all = (spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.parquet")
               .parquet(a.input))
missing_cols = [c for c in [ID] + ALL_COLS if c not in raw_all.columns]
if missing_cols:
    raise ValueError("Thieu cot trong du lieu: " + ", ".join(missing_cols))
raw = raw_all.select(ID, *ALL_COLS).withColumn("_h", F.pmod(F.xxhash64(F.col(ID)), F.lit(BUCKETS)))

# ------------------------------------------------------------------ 2) Phan tich truoc xu ly
log("LUOT 1/4: thong ke + du lieu thieu truoc xu ly (quet toan bo du lieu)")
total_before, before_missing, before_stats = full_report(raw)
log(f"  so ban ghi truoc xu ly: {total_before:,} | so cot phan tich: {len(ALL_COLS)}")
plot_overview(total_before, BEFORE_DIR)
before_missing.to_csv(BEFORE_DIR / "missing_report_before.csv", index=False)
plot_missing(before_missing, "Ty le du lieu thieu truoc tien xu ly", BEFORE_DIR / "04_missing_before.png")
before_stats.to_csv(BEFORE_DIR / "numeric_statistics_before.csv", index=False)

log("LUOT 2/4: lay mau de ve bieu do (theo hash order_id)")
thr = max(1, round(BUCKETS * min(1.0, a.sample_size / max(total_before, 1))))
before_pdf = raw.filter(F.col("_h") < thr).select(ID, *ALL_COLS).toPandas()
log(f"  mau: {len(before_pdf):,} dong")
before_pdf.drop(columns=[ID]).describe(include="all").transpose().to_csv(
    BEFORE_DIR / "descriptive_statistics_before.csv")
plot_numeric_histograms(before_pdf, NUM, "Truoc tien xu ly", BEFORE_DIR)
plot_categorical(before_pdf, CAT, "Truoc tien xu ly", BEFORE_DIR)
plot_target(before_pdf, "Truoc tien xu ly", BEFORE_DIR)

# ------------------------------------------------------------------ 3) Tien xu ly
log("LUOT 3/4: tinh median tren tap train (hash order_id, giong luc train)")
cleaned = clean_columns(raw)
train = cleaned.filter(((F.col("_h") % TEST_MOD) != 0) & (F.col("_h") < MED_BUCKETS))   # train va ~1% dong
quantiles = train.approxQuantile(NUM, [0.5], 0.001)
medians = {c: float(v[0]) if v else 0.0 for c, v in zip(NUM, quantiles)}
log("  median: " + ", ".join(f"{k}={v:.2f}" for k, v in list(medians.items())[:4]) + " ...")

transformed = impute(cleaned, medians).select(ID, "_h", *ALL_COLS)
if a.write_processed:
    processed_dir = OUT / "processed_parquet"
    log(f"Ghi du lieu sau xu ly vao {processed_dir} (rat nang)")
    transformed.write.mode("overwrite").parquet(str(processed_dir))
    transformed = spark.read.parquet(str(processed_dir))

# ------------------------------------------------------------------ 4) Phan tich sau xu ly
log("LUOT 4/4: thong ke + du lieu thieu sau xu ly (quet toan bo du lieu)")
total_after, after_missing, after_stats = full_report(transformed)
log(f"  so ban ghi sau tien xu ly: {total_after:,}")
after_missing.to_csv(AFTER_DIR / "missing_report_after.csv", index=False)
plot_missing(after_missing, "Ty le du lieu thieu sau tien xu ly", AFTER_DIR / "04_missing_after.png")
after_stats.to_csv(AFTER_DIR / "numeric_statistics_after.csv", index=False)

# mau "sau": ap dung dung cac buoc lam sach/dien khuyet len chinh tap mau "truoc" (cung tap dong)
after_pdf = impute(clean_columns(spark.createDataFrame(before_pdf.assign(_h=0))), medians).select(*ALL_COLS).toPandas()
after_pdf.describe(include="all").transpose().to_csv(AFTER_DIR / "descriptive_statistics_after.csv")
plot_numeric_histograms(after_pdf, NUM, "Sau tien xu ly", AFTER_DIR)
plot_categorical(after_pdf, CAT, "Sau tien xu ly", AFTER_DIR)
plot_target(after_pdf, "Sau tien xu ly", AFTER_DIR)

# ------------------------------------------------------------------ 5) So sanh truoc/sau
log("Ve bieu do so sanh truoc/sau")
plot_row_status(total_before, total_after, COMPARE_DIR)
plot_missing_comparison(before_missing, after_missing, COMPARE_DIR)
compare_cols = ["product_rating_avg", "product_reviews_count", "stock_quantity",
                "unit_price_usd", "quantity", "discount_percent", TARGET]
plot_distribution_comparison(before_pdf, after_pdf, compare_cols, COMPARE_DIR)

comparison = before_stats[before_stats["column"].isin(compare_cols)].merge(
    after_stats[after_stats["column"].isin(compare_cols)], on="column", suffixes=("_before", "_after"))
comparison.to_csv(COMPARE_DIR / "comparison_numeric_statistics.csv", index=False)
means = comparison.dropna(subset=["mean_before", "mean_after"])
if not means.empty:
    x, width = range(len(means)), 0.38
    plt.figure(figsize=(13, 6))
    plt.bar([i - width / 2 for i in x], means["mean_before"], width, label="Truoc")
    plt.bar([i + width / 2 for i in x], means["mean_after"], width, label="Sau")
    plt.xticks(list(x), means["column"], rotation=50, ha="right")
    plt.ylabel("Gia tri trung binh")
    plt.title("So sanh gia tri trung binh truoc va sau tien xu ly")
    plt.legend()
    save_fig(COMPARE_DIR / "comparison_means.png")

summary = {
    "input_path": a.input, "output_path": str(OUT), "max_files": a.max_files,
    "rows_before": int(total_before), "rows_after": int(total_after),
    "rows_removed_net": int(total_before - total_after),
    "columns_analyzed": ALL_COLS, "numeric_columns": NUM, "categorical_columns": CAT, "target": TARGET,
    "sample_size_for_plots": int(len(before_pdf)), "sampling": "hash(order_id), cung tap dong cho truoc/sau",
    "rating_valid_range": [RATING_MIN, RATING_MAX],
    "numeric_missing_strategy": "Median estimated from train split (hash(order_id) 80/20)",
    "categorical_missing_strategy": "__MISSING__", "train_validation_ratio": [0.8, 0.2],
    "median_values": medians,
    "median_estimated_from": f"~{100 * MED_BUCKETS / BUCKETS:.0f}% dong lay theo hash(order_id) (thong ke median trong bang so sanh)",
}
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

log("HOAN TAT. Thu muc ket qua: " + str(OUT))
hb.stop()
spark.stop()