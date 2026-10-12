# -*- coding: utf-8 -*-
"""
KIEM TRA CHAT LUONG DU LIEU 500M DONG: gia tri rong / NaN, gia tri khong hop le, trung lap.

Chay thu nhanh (khong day du, trung lap chi dung voi tap file da chon):
  python kiem_tra_du_lieu.py --input D:\\doan\\BigData\\Dataset\\ecommerce_500m --output results\\kiem_tra --max-files 40
Chay day du, chi phan rong/khong hop le (nhanh):
  python kiem_tra_du_lieu.py --input ... --output results\\kiem_tra --skip-duplicates --driver-memory 20g --cores 8 --tmp-dir D:\\spark_tmp
Chay phan trung lap (nang hon, co shuffle):
  python kiem_tra_du_lieu.py --input ... --output results\\kiem_tra --only-duplicates --driver-memory 20g --cores 8 --tmp-dir D:\\spark_tmp

Ket qua: in ra man hinh va ghi vao thu muc --output:
  kiem_tra_du_lieu.json, missing_by_column.csv, invalid_values.csv
"""
import argparse
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--max-files", type=int, default=0, help="chi dung N file rai deu (0 = tat ca)")
p.add_argument("--expect-rows", type=int, default=500_000_000)
p.add_argument("--skip-duplicates", action="store_true", help="bo qua phan trung lap")
p.add_argument("--only-duplicates", action="store_true", help="chi kiem tra trung lap")
p.add_argument("--driver-memory", default="20g")
p.add_argument("--cores", default="8")
p.add_argument("--tmp-dir", default="spark_tmp")
a = p.parse_args()

ID = "order_id"
NUM = [
    "order_year", "order_month", "order_day", "order_hour", "order_minute", "order_second",
    "age", "customer_loyalty_score", "total_orders_by_customer",
    "product_rating_avg", "product_reviews_count", "stock_quantity",
    "unit_price_usd", "quantity", "discount_percent", "delivery_days",
]
CAT = [
    "is_weekend", "gender", "customer_segment", "category", "sub_category", "brand",
    "shipping_method", "payment_method", "installment_plan", "campaign_source",
    "device_type", "traffic_source",
]
TARGET = "total_price_usd"
ALL_COLS = NUM + CAT + [TARGET]
OUT = Path(a.output)
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(msg):
    print(time.strftime("%H:%M:%S"), f"| [{(time.time() - T0) / 60:5.1f} phut]", msg, flush=True)


os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
if os.name == "nt":
    hh = os.environ.get("HADOOP_HOME") or "C:\\hadoop"
    if os.path.isdir(os.path.join(hh, "bin")):
        os.environ["HADOOP_HOME"] = hh
        os.environ["PATH"] = os.path.join(hh, "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, functions as F  # noqa: E402

Path(a.tmp_dir).mkdir(parents=True, exist_ok=True)
spark = (
    SparkSession.builder.appName("Kiem_tra_du_lieu").master(f"local[{a.cores}]")
    .config("spark.driver.memory", a.driver_memory)
    .config("spark.local.dir", str(Path(a.tmp_dir).resolve()))
    .config("spark.sql.files.maxPartitionBytes", "64m")
    .config("spark.sql.shuffle.partitions", "400")
    .config("spark.cleaner.periodicGC.interval", "5min")
    .config("spark.network.timeout", "3600s")
    .config("spark.executor.heartbeatInterval", "600s")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


class Heartbeat(threading.Thread):
    """Moi 60 giay in stage/task Spark dang chay."""

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

# ------------------------------------------------------------------ doc du lieu
if a.max_files > 0:
    all_files = sorted(glob.glob(os.path.join(a.input, "**", "*.parquet"), recursive=True))
    if not all_files:
        raise FileNotFoundError(f"Khong tim thay file parquet trong {a.input}")
    if a.max_files < len(all_files):
        step = len(all_files) / a.max_files
        files = [all_files[int(i * step)] for i in range(a.max_files)]
    else:
        files = all_files
    df = spark.read.parquet(*[Path(f).resolve().as_uri() for f in files])
    log(f"Dung {len(files)}/{len(all_files)} file (rai deu)")
else:
    df = (spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.parquet")
          .parquet(a.input))
absent = [c for c in [ID] + ALL_COLS if c not in df.columns]
if absent:
    raise SystemExit("Thieu cot trong du lieu: " + ", ".join(absent))
df = df.select(ID, *ALL_COLS)

result = {"input": a.input, "max_files": a.max_files}

# ------------------------------------------------------------------ 1) rong / NaN / khong hop le
if not a.only_duplicates:
    log("LUOT 1: dem gia tri rong/NaN va gia tri khong hop le tren moi cot (quet toan bo)")
    num_cols = NUM + [TARGET]
    exprs = [F.count(F.lit(1)).alias("rows")]
    for c in num_cols:
        x = F.col(c).cast("double")
        exprs += [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__null"),
                  F.sum(F.when(F.isnan(x), 1).otherwise(0)).alias(f"{c}__nan")]
    for c in CAT + [ID]:
        s = F.col(c).cast("string")
        exprs += [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__null"),
                  F.sum(F.when(F.trim(s) == "", 1).otherwise(0)).alias(f"{c}__empty")]

    def n(c):
        return F.col(c).cast("double")

    rules = {
        "product_rating_avg ngoai [0, 5]": (n("product_rating_avg") < 0) | (n("product_rating_avg") > 5),
        "product_reviews_count < 0": n("product_reviews_count") < 0,
        "stock_quantity < 0": n("stock_quantity") < 0,
        "quantity < 0": n("quantity") < 0,
        "total_orders_by_customer < 0": n("total_orders_by_customer") < 0,
        "delivery_days < 0": n("delivery_days") < 0,
        "cot dem khong phai so nguyen": ((n("quantity") % 1) != 0) | ((n("stock_quantity") % 1) != 0)
                                        | ((n("product_reviews_count") % 1) != 0)
                                        | ((n("total_orders_by_customer") % 1) != 0)
                                        | ((n("delivery_days") % 1) != 0),
        "age < 0 hoac > 120": (n("age") < 0) | (n("age") > 120),
        "discount_percent ngoai [0, 100]": (n("discount_percent") < 0) | (n("discount_percent") > 100),
        "unit_price_usd <= 0": n("unit_price_usd") <= 0,
        "total_price_usd <= 0 (bien muc tieu)": n(TARGET) <= 0,
        "order_month ngoai [1, 12]": (n("order_month") < 1) | (n("order_month") > 12),
        "order_day ngoai [1, 31]": (n("order_day") < 1) | (n("order_day") > 31),
        "order_hour ngoai [0, 23]": (n("order_hour") < 0) | (n("order_hour") > 23),
        "order_minute ngoai [0, 59]": (n("order_minute") < 0) | (n("order_minute") > 59),
        "order_second ngoai [0, 59]": (n("order_second") < 0) | (n("order_second") > 59),
    }
    for name, cond in rules.items():
        exprs.append(F.sum(F.when(cond, 1).otherwise(0)).alias("rule__" + name))
    v = df.agg(*exprs).first().asDict()
    total = int(v["rows"])
    log(f"  tong so dong: {total:,}" + ("" if a.max_files or total == a.expect_rows
                                       else f"  (KHAC {a.expect_rows:,} du kien!)"))

    rows = []
    for c in num_cols:
        nl, nn = int(v[f"{c}__null"] or 0), int(v[f"{c}__nan"] or 0)
        rows.append({"column": c, "type": "so", "null": nl, "nan": nn, "empty": 0,
                     "missing_total": nl + nn, "missing_pct": 100 * (nl + nn) / max(total, 1)})
    for c in CAT + [ID]:
        nl, em = int(v[f"{c}__null"] or 0), int(v[f"{c}__empty"] or 0)
        rows.append({"column": c, "type": "chuoi", "null": nl, "nan": 0, "empty": em,
                     "missing_total": nl + em, "missing_pct": 100 * (nl + em) / max(total, 1)})
    missing = pd.DataFrame(rows).sort_values("missing_total", ascending=False)
    missing.to_csv(OUT / "missing_by_column.csv", index=False)
    invalid = pd.DataFrame([{"rule": k, "rows": int(v["rule__" + k] or 0),
                             "pct": 100 * int(v["rule__" + k] or 0) / max(total, 1)} for k in rules])
    invalid.to_csv(OUT / "invalid_values.csv", index=False)

    bad_cols = missing[missing["missing_total"] > 0]
    bad_rules = invalid[invalid["rows"] > 0]
    print("\n=== GIA TRI RONG / NaN ===")
    print("Khong co cot nao bi thieu." if bad_cols.empty else bad_cols.to_string(index=False))
    print("\n=== GIA TRI KHONG HOP LE ===")
    print("Khong vi pham quy tac nao." if bad_rules.empty else bad_rules.to_string(index=False))
    result.update({"rows": total, "columns_with_missing": int(len(bad_cols)),
                   "missing_cells_total": int(missing["missing_total"].sum()),
                   "rules_violated": int(len(bad_rules)),
                   "invalid_rows_by_rule": {r.rule: int(r.rows) for r in invalid.itertuples() if r.rows}})

# ------------------------------------------------------------------ 2) trung lap
if not a.skip_duplicates:
    log("LUOT 2: kiem tra trung order_id (groupBy tren 1 cot, co shuffle)")
    g = (df.select(ID).groupBy(ID).count().filter(F.col("count") > 1)
         .agg(F.count(F.lit(1)).alias("ids"), F.sum(F.col("count") - 1).alias("extra")).first())
    dup_ids, dup_extra = int(g["ids"] or 0), int(g["extra"] or 0)
    log(f"  order_id bi trung: {dup_ids:,} gia tri, {dup_extra:,} dong thua")

    log("LUOT 3: kiem tra trung ca dong (29 cot, khong tinh order_id) bang xxhash64")
    fp = df.select(F.xxhash64(*[F.col(c) for c in ALL_COLS]).alias("fp"))
    g2 = (fp.groupBy("fp").count().filter(F.col("count") > 1)
          .agg(F.count(F.lit(1)).alias("keys"), F.sum(F.col("count") - 1).alias("extra")).first())
    dup_rows_keys, dup_rows_extra = int(g2["keys"] or 0), int(g2["extra"] or 0)
    log(f"  dong trung 29 cot: {dup_rows_keys:,} nhom, {dup_rows_extra:,} dong thua")
    print("\n=== TRUNG LAP ===")
    print(f"order_id trung: {dup_ids:,} gia tri ({dup_extra:,} dong thua)")
    print(f"Dong trung ca 29 cot (ngoai order_id): {dup_rows_keys:,} nhom ({dup_rows_extra:,} dong thua)")
    result.update({"duplicate_order_id_values": dup_ids, "duplicate_order_id_extra_rows": dup_extra,
                   "duplicate_feature_row_groups": dup_rows_keys, "duplicate_feature_row_extra": dup_rows_extra,
                   "duplicate_check_note": "dong trung dua tren xxhash64 64-bit, xac suat bao nham rat nho"})

with open(OUT / "kiem_tra_du_lieu.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
log("HOAN TAT. Ket qua trong " + str(OUT))
hb.stop()
spark.stop()