# -*- coding: utf-8 -*-
"""
train_all_500m.py (v2) - Train 3 mo hinh tren toan bo du lieu (RF, GBT, XGBoost).

Diem khac so voi ban cu:
  * Chia train/test bang hash(order_id) thay cho randomSplit: khong phai sort toan bo du lieu
    (randomSplit sort theo ca 29 cot -> tran RAM) va ca 3 mo hinh dung CHUNG mot tap test.
  * Khong .cache() ca train/test (vector 28 cot x 400M dong ~ 100GB), Spark tu persist du lieu da bin.
  * Moi mo hinh chay trong 1 tien trinh rieng -> JVM cua Spark nhả RAM truoc khi XGBoost chay.
  * Ghi ket qua ngay sau tung mo hinh (metrics.json, feature_importance.json,
    actual_vs_predicted_sample.csv -> dung duoc voi plot_results.py).
  * XGBoost doc parquet tung file qua DataIter, khong nap ca 500M dong vao RAM.

Chay thu 1% truoc:
  python train_all_500m.py --input E:\\Dataset\\ecommerce_500m --output results\\t1pct --sample 0.01
Chay day du:
  python train_all_500m.py --input E:\\Dataset\\ecommerce_500m --output results\\full_500m
Chi 1 mo hinh:
  python train_all_500m.py --input ... --output ... --models xgb
Chi kiem tra du lieu:
  python train_all_500m.py --input ... --output ... --only-preflight --expect-rows 500000000
"""
import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import traceback
from pathlib import Path

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
ID = "order_id"
SEED = 42
BUCKETS = 10_000   # bucket = hash(order_id) mod 10000
TEST_MOD = 5       # bucket % 5 == 0 -> test (20%), con lai -> train (80%)
FEATURE_NAMES = NUM + [c + "_idx" for c in CAT]
MODEL_DIRS = {"rf": "random_forest", "gbt": "gbt", "xgb": "xgboost"}
MODEL_LABELS = {"rf": "Random Forest", "gbt": "GBTRegressor", "xgb": "XGBoost"}


def log(msg):
    print(time.strftime("%H:%M:%S"), "|", msg, flush=True)


def ram_text():
    """Tra ve chuoi RAM dang dung neu co psutil (pip install psutil), nguoc lai tra ve rong."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f" | RAM {vm.used / 1e9:.1f}/{vm.total / 1e9:.1f} GB ({vm.percent:.0f}%)"
    except Exception:
        return ""


def dir_size_gb(path):
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total / 1e9


class Heartbeat(threading.Thread):
    """Moi N giay in ra: so phut da chay, stage/task cua Spark dang chay, RAM, dung luong thu muc tam."""

    def __init__(self, spark, tmp_dir, every=60):
        super().__init__(daemon=True)
        self.spark, self.tmp_dir, self.every = spark, str(tmp_dir), every
        self._halt = threading.Event()

    def stop(self):
        self._halt.set()

    def run(self):
        t0, last_disk = time.time(), time.time()
        while not self._halt.wait(self.every):
            try:
                tr = self.spark.sparkContext.statusTracker()
                parts = []
                for sid in list(tr.getActiveStageIds())[:3]:
                    si = tr.getStageInfo(sid)
                    if si is not None and si.numTasks:
                        parts.append(f"stage {sid}: {si.numCompletedTasks}/{si.numTasks} task "
                                     f"({100 * si.numCompletedTasks / si.numTasks:.0f}%)")
                msg = f"  ... dang chay {(time.time() - t0) / 60:.0f} phut | " + (", ".join(parts) or "chuyen stage")
                msg += ram_text()
                if time.time() - last_disk > 300:      # quet thu muc tam moi 5 phut
                    last_disk = time.time()
                    free = shutil.disk_usage(self.tmp_dir).free / 1e9
                    msg += f" | tmp {dir_size_gb(self.tmp_dir):.1f} GB, o dia con {free:.0f} GB"
                log(msg)
            except Exception:
                pass


# ----------------------------------------------------------------------------------------
# Tien ich chung
# ----------------------------------------------------------------------------------------
class PhaseTicker(threading.Thread):
    """Moi N giay in 'van dang chay <giai doan>' + RAM, ke ca luc XGBoost dang tinh trong C++ (khong co dong log nao)."""

    def __init__(self, every=60):
        super().__init__(daemon=True)
        self.every, self.phase, self._halt = every, "khoi dong", threading.Event()
        self.t0 = time.time()

    def stop(self):
        self._halt.set()

    def run(self):
        while not self._halt.wait(self.every):
            log(f"  ... van dang chay: {self.phase} | tong {(time.time() - self.t0) / 60:.0f} phut{ram_text()}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="thu muc goc chua cac folder con / file parquet")
    p.add_argument("--output", required=True, help="thu muc luu ket qua")
    p.add_argument("--models", default="rf,gbt,xgb", help="vd rf,gbt,xgb hoac chi xgb")
    p.add_argument("--sample", type=float, default=1.0, help="ti le dong de chay thu (1.0 = toan bo)")
    p.add_argument("--expect-rows", type=int, default=0, help="neu >0, dung lai khi tong dong khac gia tri nay")
    p.add_argument("--only-preflight", action="store_true", help="chi kiem tra du lieu roi dung")
    # Spark
    p.add_argument("--driver-memory", default="20g")
    p.add_argument("--cores", default="8", help="so nhan cho Spark (giam neu bi OOM)")
    p.add_argument("--tmp-dir", default="spark_tmp", help="thu muc Spark ghi tam khi tran RAM (nen de o o co nhieu cho)")
    p.add_argument("--rf-trees", type=int, default=20)
    p.add_argument("--rf-depth", type=int, default=10)
    p.add_argument("--gbt-iter", type=int, default=30)
    p.add_argument("--gbt-depth", type=int, default=6)
    # XGBoost
    p.add_argument("--xgb-rounds", type=int, default=100)
    p.add_argument("--xgb-depth", type=int, default=8)
    p.add_argument("--xgb-eta", type=float, default=0.1)
    p.add_argument("--xgb-max-bin", type=int, default=256)
    p.add_argument("--xgb-threads", type=int, default=0, help="0 = tat ca nhan CPU")
    p.add_argument("--xgb-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--xgb-cache-dir", default="", help="noi luu cache nhi phan cua XGBoost (mac dinh: <tmp-dir>/xgb_cache)")
    p.add_argument("--xgb-keep-cache", action="store_true", help="giu cache de lan sau khoi doc lai parquet (~56GB cho 500M dong)")
    p.add_argument("--xgb-io-threads", type=int, default=6, help="so luong doc/giai ma file parquet song song")
    p.add_argument("--xgb-external", action="store_true",
                   help="dung external memory (ExtMemQuantileDMatrix) khi RAM khong du; can xgboost >= 3.0")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--heartbeat-sec", type=int, default=60, help="chu ky in tien do (giay)")
    return p.parse_args()


def list_parquet(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"Khong tim thay file .parquet nao trong {root}")
    return files


def preflight(a):
    """Doc footer cua tat ca file (rat nhanh): tong dong, thieu cot, schema khong dong nhat."""
    import pyarrow.parquet as pq
    files = list_parquet(a.input)
    need = [ID] + NUM + CAT + [TARGET]
    ref_sig, total, bad = None, 0, []
    for f in files:
        try:
            pf = pq.ParquetFile(f)
            total += pf.metadata.num_rows
            sch = pf.schema_arrow
            sig = tuple((n, str(sch.field(n).type).replace("large_string", "string")) for n in need)
        except Exception as e:  # file hong / thieu cot
            bad.append((f, repr(e)[:120]))
            continue
        if ref_sig is None:
            ref_sig = sig
        elif sig != ref_sig:
            diff = [x for x, y in zip(sig, ref_sig) if x != y]
            bad.append((f, f"schema khac: {diff[:3]}"))
    log(f"Preflight: {len(files)} file, tong {total:,} dong, {len(bad)} file loi/khac schema")
    for f, why in bad[:10]:
        log(f"  LOI: {f} -> {why}")
    if bad:
        raise SystemExit("Co file loi hoac schema khong dong nhat, hay sua truoc khi train.")
    if a.expect_rows and total != a.expect_rows:
        raise SystemExit(f"Tong dong = {total:,} khac --expect-rows = {a.expect_rows:,}")
    return files, total


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_outputs(out_dir, key, a, metrics, params, importance, sample_df, extra_env):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "model_parameters.json", params)
    imp = dict(sorted(importance.items(), key=lambda kv: -kv[1]))
    write_json(out_dir / "feature_importance.json", imp)  # plot_results.py doc duoc file nay
    sample_df.to_csv(out_dir / "actual_vs_predicted_sample.csv", index=False)
    env = {"os": platform.platform(), "python": sys.version.split()[0], **extra_env}
    with open(out_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        for section in (metrics, params, env):
            for k, v in section.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")


# ----------------------------------------------------------------------------------------
# Spark: Random Forest va GBT
# ----------------------------------------------------------------------------------------
def build_spark(a, name):
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    if os.name == "nt":
        hh = os.environ.get("HADOOP_HOME") or "C:\\hadoop"
        if os.path.isdir(os.path.join(hh, "bin")):
            os.environ["HADOOP_HOME"] = hh
            os.environ["PATH"] = os.path.join(hh, "bin") + os.pathsep + os.environ.get("PATH", "")
        else:
            raise SystemExit(f"Khong thay {hh}\\bin (can winutils.exe va hadoop.dll). Dat bien moi truong HADOOP_HOME.")
    Path(a.tmp_dir).mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master(f"local[{a.cores}]").appName(name)
        .config("spark.driver.memory", a.driver_memory)
        .config("spark.local.dir", str(Path(a.tmp_dir).resolve()))
        .config("spark.sql.files.maxPartitionBytes", "32m")   # task nhe hon -> it tran heap
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.cleaner.periodicGC.interval", "5min")   # don file shuffle cu thuong xuyen hon (mac dinh 30 phut)
        .config("spark.network.timeout", "3600s")
        .config("spark.executor.heartbeatInterval", "600s")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_split(spark, a):
    from pyspark.sql import functions as F
    df = (
        spark.read.option("recursiveFileLookup", "true").option("pathGlobFilter", "*.parquet")
        .parquet(a.input).select(ID, *NUM, *CAT, TARGET)
    )
    df = df.withColumn("_h", F.pmod(F.xxhash64(F.col(ID)), F.lit(BUCKETS)))
    if a.sample < 1.0:
        df = df.filter(F.col("_h") < max(1, int(a.sample * BUCKETS)))
    is_test = (F.col("_h") % TEST_MOD) == 0
    # dem chi can doc cot order_id nen rat nhanh
    row = df.agg(F.count(F.lit(1)).alias("n"), F.sum(F.when(is_test, 1).otherwise(0)).alias("nt")).first()
    n_all, n_test = int(row["n"]), int(row["nt"] or 0)
    # Moi file la 1 partition (2000 task/luot). Khi chay mau 1% moi task chi con ~2000 dong nen
    # thoi gian bi chi phi lap lich task chiem het. Gop lai thanh it partition hon.
    cores = int(a.cores) if str(a.cores).isdigit() else (os.cpu_count() or 8)
    n_train = n_all - n_test
    # it partition hon -> it file shuffle hon (moi luot cua cay ghi ra 1 file/partition)
    train = df.filter(~is_test).drop(ID, "_h").coalesce(max(cores * 4, n_train // 2_000_000))
    test = df.filter(is_test).drop(ID, "_h").coalesce(max(cores * 4, n_test // 2_000_000))
    return train, test, n_all, n_test


def run_spark_model(key, a, out_dir):
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler
    from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
    from pyspark.sql import functions as F
    import math

    spark = build_spark(a, f"train_{key}")
    hb = Heartbeat(spark, a.tmp_dir, a.heartbeat_sec)
    hb.start()
    try:
        log(f"[{MODEL_LABELS[key]}] doc du lieu va chia train/test")
        train, test, n_all, n_test = load_split(spark, a)
        log(f"  tong {n_all:,} dong | test {n_test:,} | train {n_all - n_test:,}")

        log("  fit StringIndexer tren tap train (1 luot quet)")
        idx_cols = [c + "_idx" for c in CAT]
        prep = Pipeline(stages=[
            StringIndexer(inputCols=CAT, outputCols=idx_cols, handleInvalid="keep"),
            VectorAssembler(inputCols=NUM + idx_cols, outputCol="features"),
        ]).fit(train)
        max_bins = max(32, max(len(l) for l in prep.stages[0].labelsArray) + 1)
        train_p = prep.transform(train).select("features", TARGET)
        test_p = prep.transform(test).select("features", TARGET)

        if key == "rf":
            est = RandomForestRegressor(featuresCol="features", labelCol=TARGET, numTrees=a.rf_trees,
                                        maxDepth=a.rf_depth, maxBins=max_bins, seed=SEED)
            params = {"algorithm": "RandomForestRegressor (Spark ML)", "numTrees": a.rf_trees,
                      "maxDepth": a.rf_depth, "maxBins": max_bins}
        else:
            est = GBTRegressor(featuresCol="features", labelCol=TARGET, maxIter=a.gbt_iter,
                               maxDepth=a.gbt_depth, maxBins=max_bins, seed=SEED)
            params = {"algorithm": "GBTRegressor (Spark ML)", "maxIter": a.gbt_iter,
                      "maxDepth": a.gbt_depth, "maxBins": max_bins}
        params.update({"seed": SEED, "split": "hash(order_id) 80/20", "sample_fraction": a.sample})

        log("  bat dau train (co the rat lau, xem tien do o Spark log)")
        t0 = time.time()
        model = est.fit(train_p)
        train_time = time.time() - t0
        log(f"  train xong sau {train_time / 60:.1f} phut")

        t0 = time.time()
        pred = model.transform(test_p).select(TARGET, "prediction")
        err = F.col("prediction") - F.col(TARGET)
        m = pred.agg(F.count(F.lit(1)).alias("n"), F.avg(err * err).alias("mse"),
                     F.avg(F.abs(err)).alias("mae"), F.var_pop(TARGET).alias("var_y")).first()
        rmse = math.sqrt(m["mse"])
        r2 = (1.0 - m["mse"] / m["var_y"]) if m["var_y"] else None
        frac = min(1.0, 60_000 / max(1, m["n"]))
        sample_df = pred.sample(False, frac, SEED).limit(50_000).toPandas()
        eval_time = time.time() - t0

        metrics = {"model": MODEL_LABELS[key], "rows": n_all, "train_rows": n_all - n_test,
                   "test_rows": int(m["n"]), "rmse": rmse, "mae": m["mae"], "r2": r2,
                   "train_time_sec": round(train_time, 2), "eval_time_sec": round(eval_time, 2)}
        importance = dict(zip(FEATURE_NAMES, map(float, model.featureImportances.toArray())))
        env = {"spark": spark.version, "driver_memory": a.driver_memory, "cores": a.cores}
        save_outputs(out_dir, key, a, metrics, params, importance, sample_df, env)
        log(f"[{MODEL_LABELS[key]}] RMSE={rmse:.4f} MAE={m['mae']:.4f} R2={r2:.4f}")

        if not a.no_save_model:
            try:
                model.write().overwrite().save(str(out_dir / "model"))
            except Exception as e:  # tren Windows co the loi winutils; ket qua da luu o tren
                log(f"  Khong luu duoc model Spark (bo qua): {repr(e)[:150]}")
    finally:
        hb.stop()
        spark.stop()


# ----------------------------------------------------------------------------------------
# XGBoost (khong qua Spark): doc tung file parquet qua DataIter
# ----------------------------------------------------------------------------------------
def spark_bucket(ids):
    """Tinh bucket giong het Spark: pmod(xxhash64(order_id), BUCKETS) (XXH64, seed 42)."""
    import numpy as np
    from xxhash import xxh64_intdigest
    out = np.empty(len(ids), dtype=np.int64)
    for i, s in enumerate(ids):
        v = xxh64_intdigest(s.encode("utf-8"), 42)
        if v >= 1 << 63:
            v -= 1 << 64
        out[i] = v % BUCKETS          # % cua Python voi so duong luon >= 0 (= pmod)
    return out


def build_cat_values(files):
    """Lay danh sach gia tri cua 12 cot categorical tu ~40 file rai deu."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    picked = files[:: max(1, len(files) // 40)]
    vals = {c: set() for c in CAT}
    for f in picked:
        t = pq.read_table(f, columns=CAT)
        for c in CAT:
            vals[c].update(pc.unique(t[c]).to_pylist())
    return {c: pa.array(sorted(v for v in vals[c] if v is not None)) for c in CAT}


def run_xgb(a, out_dir):
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import xgboost as xgb

    files = list_parquet(a.input)
    cat_values = build_cat_values(files)
    cols = [ID] + NUM + CAT + [TARGET]
    threshold = max(1, int(a.sample * BUCKETS)) if a.sample < 1.0 else BUCKETS + 1

    cache_dir = Path(a.xgb_cache_dir) if a.xgb_cache_dir else Path(a.tmp_dir) / "xgb_cache"
    cdir = cache_dir / f"s{threshold}"
    meta = {"files": len(files), "threshold": threshold, "features": FEATURE_NAMES}
    marker = cdir / "DONE.json"

    def to_xy(t):
        X = np.empty((t.num_rows, len(FEATURE_NAMES)), dtype=np.float32)
        for j, c in enumerate(NUM):
            X[:, j] = t[c].to_numpy(zero_copy_only=False)
        for j, c in enumerate(CAT):
            codes = pc.index_in(t[c], value_set=cat_values[c])   # gia tri la -> null -> NaN
            X[:, len(NUM) + j] = codes.to_numpy(zero_copy_only=False)
        return X, t[TARGET].to_numpy(zero_copy_only=False).astype(np.float32)

    def build_cache_for_file(i):
        """Doc 1 file parquet, tach train/test theo hash(order_id) va ghi .npy (nhanh de doc lai nhieu luot)."""
        t = pq.read_table(files[i], columns=cols)
        h = spark_bucket(t[ID].to_numpy(zero_copy_only=False))
        keep = h < threshold
        is_test = (h % TEST_MOD) == 0
        for tag, mask in (("tr", keep & ~is_test), ("te", keep & is_test)):
            if mask.any():
                X, y = to_xy(t.filter(pa.array(mask)))
                np.save(cdir / f"{i:05d}_x{tag}.npy", X)
                np.save(cdir / f"{i:05d}_y{tag}.npy", y)

    def prepare_cache():
        cdir.mkdir(parents=True, exist_ok=True)
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) == meta:
            log(f"  dung lai cache co san o {cdir}")
            return
        free = shutil.disk_usage(cdir).free / 1e9
        log(f"[XGBoost] buoc 1/2: doc parquet 1 luot, ghi cache nhi phan vao {cdir} (o dia con {free:.0f} GB)")
        t0, last = time.time(), 0.0
        with ThreadPoolExecutor(max_workers=max(1, a.xgb_io_threads)) as ex:
            futs = [ex.submit(build_cache_for_file, i) for i in range(len(files))]
            for k, f in enumerate(futs, 1):
                f.result()
                if k % max(1, min(25, len(files) // 20)) == 0 or k == len(files) or time.time() - last > 30:
                    last = time.time()
                    el = time.time() - t0
                    log(f"  cache {k}/{len(files)} file ({100 * k / len(files):.0f}%) | {el / 60:.1f} phut | "
                        f"con khoang {el / k * (len(files) - k) / 60:.0f} phut{ram_text()}")
        marker.write_text(json.dumps(meta), encoding="utf-8")

    def load_cached(i, want_test):
        tag = "te" if want_test else "tr"
        fx = cdir / f"{i:05d}_x{tag}.npy"
        if not fx.exists():
            return None
        return np.load(fx), np.load(cdir / f"{i:05d}_y{tag}.npy")

    class FileIter(xgb.DataIter):
        def __init__(self, want_test, cache_prefix=None):
            self.want_test, self.i = want_test, 0
            self.name = "test" if want_test else "train"
            self.done_passes, self.t0, self.last_log = 0, time.time(), 0.0
            self.every = max(1, min(25, len(files) // 20))
            self.pool = ThreadPoolExecutor(max_workers=max(1, a.xgb_io_threads))
            self.futs, self.submitted = deque(), 0
            super().__init__(cache_prefix=cache_prefix)

        def _report(self):
            el = time.time() - self.t0
            eta = el / self.i * (len(files) - self.i)
            self.last_log = time.time()
            log(f"  doc {self.name} luot {self.done_passes + 1}: {self.i}/{len(files)} file "
                f"({100 * self.i / len(files):.0f}%) | {el / 60:.1f} phut | con khoang {eta / 60:.0f} phut{ram_text()}")

        def _fill(self):
            # doc truoc vai file song song (giai ma parquet nhả GIL) trong khi XGBoost xu ly file hien tai
            while len(self.futs) < a.xgb_io_threads * 2 and self.submitted < len(files):
                self.futs.append(self.pool.submit(load_cached, self.submitted, self.want_test))
                self.submitted += 1

        def next(self, input_data):
            while self.i < len(files):
                self._fill()
                item = self.futs.popleft().result()
                self.i += 1
                if self.i % self.every == 0 or self.i == len(files) or time.time() - self.last_log > 30:
                    self._report()
                if item is not None:
                    input_data(data=item[0], label=item[1], feature_names=FEATURE_NAMES)
                    return True
            self.done_passes += 1
            return False

        def reset(self):
            for f in self.futs:
                f.cancel()
            self.futs.clear()
            self.i, self.submitted = 0, 0
            self.t0, self.last_log = time.time(), 0.0

    threads = a.xgb_threads or (os.cpu_count() or 8)
    ticker = PhaseTicker(a.heartbeat_sec)
    ticker.start()
    ticker.phase = "buoc 1/2: tao cache nhi phan tu parquet"
    prepare_cache()
    ticker.phase = "buoc 2/2: dung DMatrix tu cache (XGBoost doc ~4 luot train + ~3 luot test)"
    log("[XGBoost] buoc 2/2: dung DMatrix tu cache; XGBoost tu doc nhieu luot nen se thay 'luot 1..4'")
    t0 = time.time()
    if a.xgb_external:
        tmp = Path(a.tmp_dir).resolve()
        tmp.mkdir(parents=True, exist_ok=True)
        dtrain = xgb.ExtMemQuantileDMatrix(FileIter(False, str(tmp / "xgb_train")),
                                           max_bin=a.xgb_max_bin, nthread=threads)
        dtest = xgb.ExtMemQuantileDMatrix(FileIter(True, str(tmp / "xgb_test")),
                                          max_bin=a.xgb_max_bin, nthread=threads, ref=dtrain)
    else:
        dtrain = xgb.QuantileDMatrix(FileIter(False), max_bin=a.xgb_max_bin, nthread=threads)
        dtest = xgb.QuantileDMatrix(FileIter(True), max_bin=a.xgb_max_bin, nthread=threads, ref=dtrain)
    n_train, n_test = dtrain.num_row(), dtest.num_row()
    log(f"  train {n_train:,} dong | test {n_test:,} dong | dung DMatrix mat {(time.time() - t0) / 60:.1f} phut")

    if not a.xgb_keep_cache and not a.xgb_external:   # DMatrix da nam trong RAM, cache khong con can
        shutil.rmtree(cdir, ignore_errors=True)
        log("  da xoa cache nhi phan (dung --xgb-keep-cache neu muon giu)")

    params = {"objective": "reg:squarederror", "tree_method": "hist", "device": a.xgb_device,
              "max_depth": a.xgb_depth, "eta": a.xgb_eta, "max_bin": a.xgb_max_bin,
              "seed": SEED, "nthread": threads}
    class Progress(xgb.callback.TrainingCallback):
        def __init__(self, total):
            self.total, self.t0 = total, time.time()

        def after_iteration(self, model, epoch, evals_log):
            done = epoch + 1
            if True:   # in moi vong: 1 vong tren 400M dong co the mat nhieu phut
                el = time.time() - self.t0
                log(f"  vong {done}/{self.total} | {el / 60:.1f} phut | "
                    f"con khoang {el / done * (self.total - done) / 60:.0f} phut{ram_text()}")
            return False

    log("  bat dau train")
    ticker.phase = "train XGBoost (moi vong co the mat vai phut)"
    t0 = time.time()
    booster = xgb.train(params, dtrain, num_boost_round=a.xgb_rounds, callbacks=[Progress(a.xgb_rounds)])
    train_time = time.time() - t0
    log(f"  train xong sau {train_time / 60:.1f} phut")

    ticker.phase = "du doan va tinh metrics tren tap test"
    t0 = time.time()
    pred = booster.predict(dtest).astype(np.float64)
    y = dtest.get_label().astype(np.float64)
    err = pred - y
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    var_y = float(np.var(y))
    rmse = mse ** 0.5
    r2 = 1.0 - mse / var_y if var_y else None
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), size=min(50_000, len(y)), replace=False)
    import pandas as pd
    sample_df = pd.DataFrame({TARGET: y[idx], "prediction": pred[idx]})
    eval_time = time.time() - t0

    gain = booster.get_score(importance_type="gain")
    tot = sum(gain.values()) or 1.0
    importance = {n: float(gain.get(n, 0.0)) / tot for n in FEATURE_NAMES}

    metrics = {"model": MODEL_LABELS["xgb"], "rows": n_train + n_test, "train_rows": n_train,
               "test_rows": n_test, "rmse": rmse, "mae": mae, "r2": r2,
               "train_time_sec": round(train_time, 2), "eval_time_sec": round(eval_time, 2)}
    out = dict(params, num_boost_round=a.xgb_rounds, split="hash(order_id) 80/20",
               sample_fraction=a.sample, external_memory=a.xgb_external,
               categorical_encoding="ordinal codes (khong dung native categorical)")
    env = {"xgboost": xgb.__version__, "threads": threads}
    save_outputs(out_dir, "xgb", a, metrics, out, importance, sample_df, env)
    ticker.stop()
    log(f"[XGBoost] RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")
    if not a.no_save_model:
        try:
            booster.save_model(str(out_dir / "model_xgboost.ubj"))
        except Exception as e:
            log(f"  Khong luu duoc model XGBoost (bo qua): {repr(e)[:150]}")


# ----------------------------------------------------------------------------------------
def argv_without_models():
    out, skip = [], False
    for tok in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if tok == "--models":
            skip = True
            continue
        if tok.startswith("--models="):
            continue
        out.append(tok)
    return out


def main():
    a = parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_DIRS:
            raise SystemExit(f"Mo hinh khong hop le: {m} (chon trong rf, gbt, xgb)")
    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)

    preflight(a)
    if a.only_preflight:
        return

    if len(models) > 1:
        # moi mo hinh 1 tien trinh rieng de giai phong RAM cua JVM truoc khi chay mo hinh tiep theo
        for m in models:
            log(f"===== BAT DAU {MODEL_LABELS[m]} =====")
            rc = subprocess.run([sys.executable, os.path.abspath(__file__), *argv_without_models(),
                                 "--models", m]).returncode
            log(f"===== {MODEL_LABELS[m]} ket thuc, ma thoat {rc} =====")
        summary = []
        for m in models:
            p = out_root / MODEL_DIRS[m] / "metrics.json"
            if p.exists():
                summary.append(json.load(open(p, encoding="utf-8")))
        write_json(out_root / "final_summary.json", summary)
        for s in summary:
            log(s)
        return

    # chay 1 mo hinh trong tien trinh nay
    m = models[0]
    out_dir = out_root / MODEL_DIRS[m]
    try:
        if m == "xgb":
            run_xgb(a, out_dir)
        else:
            run_spark_model(m, a, out_dir)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()