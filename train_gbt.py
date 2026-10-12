# -*- coding: utf-8 -*-
"""
train_gbt.py - Training Gradient-Boosted Trees Regression (Spark ML) theo "Quy ước training".

  * Đọc trực tiếp folder Parquet bằng Spark (không dùng Pandas cho dữ liệu lớn)
  * 28 feature + 1 target (total_price_usd) = 29 cột, kiểm tra dataset trước khi train
  * Chia Train/Test 80/20, seed 42; mọi bước preprocessing chỉ fit trên tập Train
  * Metrics: RMSE, MAE, R2, Training Time, Evaluation Time
  * Không hard-code số dòng / đường dẫn: dùng --input, --output (chạy 100M hay 500M giữ nguyên code)

Ví dụ (PowerShell):
  python train_gbt.py `
      --input "data/ecommerce_100m" `
      --output "results/gbt_100m" `
      --driver-memory 12g

Gộp nhiều folder/file Parquet để train chung một model:
  python train_gbt.py `
      --input "data/ecommerce_100m_0" "data/ecommerce_100m_400" `
             "data/ecommerce_100m_800" "data/ecommerce_100m_1200" `
             "data/ecommerce_100m_1600.parquet" `
      --output "results/gbt_500m"

Chạy thử trên sample nhỏ để tune trước:
  python train_gbt.py --input data/ecommerce_100m --output results/gbt_sample1pct --sample-fraction 0.01

Chỉ kiểm tra dataset:
  python train_gbt.py --input data/ecommerce_100m --output results/check --validate-only

Thư mục --output sẽ chứa: metrics.json, model_parameters.json, run_summary.txt,
feature_importance.json, predictions_sample.csv, model/, logs/
"""
import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# ----------------------------------------------------------------------------
# Schema theo quy ước (mục 3, 4)
# ----------------------------------------------------------------------------
TARGET = "total_price_usd"

INTEGER_FEATURES = [
    "order_year", "order_month", "order_day", "order_hour", "order_minute", "order_second",
    "age", "total_orders_by_customer", "product_reviews_count", "stock_quantity",
    "quantity", "discount_percent", "delivery_days",
]
FLOAT_FEATURES = ["customer_loyalty_score", "product_rating_avg", "unit_price_usd"]
CATEGORICAL_FEATURES = [
    "is_weekend", "gender", "customer_segment", "category", "sub_category", "brand",
    "shipping_method", "payment_method", "installment_plan",
    "campaign_source", "device_type", "traffic_source",
]
NUMERIC_FEATURES = INTEGER_FEATURES + FLOAT_FEATURES            # 16
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES              # 28
TRAIN_COLUMNS = FEATURES + [TARGET]                             # 29

# Giới hạn dữ liệu (mục 5): (min, max) - None = không giới hạn
RANGE_RULES = {
    "age": (1, 100), "quantity": (1, 100), "discount_percent": (0, 100), "delivery_days": (0, 60),
    "product_reviews_count": (0, None), "stock_quantity": (0, None),
    "customer_loyalty_score": (0, None), "product_rating_avg": (0, None),
    "unit_price_usd": (0, None), "total_price_usd": (0, None),
}
WEEKEND_VALUES = ("Yes", "No")
MODEL_NAME = "GBTRegressor"


# ----------------------------------------------------------------------------
# Tham số dòng lệnh (+ file config/gbt_config.json làm mặc định)
# ----------------------------------------------------------------------------
def parse_args():
    here = Path(__file__).resolve().parent
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(here / "config" / "gbt_config.json"))
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(parents=[pre], description="Train GBT Regression bằng Spark ML.")
    p.add_argument("--input", nargs="+", required=True,
                   help="Một hoặc nhiều folder/file Parquet đầu vào.")
    p.add_argument("--output", required=True, help="Thư mục kết quả của lần chạy (vd: results/gbt_100m).")
    p.add_argument("--model-dir", default=None, help="Nơi lưu model (mặc định: <output>/model).")
    p.add_argument("--log-dir", default=None, help="Nơi lưu log (mặc định: <output>/logs).")

    g = p.add_argument_group("Dữ liệu / chia tập")
    g.add_argument("--train-ratio", type=float, default=0.8, help="Tỷ lệ Train (mặc định 0.8).")
    g.add_argument("--seed", type=int, default=42, help="Seed chia Train/Test và seed model (mặc định 42).")
    g.add_argument("--sample-fraction", type=float, default=1.0,
                   help="<1.0: chỉ dùng một phần dataset (để tune thử). Mặc định 1.0 = full.")
    g.add_argument("--sample-seed", type=int, default=42, help="Seed lấy sample.")

    m = p.add_argument_group("Hyperparameter GBT")
    m.add_argument("--max-iter", type=int, default=20)
    m.add_argument("--max-depth", type=int, default=5)
    m.add_argument("--max-bins", type=int, default=32)
    m.add_argument("--step-size", type=float, default=0.1)
    m.add_argument("--subsampling-rate", type=float, default=1.0)
    m.add_argument("--min-instances-per-node", type=int, default=1)
    m.add_argument("--feature-subset-strategy", default="all")

    s = p.add_argument_group("Spark / tài nguyên (được phép đổi khi lên 500M)")
    s.add_argument("--master", default="local[*]")
    s.add_argument("--driver-memory", default=None, help="vd: 12g")
    s.add_argument("--executor-memory", default=None, help="vd: 8g (chỉ có ý nghĩa khi chạy cluster)")
    s.add_argument("--shuffle-partitions", type=int, default=None)
    s.add_argument("--app-name", default="GBT_Training")
    s.add_argument("--no-cache", action="store_true", help="Không persist dữ liệu đã preprocess.")

    o = p.add_argument_group("Tuỳ chọn khác")
    o.add_argument("--validate-only", action="store_true", help="Chỉ kiểm tra dataset rồi dừng.")
    o.add_argument("--no-save-model", action="store_true", help="Không lưu model.")
    o.add_argument("--predictions-sample-size", type=int, default=20000,
                   help="Số dòng Actual/Predicted lưu để vẽ biểu đồ.")
    o.add_argument("--run-note", default="", help="Ghi chú tự do (vd: 'tune sample 1%% - thử depth 6').")

    cfg_path = Path(pre_args.config)
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        valid = {a.dest for a in p._actions}
        p.set_defaults(**{k: v for k, v in cfg.items() if k in valid})
    args = p.parse_args()

    # Cho phép config cũ giữ nguyên input dưới dạng một chuỗi.
    if isinstance(args.input, str):
        args.input = [args.input]
    if not 0.0 < args.train_ratio < 1.0:
        p.error("--train-ratio phải nằm trong (0, 1)")
    if not 0.0 < args.sample_fraction <= 1.0:
        p.error("--sample-fraction phải nằm trong (0, 1]")
    return args


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
class ListHandler(logging.Handler):
    """Gom các WARNING/ERROR để ghi vào run_summary.txt (mục 27)."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(f"[{record.levelname}] {record.getMessage()}")


def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_gbt_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    warn_handler = ListHandler()
    root.addHandler(warn_handler)
    logging.captureWarnings(True)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    return logging.getLogger("train_gbt"), warn_handler, log_file


class SparkProgress:
    """Định kỳ báo tiến độ task của Spark và heartbeat cho bước hiện tại."""
    def __init__(self, spark, log, interval=15):
        self.spark = spark
        self.log = log
        self.interval = interval
        self.phase = "Khởi tạo Spark"
        self.phase_started = time.monotonic()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._report, daemon=True)

    def start(self):
        self.log.info("Tiến độ: %s", self.phase)
        self._thread.start()

    def set_phase(self, phase):
        self.phase = phase
        self.phase_started = time.monotonic()
        self.log.info("Tiến độ: %s", phase)

    def _report(self):
        tracker = self.spark.sparkContext.statusTracker()
        while not self._stop.wait(self.interval):
            elapsed = time.monotonic() - self.phase_started
            stage_progress = []
            try:
                for stage_id in tracker.getActiveStageIds():
                    stage = tracker.getStageInfo(stage_id)
                    if stage is None:
                        continue
                    total = stage.numTasks
                    completed = stage.numCompletedTasks
                    percent = (100.0 * completed / total) if total else 0.0
                    stage_progress.append(
                        f"stage {stage_id} {completed}/{total} tasks ({percent:.1f}%)"
                    )
            except Exception as e:
                self.log.warning("Không thể đọc tiến độ Spark: %s", e)
                return

            if stage_progress:
                self.log.info(
                    "Đang chạy: %s | %.0f giây | %s",
                    self.phase, elapsed, "; ".join(stage_progress),
                )
            else:
                self.log.info(
                    "Đang chạy: %s | %.0f giây | Spark chưa báo stage đang hoạt động",
                    self.phase, elapsed,
                )

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1)


def to_spark_path(p: str) -> str:
    if "://" in p or p.startswith("file:"):
        return p
    return Path(p).resolve().as_posix()


def has_parquet_magic(path: Path) -> bool:
    if path.stat().st_size < 8:
        return False
    with path.open("rb") as f:
        header = f.read(4)
        f.seek(-4, os.SEEK_END)
        footer = f.read(4)
    return header == b"PAR1" and footer == b"PAR1"


def expand_parquet_inputs(inputs):
    """Mở rộng folder local thành file Parquet để tránh Hadoop listStatus trên Windows."""
    groups = []
    for raw_path in inputs:
        if "://" in raw_path or raw_path.startswith("file:"):
            groups.append([raw_path])
            continue

        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy input: {path}")
        if path.is_dir():
            parquet_files = sorted(p for p in path.glob("*.parquet") if p.is_file())
            if not parquet_files:
                raise FileNotFoundError(f"Không tìm thấy file .parquet trực tiếp trong folder: {path}")
            selected_files = []
            for parquet_file in parquet_files:
                if has_parquet_magic(parquet_file):
                    selected_files.append(parquet_file)
                    continue

                backups = [
                    candidate
                    for candidate in path.rglob(parquet_file.name)
                    if candidate != parquet_file
                    and candidate.is_file()
                    and candidate.stat().st_size == parquet_file.stat().st_size
                    and has_parquet_magic(candidate)
                ]
                if len(backups) != 1:
                    raise ValueError(
                        f"Parquet bị hỏng hoặc không hợp lệ: {parquet_file}. "
                        f"Tìm thấy {len(backups)} bản dự phòng hợp lệ cùng tên/kích thước; "
                        "cần khôi phục file nguồn trước khi train."
                    )
                backup = backups[0]
                logging.getLogger("train_gbt").warning(
                    "File Parquet lỗi magic bytes: %s; dùng bản cùng tên/kích thước "
                    "hợp lệ ở thư mục con: %s",
                    parquet_file, backup,
                )
                selected_files.append(backup)
            groups.append([to_spark_path(str(p)) for p in selected_files])
        elif path.is_file():
            if path.suffix.lower() == ".parquet" and not has_parquet_magic(path):
                raise ValueError(f"File Parquet bị hỏng hoặc không hợp lệ: {path}")
            groups.append([to_spark_path(str(path))])
        else:
            raise ValueError(f"Input không phải file hoặc folder: {path}")
    return groups


def normalize_training_schema(df):
    """Chuẩn hóa kiểu các cột cần train trước khi union các Parquet nguồn."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset thiếu cột theo quy ước: {missing}")

    integer_types = {c: T.IntegerType() for c in INTEGER_FEATURES}
    numeric_types = {c: T.DoubleType() for c in FLOAT_FEATURES + [TARGET]}
    categorical_types = {c: T.StringType() for c in CATEGORICAL_FEATURES}
    target_types = {**integer_types, **numeric_types, **categorical_types}
    return df.select(
        *(F.col(c).cast(target_types[c]).alias(c) for c in TRAIN_COLUMNS)
    )


def write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def total_ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 ** 3, 1)
    except Exception:
        pass
    try:  # Linux / Colab
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 ** 2, 1)
    except Exception:
        pass
    return None


def environment_info(spark):
    import pyspark
    try:
        java_version = spark._jvm.System.getProperty("java.version")
    except Exception:
        java_version = None
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "java_version": java_version,
        "spark_version": spark.version,
        "pyspark_version": pyspark.__version__,
        "spark_worker_python": spark.sparkContext.pythonExec,
        "hadoop_home": os.environ.get("HADOOP_HOME"),
        "model_library": f"pyspark.ml.regression.GBTRegressor (Spark ML {spark.version})",
        "cpu_count": os.cpu_count(),
        "ram_gb": total_ram_gb(),
        "default_parallelism": spark.sparkContext.defaultParallelism,
    }


def build_spark(args):
    if os.name == "nt":
        hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
        if not hadoop_home:
            default_home = Path("C:/hadoop")
            if (default_home / "bin" / "winutils.exe").is_file():
                hadoop_home = str(default_home)
        if hadoop_home:
            os.environ.setdefault("HADOOP_HOME", hadoop_home)
            os.environ.setdefault("hadoop.home.dir", hadoop_home)
            hadoop_bin = str(Path(hadoop_home) / "bin")
            path_entries = os.environ.get("PATH", "").split(os.pathsep)
            if Path(hadoop_bin).is_dir() and hadoop_bin.casefold() not in {
                entry.casefold() for entry in path_entries
            }:
                os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")
    if args.master.startswith("local") and not os.environ.get("PYSPARK_PYTHON"):
        os.environ["PYSPARK_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession
    b = SparkSession.builder.appName(args.app_name).master(args.master)
    if args.driver_memory:
        b = b.config("spark.driver.memory", args.driver_memory)
    if args.executor_memory:
        b = b.config("spark.executor.memory", args.executor_memory)
    if args.shuffle_partitions:
        b = b.config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    # Phòng khi file Parquet cũ ghi timestamp nanosecond (Spark không đọc được kiểu này).
    b = b.config("spark.sql.legacy.parquet.nanosAsLong", "true")
    b = b.config("spark.network.timeout", "3600s").config("spark.executor.heartbeatInterval", "600s")
    b = b.config("spark.ui.showConsoleProgress", "true")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ----------------------------------------------------------------------------
# Kiểm tra dataset trước training (mục 5)
# ----------------------------------------------------------------------------
def validate_dataset(df, log):
    """Trả về (report, violations). Không tự sửa dữ liệu - chỉ báo cáo."""
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    violations = []
    report = {}

    missing = [c for c in TRAIN_COLUMNS if c not in df.columns]
    if missing:
        violations.append(f"Thiếu cột (sai tên hoặc không có): {missing}")
        return report, violations
    df = df.select(*TRAIN_COLUMNS)
    report["columns"] = len(df.columns)

    int_types = (T.ByteType, T.ShortType, T.IntegerType, T.LongType)
    float_types = (T.FloatType, T.DoubleType)
    schema = {f.name: f.dataType for f in df.schema.fields}
    for c in INTEGER_FEATURES:
        if not isinstance(schema[c], int_types):
            violations.append(f"{c}: kiểu {schema[c].simpleString()} (cần Integer)")
    for c in FLOAT_FEATURES + [TARGET]:
        if not isinstance(schema[c], float_types):
            violations.append(f"{c}: kiểu {schema[c].simpleString()} (cần Float/Double)")
    for c in CATEGORICAL_FEATURES:
        if not isinstance(schema[c], T.StringType):
            violations.append(f"{c}: kiểu {schema[c].simpleString()} (cần String/Categorical)")
    if violations:
        return report, violations

    inf = float("inf")
    exprs = [F.count(F.lit(1)).alias("__rows")]
    for c in TRAIN_COLUMNS:
        exprs.append(F.sum(F.col(c).isNull().cast("long")).alias(f"{c}__null"))
    for c in INTEGER_FEATURES + FLOAT_FEATURES + [TARGET]:
        exprs.append(F.min(c).alias(f"{c}__min"))
        exprs.append(F.max(c).alias(f"{c}__max"))
    for c in FLOAT_FEATURES + [TARGET]:
        exprs.append(F.sum(F.isnan(F.col(c)).cast("long")).alias(f"{c}__nan"))
        exprs.append(F.sum(F.col(c).isin(inf, -inf).cast("long")).alias(f"{c}__inf"))
    for c in CATEGORICAL_FEATURES:
        exprs.append(F.sum((F.trim(F.col(c)) == "").cast("long")).alias(f"{c}__empty"))
        exprs.append(F.approx_count_distinct(c).alias(f"{c}__distinct"))
    exprs.append(F.sum((~F.col("is_weekend").isin(*WEEKEND_VALUES)).cast("long")).alias("is_weekend__invalid"))

    log.info("Đang quét dataset để kiểm tra (1 lượt)...")
    r = df.agg(*exprs).first().asDict()

    rows = r["__rows"]
    report["rows"] = rows
    if rows == 0:
        violations.append("Dataset rỗng (0 dòng).")
        return report, violations

    report["nulls"] = {c: r[f"{c}__null"] or 0 for c in TRAIN_COLUMNS}
    report["nan"] = {c: r[f"{c}__nan"] or 0 for c in FLOAT_FEATURES + [TARGET]}
    report["inf"] = {c: r[f"{c}__inf"] or 0 for c in FLOAT_FEATURES + [TARGET]}
    report["min"] = {c: r[f"{c}__min"] for c in INTEGER_FEATURES + FLOAT_FEATURES + [TARGET]}
    report["max"] = {c: r[f"{c}__max"] for c in INTEGER_FEATURES + FLOAT_FEATURES + [TARGET]}
    report["categorical_approx_distinct"] = {c: r[f"{c}__distinct"] for c in CATEGORICAL_FEATURES}

    for c, n in report["nulls"].items():
        if n:
            violations.append(f"{c}: {n:,} giá trị NULL")
    for c, n in report["nan"].items():
        if n:
            violations.append(f"{c}: {n:,} giá trị NaN")
    for c, n in report["inf"].items():
        if n:
            violations.append(f"{c}: {n:,} giá trị Infinity")
    for c in CATEGORICAL_FEATURES:
        if r[f"{c}__empty"]:
            violations.append(f"{c}: {r[f'{c}__empty']:,} giá trị chuỗi rỗng")
    if r["is_weekend__invalid"]:
        violations.append(f"is_weekend: {r['is_weekend__invalid']:,} giá trị ngoài {WEEKEND_VALUES}")
    for c, (lo, hi) in RANGE_RULES.items():
        mn, mx = report["min"][c], report["max"][c]
        if mn is not None and lo is not None and mn < lo:
            violations.append(f"{c}: min = {mn} < {lo}")
        if mx is not None and hi is not None and mx > hi:
            violations.append(f"{c}: max = {mx} > {hi}")
    return report, violations


# ----------------------------------------------------------------------------
# run_summary.txt
# ----------------------------------------------------------------------------
def write_run_summary(path: Path, ctx: dict, status: str, error: str = None):
    g = lambda k: ctx.get(k, "N/A")
    lines = [
        "=" * 70,
        f"RUN SUMMARY - {MODEL_NAME}",
        "=" * 70,
        f"Trạng thái        : {status}",
        f"Thời điểm chạy    : {g('run_timestamp')}",
        f"Ghi chú           : {g('run_note') or '-'}",
        "",
        f"Model             : {MODEL_NAME} (Gradient-Boosted Trees Regression)",
        f"Dataset           : {g('input')}",
        f"Dataset rows      : {g('dataset_rows')}" + (f"  (sample {ctx.get('sample_fraction')})" if ctx.get("is_sample") else ""),
        f"Train rows        : {g('train_rows')}",
        f"Test rows         : {g('test_rows')}",
        f"Feature count     : {len(FEATURES)}",
        f"Feature list      : {', '.join(FEATURES)}",
        f"Target            : {TARGET}",
        f"Train/Test ratio  : {g('train_ratio')} / {g('test_ratio')}",
        f"Seed              : {g('seed')}",
        "",
        f"Model parameters  : {json.dumps(ctx.get('model_params', {}), ensure_ascii=False)}",
        "",
        f"Spark master      : {g('master')}",
        f"Driver memory     : {g('driver_memory')}",
        f"Executor memory   : {g('executor_memory')}",
        f"Partitions        : input={g('input_partitions')}, shuffle={g('shuffle_partitions')}",
        f"Machine/Cluster   : {json.dumps(ctx.get('environment', {}), ensure_ascii=False)}",
        "Spark configuration:",
    ]
    for k, v in sorted(ctx.get("spark_conf", [])):
        lines.append(f"    {k} = {v}")
    lines += [
        "",
        f"RMSE              : {g('rmse')}",
        f"MAE               : {g('mae')}",
        f"R2                : {g('r2')}",
        f"Training time (s) : {g('training_seconds')}",
        f"Evaluation time(s): {g('evaluation_seconds')}",
        f"Preprocessing (s) : {g('preprocessing_seconds')}",
        f"Total time (s)    : {g('total_seconds')}",
        "",
        f"Model saved       : {g('model_saved')}",
        f"Cardinality (cat) : {json.dumps(ctx.get('cardinality', {}), ensure_ascii=False)}",
        "Top feature importance:",
    ]
    for name, val in ctx.get("top_importance", []):
        lines.append(f"    {name:30s} {val:.4f}")
    if ctx.get("validation_note"):
        lines += ["", f"Kiểm tra dataset  : {ctx['validation_note']}"]
    lines += ["", "Warning / lỗi quan trọng:"]
    msgs = ctx.get("warnings", [])
    lines += [f"    {m}" for m in msgs] if msgs else ["    (không có)"]
    if error:
        lines += ["", "LỖI:", error]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir) if args.model_dir else out_dir / "model"
    log_dir = Path(args.log_dir) if args.log_dir else out_dir / "logs"
    log, warn_handler, log_file = setup_logging(log_dir)

    ctx = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_note": args.run_note,
        "input": args.input[0] if len(args.input) == 1 else args.input,
        "train_ratio": args.train_ratio, "test_ratio": round(1 - args.train_ratio, 10),
        "seed": args.seed, "master": args.master,
        "driver_memory": args.driver_memory or "mặc định Spark",
        "executor_memory": args.executor_memory or "mặc định / không có",
        "shuffle_partitions": args.shuffle_partitions or "mặc định Spark",
        "is_sample": args.sample_fraction < 1.0, "sample_fraction": args.sample_fraction,
        "model_saved": False, "warnings": [],
    }
    spark = None
    progress = None
    t_start = time.perf_counter()
    try:
        spark_input_groups = expand_parquet_inputs(args.input)
        log.info(
            "Đã mở rộng %d input thành %d file Parquet cụ thể",
            len(args.input), sum(len(group) for group in spark_input_groups),
        )

        from pyspark import StorageLevel
        from pyspark.ml import Pipeline, PipelineModel
        from pyspark.ml.feature import StringIndexer, VectorAssembler
        from pyspark.ml.regression import GBTRegressor
        from pyspark.sql import functions as F

        log.info("Khởi tạo Spark (master=%s)", args.master)
        spark = build_spark(args)
        progress = SparkProgress(spark, log)
        progress.start()
        ctx["environment"] = environment_info(spark)
        ctx["spark_conf"] = spark.sparkContext.getConf().getAll()
        log.info("Môi trường: %s", json.dumps(ctx["environment"], ensure_ascii=False))

        # 1. Parquet
        progress.set_phase("Mở các nguồn Parquet và đọc schema")
        normalized_sources = []
        for index, file_group in enumerate(spark_input_groups, start=1):
            log.info(
                "Đọc nguồn %d/%d (%d file Parquet)",
                index, len(spark_input_groups), len(file_group),
            )
            source = spark.read.parquet(*file_group)
            normalized_sources.append(normalize_training_schema(source))
        df = normalized_sources[0]
        for source in normalized_sources[1:]:
            df = df.unionByName(source)
        ctx["input_partitions"] = df.rdd.getNumPartitions()
        log.info("Đã đọc schema; số partition đầu vào: %s", ctx["input_partitions"])

        # 2. Select 29 cột (+ kiểm tra tên cột)
        df = df.select(*TRAIN_COLUMNS)

        if args.sample_fraction < 1.0:
            log.info("Dùng SAMPLE %.4f (seed=%d) - chỉ để tune, không phải kết quả full", args.sample_fraction, args.sample_seed)
            df = df.sample(withReplacement=False, fraction=args.sample_fraction, seed=args.sample_seed)

        # 3. Validate
        progress.set_phase("Quét và kiểm tra dữ liệu")
        report, violations = validate_dataset(df, log)
        write_json(out_dir / "validation_report.json", {"report": report, "violations": violations})
        if violations:
            for v in violations:
                log.error("VI PHẠM: %s", v)
            ctx["warnings"] = warn_handler.messages
            ctx["validation_note"] = f"{len(violations)} vi phạm - xem validation_report.json"
            write_run_summary(out_dir / "run_summary.txt", ctx, "DỪNG: dataset vi phạm quy định")
            log.error("Dataset vi phạm quy định -> dừng. Báo lại nhóm trước khi sửa dữ liệu.")
            return 2
        dataset_rows = report["rows"]
        ctx["dataset_rows"] = dataset_rows
        ctx["validation_note"] = "đạt (29 cột, kiểu dữ liệu, NULL/NaN/Infinity, giới hạn, categorical)"
        log.info("Dataset hợp lệ: %s dòng, %d cột", f"{dataset_rows:,}", report["columns"])
        log.info("Cardinality gần đúng: %s", report["categorical_approx_distinct"])
        if args.validate_only:
            ctx["warnings"] = warn_handler.messages
            write_run_summary(out_dir / "run_summary.txt", ctx, "HOÀN THÀNH (chỉ validate)")
            return 0

        # 4. Train/Test split (seed theo quy ước)
        train_df, test_df = df.randomSplit([args.train_ratio, 1 - args.train_ratio], seed=args.seed)

        # 5. Preprocessing: chỉ FIT trên Train
        t0 = time.perf_counter()
        idx_cols = [f"{c}_idx" for c in CATEGORICAL_FEATURES]
        indexer = StringIndexer(inputCols=CATEGORICAL_FEATURES, outputCols=idx_cols, handleInvalid="keep")
        assembler = VectorAssembler(inputCols=NUMERIC_FEATURES + idx_cols, outputCol="features", handleInvalid="error")
        progress.set_phase("Fit bộ tiền xử lý trên tập Train")
        pre_model = Pipeline(stages=[indexer, assembler]).fit(train_df)

        labels = pre_model.stages[0].labelsArray
        cardinality = {c: len(l) for c, l in zip(CATEGORICAL_FEATURES, labels)}
        ctx["cardinality"] = cardinality
        need_bins = max(cardinality.values()) + 1  # +1 bucket cho giá trị lạ (handleInvalid="keep")
        log.info("Cardinality (exact, từ Train): %s", cardinality)
        if args.max_bins < need_bins:
            raise SystemExit(f"--max-bins={args.max_bins} < số category lớn nhất + 1 = {need_bins}. "
                             f"Tăng --max-bins lên >= {need_bins} (và ghi lại thay đổi parameter).")

        train_feat = pre_model.transform(train_df).select("features", TARGET)
        test_feat = pre_model.transform(test_df).select("features", TARGET)
        if not args.no_cache:
            train_feat = train_feat.persist(StorageLevel.MEMORY_AND_DISK)
            test_feat = test_feat.persist(StorageLevel.MEMORY_AND_DISK)
        progress.set_phase("Biến đổi và cache dữ liệu Train/Test")
        train_rows = train_feat.count()
        test_rows = test_feat.count()
        preprocessing_seconds = time.perf_counter() - t0
        ctx.update(train_rows=train_rows, test_rows=test_rows, preprocessing_seconds=round(preprocessing_seconds, 3))
        log.info("Train rows: %s | Test rows: %s | preprocessing %.1fs", f"{train_rows:,}", f"{test_rows:,}", preprocessing_seconds)

        # 6. Train GBT
        gbt = GBTRegressor(
            featuresCol="features", labelCol=TARGET, predictionCol="prediction",
            maxIter=args.max_iter, maxDepth=args.max_depth, maxBins=args.max_bins,
            stepSize=args.step_size, subsamplingRate=args.subsampling_rate,
            minInstancesPerNode=args.min_instances_per_node,
            featureSubsetStrategy=args.feature_subset_strategy, seed=args.seed,
        )
        model_params = {
            "model": MODEL_NAME, "maxIter": args.max_iter, "maxDepth": args.max_depth,
            "maxBins": args.max_bins, "stepSize": args.step_size, "seed": args.seed,
            "subsamplingRate": args.subsampling_rate, "minInstancesPerNode": args.min_instances_per_node,
            "featureSubsetStrategy": args.feature_subset_strategy,
        }
        ctx["model_params"] = model_params
        log.info("Bắt đầu training GBT: %s", json.dumps(model_params))
        progress.set_phase(f"Huấn luyện GBT ({args.max_iter} vòng lặp tối đa)")
        t0 = time.perf_counter()
        gbt_model = gbt.fit(train_feat)
        training_seconds = time.perf_counter() - t0
        log.info("Training xong: %.1fs", training_seconds)

        # 7. Evaluation (1 lượt: RMSE, MAE, R2)
        progress.set_phase("Đánh giá model trên tập Test")
        t0 = time.perf_counter()
        pred = gbt_model.transform(test_feat)
        err = F.col("prediction") - F.col(TARGET)
        m = pred.agg(
            F.count(F.lit(1)).alias("n"),
            F.avg(err * err).alias("mse"),
            F.avg(F.abs(err)).alias("mae"),
            F.var_pop(TARGET).alias("var_y"),
        ).first()
        evaluation_seconds = time.perf_counter() - t0
        rmse = math.sqrt(m["mse"])
        mae = m["mae"]
        r2 = (1.0 - m["mse"] / m["var_y"]) if m["var_y"] and m["var_y"] > 0 else None
        log.info("RMSE=%.6f | MAE=%.6f | R2=%s | eval %.1fs", rmse, mae, r2, evaluation_seconds)

        # 8. Lưu kết quả
        total_seconds = time.perf_counter() - t_start
        metrics = {
            "model": MODEL_NAME, "dataset_rows": dataset_rows, "train_rows": train_rows,
            "test_rows": test_rows, "feature_count": len(FEATURES), "target": TARGET,
            "train_ratio": args.train_ratio, "test_ratio": round(1 - args.train_ratio, 10),
            "seed": args.seed, "rmse": rmse, "mae": mae, "r2": r2,
            "training_seconds": round(training_seconds, 3),
            "evaluation_seconds": round(evaluation_seconds, 3),
            # --- thông tin bổ sung ---
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "total_seconds": round(total_seconds, 3),
            "is_sample": args.sample_fraction < 1.0, "sample_fraction": args.sample_fraction,
            "input": ctx["input"], "run_timestamp": ctx["run_timestamp"],
        }
        write_json(out_dir / "metrics.json", metrics)
        all_params = {p.name: v for p, v in gbt_model.extractParamMap().items()}
        write_json(out_dir / "model_parameters.json", {**model_params, "all_spark_params": all_params})

        importances = gbt_model.featureImportances.toArray().tolist()
        names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        imp = sorted(zip(names, importances), key=lambda x: -x[1])
        write_json(out_dir / "feature_importance.json", [{"feature": n, "importance": v} for n, v in imp])
        ctx["top_importance"] = imp[:10]

        try:  # sample Actual vs Predicted để trực quan hóa (không cần Pandas)
            progress.set_phase("Lấy mẫu prediction để xuất CSV")
            n_s = args.predictions_sample_size
            frac = min(1.0, (n_s * 1.5) / max(test_rows, 1))
            rows = pred.select(TARGET, "prediction").sample(False, frac, args.seed).limit(n_s).collect()
            with open(out_dir / "predictions_sample.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["actual", "predicted", "residual"])
                for r in rows:
                    w.writerow([r[0], r[1], r[1] - r[0]])
        except Exception as e:  # noqa
            log.warning("Không lưu được predictions_sample.csv: %s", e)

        if not args.no_save_model:
            try:
                progress.set_phase("Lưu model")
                final_model = PipelineModel(stages=list(pre_model.stages) + [gbt_model])
                final_model.write().overwrite().save(to_spark_path(str(model_dir)))
                ctx["model_saved"] = str(model_dir)
                log.info("Đã lưu model: %s", model_dir)
            except Exception as e:  # noqa
                log.warning("Không lưu được model (%s). Trên Windows thường cần cấu hình HADOOP_HOME/winutils; "
                            "có thể chạy lại với --no-save-model.", e)
        else:
            ctx["model_saved"] = "không (--no-save-model)"

        ctx.update(rmse=rmse, mae=mae, r2=r2, training_seconds=round(training_seconds, 3),
                   evaluation_seconds=round(evaluation_seconds, 3), total_seconds=round(total_seconds, 3),
                   warnings=warn_handler.messages)
        write_run_summary(out_dir / "run_summary.txt", ctx, "HOÀN THÀNH")

        log.info("-" * 70)
        log.info("Model | Rows | Train Rows | Test Rows | RMSE | MAE | R2 | Train Time | Eval Time")
        log.info("%s | %s | %s | %s | %.4f | %.4f | %s | %.1fs | %.1fs", MODEL_NAME, f"{dataset_rows:,}",
                 f"{train_rows:,}", f"{test_rows:,}", rmse, mae,
                 f"{r2:.4f}" if r2 is not None else "N/A", training_seconds, evaluation_seconds)
        log.info("Kết quả lưu tại: %s", out_dir.resolve())
        return 0

    except SystemExit as e:
        if e.code not in (0, None) and not isinstance(e.code, int):
            log.error("%s", e.code)
            ctx["warnings"] = warn_handler.messages
            write_run_summary(out_dir / "run_summary.txt", ctx, "THẤT BẠI", error=str(e.code))
            return 1
        raise
    except Exception as e:  # noqa
        log.exception("Lỗi khi chạy: %s", e)
        ctx["warnings"] = warn_handler.messages
        write_run_summary(out_dir / "run_summary.txt", ctx, "THẤT BẠI", error=repr(e))
        return 1
    finally:
        if progress is not None:
            progress.stop()
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())
