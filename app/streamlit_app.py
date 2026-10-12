"""Vietnamese Streamlit interface for the trained ecommerce price model."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.predictor import (
    CATEGORICAL_FEATURES,
    build_feature_vector,
    load_categories,
    load_category_indexer,
    load_model,
    load_spark_tree_model,
    predict_price,
    predict_spark_price,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "full_500m"
CATEGORIES_PATH = Path(__file__).resolve().parent / "category_values.json"
MODEL_CONFIGS = {
    "random_forest": {
        "label": "Random Forest",
        "directory": RESULTS_DIR / "random_forest",
        "kind": "spark",
    },
    "gbt": {
        "label": "GBT",
        "directory": RESULTS_DIR / "gbt",
        "kind": "spark",
    },
    "xgboost": {
        "label": "XGBoost",
        "directory": RESULTS_DIR / "xgboost",
        "kind": "xgboost",
    },
}

CATEGORY_LABELS = {
    "gender": "Giới tính",
    "customer_segment": "Phân khúc khách hàng",
    "category": "Danh mục sản phẩm",
    "sub_category": "Danh mục con",
    "brand": "Thương hiệu",
    "shipping_method": "Phương thức giao hàng",
    "payment_method": "Phương thức thanh toán",
    "installment_plan": "Gói trả góp",
    "campaign_source": "Nguồn chiến dịch",
    "device_type": "Thiết bị",
    "traffic_source": "Nguồn truy cập",
}

st.set_page_config(
    page_title="Order Intelligence | Dự đoán đơn hàng",
    page_icon="🛍️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17251f;
        --muted: #76847d;
        --green: #156b4c;
        --green-dark: #0d5038;
        --line: #e8ede9;
        --surface: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(219, 239, 226, .52), transparent 30%),
            #f6f8f6;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.25rem; }
    .block-container {
        max-width: 1200px;
        padding: 2.3rem 2rem 4rem;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.35rem 2.7rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, .14);
        border-radius: 24px;
        background: linear-gradient(118deg, #103b2c 0%, #15543c 58%, #21684c 100%);
        box-shadow: 0 18px 48px rgba(19, 65, 45, .15);
        color: white;
    }

    .hero::after {
        position: absolute;
        top: -115px;
        right: -35px;
        width: 360px;
        height: 360px;
        border: 1px solid rgba(255, 255, 255, .11);
        border-radius: 50%;
        box-shadow: 0 0 0 38px rgba(255, 255, 255, .035),
                    0 0 0 78px rgba(255, 255, 255, .025);
        content: "";
        pointer-events: none;
    }

    .hero-eyebrow {
        margin: 0 0 .8rem;
        color: #b9ddc7;
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .17em;
        text-transform: uppercase;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 0;
        color: #fff;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.18;
    }

    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 680px;
        margin: .85rem 0 0;
        color: rgba(245, 250, 246, .77);
        font-size: .98rem;
        line-height: 1.7;
    }

    .hero-chip {
        position: relative;
        z-index: 1;
        display: inline-flex;
        align-items: center;
        gap: .48rem;
        margin-top: 1.25rem;
        padding: .42rem .8rem;
        border: 1px solid rgba(221, 249, 231, .23);
        border-radius: 999px;
        background: rgba(231, 255, 239, .09);
        color: #e6f6eb;
        font-size: .78rem;
        font-weight: 600;
    }

    .section-kicker {
        margin: .25rem 0 .7rem;
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
    }

    [data-testid="stMetric"] {
        min-height: 118px;
        padding: 1.05rem 1.2rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--surface);
        box-shadow: 0 5px 18px rgba(31, 55, 41, .035);
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: .81rem;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.2rem, 2vw, 1.75rem);
        font-weight: 700;
    }

    .model-note {
        margin: .65rem 0 1.7rem;
        color: var(--muted);
        font-size: .83rem;
    }

    [data-testid="stForm"] {
        padding: 1.7rem 1.8rem 1.45rem;
        border: 1px solid var(--line);
        border-radius: 20px;
        background: var(--surface);
        box-shadow: 0 9px 32px rgba(31, 55, 41, .055);
    }

    .form-title {
        margin: 0;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -.035em;
    }

    .form-description {
        margin: .35rem 0 1.25rem;
        color: var(--muted);
        font-size: .87rem;
    }

    .group-title {
        margin: .25rem 0 .25rem;
        color: var(--green-dark);
        font-family: 'Manrope', sans-serif;
        font-size: .97rem;
        font-weight: 800;
        letter-spacing: -.015em;
    }

    .group-copy {
        margin: 0 0 .85rem;
        color: var(--muted);
        font-size: .78rem;
        line-height: 1.55;
    }

    [data-testid="stWidgetLabel"] p {
        color: #46564d;
        font-size: .81rem;
        font-weight: 600;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input {
        border-color: #e2e9e3;
        border-radius: 10px;
        background: #fbfcfb;
    }

    [data-baseweb="select"] > div {
        border-color: #e2e9e3;
        border-radius: 10px;
        background: #fbfcfb;
    }

    [data-testid="stFormSubmitButton"] button {
        min-height: 3.05rem;
        margin-top: .65rem;
        border: 0;
        border-radius: 11px;
        background: linear-gradient(110deg, #176d4d, #21845d);
        box-shadow: 0 7px 16px rgba(21, 107, 76, .2);
        color: white;
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        border: 0;
        background: linear-gradient(110deg, #125d41, #1d7653);
        box-shadow: 0 10px 20px rgba(21, 107, 76, .25);
        color: white;
    }

    .prediction-card {
        margin-top: 1.15rem;
        padding: 1.35rem 1.55rem;
        border: 1px solid #cde8d7;
        border-radius: 16px;
        background: linear-gradient(115deg, #eff8f1, #f8fcf8);
    }

    .prediction-label {
        margin: 0 0 .25rem;
        color: #577363;
        font-size: .8rem;
        font-weight: 600;
    }

    .prediction-value {
        margin: 0;
        color: #125b3d;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.7rem, 4vw, 2.4rem);
        font-weight: 800;
        letter-spacing: -.045em;
    }

    .prediction-footnote {
        margin: .5rem 0 0;
        color: #718378;
        font-size: .79rem;
        line-height: 1.6;
    }

    [data-testid="stExpander"] {
        margin-top: 1.1rem;
        border: 1px solid var(--line);
        border-radius: 15px;
        background: var(--surface);
    }

    @media (max-width: 700px) {
        .block-container { padding: 1.1rem 1rem 2.5rem; }
        .hero { padding: 1.7rem 1.45rem; border-radius: 19px; }
        [data-testid="stForm"] { padding: 1.2rem 1rem 1rem; }
    }
    </style>

    <section class="hero">
        <p class="hero-eyebrow">E-commerce · Machine learning</p>
        <h1>Order Intelligence</h1>
        <p class="hero-copy">
            So sánh dự đoán giá trị đơn hàng từ Random Forest, GBT và XGBoost,
            được huấn luyện trên 500 triệu giao dịch thương mại điện tử.
        </p>
        <span class="hero-chip">✦ &nbsp; Random Forest · GBT · XGBoost</span>
    </section>
    """,
    unsafe_allow_html=True,
)

if not CATEGORIES_PATH.is_file():
    st.error(f"Không tìm thấy danh mục mã hóa: `{CATEGORIES_PATH}`")
    st.info("Xem `app/README.md` để biết cấu trúc file và cách khởi chạy.")
    st.stop()


def model_artifact_path(model_key: str) -> Path:
    config = MODEL_CONFIGS[model_key]
    if config["kind"] == "xgboost":
        return config["directory"] / "model_xgboost.ubj"
    return config["directory"] / "model"


def model_is_available(model_key: str) -> bool:
    config = MODEL_CONFIGS[model_key]
    if not model_artifact_path(model_key).exists():
        return False
    if config["kind"] == "spark":
        return (config["directory"] / "category_indexer.json").is_file()
    return True


def configure_spark_environment():
    if os.name != "nt":
        return

    hadoop_home = os.environ.get("HADOOP_HOME") or os.environ.get("hadoop.home.dir")
    if not hadoop_home:
        default_home = Path("C:/hadoop")
        if (default_home / "bin" / "winutils.exe").is_file():
            hadoop_home = str(default_home)
    if not hadoop_home or not (Path(hadoop_home) / "bin" / "winutils.exe").is_file():
        raise RuntimeError(
            "Dự đoán bằng Random Forest/GBT trên Windows cần HADOOP_HOME trỏ "
            "đến thư mục Hadoop có bin\\winutils.exe."
        )

    os.environ.setdefault("HADOOP_HOME", hadoop_home)
    os.environ.setdefault("hadoop.home.dir", hadoop_home)
    hadoop_bin = str(Path(hadoop_home) / "bin")
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if hadoop_bin.casefold() not in {entry.casefold() for entry in path_entries}:
        os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")


AVAILABLE_MODELS = [
    key for key in MODEL_CONFIGS if model_is_available(key)
]
UNAVAILABLE_MODELS = [
    key for key in MODEL_CONFIGS if not model_is_available(key)
]

if UNAVAILABLE_MODELS:
    unavailable_labels = ", ".join(
        MODEL_CONFIGS[key]["label"] for key in UNAVAILABLE_MODELS
    )
    st.warning(
        f"Chưa thể dự đoán bằng {unavailable_labels}: thiếu model hoặc mapping "
        "categorical lưu cùng model. Hãy train lại các model Spark bằng "
        "`train_all_500m.py` để tạo mapping chính xác."
    )
if not AVAILABLE_MODELS:
    st.error("Không có model nào sẵn sàng để dự đoán.")
    st.stop()


@st.cache_resource
def get_model(model_path: str):
    return load_model(Path(model_path))


@st.cache_resource
def get_spark_model(model_path: str, model_type: str):
    configure_spark_environment()
    get_spark_session()
    return load_spark_tree_model(Path(model_path), model_type)


@st.cache_resource
def get_spark_session():
    from pyspark.sql import SparkSession

    configure_spark_environment()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("OrderIntelligenceInference")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


@st.cache_data
def get_categories(categories_path: str):
    return load_categories(Path(categories_path))


@st.cache_data
def get_category_orders(mapping_path: str):
    return load_category_indexer(Path(mapping_path))


metrics_by_model = {}
importance_by_model = {}
for model_key, config in MODEL_CONFIGS.items():
    metrics_path = config["directory"] / "metrics.json"
    importance_path = config["directory"] / "feature_importance.json"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics_by_model[model_key] = json.load(metrics_file)
    if importance_path.is_file():
        with importance_path.open(encoding="utf-8") as importance_file:
            importance_by_model[model_key] = json.load(importance_file)

st.markdown('<p class="section-kicker">Hiệu suất mô hình</p>', unsafe_allow_html=True)
if metrics_by_model:
    metric_rows = []
    for model_key, metrics in metrics_by_model.items():
        metric_rows.append(
            {
                "Mô hình": MODEL_CONFIGS[model_key]["label"],
                "Dữ liệu": f'{metrics["rows"] / 1_000_000:.0f} triệu dòng',
                "MAE (USD)": metrics["mae"],
                "RMSE (USD)": metrics["rmse"],
                "R²": metrics["r2"],
            }
        )
    st.dataframe(
        pd.DataFrame(metric_rows).set_index("Mô hình"),
        use_container_width=True,
    )
    st.markdown(
        '<p class="model-note">Chỉ số đánh giá trên tập kiểm tra; kết quả thực tế '
        'có thể thay đổi tùy dữ liệu đầu vào.</p>',
        unsafe_allow_html=True,
    )

categories = get_categories(str(CATEGORIES_PATH))

with st.form("prediction_form"):
    st.markdown('<h2 class="form-title">Tạo dự đoán</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="form-description">Chọn một hoặc nhiều mô hình, sau đó nhập '
        'thông tin để so sánh tổng giá trị đơn hàng dự đoán.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("**Mô hình dự đoán** · Có thể chọn nhiều mô hình")
    if UNAVAILABLE_MODELS:
        available_labels = ", ".join(
            MODEL_CONFIGS[key]["label"] for key in AVAILABLE_MODELS
        )
        unavailable_labels = ", ".join(
            MODEL_CONFIGS[key]["label"] for key in UNAVAILABLE_MODELS
        )
        st.info(
            f"Hiện chỉ chọn được {available_labels}. "
            f"{unavailable_labels} sẽ xuất hiện sau khi train lại để lưu mapping "
            "categorical chính xác."
        )
    selected_models = st.multiselect(
        "Chọn mô hình",
        options=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS,
        format_func=lambda key: MODEL_CONFIGS[key]["label"],
        placeholder="Chọn một hoặc nhiều mô hình",
        label_visibility="collapsed",
    )
    order_column, customer_column, product_column = st.columns([1, 1, 1.12])

    with order_column:
        st.markdown('<h3 class="group-title">Đơn hàng</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="group-copy">Thời gian và các thông số đơn hàng.</p>',
            unsafe_allow_html=True,
        )
        order_date = st.date_input("Ngày đặt hàng", value=date(2024, 1, 1))
        order_time = st.time_input("Giờ đặt hàng", value=time(12, 0))
        order_second = st.number_input(
            "Giây đặt hàng", min_value=0, max_value=59, value=0
        )
        unit_price_usd = st.number_input(
            "Đơn giá sản phẩm (USD)", min_value=0.0, value=50.0, step=1.0
        )
        quantity = st.number_input(
            "Số lượng", min_value=1, max_value=100, value=1, step=1
        )
        discount_percent = st.number_input(
            "Giảm giá (%)", min_value=0, max_value=100, value=0, step=1
        )
        stock_quantity = st.number_input(
            "Tồn kho", min_value=0, value=100, step=1
        )
        delivery_days = st.number_input(
            "Thời gian giao (ngày)", min_value=0, max_value=60, value=3, step=1
        )

    with customer_column:
        st.markdown('<h3 class="group-title">Khách hàng</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p class="group-copy">Hồ sơ và mức độ gắn bó của khách hàng.</p>',
            unsafe_allow_html=True,
        )
        age = st.number_input("Tuổi khách hàng", min_value=1, max_value=100, value=30)
        customer_loyalty_score = st.number_input(
            "Điểm trung thành", min_value=0.0, value=0.5, step=0.01
        )
        total_orders_by_customer = st.number_input(
            "Số đơn hàng trước đây", min_value=0, value=5, step=1
        )
        categorical_values = {
            "is_weekend": "Yes" if order_date.weekday() >= 5 else "No",
        }
        categorical_values["gender"] = st.selectbox(
            CATEGORY_LABELS["gender"],
            options=categories["gender"],
            key="category_gender",
        )
        categorical_values["customer_segment"] = st.selectbox(
            CATEGORY_LABELS["customer_segment"],
            options=categories["customer_segment"],
            key="category_customer_segment",
        )
        product_rating_avg = st.number_input(
            "Điểm đánh giá trung bình", min_value=0.0, value=4.0, step=0.1
        )
        product_reviews_count = st.number_input(
            "Số lượt đánh giá", min_value=0, value=100, step=1
        )

    with product_column:
        st.markdown(
            '<h3 class="group-title">Sản phẩm &amp; thanh toán</h3>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="group-copy">Danh mục, thương hiệu và kênh mua sắm.</p>',
            unsafe_allow_html=True,
        )
        for column in CATEGORICAL_FEATURES[3:]:
            options = categories[column]
            categorical_values[column] = st.selectbox(
                CATEGORY_LABELS[column],
                options=options,
                key=f"category_{column}",
            )

    submitted = st.form_submit_button(
        "Dự đoán bằng mô hình đã chọn",
        type="primary",
        use_container_width=True,
    )

if submitted:
    numeric_values = {
        "order_year": order_date.year,
        "order_month": order_date.month,
        "order_day": order_date.day,
        "order_hour": order_time.hour,
        "order_minute": order_time.minute,
        "order_second": order_second,
        "age": age,
        "customer_loyalty_score": customer_loyalty_score,
        "total_orders_by_customer": total_orders_by_customer,
        "product_rating_avg": product_rating_avg,
        "product_reviews_count": product_reviews_count,
        "stock_quantity": stock_quantity,
        "unit_price_usd": unit_price_usd,
        "quantity": quantity,
        "discount_percent": discount_percent,
        "delivery_days": delivery_days,
    }
    try:
        if not selected_models:
            raise ValueError("Hãy chọn ít nhất một mô hình.")
        predictions = {}
        for model_key in selected_models:
            config = MODEL_CONFIGS[model_key]
            category_orders = None
            if config["kind"] == "spark":
                mapping_path = config["directory"] / "category_indexer.json"
                category_orders = get_category_orders(str(mapping_path))
            feature_vector = build_feature_vector(
                numeric_values,
                categorical_values,
                categories,
                category_orders,
            )
            model_path = model_artifact_path(model_key)
            if config["kind"] == "xgboost":
                model = get_model(str(model_path))
                predictions[model_key] = predict_price(model, feature_vector)
            else:
                model = get_spark_model(str(model_path), model_key)
                predictions[model_key] = predict_spark_price(
                    model,
                    feature_vector,
                    get_spark_session(),
                )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(f"Không thể dự đoán: {error}")
    else:
        prediction_columns = st.columns(len(predictions))
        for column, model_key in zip(prediction_columns, predictions):
            column.metric(
                f'Dự đoán · {MODEL_CONFIGS[model_key]["label"]}',
                f'${predictions[model_key]:,.2f} USD',
            )
        st.markdown(
            '<p class="model-note">Các giá trị là dự đoán độc lập của từng mô hình; '
            'giá trị thực tế có thể khác.</p>',
            unsafe_allow_html=True,
        )

if importance_by_model:
    importance_model = st.selectbox(
        "Xem độ quan trọng của mô hình",
        options=list(importance_by_model),
        format_func=lambda key: MODEL_CONFIGS[key]["label"],
    )
    with st.expander("Độ quan trọng của các đặc trưng"):
        feature_importance = importance_by_model[importance_model]
        top_features = sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:10]
        importance_frame = pd.DataFrame(top_features, columns=["Đặc trưng", "Mức độ"])
        st.bar_chart(importance_frame.set_index("Đặc trưng"))
