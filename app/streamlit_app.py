"""Vietnamese Streamlit interface for the trained ecommerce price model."""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import pandas as pd
import streamlit as st

from app.predictor import (
    CATEGORICAL_FEATURES,
    build_feature_vector,
    load_categories,
    load_model,
    predict_price,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "results" / "full_500m" / "xgboost" / "model_xgboost.ubj"
METRICS_PATH = ROOT / "results" / "full_500m" / "xgboost" / "metrics.json"
IMPORTANCE_PATH = (
    ROOT / "results" / "full_500m" / "xgboost" / "feature_importance.json"
)
CATEGORIES_PATH = Path(__file__).resolve().parent / "category_values.json"

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
            Ước tính giá trị đơn hàng với mô hình XGBoost được huấn luyện
            trên 500 triệu giao dịch thương mại điện tử.
        </p>
        <span class="hero-chip">✦ &nbsp; XGBoost · 500M giao dịch</span>
    </section>
    """,
    unsafe_allow_html=True,
)

for required_path, label in (
    (MODEL_PATH, "model XGBoost"),
    (METRICS_PATH, "chỉ số đánh giá"),
    (IMPORTANCE_PATH, "feature importance"),
    (CATEGORIES_PATH, "danh mục mã hóa"),
):
    if not required_path.is_file():
        st.error(f"Không tìm thấy {label}: `{required_path}`")
        st.info("Xem `app/README.md` để biết cấu trúc file và cách khởi chạy.")
        st.stop()


@st.cache_resource
def get_model(model_path: str):
    return load_model(Path(model_path))


@st.cache_data
def get_categories(categories_path: str):
    return load_categories(Path(categories_path))


with METRICS_PATH.open(encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)
with IMPORTANCE_PATH.open(encoding="utf-8") as importance_file:
    feature_importance = json.load(importance_file)

st.markdown('<p class="section-kicker">Hiệu suất mô hình</p>', unsafe_allow_html=True)
metric_columns = st.columns([1.05, 1.55, 1, 1])
metric_columns[0].metric("Model", metrics["model"])
metric_columns[1].metric("Dữ liệu huấn luyện", f'{metrics["rows"] / 1_000_000:.0f} triệu dòng')
metric_columns[2].metric("Sai số trung bình (MAE)", f'${metrics["mae"]:,.2f}')
metric_columns[3].metric("Độ phù hợp (R²)", f'{metrics["r2"]:.4f}')
st.markdown(
    f'<p class="model-note">RMSE trên tập kiểm tra: '
    f'<strong>${metrics["rmse"]:,.2f} USD</strong>'
    ' &nbsp;·&nbsp; Kết quả thực tế có thể thay đổi tùy dữ liệu đầu vào.</p>',
    unsafe_allow_html=True,
)

categories = get_categories(str(CATEGORIES_PATH))
model = get_model(str(MODEL_PATH))

with st.form("prediction_form"):
    st.markdown('<h2 class="form-title">Tạo dự đoán</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="form-description">Nhập các thông tin bên dưới để ước tính '
        'tổng giá trị đơn hàng.</p>',
        unsafe_allow_html=True,
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
        "Dự đoán tổng giá trị đơn hàng",
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
        feature_vector = build_feature_vector(
            numeric_values,
            categorical_values,
            categories,
        )
        prediction = predict_price(model, feature_vector)
    except ValueError as error:
        st.error(f"Không thể dự đoán: {error}")
    else:
        st.markdown(
            f'<section class="prediction-card">'
            f'<p class="prediction-label">TỔNG GIÁ TRỊ ĐƠN HÀNG ƯỚC TÍNH</p>'
            f'<p class="prediction-value">${prediction:,.2f} USD</p>'
            f'<p class="prediction-footnote">Đây là kết quả ước tính từ mô hình; '
            f'giá trị thực tế có thể khác.</p></section>',
            unsafe_allow_html=True,
        )

with st.expander("Độ quan trọng của các đặc trưng"):
    top_features = sorted(
        feature_importance.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    importance_frame = pd.DataFrame(top_features, columns=["Đặc trưng", "Mức độ"])
    st.bar_chart(importance_frame.set_index("Đặc trưng"))
