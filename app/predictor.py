"""Inference helpers for the trained ecommerce XGBoost model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import xgboost as xgb

NUMERIC_FEATURES = (
    "order_year",
    "order_month",
    "order_day",
    "order_hour",
    "order_minute",
    "order_second",
    "age",
    "customer_loyalty_score",
    "total_orders_by_customer",
    "product_rating_avg",
    "product_reviews_count",
    "stock_quantity",
    "unit_price_usd",
    "quantity",
    "discount_percent",
    "delivery_days",
)
CATEGORICAL_FEATURES = (
    "is_weekend",
    "gender",
    "customer_segment",
    "category",
    "sub_category",
    "brand",
    "shipping_method",
    "payment_method",
    "installment_plan",
    "campaign_source",
    "device_type",
    "traffic_source",
)
FEATURE_NAMES = NUMERIC_FEATURES + tuple(
    f"{name}_idx" for name in CATEGORICAL_FEATURES
)


def load_categories(path: Path) -> dict[str, list[str]]:
    """Load the exact sorted category values used to encode the training data."""
    with path.open(encoding="utf-8") as category_file:
        categories = json.load(category_file)

    if set(categories) != set(CATEGORICAL_FEATURES):
        raise ValueError(
            "Tệp danh mục phải có đúng 12 cột phân loại như lúc huấn luyện."
        )

    for column in CATEGORICAL_FEATURES:
        values = categories[column]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(
                f"Danh mục '{column}' phải là danh sách chuỗi đã sắp xếp, "
                "không trùng lặp và không rỗng."
            )
    return categories


def load_category_indexer(path: Path) -> dict[str, list[str]]:
    """Load StringIndexer label order saved with a Spark model."""
    with path.open(encoding="utf-8") as mapping_file:
        category_orders = json.load(mapping_file)

    if set(category_orders) != set(CATEGORICAL_FEATURES):
        raise ValueError("Mapping categorical phải có đúng 12 cột phân loại.")
    for column in CATEGORICAL_FEATURES:
        labels = category_orders[column]
        if (
            not isinstance(labels, list)
            or any(not isinstance(label, str) for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise ValueError(f"Mapping của '{column}' không hợp lệ.")
    return category_orders


def build_feature_vector(
    numeric_values: Mapping[str, int | float],
    categorical_values: Mapping[str, str],
    categories: Mapping[str, Sequence[str]],
    category_orders: Mapping[str, Sequence[str]] | None = None,
) -> np.ndarray:
    """Build a float32 row using the feature order and ordinal codes from training."""
    if set(numeric_values) != set(NUMERIC_FEATURES):
        raise ValueError("Thiếu hoặc thừa trường số so với schema huấn luyện.")
    if set(categorical_values) != set(CATEGORICAL_FEATURES):
        raise ValueError("Thiếu hoặc thừa trường phân loại so với schema huấn luyện.")
    if set(categories) != set(CATEGORICAL_FEATURES):
        raise ValueError("Tệp danh mục không khớp schema huấn luyện.")
    if category_orders is not None and set(category_orders) != set(CATEGORICAL_FEATURES):
        raise ValueError("Mapping categorical của model không khớp schema huấn luyện.")

    values = [float(numeric_values[name]) for name in NUMERIC_FEATURES]
    for name in CATEGORICAL_FEATURES:
        options = category_orders[name] if category_orders is not None else categories[name]
        selected = categorical_values[name]
        try:
            values.append(float(options.index(selected)))
        except ValueError as error:
            if category_orders is not None:
                values.append(float(len(options)))
            else:
                raise ValueError(
                    f"Giá trị '{selected}' không có trong danh mục '{name}'."
                ) from error

    return np.asarray(values, dtype=np.float32)


def load_model(path: Path) -> xgb.Booster:
    """Load the saved booster and reject models with an incompatible schema."""
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model đã train: {path}")

    model = xgb.Booster()
    model.load_model(str(path))
    if model.feature_names != list(FEATURE_NAMES):
        raise ValueError(
            "Schema của model không khớp thứ tự 28 feature dùng trong app."
        )
    return model


def predict_price(model: xgb.Booster, features: np.ndarray) -> float:
    """Predict one order's total price in USD."""
    if features.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Model cần đúng {len(FEATURE_NAMES)} feature.")

    matrix = xgb.DMatrix(
        features.reshape(1, -1),
        feature_names=list(FEATURE_NAMES),
    )
    return float(model.predict(matrix)[0])


def load_spark_tree_model(path: Path, model_type: str):
    """Load a saved Spark ML regression tree model."""
    if not path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy model đã train: {path}")
    if model_type == "random_forest":
        from pyspark.ml.regression import RandomForestRegressionModel

        return RandomForestRegressionModel.load(str(path))
    if model_type == "gbt":
        from pyspark.ml.regression import GBTRegressionModel

        return GBTRegressionModel.load(str(path))
    raise ValueError(f"Loại Spark model không hỗ trợ: {model_type}")


def predict_spark_price(model, features: np.ndarray, spark) -> float:
    """Predict one order with a Spark ML tree model."""
    from pyspark.ml.linalg import Vectors

    if features.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Model cần đúng {len(FEATURE_NAMES)} feature.")

    rows = spark.createDataFrame([(Vectors.dense(features.tolist()),)], ["features"])
    prediction = model.transform(rows).select("prediction").first()
    return float(prediction["prediction"])
