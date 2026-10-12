# Ứng dụng dự đoán đơn hàng

Giao diện Streamlit nhận thông tin khách hàng, sản phẩm và đơn hàng, sau đó
dùng model XGBoost đã huấn luyện để dự đoán `total_price_usd`.

## Cấu trúc

```text
Dataset/
├── app/
│   ├── README.md
│   ├── category_values.json
│   ├── predictor.py
│   ├── requirements.txt
│   ├── streamlit_app.py
│   └── tests/
│       └── test_predictor.py
├── ecommerce_500m/             # Parquet nguồn; app không quét dữ liệu khi chạy
└── results/
    └── full_500m/
        └── xgboost/
            ├── feature_importance.json
            ├── metrics.json
            └── model_xgboost.ubj
```

Model và dữ liệu lớn được dùng trực tiếp tại chỗ, không sao chép vào `app/`.
`category_values.json` lưu các giá trị phân loại đã sắp xếp đúng như lúc train,
để app tạo cùng mã số đầu vào cho 12 trường categorical của XGBoost.

## Cài đặt và chạy

Chạy các lệnh từ thư mục gốc `Dataset`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r app\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Streamlit sẽ in địa chỉ local để mở ứng dụng trong trình duyệt.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s app\tests
```

Model hiện tại đạt MAE khoảng 8,48 USD, RMSE khoảng 11,08 USD và R² khoảng
0,9992 trên tập kiểm tra theo kết quả lưu trong `results/full_500m/xgboost`.
Đây là chỉ số đánh giá trên tập kiểm tra, không đảm bảo sai số tương tự với
mọi đầu vào mới.
