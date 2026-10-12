# Ứng dụng dự đoán đơn hàng

Giao diện Streamlit nhận thông tin khách hàng, sản phẩm và đơn hàng, sau đó
dự đoán `total_price_usd` bằng một hoặc nhiều model Random Forest, GBT và XGBoost.

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
        ├── random_forest/
        │   ├── category_indexer.json
        │   ├── metrics.json
        │   └── model/
        ├── gbt/
        │   ├── category_indexer.json
        │   ├── metrics.json
        │   └── model/
        └── xgboost/
            ├── feature_importance.json
            ├── metrics.json
            └── model_xgboost.ubj
```

Model và dữ liệu lớn được dùng trực tiếp tại chỗ, không sao chép vào `app/`.
`category_values.json` lưu các giá trị phân loại đã sắp xếp đúng như lúc train,
để app tạo cùng mã số đầu vào cho 12 trường categorical của XGBoost.
Hai model Spark dùng thứ tự nhãn do `StringIndexer` học trong tập train; thứ tự
này được lưu riêng theo từng model trong `category_indexer.json`.

Nếu model Random Forest/GBT hiện có chưa có `category_indexer.json`, có thể dựng
lại mapping bằng cách fit riêng `StringIndexer` trên đúng tập train (chia theo
hash `order_id` như lần chạy cũ). Thao tác này không huấn luyện lại cây RF/GBT:

```powershell
.\.venv\Scripts\python.exe train_all_500m.py `
    --input ecommerce_500m `
    --output results\full_500m `
    --models rf,gbt `
    --rebuild-category-indexers
```

Lệnh đọc lại `order_id` và 12 cột categorical của dữ liệu để khớp chính xác
mapping đã học trước đây; bước này vẫn cần quét dữ liệu nhưng bỏ qua toàn bộ
thời gian train cây. Chỉ dùng với đúng dữ liệu, split và sample fraction đã dùng
để tạo model. Sau khi mapping được tạo, khởi động lại app; mặc định app dự đoán
bằng tất cả model sẵn sàng và có thể bỏ chọn để chỉ chạy một hoặc hai mô hình.
Trên Windows, dự đoán bằng Random Forest/GBT cần Java và Hadoop Windows
utilities (`bin\winutils.exe`); app tự nhận thư mục `C:\hadoop` nếu có, nếu
không hãy đặt biến môi trường `HADOOP_HOME` tới thư mục Hadoop tương ứng.

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

Bảng hiệu suất trong giao diện lấy chỉ số MAE, RMSE và R² đã lưu trong từng
thư mục kết quả. Đây là chỉ số trên tập kiểm tra, không đảm bảo sai số tương tự
với mọi đầu vào mới.
