# Customer Churn Prediction - Production-ready MLOps Pipeline

Dự án phát triển và nâng cấp mô hình dự đoán tỷ lệ rời bỏ khách hàng (Customer Churn Prediction - Binary Classification) thành một hệ thống có tính ứng dụng thực tế cao (Production-ready MLOps pipeline) phục vụ làm dự án ứng tuyển vị trí **AI Engineer Intern**.

- **Họ và tên**: Trần Tuấn Việt
- **MSSV**: 2221050021
- **Các mô hình sử dụng**: Logistic Regression, Random Forest, XGBoost
- **Công cụ MLOps & Deployment**: MLflow, FastAPI, Docker, Pydantic

---

## 📂 Cấu trúc thư mục dự án sau refactor

```text
Customer-Churn-Rate---AI-for-Marketing-HUMG-/
├── data/
│   └── Churn Modeling.csv          # Dataset gốc (10,000 dòng, 14 cột)
├── src/
│   ├── __init__.py                 # Đánh dấu package Python
│   ├── data_pipeline.py            # End-to-end data processing (clean, encode, SMOTE, scale)
│   └── train.py                    # Huấn luyện model + MLflow experiment tracking
├── app.py                          # FastAPI endpoint phục vụ mô hình (/predict)
├── Dockerfile                      # File cấu hình đóng gói ứng dụng sang Docker
├── .dockerignore                   # Danh sách file loại trừ khỏi Docker build
├── requirements.txt                # Danh sách các thư viện cần thiết
├── README.md                       # Hướng dẫn chi tiết sử dụng hệ thống
├── models/                         # Thư mục chứa các model artifacts
│   ├── Churn Model.ipynb           # Notebook phân tích gốc để đối chiếu
│   └── best_churn_model.pkl        # Best model + Scaler đã lưu sau khi train
└── mlruns/                         # Thư mục lưu dữ liệu tracking của MLflow
```

---

## ⚡ Hướng dẫn cài đặt & sử dụng với `uv` (WSL / Local)

### 1. Chuẩn bị môi trường ảo và dependencies bằng `uv`
Tạo môi trường ảo siêu tốc và cài đặt các thư viện cần thiết bằng công cụ `uv`:
```bash
# Tạo môi trường ảo
uv venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt các thư viện cực nhanh qua uv pip
uv pip install -r requirements.txt
```

### 2. Huấn luyện mô hình & Theo dõi qua MLflow
Chạy script huấn luyện để tự động tiền xử lý dữ liệu, huấn luyện 3 mô hình (`Logistic Regression`, `Random Forest`, và `XGBoost`), sau đó chọn mô hình tốt nhất dựa trên F1-score và lưu lại:
```bash
uv run python3 -m src.train --data-path "data/Churn Modeling.csv"
```
Kết quả huấn luyện và các tham số sẽ được ghi lại chi tiết vào cơ sở dữ liệu SQLite cục bộ `mlflow.db`. Bạn có thể mở giao diện quản lý thử nghiệm của **MLflow** bằng lệnh:
```bash
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db
```
Sau đó mở trình duyệt tại địa chỉ [http://localhost:5000](http://localhost:5000) để trực quan hóa, so sánh các tham số (Hyperparameters), độ đo (Accuracy, F1-weighted, ROC-AUC) và xem artifacts.

### 3. Khởi chạy FastAPI Server (Serving)
Chạy API phục vụ mô hình trên local bằng **Uvicorn** thông qua `uv`:
```bash
uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
- API Documents (Swagger UI): [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🐳 Đóng gói & Chạy ứng dụng bằng Docker

Hệ thống đã được cấu hình Dockerfile tối ưu hóa, tự động chạy pipeline huấn luyện ngay khi build image để đảm bảo có artifact phục vụ tức thì.

### 1. Build Docker Image
```bash
docker build -t churn-prediction-api .
```

### 2. Khởi chạy Docker Container
```bash
docker run -d -p 8000:8000 --name churn-api-container churn-prediction-api
```

Sau khi container khởi chạy, ứng dụng FastAPI sẽ lắng nghe trên cổng `8000` của máy host.

---

## 🧪 Kiểm thử API (Request/Response)

Gửi một request `POST /predict` bằng lệnh `curl` ở Terminal hoặc công cụ API client (Postman, Insomnia) để kiểm tra hoạt động dự đoán:

### Request mẫu
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88
  }'
```

### Response mẫu (Dự kiến)
```json
{
  "churn": "Yes",
  "probability": 0.8124,
  "model_name": "XGBoost"
}
```
*(Lưu ý: Kết quả thực tế phụ thuộc vào việc huấn luyện mô hình học máy tốt nhất trên tập dữ liệu).*