# Sử dụng Python 3.11-slim làm base image để tối ưu hóa kích thước container
FROM python:3.11-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Thiết lập các biến môi trường
# PYTHONUNBUFFERED: Đảm bảo output log hiển thị trực tiếp và không bị đệm
# PYTHONDONTWRITEBYTECODE: Tránh tạo file .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Cài đặt các gói hệ thống cần thiết (như compiler cho xgboost/lightgbm nếu cần, tuy nhiên xgboost prebuilt wheels không bắt buộc lắm nhưng vẫn nên cài build-essential nếu cần)
# Ở đây ta giữ gọn nhẹ nhất có thể
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt trước để tận dụng Docker cache layer
COPY requirements.txt .

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn của dự án vào container
COPY src/ ./src/
COPY data/ ./data/
COPY app.py .

# Tạo thư mục models để lưu trữ model artifact nếu chưa có
RUN mkdir -p models

# Chạy huấn luyện mô hình khi build image để đảm bảo có model file sẵn sàng chạy API
# (Hoặc nếu muốn tách biệt, có thể chạy container riêng để train, ở đây ta chạy huấn luyện ngay để tạo model artifact ban đầu)
RUN python -m src.train

# Expose port 8000 của container
EXPOSE 8000

# Lệnh khởi chạy ứng dụng FastAPI bằng Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
