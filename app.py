"""
app.py - FastAPI Application phục vụ mô hình Customer Churn Prediction.

Ứng dụng cung cấp:
  - Endpoint POST /predict: Nhận thông tin khách hàng, thực hiện transform và dự đoán churn
  - Endpoint GET /health: Kiểm tra tình trạng hoạt động của API và mô hình
  - Endpoint GET /: Giao diện chào mừng thân thiện

Author: Trần Tuấn Việt (MSSV: 2221050021)
"""

import os
import logging
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Khởi tạo FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API dự đoán tỷ lệ rời bỏ của khách hàng (Customer Churn Rate) - AI for Marketing",
    version="1.0.0"
)

# Đường dẫn file model đã huấn luyện
MODEL_PATH = "models/best_churn_model.pkl"

# Biến toàn cục để lưu trữ artifact đã load
model_artifact = None

# ============================================================================
# Định nghĩa Pydantic Schemas cho Input/Output Validation
# ============================================================================

class CustomerInput(BaseModel):
    CreditScore: int = Field(..., ge=300, le=850, description="Điểm tín dụng của khách hàng", example=619)
    Geography: str = Field(..., description="Quốc gia của khách hàng (France, Germany, Spain)", example="France")
    Gender: str = Field(..., description="Giới tính của khách hàng (Male, Female)", example="Female")
    Age: int = Field(..., ge=18, le=100, description="Tuổi của khách hàng", example=42)
    Tenure: int = Field(..., ge=0, le=10, description="Số năm gắn bó của khách hàng với ngân hàng", example=2)
    Balance: float = Field(..., ge=0.0, description="Số dư tài khoản", example=0.0)
    NumOfProducts: int = Field(..., ge=1, le=4, description="Số lượng sản phẩm khách hàng đang sử dụng", example=1)
    HasCrCard: int = Field(..., ge=0, le=1, description="Khách hàng có thẻ tín dụng hay không (0: Không, 1: Có)", example=1)
    IsActiveMember: int = Field(..., ge=0, le=1, description="Khách hàng có hoạt động tích cực không (0: Không, 1: Có)", example=1)
    EstimatedSalary: float = Field(..., ge=0.0, description="Mức lương ước tính hàng năm", example=101348.88)

class PredictionOutput(BaseModel):
    churn: str = Field(..., description="Kết quả dự đoán rời bỏ (Yes: Rời bỏ, No: Ở lại)")
    probability: float = Field(..., description="Xác suất rời bỏ của khách hàng (từ 0.0 đến 1.0)")
    model_name: str = Field(..., description="Tên mô hình học máy được sử dụng để dự đoán")

# ============================================================================
# Startup Event: Load Model & Scaler
# ============================================================================

@app.on_event("startup")
def load_model():
    """
    Tự động load mô hình tốt nhất cùng scaler và metadata khi FastAPI server khởi chạy.
    """
    global model_artifact
    logger.info("Đang khởi động ứng dụng...")
    
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            f"Không tìm thấy file mô hình tại '{MODEL_PATH}'. "
            "Vui lòng chạy script huấn luyện trước: 'python -m src.train'"
        )
        return
        
    try:
        model_artifact = joblib.load(MODEL_PATH)
        logger.info(
            f"Đã load thành công mô hình '{model_artifact.get('model_name')}' từ '{MODEL_PATH}'."
        )
        logger.info(f"Độ đo F1-weighted khi huấn luyện: {model_artifact.get('metrics', {}).get('f1_score_weighted', 'N/A')}")
    except Exception as e:
        logger.error(f"Lỗi khi load mô hình: {str(e)}")
        raise e

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
def read_root():
    """
    Trang chào mừng API.
    """
    return {
        "message": "Chào mừng bạn đến với Customer Churn Prediction API!",
        "docs_url": "/docs",
        "health_check": "/health",
        "status": "Running"
    }

@app.get("/health", tags=["General"])
def health_check():
    """
    Endpoint kiểm tra sức khỏe hệ thống (Health check).
    Trả về thông tin trạng thái API và mô hình đã được tải hay chưa.
    """
    if model_artifact is None:
        return {
            "status": "Degraded",
            "message": "API đang chạy nhưng mô hình học máy chưa được tải. Hãy chạy pipeline huấn luyện trước."
        }
    return {
        "status": "Healthy",
        "model_loaded": True,
        "model_name": model_artifact.get("model_name"),
        "metrics": model_artifact.get("metrics")
    }

@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK, tags=["Prediction"])
def predict(payload: CustomerInput):
    """
    Thực hiện dự đoán tỷ lệ rời bỏ (Churn) của một khách hàng cụ thể.

    Nhận thông tin chi tiết của khách hàng dưới định dạng JSON,
    sau đó thực hiện các bước tiền xử lý tương đương pipeline huấn luyện
    và trả về kết quả dự đoán (Yes/No) kèm xác suất.
    """
    # Kiểm tra xem mô hình đã được tải chưa
    if model_artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mô hình dự đoán hiện chưa sẵn sàng. Vui lòng kiểm tra lại trạng thái API tại endpoint /health"
        )
        
    try:
        # Lấy các thành phần cần thiết từ artifact
        model = model_artifact["model"]
        scaler = model_artifact["scaler"]
        feature_names = model_artifact["feature_names"]
        model_name = model_artifact["model_name"]
        
        # 1. Chuyển đổi dữ liệu từ payload sang Dictionary ban đầu
        input_data = payload.dict()
        
        # 2. Xử lý logic mã hóa giống hệt trong data_pipeline.py
        # Mã hóa cột Gender: Female -> 0, Male -> 1
        gender_str = input_data["Gender"].strip()
        if gender_str.lower() == "female":
            input_data["Gender"] = 0
        elif gender_str.lower() == "male":
            input_data["Gender"] = 1
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Giới tính '{gender_str}' không hợp lệ. Chỉ chấp nhận 'Male' hoặc 'Female'."
            )
            
        # Tiền xử lý One-Hot Encoding cho Geography
        # Các cột dummy trong tập huấn luyện sau get_dummies(drop_first=True) là:
        # 'Geography_Germany', 'Geography_Spain' (nếu France là baseline)
        geography_str = input_data["Geography"].strip().lower()
        
        # Thiết lập mặc định cho các biến dummy
        input_data["Geography_Germany"] = 0
        input_data["Geography_Spain"] = 0
        
        if geography_str == "germany":
            input_data["Geography_Germany"] = 1
        elif geography_str == "spain":
            input_data["Geography_Spain"] = 1
        elif geography_str == "france":
            # Baseline, cả hai biến dummy đều bằng 0
            pass
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Quốc gia '{input_data['Geography']}' không hợp lệ. Chỉ hỗ trợ 'France', 'Germany', 'Spain'."
            )
            
        # Xóa cột gốc Geography đi để chuẩn bị matching features
        del input_data["Geography"]
        
        # 3. Tạo DataFrame đầu vào từ dữ liệu đã qua tiền xử lý
        input_df = pd.DataFrame([input_data])
        
        # Đảm bảo thứ tự các cột khớp chính xác với tập huấn luyện
        # Điền 0 cho những cột bị thiếu nếu có (tuy nhiên ở đây chúng ta đã khớp chính xác)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Sắp xếp lại thứ tự cột đúng theo tập huấn luyện
        input_df = input_df[feature_names]
        
        # 4. Chuẩn hóa (scale) các đặc trưng
        scaled_features = scaler.transform(input_df)
        
        # 5. Thực hiện dự đoán
        prediction = model.predict(scaled_features)[0]
        # Lấy xác suất của class 1 (Exited = 1)
        probability = float(model.predict_proba(scaled_features)[0][1])
        
        # Kết quả nhãn
        churn_label = "Yes" if prediction == 1 else "No"
        
        return PredictionOutput(
            churn=churn_label,
            probability=probability,
            model_name=model_name
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Đã xảy ra lỗi hệ thống khi xử lý yêu cầu: {str(e)}"
        )
