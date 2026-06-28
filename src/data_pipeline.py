"""
data_pipeline.py - Module xử lý dữ liệu cho Customer Churn Prediction.

Module này chịu trách nhiệm toàn bộ pipeline xử lý dữ liệu:
  - Load dữ liệu thô từ CSV
  - Loại bỏ các cột không liên quan (RowNumber, CustomerId, Surname)
  - Mã hóa biến phân loại (LabelEncoder cho Gender, OneHotEncoding cho Geography)
  - Xử lý mất cân bằng dữ liệu bằng SMOTE
  - Chuẩn hóa features bằng StandardScaler

Author: Trần Tuấn Việt (MSSV: 2221050021)
"""

import logging
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Hằng số (Constants)
# ============================================================================

# Các cột cần loại bỏ vì không có giá trị dự đoán
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]

# Cột mục tiêu (target column)
TARGET_COLUMN = "Exited"

# Cột cần LabelEncode
LABEL_ENCODE_COLUMN = "Gender"

# Cột cần OneHotEncode
ONEHOT_ENCODE_COLUMN = "Geography"


# ============================================================================
# Các hàm xử lý dữ liệu
# ============================================================================


def load_data(file_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ file CSV và loại bỏ các cột không cần thiết.

    Args:
        file_path: Đường dẫn tới file CSV chứa dữ liệu khách hàng.

    Returns:
        DataFrame đã loại bỏ các cột RowNumber, CustomerId, Surname.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu file thiếu các cột bắt buộc.
    """
    logger.info(f"Đang đọc dữ liệu từ: {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Đọc thành công {len(df)} dòng, {len(df.columns)} cột.")

    # Kiểm tra các cột bắt buộc có tồn tại không
    required_columns = COLUMNS_TO_DROP + [TARGET_COLUMN, LABEL_ENCODE_COLUMN, ONEHOT_ENCODE_COLUMN]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"File CSV thiếu các cột bắt buộc: {missing_cols}")

    # Loại bỏ các cột không cần thiết cho việc dự đoán
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    logger.info(f"Đã loại bỏ cột: {COLUMNS_TO_DROP}. Còn lại {len(df.columns)} cột.")

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa biến phân loại (categorical encoding).

    Thực hiện 2 bước:
      1. LabelEncoder cho cột Gender: Female → 0, Male → 1
      2. OneHotEncoding cho cột Geography: tạo Geography_Germany, Geography_Spain
         (drop_first=True để tránh multicollinearity, France là baseline)

    Args:
        df: DataFrame đã load và loại bỏ cột không cần thiết.

    Returns:
        DataFrame với các biến phân loại đã được mã hóa.
    """
    df = df.copy()  # Tránh thay đổi DataFrame gốc

    # 1. LabelEncoder cho Gender
    le = LabelEncoder()
    df[LABEL_ENCODE_COLUMN] = le.fit_transform(df[LABEL_ENCODE_COLUMN])
    logger.info(f"LabelEncode '{LABEL_ENCODE_COLUMN}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # 2. OneHotEncoding cho Geography (drop_first=True → France là baseline)
    df = pd.get_dummies(df, columns=[ONEHOT_ENCODE_COLUMN], drop_first=True)

    # Đảm bảo các cột dummy là kiểu int (0/1) thay vì bool
    dummy_cols = [col for col in df.columns if col.startswith(f"{ONEHOT_ENCODE_COLUMN}_")]
    for col in dummy_cols:
        df[col] = df[col].astype(int)

    logger.info(f"OneHotEncode '{ONEHOT_ENCODE_COLUMN}': tạo cột {dummy_cols}")
    logger.info(f"Sau preprocessing: {len(df.columns)} cột - {list(df.columns)}")

    return df


def split_and_balance(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tách dữ liệu thành train/test và xử lý mất cân bằng bằng SMOTE.

    SMOTE (Synthetic Minority Over-sampling Technique) tạo thêm mẫu synthetic
    cho class thiểu số (Churn = Yes) để cân bằng tỷ lệ với class đa số.
    Chỉ áp dụng SMOTE trên tập train để tránh data leakage.

    Args:
        df: DataFrame đã preprocessing.
        target_col: Tên cột mục tiêu.
        test_size: Tỷ lệ dữ liệu dành cho test (mặc định 20%).
        random_state: Seed để đảm bảo reproducibility.

    Returns:
        Tuple gồm (X_train, X_test, y_train, y_test).
        X_train đã được oversample bằng SMOTE.
    """
    # Tách features (X) và target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    logger.info(f"Phân phối target trước khi split: {dict(y.value_counts())}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu")

    # SMOTE oversampling trên tập train (KHÔNG áp dụng trên test → tránh data leakage)
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logger.info(
        f"Sau SMOTE: Train {len(X_train)} → {len(X_train_resampled)} mẫu "
        f"(phân phối: {dict(pd.Series(y_train_resampled).value_counts())})"
    )

    return X_train_resampled, X_test, y_train_resampled, y_test


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Chuẩn hóa features bằng StandardScaler (zero mean, unit variance).

    Scaler được fit trên tập train và transform cả train + test.
    Scaler cũng được trả về để lưu lại cho inference.

    Args:
        X_train: Features tập train.
        X_test: Features tập test.

    Returns:
        Tuple gồm (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("Đã chuẩn hóa features bằng StandardScaler.")
    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, List[str]]:
    """
    Chạy toàn bộ pipeline xử lý dữ liệu end-to-end.

    Pipeline thực hiện các bước:
      1. Load dữ liệu → loại bỏ cột không cần
      2. Mã hóa biến phân loại (Label + OneHot encoding)
      3. Tách train/test + SMOTE oversampling
      4. Chuẩn hóa features bằng StandardScaler

    Args:
        data_path: Đường dẫn tới file CSV.
        test_size: Tỷ lệ dữ liệu test.
        random_state: Seed cho reproducibility.

    Returns:
        Tuple gồm:
          - X_train_scaled (np.ndarray): Features train đã chuẩn hóa
          - X_test_scaled (np.ndarray): Features test đã chuẩn hóa
          - y_train (pd.Series): Labels train
          - y_test (pd.Series): Labels test
          - scaler (StandardScaler): Scaler đã fit (dùng cho inference)
          - feature_names (List[str]): Danh sách tên features (dùng cho inference)
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU DATA PIPELINE")
    logger.info("=" * 60)

    # Bước 1: Load dữ liệu
    df = load_data(data_path)

    # Bước 2: Mã hóa biến phân loại
    df = preprocess_features(df)

    # Bước 3: Tách train/test + SMOTE
    X_train, X_test, y_train, y_test = split_and_balance(
        df, test_size=test_size, random_state=random_state
    )

    # Lưu lại danh sách tên features trước khi scale
    feature_names = list(X_train.columns)

    # Bước 4: Chuẩn hóa features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    logger.info("=" * 60)
    logger.info("HOÀN THÀNH DATA PIPELINE")
    logger.info("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
