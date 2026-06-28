"""
train.py - Module huấn luyện mô hình Customer Churn Prediction với MLflow tracking.

Module này thực hiện:
  - Gọi data pipeline để xử lý dữ liệu
  - Huấn luyện 3 mô hình: Logistic Regression, Random Forest, XGBoost
  - Log hyperparameters, metrics, model artifacts qua MLflow
  - Chọn mô hình tốt nhất (theo F1-score) và lưu thành file .pkl

Sử dụng:
  python -m src.train --data-path "data/Churn Modeling.csv"

Author: Trần Tuấn Việt (MSSV: 2221050021)
"""

import argparse
import logging
import os
from typing import Dict, Any, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.data_pipeline import run_pipeline

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Hằng số
# ============================================================================

# Thư mục lưu model artifacts
MODEL_DIR = "models"
MODEL_ARTIFACT_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")

# MLflow experiment name
EXPERIMENT_NAME = "customer-churn-prediction"

# ============================================================================
# Định nghĩa các mô hình và hyperparameters
# ============================================================================


def get_models() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Trả về dictionary chứa các mô hình cần huấn luyện cùng hyperparameters.

    Returns:
        Dict với key là tên model, value là tuple (model_instance, params_dict).
    """
    models = {
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
            {
                "model_type": "LogisticRegression",
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
            },
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            {
                "model_type": "RandomForest",
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {
                "model_type": "XGBoost",
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
        ),
    }
    return models


# ============================================================================
# Hàm huấn luyện và đánh giá
# ============================================================================


def evaluate_model(
    model, X_test: np.ndarray, y_test: pd.Series
) -> Dict[str, float]:
    """
    Đánh giá mô hình trên tập test.

    Tính toán các metrics:
      - Accuracy
      - F1-score (weighted): phù hợp cho bài toán imbalanced
      - ROC-AUC score

    Args:
        model: Mô hình đã train.
        X_test: Features tập test (đã scale).
        y_test: Labels tập test.

    Returns:
        Dict chứa các metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score_weighted": f1_score(y_test, y_pred, average="weighted"),
        "f1_score_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # In classification report chi tiết
    report = classification_report(y_test, y_pred, target_names=["Not Churn", "Churn"])
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


def train_and_log_model(
    model_name: str,
    model,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Huấn luyện một mô hình và log kết quả vào MLflow.

    Mỗi model được track trong một MLflow run riêng biệt, bao gồm:
      - Hyperparameters (log_params)
      - Metrics: accuracy, F1, ROC-AUC (log_metrics)
      - Model artifact (log_model)

    Args:
        model_name: Tên mô hình (dùng làm run name).
        model: Instance của model chưa train.
        params: Dict hyperparameters.
        X_train, y_train: Dữ liệu train.
        X_test, y_test: Dữ liệu test.

    Returns:
        Dict chứa metrics của model.
    """
    with mlflow.start_run(run_name=model_name):
        logger.info(f"{'=' * 60}")
        logger.info(f"ĐANG HUẤN LUYỆN: {model_name}")
        logger.info(f"{'=' * 60}")

        # Log hyperparameters
        mlflow.log_params(params)

        # Huấn luyện mô hình
        model.fit(X_train, y_train)
        logger.info(f"Huấn luyện {model_name} hoàn tất.")

        # Đánh giá mô hình
        metrics = evaluate_model(model, X_test, y_test)

        # Log metrics vào MLflow
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics - {model_name}: {metrics}")

        # Log model artifact vào MLflow
        mlflow.sklearn.log_model(
            model, 
            artifact_path=model_name,
            serialization_format="pickle"
        )
        logger.info(f"Đã log model artifact '{model_name}' vào MLflow.")

    return metrics


def save_best_model(
    model, scaler, feature_names, model_name: str, metrics: Dict[str, float]
) -> str:
    """
    Lưu model tốt nhất cùng scaler và metadata vào file .pkl.

    File .pkl chứa dictionary:
      {
        "model": trained_model,
        "scaler": fitted_scaler,
        "feature_names": list_of_feature_names,
        "model_name": model_class_name,
        "metrics": evaluation_metrics
      }

    Điều này đảm bảo API inference có thể tái tạo chính xác pipeline transform.

    Args:
        model: Model đã train.
        scaler: StandardScaler đã fit.
        feature_names: Danh sách tên features.
        model_name: Tên mô hình.
        metrics: Dict metrics.

    Returns:
        Đường dẫn file .pkl đã lưu.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": model_name,
        "metrics": metrics,
    }

    joblib.dump(artifact, MODEL_ARTIFACT_PATH)
    logger.info(f"Đã lưu best model ({model_name}) vào: {MODEL_ARTIFACT_PATH}")
    logger.info(f"Best model metrics: {metrics}")

    return MODEL_ARTIFACT_PATH


# ============================================================================
# Main training pipeline
# ============================================================================


def run_training(data_path: str) -> None:
    """
    Chạy toàn bộ pipeline huấn luyện:
      1. Xử lý dữ liệu qua data_pipeline
      2. Huấn luyện tất cả models với MLflow tracking
      3. Chọn best model theo F1-score và lưu artifact

    Args:
        data_path: Đường dẫn tới file CSV dữ liệu.
    """
    # ── Bước 1: Data Pipeline ──
    X_train, X_test, y_train, y_test, scaler, feature_names = run_pipeline(data_path)

    # ── Bước 2: Cấu hình MLflow ──
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow experiment: '{EXPERIMENT_NAME}' | Tracking URI: sqlite:///mlflow.db")

    # ── Bước 3: Huấn luyện các mô hình ──
    models = get_models()
    results = {}  # {model_name: {"model": model, "metrics": metrics}}

    for model_name, (model, params) in models.items():
        metrics = train_and_log_model(
            model_name=model_name,
            model=model,
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        results[model_name] = {"model": model, "metrics": metrics}

    # ── Bước 4: Chọn best model ──
    logger.info("=" * 60)
    logger.info("SO SÁNH KẾT QUẢ CÁC MÔ HÌNH")
    logger.info("=" * 60)

    # Tạo bảng so sánh
    comparison = pd.DataFrame(
        {name: result["metrics"] for name, result in results.items()}
    ).T
    logger.info(f"\n{comparison.to_string()}")

    # Chọn model có F1-score (weighted) cao nhất
    best_model_name = max(results, key=lambda x: results[x]["metrics"]["f1_score_weighted"])
    best_model = results[best_model_name]["model"]
    best_metrics = results[best_model_name]["metrics"]

    logger.info(f"\n{'=' * 60}")
    logger.info(f"BEST MODEL: {best_model_name} (F1-weighted: {best_metrics['f1_score_weighted']:.4f})")
    logger.info(f"{'=' * 60}")

    # ── Bước 5: Lưu best model ──
    save_best_model(
        model=best_model,
        scaler=scaler,
        feature_names=feature_names,
        model_name=best_model_name,
        metrics=best_metrics,
    )

    logger.info("\n🎉 HOÀN THÀNH TRAINING PIPELINE!")
    logger.info(f"   Model artifact: {MODEL_ARTIFACT_PATH}")
    logger.info(f"   MLflow UI: chạy 'mlflow ui --port 5000' để xem chi tiết.")


# ============================================================================
# CLI Entry Point
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình Customer Churn Prediction với MLflow tracking."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/Churn Modeling.csv",
        help="Đường dẫn tới file CSV dữ liệu (mặc định: data/Churn Modeling.csv)",
    )
    args = parser.parse_args()

    run_training(args.data_path)
