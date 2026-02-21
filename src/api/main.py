from fastapi import FastAPI, HTTPException
from typing import List
import mlflow.pyfunc
import pandas as pd
import re
import os
import joblib
from src.api.pydantic_models import CustomerData, RiskPredictionResponse

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict credit risk using MLflow models",
    version="1.0.0"
)

def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in X.columns]
    return X

# --- Load model ---


# Path to your model
MODEL_PATH = os.path.join("/Users/elbethelzewdie/Downloads/credit-risk-analysis/Credit-risk-analysis/model", "LogisticRegression_best_model1.pkl")

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


# --- Endpoints ---
@app.post("/predict", response_model=List[RiskPredictionResponse])
def predict(customers: List[CustomerData]):
    try:
        input_df = pd.DataFrame([c.dict() for c in customers])
        input_df = clean_feature_names(input_df)

        # Ensure feature alignment
        input_df = input_df[model.feature_names_in_]

        predictions = model.predict_proba(input_df)[:, 1]

        response = [
            RiskPredictionResponse(
                CustomerId=row.CustomerId,
                risk_probability=float(pred)
            )
            for row, pred in zip(customers, predictions)
        ]

        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
