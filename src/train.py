# train.py
"""
Reusable module to train multiple ML models with MLflow tracking
"""

import pandas as pd
import re
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

RANDOM_STATE = 42

# -----------------------------
# Clean feature names
# -----------------------------
def clean_feature_names(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X.columns = [
        re.sub(r"[^A-Za-z0-9_]+", "_", col)
        for col in X.columns
    ]
    return X

# --------------------------
# Evaluation function
# --------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


# --------------------------
# Train function
# --------------------------
def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Trains multiple ML models and logs results to MLflow.

    Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector

    Returns:
        dict: Evaluation metrics per model
    """
   
    # --------------------------
    # Split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    metrics_dict = {}
    mlflow.set_experiment("Credit_Risk_Modeling")

    min_class_count = np.bincount(y_train).min()
    cv_folds = min(5, min_class_count)

    # --------------------------
    # Logistic Regression
    # --------------------------
    with mlflow.start_run(run_name="LogisticRegression_GridSearch"):
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)

        param_grid = {
            "C": [0.01, 0.1, 1, 10],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        }

        grid = GridSearchCV(
            lr,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv_folds,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # --------------------------
        # Save best model as pickle
        # --------------------------
        with open("/Users/elbethelzewdie/Downloads/credit-risk-analysis/Credit-risk-analysis/models/LogisticRegression_best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("✅ Logistic Regression model saved to LogisticRegression_best_model.pkl")

        metrics_dict["LogisticRegression"] = evaluate_model(
            best_model, X_test, y_test
        )

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics_dict["LogisticRegression"])
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="LogisticRegression")


    # --------------------------
    # Decision Tree
    # --------------------------
    with mlflow.start_run(run_name="DecisionTree_GridSearch"):
        dt = DecisionTreeClassifier(random_state=RANDOM_STATE)

        param_grid = {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        }

        grid = GridSearchCV(
            dt,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv_folds,
            n_jobs=-1,
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics_dict["DecisionTree"] = evaluate_model(
            best_model, X_test, y_test
        )

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics_dict["DecisionTree"])
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="DecisionTree")


    # --------------------------
    # Random Forest
    # --------------------------
    with mlflow.start_run(run_name="RandomForest_RandomSearch"):
        rf = RandomForestClassifier(random_state=RANDOM_STATE)

        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        search = RandomizedSearchCV(
            rf,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=cv_folds,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        metrics_dict["RandomForest"] = evaluate_model(
            best_model, X_test, y_test
        )

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics_dict["RandomForest"])
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="RandomForest")


    # --------------------------
    # XGBoost
    # --------------------------
    with mlflow.start_run(run_name="XGBoost_RandomSearch"):
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )

        param_dist = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
        }

        search = RandomizedSearchCV(
            xgb,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=cv_folds,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        metrics_dict["XGBoost"] = evaluate_model(
            best_model, X_test, y_test
        )

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics_dict["XGBoost"])
        mlflow.sklearn.log_model(best_model, "model", registered_model_name="XGBoost")


    return metrics_dict
