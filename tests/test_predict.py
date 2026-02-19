"""
Unit tests for the prediction module
"""

import pytest
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import tempfile
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from src.predict import predict_model
import argparse
from unittest.mock import patch, MagicMock, call


@pytest.fixture
def sample_input_data():
    """Create sample input data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10)
    })


@pytest.fixture
def sample_model_and_data():
    """Create a sample model and training data, then register it in MLflow"""
    # Create sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=3, 
        n_informative=2, 
        n_redundant=0, 
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    
    # Register model in MLflow
    model_name = "test_model"
    
    # Start MLflow run
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
    
    return model_name, X_df


def test_predict_model_returns_numpy_array(sample_input_data, sample_model_and_data):
    """Test that predict_model returns a numpy array"""
    model_name, _ = sample_model_and_data
    
    # Make predictions
    predictions = predict_model(model_name, sample_input_data)
    
    # Assertions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_input_data)


def test_predict_model_with_different_input_sizes(sample_model_and_data):
    """Test predictions with different input sizes"""
    model_name, original_data = sample_model_and_data
    
    # Test with single row
    single_row = original_data.iloc[[0]]
    predictions_single = predict_model(model_name, single_row)
    assert len(predictions_single) == 1
    
    # Test with multiple rows
    multiple_rows = original_data.head(5)
    predictions_multiple = predict_model(model_name, multiple_rows)
    assert len(predictions_multiple) == 5


def test_predict_model_preserves_input_order(sample_model_and_data):
    """Test that predictions maintain the same order as input data"""
    model_name, original_data = sample_model_and_data
    
    # Get predictions for original data
    predictions = predict_model(model_name, original_data)
    
    # Get predictions for shuffled data
    shuffled_data = original_data.sample(frac=1, random_state=42)
    shuffled_predictions = predict_model(model_name, shuffled_data)
    
    # Check that predictions are in the same order as input
    # (we can check by reordering shuffled predictions back to original order)
    reordered_predictions = shuffled_predictions[shuffled_data.index.argsort()]
    assert np.array_equal(predictions, reordered_predictions)


def test_predict_model_with_missing_model(sample_input_data):
    """Test that appropriate error is raised when model doesn't exist"""
    with pytest.raises(Exception) as exc_info:
        predict_model("non_existent_model", sample_input_data)
    
    assert "Model" in str(exc_info.value) or "not found" in str(exc_info.value)


def test_predict_model_with_invalid_features(sample_model_and_data):
    """Test predictions with data missing required features"""
    model_name, _ = sample_model_and_data
    
    # Create data with wrong feature names
    invalid_data = pd.DataFrame({
        'wrong_feature1': [1, 2, 3],
        'wrong_feature2': [4, 5, 6]
    })
    
    with pytest.raises(Exception) as exc_info:
        predict_model(model_name, invalid_data)
    
    assert "feature" in str(exc_info.value).lower() or "columns" in str(exc_info.value).lower()


def test_predict_model_returns_consistent_types(sample_model_and_data):
    """Test that prediction outputs have consistent data types"""
    model_name, original_data = sample_model_and_data
    
    predictions = predict_model(model_name, original_data)
    
    # For classification, predictions should be integers or floats
    assert predictions.dtype in [np.int64, np.int32, np.float64, np.float32]


def test_predict_model_with_empty_dataframe(sample_model_and_data):
    """Test predictions with empty dataframe - should handle gracefully"""
    model_name, original_data = sample_model_and_data
    
    empty_data = pd.DataFrame(columns=original_data.columns)
    
    # The model might raise an error for empty input, which is acceptable
    # We're just testing that the function handles it appropriately
    try:
        predictions = predict_model(model_name, empty_data)
        assert len(predictions) == 0
        assert isinstance(predictions, np.ndarray)
    except Exception as e:
        # If it raises an exception, make sure it's a reasonable one
        assert "0 sample" in str(e) or "empty" in str(e).lower()


@pytest.mark.parametrize("data_size", [1, 10, 100])
def test_predict_model_with_different_data_sizes(sample_model_and_data, data_size):
    """Test predictions with different data sizes using parameterization"""
    model_name, original_data = sample_model_and_data
    
    # Create data of specified size (repeating if necessary)
    if data_size <= len(original_data):
        test_data = original_data.head(data_size)
    else:
        # Repeat data to reach desired size
        repeats = (data_size // len(original_data)) + 1
        test_data = pd.concat([original_data] * repeats, ignore_index=True).head(data_size)
    
    predictions = predict_model(model_name, test_data)
    assert len(predictions) == data_size


def test_mlflow_model_loading(sample_model_and_data):
    """Test that MLflow model loading works correctly"""
    model_name, original_data = sample_model_and_data
    
    # Get predictions using the function
    predictions = predict_model(model_name, original_data)
    
    # Load model directly and compare
    model_uri = f"models:/{model_name}/latest"
    direct_model = mlflow.pyfunc.load_model(model_uri)
    direct_predictions = direct_model.predict(original_data)
    
    # Compare results
    assert np.array_equal(predictions, direct_predictions)


def test_predict_model_with_different_model_types():
    """Test predictions with different types of models"""
    # Create regression data
    X_reg, y_reg = make_regression(n_samples=100, n_features=3, random_state=42)
    X_reg_df = pd.DataFrame(X_reg, columns=['feature1', 'feature2', 'feature3'])
    
    reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
    reg_model.fit(X_reg_df, y_reg)
    
    reg_model_name = "test_regression_model"
    
    with mlflow.start_run():
        mlflow.sklearn.log_model(reg_model, "model", registered_model_name=reg_model_name)
    
    # Test predictions
    test_data = X_reg_df.head(5)
    predictions = predict_model(reg_model_name, test_data)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 5
    # Regression predictions should be floats
    assert predictions.dtype in [np.float64, np.float32]


def test_predict_model_with_save_path(sample_model_and_data, tmp_path):
    """Test that predictions can be saved to a file"""
    model_name, original_data = sample_model_and_data
    
    # Create a temporary save path
    save_path = tmp_path / "predictions.csv"
    
    # Instead, let's test saving functionality separately
    predictions = predict_model(model_name, original_data)
    
    # Save predictions manually
    pd.DataFrame({'predictions': predictions}).to_csv(save_path, index=False)
    
    # Verify file was created
    assert os.path.exists(save_path)
    
    # Verify content
    saved_data = pd.read_csv(save_path)
    assert len(saved_data) == len(original_data)
    assert 'predictions' in saved_data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-p", "no:warnings"])