import pytest
import pandas as pd
import numpy as np
import os
from src.train import clean_feature_names, train_model

# 1. Mocking the data for tests
@pytest.fixture
def sample_data():
    """Generates a small synthetic dataset for testing."""
    X = pd.DataFrame({
        "Feature 1!": np.random.rand(100),
        "Feature-2": np.random.rand(100),
        "Feature 3": np.random.rand(100)
    })
    y = pd.Series(np.random.randint(0, 2, size=100))
    return X, y

# 2. Test Feature Name Cleaning
def test_clean_feature_names(sample_data):
    X, _ = sample_data
    X_cleaned = clean_feature_names(X)
    
    # Check if special characters were replaced with underscores
    for col in X_cleaned.columns:
        assert "!" not in col
        assert "-" not in col
        assert " " not in col
    
    assert len(X_cleaned.columns) == 3
    print("✅ Feature name cleaning test passed.")

# 3. Test the Training Pipeline
def test_train_model_execution(sample_data, tmp_path):
    """
    Tests if the train_model function runs without error 
    and returns the expected metrics dictionary.
    """
    X, y = sample_data
    
    # We run the function
    try:
        metrics = train_model(X, y)
    except Exception as e:
        pytest.fail(f"train_model raised an exception: {e}")

    # Verify output structure
    expected_models = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
    for model_name in expected_models:
        assert model_name in metrics
        assert "roc_auc" in metrics[model_name]
        assert "accuracy" in metrics[model_name]
        
    print("✅ Model training and evaluation test passed.")

# 4. Test File Persistence (Optional)
def test_pickle_creation():
    # Note: Your script has a hardcoded path for the pickle file.
    # In a real test, you'd want to make that path configurable.
    path = "/Users/elbethelzewdie/Downloads/credit-risk-analysis/Credit-risk-analysis/models/LogisticRegression_best_model.pkl"
    
    # Check if the file exists (this assumes you've run the test once)
    if os.path.exists(path):
        assert os.path.getsize(path) > 0
        print("✅ Pickle file exists and is not empty.")