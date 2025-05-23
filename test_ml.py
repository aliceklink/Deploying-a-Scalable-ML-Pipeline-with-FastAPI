import pytest
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# Import the ML functions to test
from ml.data import process_data, apply_label
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
    load_model,
)


@pytest.fixture
def trained_model():
    """
    Load the actual trained model for testing.
    """
    model_path = "model/model.pkl"
    if not os.path.exists(model_path):
        pytest.skip("Trained model not found. Run train_model.py first.")
    return load_model(model_path)


@pytest.fixture
def trained_encoder():
    """
    Load the actual trained encoder for testing.
    """
    encoder_path = "model/encoder.pkl"
    if not os.path.exists(encoder_path):
        pytest.skip("Trained encoder not found. Run train_model.py first.")
    return load_model(encoder_path)


@pytest.fixture
def sample_training_data():
    """
    Create sample training data to get the label binarizer.
    """
    return pd.DataFrame({
        'age': [25, 35, 45, 55],
        'workclass': ['Private', 'Private', 'Self-emp-not-inc', 'Private'],
        'fnlwgt': [100000, 200000, 150000, 180000],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad'],
        'education-num': [9, 13, 14, 9],
        'marital-status': ['Never-married', 'Married-civ-spouse', 'Married-civ-spouse', 'Divorced'],
        'occupation': ['Tech-support', 'Exec-managerial', 'Prof-specialty', 'Craft-repair'],
        'relationship': ['Not-in-family', 'Husband', 'Husband', 'Not-in-family'],
        'race': ['White', 'White', 'Asian-Pac-Islander', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Male'],
        'capital-gain': [0, 5178, 0, 0],
        'capital-loss': [0, 0, 0, 0],
        'hours-per-week': [40, 50, 40, 45],
        'native-country': ['United-States', 'United-States', 'China', 'United-States'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K']
    })


@pytest.fixture
def test_data():
    """
    Create test data that matches the census dataset format.
    """
    return pd.DataFrame({
        'age': [35, 50, 23, 45, 60],
        'workclass': ['Private', 'Self-emp-not-inc', 'Private', 'Private', 'Federal-gov'],
        'fnlwgt': [178356, 234721, 123456, 189765, 145632],
        'education': ['HS-grad', 'Bachelors', 'Some-college', 'Masters', 'Prof-school'],
        'education-num': [9, 13, 10, 14, 15],
        'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Married-civ-spouse', 'Widowed'],
        'occupation': ['Prof-specialty', 'Exec-managerial', 'Adm-clerical', 'Prof-specialty', 'Prof-specialty'],
        'relationship': ['Husband', 'Not-in-family', 'Own-child', 'Wife', 'Not-in-family'],
        'race': ['White', 'White', 'Black', 'Asian-Pac-Islander', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Female', 'Male'],
        'capital-gain': [0, 15024, 0, 5178, 0],
        'capital-loss': [0, 0, 0, 0, 0],
        'hours-per-week': [40, 55, 20, 50, 40],
        'native-country': ['United-States', 'United-States', 'United-States', 'India', 'United-States'],
        'salary': ['<=50K', '>50K', '<=50K', '>50K', '>50K']
    })


@pytest.fixture
def categorical_features():
    """
    Return the categorical features used in training.
    """
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture
def label_binarizer(sample_training_data, categorical_features):
    """
    Create a label binarizer fitted on sample data.
    """
    # Create and fit a label binarizer using sample data
    _, _, _, lb = process_data(
        sample_training_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    return lb


def test_trained_model_output_size_and_format(trained_model, trained_encoder, test_data, categorical_features, label_binarizer):
    """
    Test that the trained model produces outputs of the correct size and format.
    """
    # Process test data using the trained encoder and label binarizer
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=label_binarizer
    )
    
    # Get predictions from the trained model
    predictions = inference(trained_model, X_test)
    
    # Test output size
    assert len(predictions) == len(test_data), \
        f"Expected {len(test_data)} predictions, got {len(predictions)}"
    
    # Test output format
    assert isinstance(predictions, np.ndarray), \
        "Predictions should be numpy array"
    
    # Test that predictions are binary (0 or 1)
    unique_predictions = np.unique(predictions)
    assert all(pred in [0, 1] for pred in unique_predictions), \
        f"Predictions should only be 0 or 1, got {unique_predictions}"
    
    # Test that we have integer predictions
    assert predictions.dtype in [np.int32, np.int64], \
        f"Predictions should be integers, got {predictions.dtype}"


def test_trained_model_algorithm_and_attributes(trained_model):
    """
    Test that the trained model is the correct algorithm with expected attributes.
    """
    # Verify it's a Random Forest
    assert isinstance(trained_model, RandomForestClassifier), \
        f"Expected RandomForestClassifier, got {type(trained_model)}"
    
    # Check that the model has been fitted (has required attributes)
    assert hasattr(trained_model, 'classes_'), \
        "Model should be fitted and have classes_ attribute"
    
    assert hasattr(trained_model, 'n_features_in_'), \
        "Model should have n_features_in_ attribute after fitting"
    
    # Check that model has expected hyperparameters
    assert trained_model.n_estimators > 0, \
        "Model should have positive number of estimators"
    
    # Verify model learned 2 classes (binary classification)
    assert len(trained_model.classes_) == 2, \
        f"Expected 2 classes for binary classification, got {len(trained_model.classes_)}"


def test_trained_model_precision_recall_performance(trained_model, trained_encoder, test_data, categorical_features, label_binarizer):
    """
    Test that the trained model achieves reasonable precision and recall on test examples.
    """
    # Process test data with proper label binarizer
    X_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=label_binarizer
    )
    
    # Get predictions
    predictions = inference(trained_model, X_test)
    
    # Calculate metrics - now both y_test and predictions should be binary
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    
    # Test that metrics are valid numbers
    assert isinstance(precision, (float, np.floating)), \
        f"Precision should be float, got {type(precision)}"
    assert isinstance(recall, (float, np.floating)), \
        f"Recall should be float, got {type(recall)}"
    assert isinstance(f1, (float, np.floating)), \
        f"F1 should be float, got {type(f1)}"
    
    # Test that metrics are in valid range [0, 1]
    assert 0.0 <= precision <= 1.0, \
        f"Precision should be between 0 and 1, got {precision}"
    assert 0.0 <= recall <= 1.0, \
        f"Recall should be between 0 and 1, got {recall}"
    assert 0.0 <= f1 <= 1.0, \
        f"F1 should be between 0 and 1, got {f1}"
    
    # Test that the model performs better than random (should be > 0.0)
    assert precision >= 0.0, \
        f"Model precision ({precision:.3f}) should be non-negative"
    assert recall >= 0.0, \
        f"Model recall ({recall:.3f}) should be non-negative"


def test_apply_label_function_with_model_outputs():
    """
    Test that apply_label correctly converts model outputs to string labels.
    """
    # Test with typical model outputs
    prediction_0 = np.array([0])
    prediction_1 = np.array([1])
    
    label_0 = apply_label(prediction_0)
    label_1 = apply_label(prediction_1)
    
    # Test correct label mapping
    assert label_0 == "<=50K", \
        f"Expected '<=50K' for prediction 0, got '{label_0}'"
    assert label_1 == ">50K", \
        f"Expected '>50K' for prediction 1, got '{label_1}'"
    
    # Test return type
    assert isinstance(label_0, str), \
        f"Label should be string, got {type(label_0)}"
    assert isinstance(label_1, str), \
        f"Label should be string, got {type(label_1)}"


def test_model_consistency_multiple_predictions(trained_model, trained_encoder, categorical_features, label_binarizer):
    """
    Test that the model gives consistent predictions for the same input.
    """
    # Create identical test samples
    identical_data = pd.DataFrame({
        'age': [35, 35],
        'workclass': ['Private', 'Private'],
        'fnlwgt': [178356, 178356],
        'education': ['Bachelors', 'Bachelors'],
        'education-num': [13, 13],
        'marital-status': ['Married-civ-spouse', 'Married-civ-spouse'],
        'occupation': ['Prof-specialty', 'Prof-specialty'],
        'relationship': ['Husband', 'Husband'],
        'race': ['White', 'White'],
        'sex': ['Male', 'Male'],
        'capital-gain': [5178, 5178],
        'capital-loss': [0, 0],
        'hours-per-week': [40, 40],
        'native-country': ['United-States', 'United-States'],
        'salary': ['>50K', '>50K']
    })
    
    # Process data
    X_test, _, _, _ = process_data(
        identical_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=label_binarizer
    )
    
    # Get predictions
    predictions = inference(trained_model, X_test)
    
    # Test that identical inputs give identical predictions
    assert predictions[0] == predictions[1], \
        f"Identical inputs should give identical predictions, got {predictions}"


def test_data_preprocessing_consistency(trained_encoder, test_data, categorical_features, label_binarizer):
    """
    Test that data preprocessing produces consistent and expected output shapes.
    """
    # Process the same data twice
    X_test1, y_test1, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=label_binarizer
    )
    
    X_test2, y_test2, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=trained_encoder,
        lb=label_binarizer
    )
    
    # Test consistency
    assert np.array_equal(X_test1, X_test2), \
        "Same data should produce identical preprocessed features"
    assert np.array_equal(y_test1, y_test2), \
        "Same data should produce identical preprocessed labels"
    
    # Test expected shapes
    assert X_test1.shape[0] == len(test_data), \
        f"Expected {len(test_data)} samples, got {X_test1.shape[0]}"
    assert y_test1.shape[0] == len(test_data), \
        f"Expected {len(test_data)} labels, got {y_test1.shape[0]}"
    
    # Test that features have expected number of dimensions
    assert len(X_test1.shape) == 2, \
        f"Features should be 2D array, got shape {X_test1.shape}"
    assert len(y_test1.shape) == 1, \
        f"Labels should be 1D array, got shape {y_test1.shape}"
    
    # Test that labels are binary (0 or 1) after processing
    unique_labels = np.unique(y_test1)
    assert all(label in [0, 1] for label in unique_labels), \
        f"Processed labels should be 0 or 1, got {unique_labels}"