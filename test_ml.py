import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

# Import the ML functions to test
from ml.data import process_data, apply_label
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)


@pytest.fixture
def sample_data():
    """
    Create sample data for testing that mimics the census dataset structure.
    """
    data = pd.DataFrame({
        'age': [25, 35, 45, 55, 30, 40, 50, 60],
        'workclass': ['Private', 'Private', 'Self-emp-not-inc', 'Private',
                      'Private', 'Self-emp-not-inc', 'Private', 'Private'],
        'fnlwgt': [100000, 200000, 150000, 180000, 120000, 160000, 190000,
                   110000],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad',
                      'Some-college', 'Bachelors', 'Masters', 'HS-grad'],
        'education-num': [9, 13, 14, 9, 10, 13, 14, 9],
        'marital-status': ['Never-married', 'Married-civ-spouse',
                           'Married-civ-spouse', 'Divorced',
                           'Never-married', 'Married-civ-spouse',
                           'Married-civ-spouse', 'Divorced'],
        'occupation': ['Tech-support', 'Exec-managerial', 'Prof-specialty',
                       'Craft-repair', 'Adm-clerical', 'Exec-managerial',
                       'Prof-specialty', 'Craft-repair'],
        'relationship': ['Not-in-family', 'Husband', 'Husband',
                         'Not-in-family', 'Not-in-family', 'Husband',
                         'Husband', 'Not-in-family'],
        'race': ['White', 'White', 'Asian-Pac-Islander', 'White',
                 'Black', 'White', 'Asian-Pac-Islander', 'White'],
        'sex': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male',
                'Female', 'Male'],
        'capital-gain': [0, 5178, 0, 0, 0, 15024, 0, 0],
        'capital-loss': [0, 0, 0, 0, 0, 0, 0, 0],
        'hours-per-week': [40, 50, 40, 45, 35, 55, 40, 40],
        'native-country': ['United-States', 'United-States', 'China',
                           'United-States', 'United-States', 'United-States',
                           'India', 'United-States'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K', '<=50K', '>50K',
                   '>50K', '<=50K']
    })
    return data


@pytest.fixture
def categorical_features():
    """
    Return the list of categorical features used in the project.
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


def test_process_data_returns_expected_types(sample_data, categorical_features):
    """
    Test that process_data function returns the expected data types.
    This tests if ML functions return expected type of result.
    """
    # Test training mode
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Check return types
    assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
    expected_encoder_type = OneHotEncoder
    assert isinstance(encoder, expected_encoder_type), \
        "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"

    # Check shapes
    expected_rows = len(sample_data)
    msg = "X_train should have same number of rows as input data"
    assert X_train.shape[0] == expected_rows, msg
    msg = "y_train should have same number of rows as input data"
    assert y_train.shape[0] == expected_rows, msg
    assert len(y_train.shape) == 1, "y_train should be 1-dimensional"


def test_train_model_algorithm_type(sample_data, categorical_features):
    """
    Test that train_model function uses expected algorithm (Random Forest).
    This tests if the ML model uses the expected algorithm.
    """
    # Process data for training
    X_train, y_train, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    # Train model
    model = train_model(X_train, y_train)

    # Check if the model is a Random Forest
    expected_model_type = RandomForestClassifier
    assert isinstance(model, expected_model_type), \
        "Model should be RandomForestClassifier"

    # Check expected Random Forest attributes
    assert hasattr(model, 'n_estimators'), \
        "Model should have n_estimators attribute"
    assert hasattr(model, 'max_depth'), \
        "Model should have max_depth attribute"
    assert hasattr(model, 'random_state'), \
        "Model should have random_state attribute"


def test_compute_model_metrics_expected_values():
    """
    Test that compute_model_metrics function returns expected values.
    This tests if computing metrics functions return expected value.
    """
    # Perfect predictions
    y_true_perfect = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0, 0, 1, 1])

    precision, recall, f1 = compute_model_metrics(y_true_perfect,
                                                  y_pred_perfect)

    msg = f"Perfect predictions should have precision 1.0, got {precision}"
    assert precision == 1.0, msg
    msg = f"Perfect predictions should have recall 1.0, got {recall}"
    assert recall == 1.0, msg
    msg = f"Perfect predictions should have F1 1.0, got {f1}"
    assert f1 == 1.0, msg

    # Check that metrics are in valid range [0, 1]
    y_true_mixed = np.array([0, 0, 1, 1, 0, 1])
    y_pred_mixed = np.array([0, 1, 1, 0, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true_mixed, y_pred_mixed)

    msg = f"Precision should be between 0 and 1, got {precision}"
    assert 0.0 <= precision <= 1.0, msg
    msg = f"Recall should be between 0 and 1, got {recall}"
    assert 0.0 <= recall <= 1.0, msg
    msg = f"F1 should be between 0 and 1, got {f1}"
    assert 0.0 <= f1 <= 1.0, msg


def test_inference_returns_expected_format(sample_data, categorical_features):
    """
    Test that inference function returns predictions in expected format.
    This tests if ML functions return expected type of result.
    """
    # Process data and train model
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )

    model = train_model(X_train, y_train)

    # Test inference
    predictions = inference(model, X_train)

    # Check return type and format
    assert isinstance(predictions, np.ndarray), \
        "Predictions should be numpy array"
    msg = "Should have one prediction per input sample"
    assert len(predictions) == len(X_train), msg
    msg = "Predictions should be integers"
    assert predictions.dtype in [np.int32, np.int64], msg

    # Check that predictions are binary (0 or 1)
    unique_preds = np.unique(predictions)
    msg = "Predictions should only be 0 or 1"
    assert all(pred in [0, 1] for pred in unique_preds), msg


def test_apply_labels():
    """
    Test that apply_label function works correctly.
    This tests the label conversion functionality.
    """
    # Test with single prediction (how the function actually works)
    single_pred = np.array([1])
    single_label = apply_label(single_pred)
    assert single_label == ">50K", f"Expected '>50K', got {single_label}"

    # Test with zero prediction
    zero_pred = np.array([0])
    zero_label = apply_label(zero_pred)
    assert zero_label == "<=50K", f"Expected '<=50K', got {zero_label}"
