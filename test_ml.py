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


def test_train_fresh_model_and_validate_performance(sample_data, categorical_features):
    """
    Train a fresh model and validate its performance.
    This avoids version compatibility issues with saved models.
    """
    # Process data for training
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Train a fresh model
    model = train_model(X_train, y_train)
    
    # Test model type and attributes
    assert isinstance(model, RandomForestClassifier), \
        f"Expected RandomForestClassifier, got {type(model)}"
    
    # Test model has been fitted
    assert hasattr(model, 'classes_'), \
        "Model should be fitted and have classes_ attribute"
    
    # Test predictions on training data
    predictions = inference(model, X_train)
    
    # Test output format
    assert isinstance(predictions, np.ndarray), \
        "Predictions should be numpy array"
    assert len(predictions) == len(sample_data), \
        f"Expected {len(sample_data)} predictions, got {len(predictions)}"
    
    # Test that predictions are binary (0 or 1)
    unique_predictions = np.unique(predictions)
    assert all(pred in [0, 1] for pred in unique_predictions), \
        f"Predictions should only be 0 or 1, got {unique_predictions}"
    
    # Test model performance
    precision, recall, f1 = compute_model_metrics(y_train, predictions)
    
    # Test that metrics are valid
    assert 0.0 <= precision <= 1.0, \
        f"Precision should be between 0 and 1, got {precision}"
    assert 0.0 <= recall <= 1.0, \
        f"Recall should be between 0 and 1, got {recall}"
    assert 0.0 <= f1 <= 1.0, \
        f"F1 should be between 0 and 1, got {f1}"


def test_model_prediction_consistency(sample_data, categorical_features):
    """
    Test that a trained model gives consistent predictions for identical inputs.
    """
    # Process data for training
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Train a fresh model
    model = train_model(X_train, y_train)
    
    # Create identical test samples
    test_data = pd.DataFrame({
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
    
    # Process test data
    X_test, _, _, _ = process_data(
        test_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Get predictions
    predictions = inference(model, X_test)
    
    # Test that identical inputs give identical predictions
    assert predictions[0] == predictions[1], \
        f"Identical inputs should give identical predictions, got {predictions}"


def test_model_handles_different_inputs(sample_data, categorical_features):
    """
    Test that the model can handle various types of input data.
    """
    # Process data for training
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Train a fresh model
    model = train_model(X_train, y_train)
    
    # Test with different demographic profiles
    test_cases = [
        # High income profile
        {
            'age': [45], 'workclass': ['Private'], 'fnlwgt': [180000],
            'education': ['Masters'], 'education-num': [14],
            'marital-status': ['Married-civ-spouse'], 'occupation': ['Exec-managerial'],
            'relationship': ['Husband'], 'race': ['White'], 'sex': ['Male'],
            'capital-gain': [15024], 'capital-loss': [0], 'hours-per-week': [50],
            'native-country': ['United-States'], 'salary': ['>50K']
        },
        # Lower income profile
        {
            'age': [22], 'workclass': ['Private'], 'fnlwgt': [120000],
            'education': ['HS-grad'], 'education-num': [9],
            'marital-status': ['Never-married'], 'occupation': ['Adm-clerical'],
            'relationship': ['Not-in-family'], 'race': ['Black'], 'sex': ['Female'],
            'capital-gain': [0], 'capital-loss': [0], 'hours-per-week': [25],
            'native-country': ['United-States'], 'salary': ['<=50K']
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        test_df = pd.DataFrame(test_case)
        
        # Process test data
        X_test, _, _, _ = process_data(
            test_df,
            categorical_features=categorical_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb
        )
        
        # Get predictions
        predictions = inference(model, X_test)
        
        # Test that prediction is valid
        assert len(predictions) == 1, \
            f"Expected 1 prediction for test case {i}, got {len(predictions)}"
        assert predictions[0] in [0, 1], \
            f"Prediction should be 0 or 1, got {predictions[0]}"


def test_apply_label_function():
    """
    Test that apply_label correctly converts predictions to labels.
    """
    # Test with different prediction arrays
    test_cases = [
        (np.array([0]), "<=50K"),
        (np.array([1]), ">50K"),
    ]
    
    for prediction, expected_label in test_cases:
        result = apply_label(prediction)
        assert result == expected_label, \
            f"Expected '{expected_label}' for prediction {prediction[0]}, got '{result}'"
        assert isinstance(result, str), \
            f"Label should be string, got {type(result)}"


def test_data_processing_pipeline(sample_data, categorical_features):
    """
    Test that the data processing pipeline works correctly.
    """
    # Test training mode
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=True
    )
    
    # Test basic properties
    assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
    assert isinstance(encoder, OneHotEncoder), "encoder should be OneHotEncoder"
    assert isinstance(lb, LabelBinarizer), "lb should be LabelBinarizer"
    
    # Test shapes
    assert X_train.shape[0] == len(sample_data), \
        f"X_train should have {len(sample_data)} rows, got {X_train.shape[0]}"
    assert y_train.shape[0] == len(sample_data), \
        f"y_train should have {len(sample_data)} rows, got {y_train.shape[0]}"
    
    # Test inference mode
    X_test, y_test, _, _ = process_data(
        sample_data,
        categorical_features=categorical_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    
    # Test that inference mode produces same results
    assert np.array_equal(X_train, X_test), \
        "Same data should produce same features in training vs inference mode"
    assert np.array_equal(y_train, y_test), \
        "Same data should produce same labels in training vs inference mode"


def test_compute_model_metrics_function():
    """
    Test that compute_model_metrics calculates correct values.
    """
    # Test with known values
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 0])  # One false negative
    
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    
    # Test that metrics are valid numbers
    assert isinstance(precision, (float, np.floating)), \
        f"Precision should be float, got {type(precision)}"
    assert isinstance(recall, (float, np.floating)), \
        f"Recall should be float, got {type(recall)}"
    assert isinstance(f1, (float, np.floating)), \
        f"F1 should be float, got {type(f1)}"
    
    # Test ranges
    assert 0.0 <= precision <= 1.0, \
        f"Precision should be between 0 and 1, got {precision}"
    assert 0.0 <= recall <= 1.0, \
        f"Recall should be between 0 and 1, got {recall}"
    assert 0.0 <= f1 <= 1.0, \
        f"F1 should be between 0 and 1, got {f1}"
    
    # Test specific values for this case
    # Precision = TP/(TP+FP) = 2/(2+0) = 1.0
    # Recall = TP/(TP+FN) = 2/(2+1) = 0.667
    assert precision == 1.0, f"Expected precision 1.0, got {precision}"
    assert abs(recall - 2/3) < 0.01, f"Expected recall ~0.667, got {recall}"