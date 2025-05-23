"""
Functions for training, inference, and model management.
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import logging

# Optional: use this to set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn model
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """
    Loads pickle file from `path` and returns it.

    Inputs
    ------
    path : str
        Path to the pickle file.

    Returns
    -------
    model
        Loaded model or encoder.
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics when the value of a given feature is held fixed.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    column_name : str
        Column containing the fixed feature.
    slice_value : str, int, float
        Value of the feature in `column_name` to filter on.
    categorical_features : list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `data`. (default="salary")
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    model : sklearn model
        Trained machine learning model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float

    """
    # Import here to avoid circular imports
    from ml.data import process_data
    
    # Filter the data for the specific slice
    data_slice = data[data[column_name] == slice_value]

    if len(data_slice) == 0:
        logger.warning(f"No data found for {column_name}={slice_value}")
        return 0.0, 0.0, 0.0

    # Process the slice data - FIXED: properly unpack all 4 return values
    X_slice_processed, y_slice_processed, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Make predictions
    preds = inference(model, X_slice_processed)

    # Compute metrics
    precision, recall, fbeta = compute_model_metrics(y_slice_processed, preds)

    return precision, recall, fbeta