import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# Load the census.csv data
project_path = "."  # Current directory (change this to your actual path if needed)
data_path = os.path.join(project_path, "data", "census.csv")
print(f"Loading data from: {data_path}")
data = pd.read_csv(data_path)  # Load the census data

# Split the provided data to have a train dataset and a test dataset
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=42, stratify=data['salary'])
print(f"Train set size: {len(train)}, Test set size: {len(test)}")

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,                        # use the train dataset
    categorical_features=cat_features,
    label="salary",
    training=True                 # use training=True
    # do not need to pass encoder and lb as input (they will be created)
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Use the train_model function to train the model on the training dataset
print("Training model...")
model = train_model(X_train, y_train)  # Train the model using the training data

# Create model directory if it doesn't exist
os.makedirs(os.path.join(project_path, "model"), exist_ok=True)

# Save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
print(f"Model saved to: {model_path}")
print(f"Encoder saved to: {encoder_path}")

# Load the model (to verify saving/loading works)
model = load_model(model_path)

# Use the inference function to run the model inferences on the test dataset.
print("Running inference on test set...")
preds = inference(model, X_test)  # Run inference on the test dataset

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Overall Model Performance:")
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute the performance on model slices using the performance_on_categorical_slice function
print("Computing performance on categorical slices...")

# Clear the slice output file before writing new results
slice_output_path = "slice_output.txt"
with open(slice_output_path, "w") as f:
    f.write("")  # Clear the file

# Iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=test,                    # use test data
            column_name=col,              # use col (current categorical feature)
            slice_value=slicevalue,       # use slicevalue (current unique value)
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            model=model
        )
        with open(slice_output_path, "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

print(f"Slice performance results saved to: {slice_output_path}")
print("Training and evaluation complete!")
