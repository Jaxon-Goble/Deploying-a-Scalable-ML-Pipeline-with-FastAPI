# Import all unit test dependencies
import pytest
import pandas as pd

from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    inference,
    train_model
)

# Import necessary variables from train_model for testing
from train_model import data as raw_data
from train_model import cat_features as cat_features

@pytest.fixture
def test_data():
    """
    Provides the dataset for testing.
    """
    return raw_data.copy()

@pytest.fixture
def categorical_features():
    """
    Provides categorical feature names.
    """
    return cat_features


# Data
# TODO: implement the first test. Change the function name and input as needed
def test_null_values(test_data: pd.DataFrame):
    """
    Takes in a dataframe as input, and check whether or not it has zero null values.
    """
    null_count = test_data.isnull().sum().sum()

    # Run null value count test
    assert null_count == 0, f"Dataset has {null_count} null values."


# Data
# TODO: implement the second test. Change the function name and input as needed
def test_data_split(test_data: pd.DataFrame):
    """
    Checks train_test_split dataset outputs to ensure they are the correct proportions and they add up to the original dataset.
    """
    train, test = train_test_split(test_data, test_size=0.15, random_state=14)

    # Run split dataset tests
    assert train.shape[0] + test.shape[0] == test_data.shape[0]
    assert 0.149 < (test.shape[0] / test_data.shape[0]) < 0.151


# Model
# TODO: implement the third test. Change the function name and input as needed
def test_model_predictions(test_data: pd.DataFrame, categorical_features: list[str]):
    """
    Checks model predictions array for lack of results and/or incorrect results (not 0 or 1).
    """
    # Follow train_model.py process to reach model predictions
    train, test = train_test_split(test_data, test_size=0.15, random_state=14)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    # Run predictions array tests
    assert len(preds) != 0, "Predictions array is empty."
    assert set(preds) <= {0, 1}, "Incorrect values detected in predictions array."
