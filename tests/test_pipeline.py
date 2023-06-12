"""
test ML model
"""
import pytest
from tools.utils import load_data, process_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

DATA_PATH = 'data/census.csv'
MODEL_PATH = 'model/model.pkl'

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

@pytest.fixture(name='data')
def data():
    """
    data will be used for testing
    """
    yield load_data(DATA_PATH)


def test_load_data(data):
    """
    Check data
    """
    assert data.shape[0] > 0
    assert data.shape[1] > 0

def test_data_shape(data):
    """ 
    Test no null values
    """
    assert data.shape == data.dropna().shape

def test_save_model():
    """ 
    Test model 
    """
    model = joblib.load(MODEL_PATH)
    assert isinstance(model, RandomForestClassifier)

def test_process_data(data):
    """ 
    Test the data split 
    """
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, 
        encoder=encoder, lb=lb)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
