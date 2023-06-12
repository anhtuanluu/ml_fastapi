"""test the enpoints"""
from fastapi.testclient import TestClient
# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

def test_get_root():
    """ Test the root page get a succesful response"""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello, This is a test"}

def test_predict_post_below():
    """ Test post api"""
    test_input = {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States"
            }
    r = client.post("/predict", json=test_input)
    assert r.status_code == 200
    assert r.json() == {"result": "<=50K"}

def test_predict_post_above():
    test_input = {
                "age": 31,
                "workclass": "Private",
                "fnlgt": 45781,
                "education": "Masters",
                "education_num": 14,
                "marital_status": "Never-married",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Female",
                "capital_gain": 14084,
                "capital_loss": 0,
                "hours_per_week": 50,
                "native_country": "United-States"
            }
    """ Test post api"""
    r = client.post("/predict", json=test_input)
    assert r.status_code == 200
    assert r.json() == {"result": ">50K"}
