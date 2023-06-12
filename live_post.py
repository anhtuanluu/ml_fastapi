"""
Test post
"""
import requests

API_URL = "http://income-predict.onrender.com/predict"

test_input = {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": 'Bachelors',
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

r = requests.post(API_URL, json=test_input)

print("Testing app")
print(f"Code: {r.status_code}")
print(f"Body: {r.json()}")
