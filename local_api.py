import json

import requests


r = requests.get(
    url='http://127.0.0.1:8000',
    timeout=5
)


print(r.status_code)

print(r.json())



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


r = requests.post(
    url='http://127.0.0.1:8000/data/',
    json=data,
    timeout=5
)


print(r.status_code)

print(r.json())
