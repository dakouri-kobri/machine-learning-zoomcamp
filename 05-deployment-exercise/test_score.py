import requests

url = "http://localhost:9696/predict-score"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}
score_response = requests.post(url, json=client).json()
print("score_response:", score_response)