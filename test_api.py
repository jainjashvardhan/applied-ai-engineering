# test_api.py — create this file

import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    print("✅ Health check passed")

def test_valid_cv():
    payload = {
        "cv_text": """
            Sarah Chen — Senior Data Engineer
            5 years experience at Razorpay and Swiggy.
            Built Kafka-based event streaming pipeline handling 5M events/day.
            Skills: Python, Apache Kafka, Spark, dbt, Airflow, PostgreSQL, AWS.
            Led team of 3 engineers. B.Tech IIT Bombay 2018.
        """,
        "role": "Senior Data Engineer",
        "experience_level": "senior"
    }
    response = requests.post(f"{BASE_URL}/analyse-cv", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["verdict"] in ["recommend", "consider", "reject"]
    assert 0 <= data["confidence"] <= 100
    assert len(data["strengths"]) == 3
    assert len(data["interview_questions"]) == 3
    print(f"✅ Valid CV test passed | verdict: {data['verdict']} | confidence: {data['confidence']}")
    print(f"   Strengths: {data['strengths']}")

def test_empty_cv_rejected():
    payload = {"cv_text": "hi", "role": "Engineer"}
    response = requests.post(f"{BASE_URL}/analyse-cv", json=payload)
    # Should be rejected by our validator — 422 Unprocessable Entity
    assert response.status_code == 422
    print("✅ Empty CV validation test passed")

def test_missing_role():
    payload = {"cv_text": "Some CV text that is long enough to pass validation"}
    response = requests.post(f"{BASE_URL}/analyse-cv", json=payload)
    assert response.status_code == 422
    print("✅ Missing role validation test passed")

def test_invalid_experience_level():
    payload = {
        "cv_text": "A" * 100,
        "role": "Engineer",
        "experience_level": "expert"   # not in allowed values
    }
    response = requests.post(f"{BASE_URL}/analyse-cv", json=payload)
    assert response.status_code == 422
    print("✅ Invalid experience level test passed")

if __name__ == "__main__":
    print("\n🔍 Running CV Analyser API tests...\n")
    test_health()
    test_empty_cv_rejected()
    test_missing_role()
    test_invalid_experience_level()
    test_valid_cv()      # this calls Claude — costs tokens
    print("\n✅ All tests passed\n")


