import requests

BASE_URL = "http://127.0.0.1:8000/api"
def run_test():
    # Because we don't have a user token readily available in the script,
    # let's write a direct test against ml_pipeline or the predictor directly.
    from app.services.ml_pipeline import pipeline_manager
    from app.models import Assessment, User
    
    # Mock Assessment and User
    class MockUser:
        id = "test_user_id"
        email = "doctor@example.com"
        full_name = "Test Patient"
        age = 75
        gender = 1
        education = 1
        family_history = 1
        diabetes = 1
        hypertension = 1
        depression = 0
        head_injury = 0
        sleep_quality = 4
        physical_activity = 2
        smoking = 1
        alcohol_consumption = 0
        diet_quality = 3
        height = 170
        weight = 80
        
    class MockAssessment:
        id = "test_asmt_id"
        user = MockUser()
        user_id = MockUser.id
        memory_score = 0.0
        language_score = 0.0
        executive_score = 0.0
        
    res = pipeline_manager.process_assessment(MockAssessment(), [])
    pred = res.get("prediction", {})
    
    # Force High Risk for testing if not already High
    if pred.get("risk_level") != "High":
        pred["risk_level"] = "High"
        pred["probability"] = 0.99
    
    print("\n[E2E Test] Prediction Result:")
    print(pred)
    
    # Trigger the Email Logic directly to test
    if pred.get("risk_level") == "High":
        from app.services.email_service import send_risk_alert_email
        send_risk_alert_email(
            recipient_email="anugrahks2006@gmail.com",
            assessment_id="test_asmt_id",
            risk_level=pred.get("risk_level"),
            probability=pred.get("probability", 0.0),
            feature_importances=pred.get("feature_importances", []),
            patient_name="Test Patient"
        )
        
if __name__ == "__main__":
    run_test()
