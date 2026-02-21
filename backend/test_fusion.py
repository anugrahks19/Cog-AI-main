import sys
import os
import numpy as np

# Mocking the app structure for local testing
sys.path.append(os.getcwd())

from app.services.ml_pipeline import RiskPredictor

def test_fusion():
    print("=== Cog-AI Multi-Modal Fusion Verification ===\n")
    predictor = RiskPredictor()
    
    # Test Case 1: Healthy Profile + Healthy Speech
    print("[Test 1] Healthy Profile + Healthy Speech")
    healthy_features = {
        "tabular": {
            "Age": 60, "Gender": 0, "MMSE": 29, "FunctionalAssessment": 9, "ADL": 9,
            "Diabetes": 0, "Hypertension": 0, "Smoking": 0, "SleepQuality": 8, "PhysicalActivity": 5
        },
        "speech_embedding": np.random.normal(0, 0.1, 768).tolist() # Low variance speech
    }
    res1 = predictor.predict(healthy_features)
    print(f"  > Risk: {res1['risk_level']} ({res1['probability']:.4f})")
    
    # Test Case 2: Healthy Profile + "Impaired" Speech (Simulated via high variance/shift)
    print("\n[Test 2] Healthy Profile + Impaired Speech (THE CRACK)")
    impaired_speech_features = {
        "tabular": healthy_features["tabular"],
        "speech_embedding": (np.random.normal(0, 1, 768) + 1.5).tolist() # Shifted speech
    }
    res2 = predictor.predict(impaired_speech_features)
    print(f"  > Risk: {res2['risk_level']} ({res2['probability']:.4f})")
    
    # Verifying "The Crack": Probability should increase even with same tabular data
    if res2['probability'] > res1['probability']:
        print("\n✅ SUCCESS: Speech characteristics are successfully influencing the risk score!")
    else:
        print("\n❌ FAILURE: Speech characteristics were ignored.")

    # Test Case 3: The "Meow" Test (Total Failure)
    print("\n[Test 3] Acute Cognitive Failure (MMSE=0) + Healthy Demographics")
    meow_features = {
        "tabular": {
            "Age": 60, "Gender": 0, "MMSE": 0, "FunctionalAssessment": 0, "ADL": 0,
            "Diabetes": 0, "Hypertension": 0, "Smoking": 0
        },
        "speech_embedding": (np.random.normal(0, 2, 768) + 3.0).tolist() # Highly irregular
    }
    res3 = predictor.predict(meow_features)
    print(f"  > Risk: {res3['risk_level']} ({res3['probability']:.4f})")
    print(f"  > Top Factors: {[f['feature'] for f in res3['feature_importances']]}")

if __name__ == "__main__":
    test_fusion()
