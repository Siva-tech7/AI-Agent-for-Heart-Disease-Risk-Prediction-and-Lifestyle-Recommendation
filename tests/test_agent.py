import numpy as np
import pandas as pd
from src.agent import HeartDiseaseAgent, categorize_risk

class MockPreprocessor:
    def transform(self, df):
        return np.array([[1, 2, 3]])
        
def mock_predict_proba(X):
    # return shape (n, 2)
    return np.array([[0.2, 0.8]])

def test_categorize_risk():
    assert categorize_risk(0.1) == "Low"
    assert categorize_risk(0.5) == "Moderate"
    assert categorize_risk(0.9) == "High"

def test_agent_prediction():
    agent = HeartDiseaseAgent(
        preprocessor=MockPreprocessor(),
        predict_proba_fn=mock_predict_proba,
        feature_cols=["f1", "f2", "f3"],
        feature_names_out=["f1", "f2", "f3"],
        model_name="MockModel",
        explainer=None
    )
    
    patient = {"f1": 1, "f2": 2, "f3": 3}
    report = agent.predict(patient, explain=False)
    
    assert report["predicted_class"] == 1
    assert report["probability"] == 0.8
    assert report["risk_category"] == "High"
    assert "MockModel" in report["model_used"]
    assert "disclaimer" in report
    assert isinstance(report["lifestyle_recommendations"], list)
