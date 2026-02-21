from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import os
import librosa
import numpy as np
import torch
import torchaudio
import joblib

from app.models import Assessment, SpeechSample

# Trigger Reload 6


class AudioFeatureExtractor:
    def __init__(self) -> None:
        self.sample_rate = 16000
        self.mfcc_config = {"n_mfcc": 40, "n_fft": 1024, "hop_length": 512}

    def load_audio(self, file_path: Path | str) -> Tuple[np.ndarray, int]:
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.numpy().squeeze(), self.sample_rate

    def extract_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, **self.mfcc_config)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_cross = librosa.feature.zero_crossing_rate(y=audio)

        return {
            "mfcc_mean": mfcc.mean(axis=1).tolist(),
            "mfcc_std": mfcc.std(axis=1).tolist(),
            "spectral_rolloff_mean": float(spectral_rolloff.mean()),
            "zero_cross_mean": float(zero_cross.mean()),
        }


class SpeechEmbeddingExtractor:
    """
    Lazily loads Wav2Vec2 only when ENABLE_SPEECH_EMBEDDINGS=1.
    Falls back to a zero vector if the model cannot be loaded (keeps free-tier deploys stable).
    """

    def __init__(self) -> None:
        self.processor = None
        self.model = None

        enable = os.getenv("ENABLE_SPEECH_EMBEDDINGS", "0") == "1"
        if not enable:
            return

        try:
            # Import transformers lazily to avoid heavy import unless explicitly enabled
            from transformers import AutoModel, AutoProcessor  # type: ignore

            self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = AutoModel.from_pretrained("facebook/wav2vec2-base-960h")
            self.model.eval()
        except Exception:
            # Leave processor/model as None to trigger fallback
            self.processor = None
            self.model = None

    def get_embeddings(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.processor is None or self.model is None:
            # Return a stable dimensionality for downstream averaging
            return np.zeros((1, 768), dtype=np.float32)

        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()



class RiskPredictor:
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.pca: Optional[Any] = None
        self.scaler: Optional[Any] = None
        
        # Path resolution (handle relative to app module)
        base_path = Path(__file__).parent.parent
        model_path = base_path / "model.pkl"
        pca_path = base_path / "pca_transform.pkl"
        scaler_path = base_path / "scaler.pkl"

        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
            if pca_path.exists():
                self.pca = joblib.load(pca_path)
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            print(f"[RiskPredictor] Loaded Fused ML Ensemble & Transformers")
        except Exception as e:
            print(f"[RiskPredictor] Error loading models: {e}")
            self.model = None

    def _fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Heuristic fallback relying on MMSE and FunctionalAssessment which are the strongest predictors
        in the Alzheimer's Disease Dataset (Rabie El Kharoua).
        """
        tab = features.get("tabular", {})
        
        # Key Predictors
        mmse = tab.get("MMSE", 30.0)             # Range 0-30 (Lower is worse)
        func = tab.get("FunctionalAssessment", 10.0) # Range 0-10 (Lower is worse)
        adl = tab.get("ADL", 10.0)               # Range 0-10 (Lower is worse)
        mem_complaint = tab.get("MemoryComplaints", 0) # 0=No, 1=Yes
        
        # Risk Calculation (Heuristic)
        # MMSE < 24 is typically indicative of impairment
        score_risk = 0.0
        if mmse < 24: score_risk += 0.4
        if mmse < 20: score_risk += 0.3
        
        func_risk = 0.0
        if func < 6: func_risk += 0.2
        if adl < 6: func_risk += 0.1
        
        complaint_risk = 0.2 if mem_complaint == 1 else 0.0
        
        probability = 0.1 + score_risk + func_risk + complaint_risk
        probability = max(0.02, min(0.98, probability))
        
        risk_level = "Low" if probability < 0.33 else ("Medium" if probability < 0.66 else "High")

        return {
            "risk_level": risk_level,
            "probability": probability,
            "feature_importances": [
                {"feature": "MMSE (Cognitive)", "contribution": 0.45, "direction": "negative"},
                {"feature": "Functional Assessment", "contribution": 0.25, "direction": "negative"},
                {"feature": "Memory Complaints", "contribution": 0.15, "direction": "positive"},
                {"feature": "Age", "contribution": 0.15, "direction": "positive"},
            ],
            "recommendations": [
                "Share results with a clinician" if probability > 0.3 else "Continue healthy habits",
                "Focus on memory exercises" if mmse < 25 else "Maintain cognitive activity",
                "Monitor daily routine difficulties" if func < 8 else None,
            ],
        }

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.pca is None or self.scaler is None:
            return self._fallback(features)

        tab = features.get("tabular", {})
        speech_raw = features.get("speech_embedding", [0.0] * 768)
        
        # 1. Construct Tabular part (15 features)
        diabetes = tab.get("Diabetes", 0)
        hypert = tab.get("Hypertension", 0)
        smoking = tab.get("Smoking", 0)
        cardio_score = diabetes + hypert + smoking
        
        sleep = tab.get("SleepQuality", 7)
        activity = tab.get("PhysicalActivity", 5)
        lifestyle_deficit = (1 if sleep < 6 else 0) + (1 if activity < 4 else 0)

        try:
            # Match training feature order
            tabular_vec = [
                tab.get("Age", 60),
                tab.get("Gender", 0),
                tab.get("EducationLevel", 0),
                tab.get("MMSE", 30.0),
                tab.get("FunctionalAssessment", 10.0),
                tab.get("MemoryComplaints", 0),
                tab.get("ADL", 10.0),
                tab.get("FamilyHistoryAlzheimers", 0),
                tab.get("HeadInjury", 0),
                tab.get("Depression", 0),
                cardio_score,
                lifestyle_deficit,
                tab.get("BMI", 25.0),
                tab.get("AlcoholConsumption", 0.0),
                tab.get("DietQuality", 5.0)
            ]

            # 2. Multi-Modal Fusion (The "Crack")
            # PCA dimensionality reduction for speech embeddings
            speech_vec = np.array(speech_raw).reshape(1, -1)
            speech_reduced = self.pca.transform(speech_vec).flatten()
            
            # 3. Concatenate Fused Vector (Tabular + Reduced Speech)
            fused_vec = np.concatenate([tabular_vec, speech_reduced]).reshape(1, -1)
            
            # 4. Standardize
            fused_scaled = self.scaler.transform(fused_vec)

            # 5. Predict
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(fused_scaled)[0, 1])
            else:
                proba = float(self.model.predict(fused_scaled)[0])
                proba = max(0.0, min(1.0, proba))
            
            # --- LONGITUDINAL BOOST ---
            score_decline = tab.get("ScoreDecline", 0)
            if score_decline == 1:
                proba = min(0.99, proba + 0.3) 

            risk_level = "Low" if proba < 0.33 else ("Medium" if proba < 0.66 else "High")

            # Feature Importances (Dynamic Mapping)
            fi = []
            if hasattr(self.model, "feature_importances_") or hasattr(self.model, "estimators_"):
                # Composite names
                names = [
                    'Age', 'Gender', 'Education', 'MMSE', 'Functional', 'MemoryComplaints', 'ADL',
                    'FamilyHist', 'HeadInjury', 'Depression', 'CardioScore', 'LifestyleDeficit',
                    'BMI', 'Alcohol', 'DietQuality'
                ] + [f'SpeechBio_{i}' for i in range(10)]
                
                # For VotingClassifier, we take weighted importance
                try:
                    if hasattr(self.model, "estimators_"):
                        importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
                    else:
                        importances = self.model.feature_importances_
                        
                    indices = np.argsort(importances)[::-1][:5]
                    total = float(np.sum(np.abs(importances))) or 1.0
                    for i in indices:
                        fi.append({
                            "feature": names[i],
                            "contribution": float(importances[i]) / total,
                            "direction": "negative" if i in [3, 4, 6] else "positive" # Cognitive/ADL are negative drivers
                        })
                except: pass

            return {
                "risk_level": risk_level,
                "probability": proba,
                "feature_importances": fi or self._fallback(features)["feature_importances"],
                "recommendations": [
                    "Consult a neurologist for a detailed assessment" if proba > 0.5 else "Maintain a healthy lifestyle",
                    "Monitor speech and memory patterns" if proba > 0.3 else None,
                    "Incoherent speech detection triggered" if np.abs(speech_reduced).mean() > 2.0 else None,
                ],
            }
        except Exception as e:
            print(f"Prediction Error: {e}")
            return self._fallback(features)


class PipelineManager:
    def __init__(self) -> None:
        self.audio_feature_extractor = AudioFeatureExtractor()
        self.speech_embedding_extractor = SpeechEmbeddingExtractor()
        self.predictor = RiskPredictor()

    def process_assessment(self, assessment: Assessment, samples: List[SpeechSample]) -> Dict[str, Any]:
        speech_embeddings: List[np.ndarray] = []
        audio_feature_map: Dict[str, Dict[str, Any]] = {}

        transcript_tokens: List[str] = []
        languages: List[str] = []
        mismatch_flags: List[bool] = []
        language_confidences: List[float] = []

        for sample in samples:
            try:
                audio, sr = self.audio_feature_extractor.load_audio(sample.file_path)
                audio_feature_map[sample.task_id] = self.audio_feature_extractor.extract_features(audio, sr)
                speech_embeddings.append(self.speech_embedding_extractor.get_embeddings(audio, sr))
            except Exception:
                continue

            if sample.transcript: transcript_tokens.extend(sample.transcript.split())
            if sample.detected_language: languages.append(sample.detected_language)

        if speech_embeddings:
            speech_embedding = np.mean(np.concatenate(speech_embeddings, axis=0), axis=0).tolist()
        else:
            speech_embedding = [0.0] * 768

        # --- DATASET MAPPING (Alzheimer's Disease Dataset - Rabie El Kharoua) ---
        
        # 1. Age (60-90) & Demographics & Health
        user_age = 60
        gender = 0
        education = 0
        # Health Defaults
        fam_hist = 0
        diabetes = 0
        hypert = 0
        depress = 0
        head_inj = 0
        sleep = 7
        phys_act = 5
        smoking = 0
        
        if assessment.user:
            user_age = assessment.user.age or 60
            gender = assessment.user.gender or 0
            education = assessment.user.education or 0
            fam_hist = assessment.user.family_history or 0
            diabetes = assessment.user.diabetes or 0
            hypert = assessment.user.hypertension or 0
            depress = assessment.user.depression or 0
            head_inj = assessment.user.head_injury or 0
            sleep = assessment.user.sleep_quality or 7
            phys_act = assessment.user.physical_activity or 5
            smoking = assessment.user.smoking or 0
            
            # Phase 3: New Fields
            alcohol = assessment.user.alcohol_consumption or 0.0
            diet = assessment.user.diet_quality or 5.0
            h_cm = assessment.user.height or 0.0
            w_kg = assessment.user.weight or 0.0
            
            # Calculate BMI
            bmi = 25.0 # Default
            if h_cm > 0 and w_kg > 0:
                h_m = h_cm / 100.0
                bmi = w_kg / (h_m * h_m)
                # Clamp logical range
                bmi = max(10.0, min(50.0, bmi))

        # 2. MMSE (0-30)
        # Derived from Memory + Language scores (0.0 - 1.0)
        # We average them and scale to 30.
        mem_val = assessment.memory_score or 0.0
        lang_val = assessment.language_score or 0.0
        global_score = (mem_val + lang_val) / 2.0 if (mem_val + lang_val) > 0 else 0.0
        mmse_val = global_score * 30.0
        
        # 3. FunctionalAssessment (0-10)
        # Derived from Executive Score (0.0 - 1.0)
        exec_val = assessment.executive_score or 0.0
        func_val = exec_val * 10.0
        
        # 4. MemoryComplaints (0 or 1)
        # If memory score is low (< 0.4), assume complaints.
        mem_complaints = 1 if mem_val < 0.4 else 0
        
        # 5. ADL (0-10) - Activities of Daily Living
        # Strongly correlated with Executive Function.
        adl_val = exec_val * 10.0

        tabular_features = {
            "Age": user_age,
            "Gender": gender,
            "EducationLevel": education,
            # New Health Features
            "FamilyHistoryAlzheimers": fam_hist,
            "Diabetes": diabetes,
            "Hypertension": hypert,
            "Depression": depress,
            "HeadInjury": head_inj,
            "SleepQuality": sleep,
            "PhysicalActivity": phys_act,
            "Smoking": smoking,
            "AlcoholConsumption": alcohol,
            "DietQuality": diet,
            "BMI": bmi,
            # Computed
            "MMSE": mmse_val,
            "FunctionalAssessment": func_val,
            "MemoryComplaints": mem_complaints,
            "ADL": adl_val,
            # Raw scores for debug
            "raw_memory": mem_val,
            "raw_executive": exec_val,
        }

        features = {
            "audio": audio_feature_map,
            "speech_embedding": speech_embedding,
            "tabular": tabular_features,
            "transcript_tokens": transcript_tokens,
            "languages": languages,
        }

        prediction = self.predictor.predict(features)
        return {"prediction": prediction, "features": features}


pipeline_manager = PipelineManager()
