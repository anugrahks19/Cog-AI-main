import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "alzheimers_disease_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "app", "model.pkl")

def train():
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print("Error: alzheimers_disease_data.csv not found.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Selection (Now includes Phase 1 Health Context)
    # Feature Engineering (Interaction Features)
    # 1. Cardiovascular Risk Score (Combines Hypertension, Diabetes, Smoking, etc.)
    # We sum the risk factors to create a compound index.
    # Note: Cholesterol columns might be missing in our simplified feature list, 
    # but we use what we have in the dataframe (it has all columns).
    
    # Normalize Cholesterol if present, otherwise default logic
    df['CardiovascularScore'] = (
        df['Diabetes'] + 
        df['Hypertension'] + 
        df['Smoking'] + 
        (df['systolicbp'] > 140).astype(int) if 'systolicbp' in df.columns else 0
    )
    
    # 2. Lifestyle Deficit Score (Inverted: Higher is worse)
    # Poor Sleep (<6) + Low Activity (<4)
    df['LifestyleDeficit'] = (
        (df['SleepQuality'] < 6).astype(int) + 
        (df['PhysicalActivity'] < 4).astype(int)
    )

    # 1. Feature Selection (Now includes Derived Features)
    features = [
        'Age', 'Gender', 'EducationLevel', 
        'MMSE', 'FunctionalAssessment', 'MemoryComplaints', 'ADL',
        'FamilyHistoryAlzheimers', 'HeadInjury', 'Depression',
        'CardiovascularScore', 'LifestyleDeficit',
        'BMI', 'AlcoholConsumption', 'DietQuality'
    ]
    target = 'Diagnosis'
    
    print(f"Training on {len(features)} features (Inc. Interactions): {features}")
    
    X = df[features]
    y = df[target]
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Upgrade: XGBoost with Hyperparameter Tuning
    print("Starting Grid Search for XGBoost...")
    
    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    # Grid to find the mathematical optimum
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # 4. Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy (XGBoost): {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 5. Save Model
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
