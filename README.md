# Cog.ai - AI-Powered Dementia Screening Platform

**Empowering early detection through AI-driven cognitive assessment.**

Cog.ai is a non-invasive, multi-modal screening tool that uses voice biomarkers, cognitive games, and health context to predict the risk of Alzheimer's and other forms of dementia with high accuracy (+90%).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have the following installed:
*   **Node.js** (v18 or higher)
*   **Python** (v3.9 or higher)
*   **Git**

### 2. Backend Setup (The Brain)
The backend handles the AI models, database, and API logic.

1.  **Navigate to the backend folder**:
    ```bash
    cd backend
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train/Load the AI Model**:
    *   Ensure `alzheimers_disease_data.csv` is in the project root.
    *   Run the training script (this generates `app/model.pkl`):
    ```bash
    python train_model.py
    ```

5.  **Run the Server**:
    ```bash
    uvicorn app.main:app --reload
    ```
    *   The API will start at `http://localhost:8000`.

### 3. Frontend Setup (The Interface)
The frontend is a modern React + Vite application.

1.  **Open a new terminal** and navigate to the project root:
    ```bash
    cd d:\Cog-AI-main\Cog-AI-main
    ```

2.  **Install dependencies**:
    ```bash
    npm install
    ```

3.  **Configure Environment**:
    *   Create a `.env` file in the root if one doesn't exist.
    *   Add your Firebase configuration keys (see `.env.example`).

4.  **Start the App**:
    ```bash
    npm run dev
    ```
    *   The app will open at `http://localhost:5173`.

---

## 🧠 Features

### 1. Multi-Modal Assessment
*   **Cognitive Games**: Stroop Test, N-Back, Memory Sequence.
*   **Voice Analysis**: Analyzes speech patterns for micro-tremors and pauses.
*   **Health Context**: Integtrates BMI, Lifestyle, and Medical History.

### 2. AI Prediction Engine
*   **Model**: XGBoost Classifier (Tuned).
*   **Accuracy**: ~90% on validation set.
*   **Calibration**: Uses 15+ distinct health and cognitive markers.

### 3. User Dashboard
*   **Risk Report**: Clear "Low/Medium/High" risk classification.
*   **Explainable AI**: See exactly *why* a result was given (e.g., "High Sleep Deficit").
*   **Longitudinal Tracking**: Monitors decline over time.

---

## 🛠️ Tech Stack
*   **Frontend**: React, Vite, TailwindCSS, ShadcnUI.
*   **Backend**: Python, FastAPI, SQLAlchemy.
*   **AI/ML**: XGBoost, Scikit-Learn, Pandas, Numpy.
*   **Database**: SQLite (Local Dev) / PostgreSQL (Prod).
*   **Auth**: Firebase Authentication.

---

## 📜 License
This project is for educational and research purposes.
