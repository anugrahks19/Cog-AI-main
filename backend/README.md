# Cog.ai Mindful Scan Backend

This backend implements the FastAPI services, data models, and machine learning
pipeline required for the Cog.ai Mindful Scan platform. It exposes REST
endpoints for user onboarding, consent management, multimedia assessment data
capture, feature extraction, risk prediction, and result retrieval.

## Tech Stack

- **FastAPI** for API services
- **PostgreSQL + SQLAlchemy** for persistence
- **Alembic** for migrations
- **Torchaudio, Librosa, OpenSMILE, Parselmouth** for audio preprocessing
- **Transformers (wav2vec2 / IndicBERT)** for speech & text embeddings
- **scikit-learn, XGBoost, PyTorch** for classical + neural modeling
- **Prefect / Celery (optional)** for background processing and pipelines

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

Create a `.env` file with the following variables:

```
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/cogai
JWT_SECRET=super-secret
JWT_ALGORITHM=HS256
STORAGE_BUCKET=local
```

## Application Entry Point

Run the FastAPI app with:

```bash
uvicorn app.main:app --reload
```

This project uses modular routers for users, assessments, and analytics. Models
are organized under `app/models`, schemas under `app/schemas`, and business
logic inside `app/services`.

## Pipeline Overview

The backend handles the following pipeline:

1. **User Registration & Consent**: Validates inputs, stores consent metadata,
   and issues a JWT session token for subsequent requests.
2. **Assessment Initiation**: Creates assessment records and generates
   identifiers for attaching speech/audio and cognitive logs.
3. **Speech Ingestion**: Accepts uploaded audio (WebM), stores the artifacts,
   and triggers preprocessing workers.
4. **Feature Extraction**: Generates acoustic features (MFCCs, jitter, shimmer,
   pauses), transcripts via wav2vec2, and text embeddings using IndicBERT.
5. **Cognitive Log Processing**: Computes reaction times, accuracy, hesitation
   metrics, and aggregates into structured tabular features.
6. **Risk Prediction**: Loads pretrained models, fuses acoustic, textual, and
   cognitive embeddings, and returns a probability distribution over Low,
   Medium, High risk classes with explainability metadata.
7. **Results Delivery**: Persists final feature sets, prediction outputs, and
   exposes read-only endpoints for frontend consumption.

See `architecture.md` for design diagrams and detailed service descriptions.
