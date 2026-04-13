# 🧠 Mental Health Sentiment Analyzer — GCP Production

> **Production-grade NLP system** that classifies tweets into four sentiment categories using a Bidirectional LSTM model, deployed on Google Cloud Platform with full MLOps pipeline, CI/CD automation, and Docker containerization.

---

> ⚠️ **Note on Live Demo Availability**
> The live demo URL may be temporarily unavailable or unresponsive. To avoid ongoing cloud infrastructure charges, the GCP project and Cloud Run service may be paused or deleted after the initial demonstration. This is standard practice for portfolio projects to manage costs responsibly. To run the project locally or redeploy, follow the setup instructions below.
>
> **Last known live URL:** `https://mental-health-sentiment-1050024610281.us-central1.run.app`

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [MLOps Pipeline](#mlops-pipeline)
- [CI/CD Pipeline](#cicd-pipeline)
- [GCP Infrastructure](#gcp-infrastructure)
- [Local Setup](#local-setup)
- [API Reference](#api-reference)
- [Training Results](#training-results)
- [Dataset](#dataset)
- [Deployment](#deployment)

---

## 🎯 Project Overview

This project upgrades an existing local Mental Health Sentiment Analysis model into a **fully production-grade GCP deployment** with:

- Bidirectional LSTM trained on 73,996 Twitter tweets
- 4-class sentiment classification: **Positive, Negative, Neutral, Irrelevant**
- Class imbalance handled via `compute_class_weight("balanced")`
- Model artifacts versioned and stored in Google Cloud Storage
- Model registered in Vertex AI Model Registry
- Automated retraining pipeline via Vertex AI Pipelines
- Full CI/CD via GitHub Actions → Docker → Cloud Run
- Flask REST API with clean web UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEVELOPER WORKFLOW                       │
│  Google Colab (Training)  →  GCS (Artifacts)  →  Cloud Shell    │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ git push
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GITHUB ACTIONS CI/CD                        │
│   🧪 Run Tests  →  🐳 Docker Build + Push  →  🚀 Deploy        │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     GCP PRODUCTION         │
                    │                            │
                    │  Artifact Registry         │
                    │  (Docker images)           │
                    │         │                  │
                    │         ▼                  │
                    │   Cloud Run Service        │
                    │   (Flask + Gunicorn)       │
                    │         │                  │
                    │         ▼                  │
                    │   GCS Bucket               │
                    │   (Model artifacts)        │
                    │         │                  │
                    │         ▼                  │
                    │   Vertex AI Registry       │
                    │   (Model versioning)       │
                    └────────────────────────────┘
```

---

## 📁 Project Structure

```
mental-health-sentiment-gcp/
│
├── app/                          # Flask web application
│   ├── app.py                    # Main Flask app with /predict, /health endpoints
│   ├── templates/
│   │   └── index.html            # Web UI with sentiment result display
│   └── static/
│       └── styles.css            # UI styling
│
├── pipeline/                     # Vertex AI MLOps pipeline
│   ├── __init__.py
│   └── training_pipeline.py      # Kubeflow pipeline: Train → Gate → Register
│
├── scripts/                      # Utility scripts
│   ├── download_models.py        # Downloads model artifacts from GCS at startup
│   ├── upload_models_to_gcs.py   # One-time upload of artifacts to GCS
│   └── gcp_setup.sh              # Full GCP infrastructure setup script
│
├── tests/                        # Test suite
│   └── test_app.py               # 6 pytest tests with mock models (no GCS needed)
│
├── training_results/             # Saved training outputs (Git tracked)
│   ├── metrics.json              # Accuracy, F1 scores, classification report
│   ├── label_encoder.pkl         # Class label mapping
│   ├── class_weights.pkl         # Computed class weights for imbalance
│   └── training_history.pkl      # Epoch-by-epoch training history
│
├── .github/
│   └── workflows/
│       └── deploy.yml            # GitHub Actions CI/CD pipeline
│
├── train_model.py                # Full BiLSTM training script
├── Dockerfile                    # Multi-stage Docker build (Python 3.12-slim)
├── entrypoint.sh                 # Container startup: download models → start gunicorn
├── requirements.txt              # Pinned Python dependencies
├── conftest.py                   # Pytest path configuration
├── pytest.ini                    # Pytest settings
├── .gitignore                    # Excludes models, data, credentials, venv
├── .dockerignore                 # Keeps Docker image lean
└── README.md                     # This file
```

---

## 📊 Model Performance

**Model:** Bidirectional LSTM
**Training:** Google Colab CPU | 15 epochs | Batch size 128
**Dataset:** 73,996 training tweets | 1,000 validation tweets

### Overall Metrics

| Metric | Score |
|---|---|
| **Accuracy** | **94.40%** |
| **F1 Macro** | **94.28%** |
| Training Time | ~39 minutes |

### Per-Class Metrics

| Class | Precision | Recall | F1 Score | Support |
|---|---|---|---|---|
| Irrelevant | 0.9016 | 0.9593 | **0.9296** | 172 |
| Negative | 0.9545 | 0.9474 | **0.9509** | 266 |
| Neutral | 0.9704 | 0.9193 | **0.9441** | 285 |
| Positive | 0.9364 | 0.9567 | **0.9464** | 277 |

### Class Imbalance Handling

| Class | Training Samples | Weight Applied |
|---|---|---|
| Negative | 22,358 | 0.8274 (seen most → penalized least) |
| Positive | 20,655 | 0.8956 |
| Neutral | 18,108 | 1.0216 |
| Irrelevant | 12,875 | 1.4368 (seen least → penalized most) |

Class weights computed automatically using `sklearn.utils.class_weight.compute_class_weight("balanced")` and passed directly into `model.fit()`.

### Training Progression

```
Epoch  1: val_accuracy = 0.694
Epoch  2: val_accuracy = 0.832  (+13.8%)
Epoch  3: val_accuracy = 0.873
Epoch  5: val_accuracy = 0.911
Epoch  8: val_accuracy = 0.937  ← best jump
Epoch 10: val_accuracy = 0.940  ← LR reduced 0.001 → 0.0005
Epoch 13: val_accuracy = 0.942
Epoch 15: val_accuracy = 0.944  ← BEST (early stopping restored)
```

Learning rate reduced automatically by `ReduceLROnPlateau` when validation loss plateaued — allowing finer weight updates in later epochs.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Model** | Bidirectional LSTM (Keras / TensorFlow 2.19) |
| **Web Framework** | Flask 3.0 + Gunicorn |
| **Container** | Docker (Python 3.12-slim, multi-stage build) |
| **Training** | Google Colab (CPU) |
| **Model Storage** | Google Cloud Storage (GCS) |
| **Model Registry** | Vertex AI Model Registry |
| **MLOps Pipeline** | Vertex AI Pipelines (Kubeflow) |
| **CI/CD** | GitHub Actions |
| **Container Registry** | GCP Artifact Registry |
| **Serving** | GCP Cloud Run (serverless) |
| **Development** | Google Cloud Shell Editor |
| **Version Control** | GitHub + Git LFS |

---

## 🔄 MLOps Pipeline

The `pipeline/training_pipeline.py` defines a 3-step Vertex AI Kubeflow pipeline:

```
Step 1: Train BiLSTM
        ↓ reads data from GCS
        ↓ trains model
        ↓ saves artifacts to GCS (versioned + latest)
        ↓ outputs metrics.json

Step 2: Evaluate + Gate Check
        ↓ reads metrics.json
        ↓ checks: accuracy ≥ 94% AND F1 macro ≥ 93%
        ↓ logs metrics to Vertex AI dashboard
        ✅ PASS → proceeds to registration
        ❌ FAIL → pipeline stops, no deployment

Step 3: Register in Vertex AI Model Registry
        ↓ uploads model metadata
        ↓ links artifact URI (gs://mh-sentiment-models/models/{version}/)
        ↓ attaches accuracy + F1 labels
        ↓ model versioned and tracked permanently
```

**Gate Thresholds:**
```python
ACCURACY_THRESHOLD = 0.94
F1_THRESHOLD       = 0.93
```

---

## ⚙️ CI/CD Pipeline

Defined in `.github/workflows/deploy.yml`. Triggers on every push to `main`.

```
Push to main
    │
    ▼
🧪 Job 1: Run Tests
    pytest tests/ -v
    (uses mock models — no GCS needed)
    │
    ▼ (only if tests pass)
🐳 Job 2: Build & Push Docker Image
    docker build → push to Artifact Registry
    tags: :latest + :{git-sha}
    │
    ▼ (only if build succeeds)
🚀 Job 3: Deploy to Cloud Run
    gcloud run deploy
    --memory 2Gi --cpu 2
    --allow-unauthenticated
    --set-env-vars GCS_BUCKET, MODEL_DIR, MAX_LEN
    │
    ▼
✅ Smoke Test
    curl /health → must return 200
```

**GitHub Secrets Required:**

| Secret | Value |
|---|---|
| `GCP_PROJECT_ID` | `mental-health-sentiment-gcp` |
| `GCP_SA_KEY` | GitHub Actions service account JSON key |
| `GCS_BUCKET` | `mh-sentiment-models` |
| `CLOUD_RUN_SA_EMAIL` | `cloud-run-sa@mental-health-sentiment-gcp.iam.gserviceaccount.com` |

---

## ☁️ GCP Infrastructure

| Resource | Name | Purpose |
|---|---|---|
| GCP Project | `mental-health-sentiment-gcp` | Root project |
| GCS Bucket | `mh-sentiment-models` | Model artifacts + training data |
| Artifact Registry | `mental-health-sentiment-repo` | Docker images |
| Cloud Run Service | `mental-health-sentiment` | Live serving |
| Vertex AI Model | `bilstm-sentiment-20260413` | Model registry entry |
| Service Account | `cloud-run-sa` | Cloud Run → GCS access |
| Service Account | `github-actions-sa` | CI/CD → GCP access |

**GCS Bucket Structure:**
```
gs://mh-sentiment-models/
├── data/
│   ├── twitter_training.csv      (9.9MB — 73,996 tweets)
│   └── twitter_validation.csv    (161KB — 1,000 tweets)
├── models/
│   ├── sentiment_analysis_model.keras   (latest)
│   ├── tokenizer.pkl                    (latest)
│   ├── label_encoder.pkl                (latest)
│   ├── class_weights.pkl                (latest)
│   ├── metrics.json                     (latest)
│   └── 20260413_135123/                 (versioned)
│       ├── sentiment_analysis_model.keras
│       ├── tokenizer.pkl
│       ├── label_encoder.pkl
│       ├── class_weights.pkl
│       ├── metrics.json
│       └── saved_model/
│           ├── saved_model.pb
│           └── variables/
└── pipeline_root/                       (Vertex AI pipeline artifacts)
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.12
- Google Cloud SDK (`gcloud`)
- Docker
- GitHub account

### Step 1 — Clone the repo
```bash
git clone https://github.com/prem332/mental-health-sentiment-gcp.git
cd mental-health-sentiment-gcp
```

### Step 2 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — Authenticate with GCP
```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project mental-health-sentiment-gcp
```

### Step 4 — Download model artifacts from GCS
```bash
mkdir -p models
python scripts/download_models.py
```

### Step 5 — Run the app locally
```bash
MODEL_DIR=models python app/app.py
```

Visit `http://localhost:8080`

### Step 6 — Run tests
```bash
pytest tests/ -v
```

---

## 📡 API Reference

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "model": "BiLSTM-96.2%"
}
```

### `POST /predict`
Predict sentiment of input text.

**Request:**
```json
{
  "text": "@Microsoft I absolutely love the new update, best software ever!"
}
```

**Response:**
```json
{
  "text": "@Microsoft I absolutely love the new update, best software ever!",
  "sentiment": "Positive",
  "confidence": 0.9959,
  "probabilities": {
    "Irrelevant": 0.0012,
    "Negative": 0.0008,
    "Neutral": 0.0021,
    "Positive": 0.9959
  }
}
```

**Sentiment Classes:**
- `Positive` — positive opinion/emotion about a brand/topic
- `Negative` — negative opinion/complaint about a brand/topic
- `Neutral` — factual/news statement about a brand/topic
- `Irrelevant` — tweet unrelated to any specific brand/topic

---

## 📈 Training Results

All results stored in `training_results/`:

```
training_results/
├── metrics.json           # Full metrics including classification report
├── label_encoder.pkl      # LabelEncoder with class mappings
├── class_weights.pkl      # {0: 1.437, 1: 0.827, 2: 1.022, 3: 0.896}
└── training_history.pkl   # Loss + accuracy per epoch
```

**Label Encoding:**
```
0 → Irrelevant
1 → Negative
2 → Neutral
3 → Positive
```

**Model Version:** `20260413_135123`

---

## 📦 Dataset

**Source:** Twitter Sentiment Analysis dataset (Kaggle)

**Files:**
- `twitter_training.csv` — 73,996 labeled tweets
- `twitter_validation.csv` — 1,000 labeled tweets

**Format:**
```
id, entity, sentiment, text
3364, Facebook, Irrelevant, "I mentioned on Facebook..."
352,  Amazon,   Neutral,    "BBC News - Amazon boss..."
8312, Microsoft,Negative,   "@Microsoft Why do I pay..."
```

**Domain:** Tweets about specific brands/companies/games.
For best results, input tweets that reference a specific brand, product, or public entity.

---

## 🌐 Deployment

### Redeploy from scratch

**Step 1 — Run GCP setup:**
```bash
bash scripts/gcp_setup.sh
```

**Step 2 — Upload data to GCS:**
```bash
gcloud storage cp data/twitter_training.csv gs://mh-sentiment-models/data/
gcloud storage cp data/twitter_validation.csv gs://mh-sentiment-models/data/
```

**Step 3 — Train model (Google Colab recommended):**
```python
!git clone https://github.com/prem332/mental-health-sentiment-gcp.git
%cd mental-health-sentiment-gcp
!python train_model.py --bucket mh-sentiment-models --epochs 15
```

**Step 4 — Add GitHub Secrets and push to main:**
```bash
git push origin main
# GitHub Actions automatically:
# → runs tests
# → builds Docker image
# → deploys to Cloud Run
```

**Step 5 — Get live URL:**
```bash
gcloud run services describe mental-health-sentiment \
  --region us-central1 \
  --format 'value(status.url)'
```

---

## 👤 Author

**Prem Kumar** — AI/ML Engineer
- GitHub: [@prem332](https://github.com/prem332)
- Portfolio: [End_to_End_AI_ML_projects](https://github.com/prem332/End_to_End_AI_ML_projects)

---

## 📄 License

This project is for portfolio and educational purposes.