# Mental Health Sentiment Analysis GCP
Production-grade BiLSTM sentiment classifier deployed on GCP Cloud Run.

## Model Performance
- Accuracy: 94.4%
- F1 Macro: 94.28%
- Classes: Positive, Negative, Neutral, Irrelevant

## Architecture
- Model: Bidirectional LSTM
- Training: Google Colab (CPU)
- Storage: GCS
- Registry: Vertex AI Model Registry
- Serving: Cloud Run + Docker
- CI/CD: GitHub Actions
