import os, sys, logging
from google.cloud import storage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
BUCKET_NAME = os.environ["GCS_BUCKET"]
GCS_PREFIX  = os.getenv("GCS_MODEL_PREFIX", "models/")
LOCAL_DIR   = os.getenv("MODEL_DIR", "models")
ARTIFACTS   = ["sentiment_analysis_model.keras","tokenizer.pkl","label_encoder.pkl","class_weights.pkl"]
def download_models():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    for artifact in ARTIFACTS:
        local_path = os.path.join(LOCAL_DIR, artifact)
        if os.path.exists(local_path): logger.info("✅ %s exists, skipping.", artifact); continue
        bucket.blob(f"{GCS_PREFIX}{artifact}").download_to_filename(local_path)
        logger.info("✅ %s downloaded.", artifact)
    logger.info("🎉 All artifacts ready.")
if __name__ == "__main__":
    try: download_models()
    except Exception as e: logger.error("❌ %s", e); sys.exit(1)
