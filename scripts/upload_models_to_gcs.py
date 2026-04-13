import os, argparse, logging
from google.cloud import storage
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
ARTIFACTS = ["models/sentiment_analysis_model.keras","models/tokenizer.pkl",
             "models/label_encoder.pkl","models/class_weights.pkl","models/training_history.pkl"]
def upload(bucket_name, gcs_prefix="models/"):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for local_path in ARTIFACTS:
        if not os.path.exists(local_path): logger.warning("⚠️  %s not found.", local_path); continue
        blob_name = gcs_prefix + os.path.basename(local_path)
        bucket.blob(blob_name).upload_from_filename(local_path)
        logger.info("✅ Uploaded %s → gs://%s/%s", local_path, bucket_name, blob_name)
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="models/")
    args = p.parse_args()
    upload(args.bucket, args.prefix)
