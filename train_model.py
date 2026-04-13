"""
train_model.py - Full BiLSTM training script
Reads Twitter CSV from GCS, trains, saves artifacts back to GCS.
Usage: python train_model.py --bucket mh-sentiment-models --epochs 10
"""
import os, sys, json, pickle, logging, argparse, tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASSES    = ["Positive", "Negative", "Neutral", "Irrelevant"]
MAX_WORDS  = 10000
MAX_LEN    = 100
EMBED_DIM  = 64
LSTM_UNITS = 64
DROPOUT    = 0.4
BATCH_SIZE = 64
PATIENCE   = 3

# ── 1. DATA ───────────────────────────────────────────────────────────────────
def download_data_from_gcs(bucket_name, local_dir):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    train_path = os.path.join(local_dir, "twitter_training.csv")
    val_path   = os.path.join(local_dir, "twitter_validation.csv")
    logger.info("⬇️  Downloading data from gs://%s/data/...", bucket_name)
    bucket.blob("data/twitter_training.csv").download_to_filename(train_path)
    bucket.blob("data/twitter_validation.csv").download_to_filename(val_path)
    logger.info("✅ Data downloaded.")
    return train_path, val_path

def load_and_clean(train_path, val_path):
    col_names = ["id", "entity", "sentiment", "text"]
    df_train  = pd.read_csv(train_path, header=None, names=col_names)
    df_val    = pd.read_csv(val_path,   header=None, names=col_names)
    def clean(df):
        df = df.dropna(subset=["text", "sentiment"]).copy()
        df["text"]      = df["text"].astype(str).str.lower().str.strip()
        df["sentiment"] = df["sentiment"].str.strip().str.capitalize()
        return df[df["sentiment"].isin(CLASSES)]
    df_train, df_val = clean(df_train), clean(df_val)
    logger.info("Train: %d | Val: %d", len(df_train), len(df_val))
    logger.info("Class dist:\n%s", df_train["sentiment"].value_counts().to_string())
    return df_train, df_val

# ── 2. PREPROCESS ─────────────────────────────────────────────────────────────
def preprocess(df_train, df_val):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train["text"])
    X_train = pad_sequences(tokenizer.texts_to_sequences(df_train["text"]), maxlen=MAX_LEN, padding="post", truncating="post")
    X_val   = pad_sequences(tokenizer.texts_to_sequences(df_val["text"]),   maxlen=MAX_LEN, padding="post", truncating="post")

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASSES)
    y_train = label_encoder.transform(df_train["sentiment"])
    y_val   = label_encoder.transform(df_val["sentiment"])

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    logger.info("⚖️  Class weights: %s", class_weights)
    return X_train, X_val, y_train, y_val, tokenizer, label_encoder, class_weights

# ── 3. MODEL ──────────────────────────────────────────────────────────────────
def build_model(num_classes=4):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS, dropout=DROPOUT, recurrent_dropout=0.2, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS // 2, dropout=DROPOUT, recurrent_dropout=0.2)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="bilstm_sentiment")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary(print_fn=logger.info)
    return model

# ── 4. TRAIN ──────────────────────────────────────────────────────────────────
def train(model, X_train, y_train, X_val, y_val, class_weights, epochs):
    import tensorflow as tf
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]
    logger.info("🏋️  Training (epochs=%d, batch=%d)...", epochs, BATCH_SIZE)
    return model.fit(X_train, y_train, validation_data=(X_val, y_val),
                     epochs=epochs, batch_size=BATCH_SIZE,
                     class_weight=class_weights, callbacks=callbacks, verbose=1)

# ── 5. EVALUATE ───────────────────────────────────────────────────────────────
def evaluate(model, X_val, y_val, label_encoder):
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    y_pred       = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc          = accuracy_score(y_val, y_pred)
    f1_macro     = f1_score(y_val, y_pred, average="macro")
    f1_per_class = f1_score(y_val, y_pred, average=None)
    metrics = {
        "accuracy":      round(float(acc), 4),
        "f1_macro":      round(float(f1_macro), 4),
        "f1_per_class":  {cls: round(float(f), 4) for cls, f in zip(label_encoder.classes_, f1_per_class)},
        "classification_report": classification_report(y_val, y_pred, target_names=label_encoder.classes_, output_dict=True),
        "trained_at":    datetime.utcnow().isoformat(),
        "model_version": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
    }
    logger.info("✅ Accuracy: %.4f | F1 macro: %.4f", acc, f1_macro)
    logger.info("\n%s", classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    return metrics

# ── 6. SAVE + UPLOAD ──────────────────────────────────────────────────────────
def save_artifacts(model, tokenizer, label_encoder, class_weights, history, metrics, local_dir):
    paths = []
    model_path = os.path.join(local_dir, "sentiment_analysis_model.keras")
    model.save(model_path); paths.append(model_path)
    for name, obj in [("tokenizer.pkl", tokenizer), ("label_encoder.pkl", label_encoder), ("class_weights.pkl", class_weights), ("training_history.pkl", history.history)]:
        p = os.path.join(local_dir, name)
        with open(p, "wb") as f: pickle.dump(obj, f)
        paths.append(p)
    metrics_path = os.path.join(local_dir, "metrics.json")
    with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
    paths.append(metrics_path)
    logger.info("💾 Artifacts saved to %s", local_dir)
    return paths

def upload_artifacts_to_gcs(bucket_name, local_paths, version):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for local_path in local_paths:
        fname = os.path.basename(local_path)
        bucket.blob(f"models/{version}/{fname}").upload_from_filename(local_path)
        bucket.blob(f"models/{fname}").upload_from_filename(local_path)
        logger.info("⬆️  Uploaded %s (versioned + latest)", fname)
    logger.info("🎉 All artifacts uploaded.")

# ── 7. MAIN ───────────────────────────────────────────────────────────────────
def main(args):
    import tensorflow as tf
    logger.info("TF version: %s", tf.__version__)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path, val_path   = download_data_from_gcs(args.bucket, tmpdir)
        df_train, df_val       = load_and_clean(train_path, val_path)
        X_train, X_val, y_train, y_val, tokenizer, label_encoder, class_weights = preprocess(df_train, df_val)
        model                  = build_model(num_classes=len(CLASSES))
        history                = train(model, X_train, y_train, X_val, y_val, class_weights, args.epochs)
        metrics                = evaluate(model, X_val, y_val, label_encoder)
        version                = metrics["model_version"]
        artifact_dir           = os.path.join(tmpdir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        local_paths            = save_artifacts(model, tokenizer, label_encoder, class_weights, history, metrics, artifact_dir)
        upload_artifacts_to_gcs(args.bucket, local_paths, version)
        if args.metrics_output:
            with open(args.metrics_output, "w") as f: json.dump(metrics, f, indent=2)
        logger.info("=" * 50)
        logger.info("✅ DONE | Accuracy=%.4f | F1=%.4f | Version=%s", metrics["accuracy"], metrics["f1_macro"], version)
        logger.info("=" * 50)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket",         required=True)
    parser.add_argument("--epochs",         type=int, default=10)
    parser.add_argument("--metrics-output", default="")
    main(parser.parse_args())