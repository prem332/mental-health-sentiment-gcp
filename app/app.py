import os, pickle, logging, numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = Flask(__name__)
MAX_LEN   = int(os.getenv("MAX_LEN", 100))
MODEL_DIR = os.getenv("MODEL_DIR", "models")
PORT      = int(os.getenv("PORT", 8080))

def load_artifacts():
    logger.info("Loading artifacts from %s", MODEL_DIR)
    model = load_model(os.path.join(MODEL_DIR, "sentiment_analysis_model.keras"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"),     "rb") as f: tokenizer     = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f: label_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "class_weights.pkl"), "rb") as f: class_weights = pickle.load(f)
    logger.info("✅ All artifacts loaded.")
    return model, tokenizer, label_encoder, class_weights

model, tokenizer, label_encoder, class_weights = load_artifacts()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text field is required"}), 400
    seq        = tokenizer.texts_to_sequences([text])
    padded     = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    probs      = model.predict(padded, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    label      = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(probs))
    logger.info("text=%r  label=%s  confidence=%.4f", text[:60], label, confidence)
    return jsonify({
        "text": text, "sentiment": label, "confidence": round(confidence, 4),
        "probabilities": {cls: round(float(p), 4) for cls, p in zip(label_encoder.classes_, probs)}
    })

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "BiLSTM-96.2%"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)