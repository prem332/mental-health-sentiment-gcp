import os, pickle, pytest
os.environ["MODEL_DIR"] = "tests/mock_models"
os.environ.setdefault("GCS_BUCKET", "test-bucket")

@pytest.fixture(scope="session", autouse=True)
def create_mock_models():
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    d = "tests/mock_models"; os.makedirs(d, exist_ok=True)
    from tensorflow.keras.preprocessing.text import Tokenizer
    tok = Tokenizer(num_words=100)
    tok.fit_on_texts(["positive negative neutral irrelevant mental health"])
    with open(f"{d}/tokenizer.pkl","wb") as f: pickle.dump(tok,f)
    le = LabelEncoder(); le.fit(["Positive","Negative","Neutral","Irrelevant"])
    with open(f"{d}/label_encoder.pkl","wb") as f: pickle.dump(le,f)
    with open(f"{d}/class_weights.pkl","wb") as f: pickle.dump({0:1.,1:1.,2:1.,3:1.},f)
    inp = tf.keras.Input(shape=(100,))
    x   = tf.keras.layers.Embedding(100,8)(inp)
    x   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(8))(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    tf.keras.Model(inp,out).save(f"{d}/sentiment_analysis_model.keras")

@pytest.fixture()
def client(create_mock_models):
    from app.app import app; app.config["TESTING"] = True
    with app.test_client() as c: yield c

def test_health(client):          assert client.get("/health").status_code == 200
def test_index(client):           assert client.get("/").status_code == 200
def test_predict_ok(client):      assert "sentiment" in client.post("/predict", json={"text":"I feel hopeful"}).get_json()
def test_predict_no_text(client): assert client.post("/predict", json={}).status_code == 400
def test_predict_empty(client):   assert client.post("/predict", json={"text":"  "}).status_code == 400
def test_probs_sum(client):
    probs = client.post("/predict", json={"text":"anxiety is hard"}).get_json()["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-3
