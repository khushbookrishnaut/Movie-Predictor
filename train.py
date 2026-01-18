import pandas as pd
import xgboost as xgb
import joblib
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# ======================
# CONFIG
# ======================
DATASET_PATH = "final_dataset 2.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("📥 Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df[['text', 'emotion']].dropna()

# ======================
# TRAIN BPE TOKENIZER
# ======================
print("🔤 Training BPE tokenizer...")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=3000,
    special_tokens=["[UNK]", "[PAD]"]
)
tokenizer.train_from_iterator(df['text'].astype(str), trainer=trainer)

def bpe(text):
    return " ".join(tokenizer.encode(str(text)).tokens)

df["bpe_text"] = df["text"].apply(bpe)

# ======================
# TF-IDF
# ======================
print("📊 Vectorizing...")
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=3000
)
X = vectorizer.fit_transform(df["bpe_text"])

# ======================
# LABEL ENCODING
# ======================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emotion"])

# ======================
# TRAIN MODEL
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🤖 Training XGBoost...")
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(label_encoder.classes_),
    n_estimators=120,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist"
)

model.fit(X_train, y_train)

# ======================
# METRICS
# ======================
acc = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"✅ Training done | Accuracy: {acc:.2f}%")

# ======================
# SAVE EVERYTHING
# ======================
print("💾 Saving model files...")
joblib.dump(model, f"{MODEL_DIR}/model.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
tokenizer.save(f"{MODEL_DIR}/bpe_tokenizer.json")

print("🎉 All files saved successfully!")