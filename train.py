# import pandas as pd
# import xgboost as xgb
# import joblib
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import os

# # ======================
# # CONFIG
# # ======================
# DATASET_PATH = "final_dataset 2.csv"
# MODEL_DIR = "model"
# os.makedirs(MODEL_DIR, exist_ok=True)

# print("📥 Loading dataset...")
# df = pd.read_csv(DATASET_PATH)
# df = df[['text', 'emotion']].dropna()

# # ======================
# # TRAIN BPE TOKENIZER
# # ======================
# print("🔤 Training BPE tokenizer...")
# tokenizer = Tokenizer(models.BPE())
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# trainer = trainers.BpeTrainer(
#     vocab_size=3000,
#     special_tokens=["[UNK]", "[PAD]"]
# )
# tokenizer.train_from_iterator(df['text'].astype(str), trainer=trainer)

# def bpe(text):
#     return " ".join(tokenizer.encode(str(text)).tokens)

# df["bpe_text"] = df["text"].apply(bpe)

# # ======================
# # TF-IDF
# # ======================
# print("📊 Vectorizing...")
# vectorizer = TfidfVectorizer(
#     ngram_range=(1,2),
#     max_features=3000
# )
# X = vectorizer.fit_transform(df["bpe_text"])

# # ======================
# # LABEL ENCODING
# # ======================
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df["emotion"])

# # ======================
# # TRAIN MODEL
# # ======================
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# print("🤖 Training XGBoost...")
# model = xgb.XGBClassifier(
#     objective="multi:softmax",
#     num_class=len(label_encoder.classes_),
#     n_estimators=120,
#     max_depth=4,
#     learning_rate=0.1,
#     tree_method="hist"
# )

# model.fit(X_train, y_train)

# # ======================
# # METRICS
# # ======================
# acc = accuracy_score(y_test, model.predict(X_test)) * 100
# print(f"✅ Training done | Accuracy: {acc:.2f}%")

# # ======================
# # SAVE EVERYTHING
# # ======================
# print("💾 Saving model files...")
# joblib.dump(model, f"{MODEL_DIR}/model.pkl")
# joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
# joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
# tokenizer.save(f"{MODEL_DIR}/bpe_tokenizer.json")

# print("🎉 All files saved successfully!")
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
# CONFIGURATION
# ======================
# CHANGE THIS to your text file path if needed
DATASET_PATH = "train.txt"  
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======================
# 1. LOAD DATASET (TXT or CSV)
# ======================
print(f"📥 Loading dataset from {DATASET_PATH}...")

if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: File '{DATASET_PATH}' not found.")
    exit()

# Logic to handle .txt vs .csv
if DATASET_PATH.endswith('.txt'):
    # Assuming format: "I am happy;joy" (semicolon separator is common for NLP txt files)
    # If your txt file uses comma, change sep=';' to sep=','
    df = pd.read_csv(DATASET_PATH, sep=';', names=['text', 'emotion'], engine='python')
else:
    # Default CSV loading
    df = pd.read_csv(DATASET_PATH)

# Clean data
df = df[['text', 'emotion']].dropna()
print(f"✅ Loaded {len(df)} rows. Emotions found: {df['emotion'].unique()}")

# ======================
# 2. TRAIN BPE TOKENIZER
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
# 3. VECTORIZE (TF-IDF)
# ======================
print("📊 Vectorizing...")
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=3000
)
X = vectorizer.fit_transform(df["bpe_text"])

# ======================
# 4. LABEL ENCODING
# ======================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["emotion"])

# ======================
# 5. TRAIN MODEL
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
    tree_method="hist"  # Faster training
)

model.fit(X_train, y_train)

# ======================
# 6. METRICS & SAVING
# ======================
acc = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"✅ Training done | Accuracy: {acc:.2f}%")

print("💾 Saving model files...")
joblib.dump(model, f"{MODEL_DIR}/model.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
tokenizer.save(f"{MODEL_DIR}/bpe_tokenizer.json")

print("🎉 All files saved successfully in 'model/' folder!")