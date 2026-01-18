import pandas as pd
#a comnment
import numpy as np
import xgboost as xgb
import re
import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# PART 1: LOAD TRAINING DATA (.txt support)
# ==========================================
DATASET_PATH = r"C:\Users\KHUSHBOO KUMARI\OneDrive\shipathon_project\train.txt"

if not os.path.exists(DATASET_PATH):
    print(f"❌ File not found: {DATASET_PATH}")
    exit()

print("📂 Loading Training Data...")
# Loading txt: assuming 'text<TAB>mood' or 'text<COMMA>mood'
df = pd.read_csv(DATASET_PATH, sep=None, engine='python', names=['text', 'mood'], on_bad_lines='skip')
df.dropna(inplace=True)

# ==========================================
# PART 2: TRAIN THE BRAIN
# ==========================================


print("🧠 Training XGBoost Model...")

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(df['text'].astype(str), trainer=trainer)

def bpe_tokenize(text):
    return " ".join(tokenizer.encode(str(text)).tokens)

df['bpe_text'] = df['text'].apply(bpe_tokenize)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['bpe_text'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['mood'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))
model.fit(X_train, y_train)

print(f"🏆 Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.1f}%")

# ==========================================
# PART 3: CHAT PARSING LOGIC
# ==========================================
def parse_chat_history(chat_text):
    speaker_data = {}
    patterns = [
        r"\[.*?\]\s*([^:]+):\s*(.*)",  # iOS
        r"\d{1,2}/\d{1,2}/\d{2,4}.*?-\s([^:]+):\s*(.*)", # Android
        r"^([^:\d]+):\s*(.*)" # Simple
    ]
    
    for line in chat_text.split('\n'):
        for p in patterns:
            match = re.search(p, line)
            if match:
                name, msg = match.group(1).strip(), match.group(2).strip()
                if len(name) < 20:
                    speaker_data.setdefault(name, []).append(msg)
                break
    return {k: " ".join(v) for k, v in speaker_data.items()}

# ==========================================
# PART 4: RECOMMENDATIONS
# ==========================================
def get_recommendations(label, pref_type):
    label = str(label).lower()
    if any(x in label for x in ['happy', 'joy', 'fun']):
        recs = {"w": ["Panchayat", "Gullak"], "m": ["3 Idiots", "ZNMD"]}
    elif any(x in label for x in ['sad', 'grief']):
        recs = {"w": ["Aspirants", "Kota Factory"], "m": ["Taare Zameen Par", "Masaan"]}
    elif any(x in label for x in ['ang', 'rage']):
        recs = {"w": ["Mirzapur", "The Boys"], "m": ["Gangs of Wasseypur", "Animal"]}
    else:
        recs = {"w": ["Family Man", "Scam 1992"], "m": ["Swades", "Lagaan"]}
    
    return recs.get(pref_type, recs["m"])

# ==========================================
# PART 5: THE INTERFACE
# ==========================================
print("\n" + "="*40)
print("💬 PASTE CHAT HISTORY (Type 'END' to finish)")
print("="*40)

user_lines = []
while True:
    line = input()
    if line.strip().upper() == 'END': break
    user_lines.append(line)

speakers = parse_chat_history("\n".join(user_lines))

if not speakers:
    print("❌ No messages found. Check your chat format!")
else:
    names = list(speakers.keys())
    for i, name in enumerate(names): print(f"{i+1}. {name}")
    
    idx = int(input(f"\nSelect person (1-{len(names)}): ")) - 1
    selected = names[idx]
    
    # Predict
    bpe_input = bpe_tokenize(speakers[selected])
    mood_idx = model.predict(vectorizer.transform([bpe_input]))[0]
    mood = label_encoder.inverse_transform([mood_idx])[0]
    
    print(f"\n✨ {selected}'s Mood: {mood.upper()}")
    pref = input("Choice: Movie (m) or Series (w)? ").lower()
    print(f"🍿 Recommendations: {', '.join(get_recommendations(mood, pref))}")