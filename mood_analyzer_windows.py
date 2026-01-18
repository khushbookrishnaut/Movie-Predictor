# import pandas as pd
# import numpy as np
# import xgboost as xgb
# import re
# import os
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # ==========================================
# # PART 1: LOAD TRAINING DATA (.txt support)
# # ==========================================
# DATASET_PATH = r"C:\Users\KHUSHBOO KUMARI\OneDrive\shipathon_project\train.txt"

# if not os.path.exists(DATASET_PATH):
#     print(f" File not found: {DATASET_PATH}")
#     exit()

# print(" Loading Training Data...")
# # Loading txt: assuming 'text<TAB>mood' or 'text<COMMA>mood'
# df = pd.read_csv(DATASET_PATH, sep=None, engine='python', names=['text', 'mood'], on_bad_lines='skip')
# df.dropna(inplace=True)

# # ==========================================
# # PART 2: TRAIN THE BRAIN
# # ==========================================


# print("Training XGBoost Model...")

# tokenizer = Tokenizer(models.BPE())
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]"])
# tokenizer.train_from_iterator(df['text'].astype(str), trainer=trainer)

# def bpe_tokenize(text):
#     return " ".join(tokenizer.encode(str(text)).tokens)

# df['bpe_text'] = df['text'].apply(bpe_tokenize)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
# X = vectorizer.fit_transform(df['bpe_text'])
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(df['mood'])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))
# model.fit(X_train, y_train)

# print(f" Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.1f}%")

# # ==========================================
# # PART 3: CHAT PARSING LOGIC
# # ==========================================
# def parse_chat_history(chat_text):
#     speaker_data = {}
#     patterns = [
#         r"\[.*?\]\s*([^:]+):\s*(.*)",  # iOS
#         r"\d{1,2}/\d{1,2}/\d{2,4}.*?-\s([^:]+):\s*(.*)", # Android
#         r"^([^:\d]+):\s*(.*)" # Simple
#     ]
    
#     for line in chat_text.split('\n'):
#         for p in patterns:
#             match = re.search(p, line)
#             if match:
#                 name, msg = match.group(1).strip(), match.group(2).strip()
#                 if len(name) < 20:
#                     speaker_data.setdefault(name, []).append(msg)
#                 break
#     return {k: " ".join(v) for k, v in speaker_data.items()}

# # ==========================================
# # PART 4: RECOMMENDATIONS
# # ==========================================
# # def get_recommendations(label, pref_type):
# #     label = str(label).lower()
# #     if any(x in label for x in ['happy', 'joy', 'fun']):
# #         recs = {"w": ["Panchayat", "Gullak"], "m": ["3 Idiots", "ZNMD"]}
# #     elif any(x in label for x in ['sad', 'grief']):
# #         recs = {"w": ["Aspirants", "Kota Factory"], "m": ["Taare Zameen Par", "Masaan"]}
# #     elif any(x in label for x in ['ang', 'rage']):
# #         recs = {"w": ["Mirzapur", "The Boys"], "m": ["Gangs of Wasseypur", "Animal"]}
# #     else:
# #         recs = {"w": ["Family Man", "Scam 1992"], "m": ["Swades", "Lagaan"]}
    
# #     return recs.get(pref_type, recs["m"])
# import random

# def get_recommendations(label, pref_type):
#     label = str(label).lower()
#     pref_type = pref_type.lower() # 'm' for movie, 'w' for web series

#     # ==========================
#     # 1. JOY (Soulful, Inspirational, Uplifting)
#     # ==========================
#     joy_movies = [
#         "3 Idiots", "The Intouchables", "Zindagi Na Milegi Dobara", "Forrest Gump", "Queen", 
#         "Dear Zindagi", "Good Will Hunting", "The Secret Life of Walter Mitty", "Chef", 
#         "English Vinglish", "Piku", "Amélie", "Ratatouille", "Paddington 2", "Little Miss Sunshine", 
#         "Udaan", "Wake Up Sid", "Munna Bhai M.B.B.S.", "Lage Raho Munna Bhai", "Swades", 
#         "Iqbal", "Chak De! India", "Dangal", "Rocky", "The Shawshank Redemption", 
#         "The Pursuit of Happyness", "It's a Wonderful Life", "My Neighbor Totoro", 
#         "Kiki's Delivery Service", "Soul", "CODA", "Green Book", "Lion"
#     ]
#     joy_series = [
#         "Panchayat", "Ted Lasso", "Gullak", "Modern Family", "Schitt's Creek", "The Good Place", 
#         "Queer Eye", "Parks and Recreation", "Brooklyn Nine-Nine", "The Marvelous Mrs. Maisel", 
#         "Anne with an E", "Abbott Elementary", "Trying", "Kim's Convenience", "The Durrells", 
#         "All Creatures Great and Small", "Somebody Feed Phil", "The Great British Bake Off", 
#         "Old Enough!", "Yeh Meri Family", "Sarabhai vs Sarabhai", "Pushpavalli", "Kota Factory", 
#         "Aspirants", "Little Things", "Gilmore Girls", "Friends", "Seinfeld", "Superstore", 
#         "Derry Girls", "Heartstopper", "Unbreakable Kimmy Schmidt"
#     ]

#     # ==========================
#     # 2. HAPPY (Cheery, Comedy, Feel-Good)
#     # ==========================
#     happy_movies = [
#         "Hera Pheri", "Home Alone", "Jab We Met", "Golmaal: Fun Unlimited", "Minions", 
#         "Dhamaal", "Welcome", "Phir Hera Pheri", "Andaz Apna Apna", "Chup Chup Ke", 
#         "Bhool Bhulaiyaa", "Chennai Express", "Housefull", "Singh Is Kinng", "Namastey London", 
#         "Partner", "No Entry", "Masti", "Bhagam Bhag", "Malamaal Weekly", "Hungama", 
#         "Hulchul", "De Dana Dan", "Garam Masala", "Khatta Meetha", "Awara Paagal Deewana", 
#         "Shrek", "Kung Fu Panda", "Madagascar", "Ice Age", "Despicable Me", "The Mask", 
#         "Dumb and Dumber", "Superbad", "The Hangover", "Barbie", "Free Guy"
#     ]
#     happy_series = [
#         "Friends", "Sarabhai vs Sarabhai", "Brooklyn Nine-Nine", "The Big Bang Theory", 
#         "Khichdi", "How I Met Your Mother", "Seinfeld", "The Office (US)", "Parks and Recreation", 
#         "New Girl", "Community", "Arrested Development", "It's Always Sunny in Philadelphia", 
#         "Curb Your Enthusiasm", "Silicon Valley", "Veep", "30 Rock", "What We Do in the Shadows", 
#         "The IT Crowd", "Sex Education", "Never Have I Ever", "Emily in Paris", "The Mindy Project", 
#         "2 Broke Girls", "Two and a Half Men", "Taarak Mehta Ka Ooltah Chashmah", "Bhabiji Ghar Par Hain!",
#         "Workin' Moms", "Young Sheldon", "Rick and Morty", "South Park"
#     ]

#     # ==========================
#     # 3. LOVE (Romance, Deep Connection)
#     # ==========================
#     love_movies = [
#         "Dilwale Dulhania Le Jayenge", "Titanic", "Sita Ramam", "La La Land", "Yeh Jawaani Hai Deewani", 
#         "About Time", "Veer-Zaara", "Kal Ho Naa Ho", "Kuch Kuch Hota Hai", "Kabhi Khushi Kabhie Gham", 
#         "Mohabbatein", "Dil To Pagal Hai", "Hum Dil De Chuke Sanam", "Devdas", "Goliyon Ki Raasleela Ram-Leela", 
#         "Bajirao Mastani", "Padmaavat", "Aashiqui 2", "The Notebook", "Pride and Prejudice", 
#         "The Fault in Our Stars", "Me Before You", "500 Days of Summer", "Eternal Sunshine of the Spotless Mind", 
#         "Her", "Before Sunrise", "Before Sunset", "Before Midnight", "Rockstar", "Barfi!", 
#         "Laila Majnu", "Raanjhanaa", "2 States", "Crazy Rich Asians", "Past Lives"
#     ]
#     love_series = [
#         "Little Things", "Bridgerton", "Mismatched", "Emily in Paris", "Modern Love", 
#         "Flames", "Permanent Roommates", "This Is Us", "Grey's Anatomy", "Outlander", 
#         "The Summer I Turned Pretty", "Heartstopper", "Sex and the City", "Gossip Girl", 
#         "The Vampire Diaries", "The O.C.", "One Tree Hill", "Dawson's Creek", "Gilmore Girls", 
#         "Normal People", "Fleabag", "Lovesick", "Dash & Lily", "Bandish Bandits", 
#         "Made in Heaven", "Four More Shots Please!", "Broken But Beautiful", "Feels Like Ishq", 
#         "College Romance", "K-Drama: Crash Landing on You", "K-Drama: Business Proposal", "K-Drama: Hometown Cha-Cha-Cha"
#     ]

#     # ==========================
#     # 4. FUN (Excitement, Masala, Action)
#     # ==========================
#     fun_movies = [
#         "Avengers: Endgame", "Stree 2", "Bhool Bhulaiyaa 2", "Jawan", "Guardians of the Galaxy", 
#         "RRR", "Deadpool", "Pathaan", "War", "Tiger 3", "K.G.F: Chapter 1", "K.G.F: Chapter 2", 
#         "Pushpa: The Rise", "Baahubali: The Beginning", "Baahubali 2: The Conclusion", "Kantara", 
#         "Vikram", "Leo", "Jailer", "Dhoom", "Dhoom 2", "Don", "Kick", "Wanted", "Dabangg", 
#         "Singham", "Simmba", "Sooryavanshi", "Fast & Furious 7", "Mission: Impossible - Fallout", 
#         "Jurassic Park", "Spider-Man: No Way Home", "Iron Man", "Top Gun: Maverick", "Bullet Train"
#     ]
#     fun_series = [
#         "The Family Man", "Stranger Things", "Farzi", "Money Heist", "The Boys", "Loki", 
#         "Sex Education", "The Umbrella Academy", "The Mandalorian", "Andor", "WandaVision", 
#         "Hawkeye", "Moon Knight", "Ms. Marvel", "Daredevil", "The Punisher", "Reacher", 
#         "Jack Ryan", "The Night Agent", "Citadel", "Rana Naidu", "Guns & Gulaabs", 
#         "Commando", "Special Ops", "The Freelancer", "The Night Manager (India)", "Lupin", 
#         "Slow Horses", "Gen V", "One Piece (Live Action)", "Cobra Kai", "Warrior"
#     ]

#     # ==========================
#     # 5. SURPRISE (Twists, Mind-bending, Thriller)
#     # ==========================
#     surprise_movies = [
#         "Drishyam", "Drishyam 2", "Inception", "Andhadhun", "Kahaani", "Parasite", "Shutter Island", 
#         "Talaash", "The Sixth Sense", "The Prestige", "Memento", "Se7en", "Gone Girl", 
#         "Primal Fear", "The Usual Suspects", "Oldboy", "Arrival", "Get Out", "A Quiet Place", 
#         "Hereditary", "The Others", "Saw", "Knives Out", "Glass Onion", "Searching", 
#         "Missing", "Badla", "Ittefaq", "Race", "Don (2006)", "Tenet", "Interstellar", "Coherence"
#     ]
#     surprise_series = [
#         "Dark", "Squid Game", "Asur", "Black Mirror", "Severance", "Westworld", "1899", 
#         "Sherlock", "Lost", "The Leftovers", "Mr. Robot", "Orphan Black", "Sense8", "The OA", 
#         "Yellowjackets", "The Haunting of Hill House", "Midnight Mass", "Behind Her Eyes", 
#         "Archive 81", "3 Body Problem", "Manifest", "Fringe", "The X-Files", "Twin Peaks", 
#         "Wayward Pines", "Alice in Borderland", "Kingdom", "Sweet Home", "The Silent Sea", 
#         "Dexter", "Hannibal", "Mindhunter"
#     ]

#     # ==========================
#     # 6. HATE (Revenge, Intense Rivalry, Dark Crime)
#     # ==========================
#     hate_movies = [
#         "Gangs of Wasseypur", "Gangs of Wasseypur 2", "Gone Girl", "Haider", "Badlapur", 
#         "Joker", "V for Vendetta", "Raman Raghav 2.0", "Nightcrawler", "American Psycho", 
#         "There Will Be Blood", "No Country for Old Men", "A Clockwork Orange", "Taxi Driver", 
#         "Scarface", "The Godfather", "Goodfellas", "Casino", "The Departed", "Reservoir Dogs", 
#         "Pulp Fiction", "Kill Bill: Vol. 1", "Django Unchained", "Inglourious Basterds", 
#         "Satya", "Company", "Sarkar", "Raajneeti", "Omkara", "Maqbool", "Ugly", "NH10"
#     ]
#     hate_series = [
#         "Mirzapur", "Succession", "Paatal Lok", "Game of Thrones", "House of Cards", 
#         "Sacred Games", "Peaky Blinders", "Breaking Bad", "Better Call Saul", "Ozark", 
#         "Narcos", "Narcos: Mexico", "The Wire", "The Sopranos", "Boardwalk Empire", 
#         "Sons of Anarchy", "Mad Men", "Yellowstone", "Billions", "Rome", "Spartacus", 
#         "Vikings", "The Last Kingdom", "Warrior", "Banshee", "Gangs of London", "McMafia", 
#         "Fargo", "True Detective", "Tabbar", "Delhi Crime"
#     ]

#     # ==========================
#     # 7. ANGER (Rage, Action, Aggression)
#     # ==========================
#     anger_movies = [
#         "Animal", "John Wick", "John Wick: Chapter 4", "Kabir Singh", "Mad Max: Fury Road", 
#         "Gladiator", "Agneepath", "Fight Club", "Rambo", "Rocky", "The Terminator", 
#         "Predator", "300", "Logan", "The Raid", "Ong Bak", "Ip Man", "Kill (2024)", 
#         "Monkey Man", "Nobody", "Taken", "Man on Fire", "The Equalizer", "Wrath of Man", 
#         "Shootout at Lokhandwala", "Shootout at Wadala", "Satya", "Gulaal", "Rakta Charitra", 
#         "Bandit Queen", "Singham", "Gadar: Ek Prem Katha", "KGF"
#     ]
#     anger_series = [
#         "The Boys", "Breaking Bad", "Reacher", "Daredevil", "The Punisher", "Vikings", 
#         "Warrior", "Banshee", "Spartacus", "Kingdom", "Cobra Kai", "Primal", "Blue Eye Samurai", 
#         "Invincible", "Gangs of London", "Tokyo Vice", "Tulsa King", "Mayor of Kingstown", 
#         "Yellowstone", "1883", "1923", "Peaky Blinders", "Sons of Anarchy", "Mayans M.C.", 
#         "Snowfall", "The Shield", "Southland", "Bosch", "Justified", "Terminal List"
#     ]

#     # ==========================
#     # 8. SAD (Grief, Emotional, Cry)
#     # ==========================
#     sad_movies = [
#         "Kal Ho Naa Ho", "The Pursuit of Happyness", "Taare Zameen Par", "Schindler's List", 
#         "Grave of the Fireflies", "October", "The Sky Is Pink", "Hachi: A Dog's Tale", 
#         "Marley & Me", "The Green Mile", "Manchester by the Sea", "Blue Valentine", 
#         "Requiem for a Dream", "The Boy in the Striped Pyjamas", "Life Is Beautiful", 
#         "Atonement", "The Pianist", "12 Years a Slave", "Moonlight", "Roma", "Lion", 
#         "Capernaum", "Rang De Basanti", "Anand", "Sadma", "Masoom", "Lootera", "Guzaarish", 
#         "Black", "Devdas", "Up (Opening Scene)", "Coco", "Inside Out"
#     ]
#     sad_series = [
#         "Aspirants", "This Is Us", "Kota Factory", "After Life", "Maid", "The Crown", 
#         "13 Reasons Why", "Normal People", "BoJack Horseman", "The Handmaid's Tale", 
#         "When They See Us", "Unbelievable", "Dopesick", "Chernobyl", "Band of Brothers", 
#         "The Pacific", "Five Days at Memorial", "Patrick Melrose", "I Know This Much Is True", 
#         "Mare of Easttown", "Broadchurch", "The Leftovers", "Six Feet Under", "Shameless", 
#         "Pose", "It's a Sin", "Violet Evergarden", "Clannad", "Your Lie in April", "Move to Heaven"
#     ]

#     # ==========================
#     # 9. FEAR (Horror, Anxiety)
#     # ==========================
#     fear_movies = [
#         "Tumbbad", "The Conjuring", "Raaz", "A Quiet Place", "Hereditary", "It", "Pari", 
#         "The Exorcist", "The Shining", "Psycho", "Halloween", "The Texas Chain Saw Massacre", 
#         "A Nightmare on Elm Street", "Scream", "Insidious", "Sinister", "The Ring", 
#         "The Grudge", "Saw", "Hostel", "Get Out", "Us", "Nope", "Midsommar", "The Witch", 
#         "1920", "Ragini MMS", "Bhool Bhulaiyaa", "Stree", "Pizza", "Train to Busan", 
#         "Evil Dead Rise", "Barbarian", "Smile", "Talk to Me"
#     ]
#     fear_series = [
#         "The Haunting of Hill House", "Ghoul", "All of Us Are Dead", "Marianne", "Betaal", 
#         "Typewriter", "Midnight Mass", "The Haunting of Bly Manor", "The Fall of the House of Usher", 
#         "American Horror Story", "Penny Dreadful", "The Walking Dead", "Fear the Walking Dead", 
#         "The Last of Us", "Kingdom", "Sweet Home", "Hellbound", "Parasyte: The Grey", 
#         "Alice in Borderland", "Stranger Things", "Black Mirror", "Guillermo del Toro's Cabinet of Curiosities", 
#         "Creepshow", "Channel Zero", "Slasher", "Bates Motel", "Hannibal", "From", "Yellowjackets", "Archive 81"
#     ]

#     # ==========================
#     # LOGIC TO SELECT CATEGORY
#     # ==========================
#     options = []
    
#     if 'joy' in label:
#         options = joy_series if pref_type == 'w' else joy_movies
#     elif 'happy' in label:
#         options = happy_series if pref_type == 'w' else happy_movies
#     elif 'love' in label:
#         options = love_series if pref_type == 'w' else love_movies
#     elif 'fun' in label:
#         options = fun_series if pref_type == 'w' else fun_movies
#     elif 'surprise' in label:
#         options = surprise_series if pref_type == 'w' else surprise_movies
#     elif 'hate' in label:
#         options = hate_series if pref_type == 'w' else hate_movies
#     elif 'ang' in label or 'rage' in label or 'mad' in label:
#         options = anger_series if pref_type == 'w' else anger_movies
#     elif 'sad' in label or 'cry' in label or 'depres' in label:
#         options = sad_series if pref_type == 'w' else sad_movies
#     elif 'fear' in label or 'scared' in label:
#         options = fear_series if pref_type == 'w' else fear_movies
#     else:
#         # Default Mix for Neutral/Unknown
#         default_movies = ["Interstellar", "Kantara", "The Dark Knight", "Lagaan", "Dangal", 
#                           "Swades", "Top Gun: Maverick", "Avatar: The Way of Water", "Dune", 
#                           "Oppenheimer", "Barbie", "Inception", "Parasite", "RRR", "Zindagi Na Milegi Dobara",
#                           "3 Idiots", "Avengers: Endgame", "Jawan", "Stree 2", "Drishyam", "Titanic",
#                           "The Godfather", "Pulp Fiction", "Schindler's List", "Forrest Gump", "The Matrix",
#                           "Spirited Away", "Your Name", "Spider-Man: Across the Spider-Verse", "Everything Everywhere All At Once"]
        
#         default_series = ["Sherlock", "Narcos", "Scam 1992", "Rocket Boys", "Band of Brothers", 
#                           "Chernobyl", "True Detective", "The Wire", "Breaking Bad", "Game of Thrones", 
#                           "The Crown", "Stranger Things", "The Boys", "The Mandalorian", "The Last of Us", 
#                           "Succession", "The Bear", "The White Lotus", "Fargo", "Better Call Saul", 
#                           "Mindhunter", "Dark", "Black Mirror", "Peaky Blinders", "Arcane", 
#                           "Blue Eye Samurai", "Shōgun", "Ripley", "Baby Reindeer", "House of the Dragon"]
        
#         options = default_series if pref_type == 'w' else default_movies

#     # -------------------------------------------
#     # RANDOM SELECTION LOGIC
#     # -------------------------------------------
#     # Select 4 random recommendations from the pool of 30+
#     if len(options) >= 4:
#         return random.sample(options, 4)
#     else:
#         return options


# # ==========================================
# # PART 5: THE INTERFACE
# # ==========================================
# print("\n" + "="*40)
# print("PASTE CHAT HISTORY (Type 'END' to finish)")
# print("="*40)

# user_lines = []
# while True:
#     line = input()
#     if line.strip().upper() == 'END': break
#     user_lines.append(line)

# speakers = parse_chat_history("\n".join(user_lines))

# if not speakers:
#     print(" No messages found. Check your chat format!")
# else:
#     names = list(speakers.keys())
#     for i, name in enumerate(names): print(f"{i+1}. {name}")
    
#     idx = int(input(f"\nSelect person (1-{len(names)}): ")) - 1
#     selected = names[idx]
    
#     # Predict
#     bpe_input = bpe_tokenize(speakers[selected])
#     mood_idx = model.predict(vectorizer.transform([bpe_input]))[0]
#     mood = label_encoder.inverse_transform([mood_idx])[0]
    
#     print(f"\n {selected}'s Mood: {mood.upper()}")
#     pref = input("Choice: Movie (m) or Series (w)? ").lower()
#     print(f" Recommendations: {', '.join(get_recommendations(mood, pref))}")
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import os
import random
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ==========================================
# PART 1: LOAD YOUR SPECIFIC DATASET
# ==========================================
# Update this path if the file is in a different folder
DATASET_PATH = r"C:\Users\KHUSHBOO KUMARI\OneDrive\shipathon_project\final_dataset 2.csv"

# Fallback: Check current directory if absolute path fails
if not os.path.exists(DATASET_PATH):
    if os.path.exists("final_dataset 2.csv"):
        DATASET_PATH = "final_dataset 2.csv"
    else:
        print(f"❌ File not found: {DATASET_PATH}")
        print("Please make sure 'final_dataset 2.csv' is in the project folder.")
        exit()

print(f"📂 Loading Data from: {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# Clean data
df = df[['text', 'emotion']].dropna()
print(f"✅ Data Loaded: {len(df)} rows.")
print(f"   Emotions detected: {df['emotion'].unique()}")

# ==========================================
# PART 2: TRAIN THE BRAIN
# ==========================================
print("🧠 Training XGBoost Model...")

# Train BPE Tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(df['text'].astype(str), trainer=trainer)

def bpe_tokenize(text):
    return " ".join(tokenizer.encode(str(text)).tokens)

df['bpe_text'] = df['text'].apply(bpe_tokenize)

# Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['bpe_text'])

# Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['emotion'])

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))
model.fit(X_train, y_train)

# Accuracy on data it learned from
train_acc = accuracy_score(y_train, model.predict(X_train)) * 100
print(f"🧠 Training Accuracy: {train_acc:.1f}%")

# Accuracy on new, unseen data (The real test)
test_acc = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"🏆 Testing Accuracy:  {test_acc:.1f}%")

# ==========================================
# PART 3: CHAT PARSING LOGIC
# ==========================================
def parse_chat_history(chat_text):
    speaker_data = {}
    patterns = [
        r"\[.*?\]\s*([^:]+):\s*(.*)",  # iOS [Date] Name: Msg
        r"\d{1,2}/\d{1,2}/\d{2,4}.*?-\s([^:]+):\s*(.*)", # Android
        r"^([^:\d]+):\s*(.*)" # Simple Name: Msg
    ]
    
    for line in chat_text.split('\n'):
        for p in patterns:
            match = re.search(p, line)
            if match:
                name, msg = match.group(1).strip(), match.group(2).strip()
                # Filter out system messages/long headers
                if len(name) < 20: 
                    speaker_data.setdefault(name, []).append(msg)
                break
    return {k: " ".join(v) for k, v in speaker_data.items()}

# ==========================================
# PART 4: RECOMMENDATIONS (Updated for your file)
# ==========================================

import random

def get_recommendations(label, pref_type):
    label = str(label).lower()
    pref_type = pref_type.lower() # 'm' for movie, 'w' for web series

    # ==========================
    # 1. JOY (Soulful, Inspirational, Uplifting)
    # ==========================
    joy_movies = [
        "3 Idiots", "The Intouchables", "Zindagi Na Milegi Dobara", "Forrest Gump", "Queen", 
        "Dear Zindagi", "Good Will Hunting", "The Secret Life of Walter Mitty", "Chef", 
        "English Vinglish", "Piku", "Amélie", "Ratatouille", "Paddington 2", "Little Miss Sunshine", 
        "Udaan", "Wake Up Sid", "Munna Bhai M.B.B.S.", "Lage Raho Munna Bhai", "Swades", 
        "Iqbal", "Chak De! India", "Dangal", "Rocky", "The Shawshank Redemption", 
        "The Pursuit of Happyness", "It's a Wonderful Life", "My Neighbor Totoro", 
        "Kiki's Delivery Service", "Soul", "CODA", "Green Book", "Lion"
    ]
    joy_series = [
        "Panchayat", "Ted Lasso", "Gullak", "Modern Family", "Schitt's Creek", "The Good Place", 
        "Queer Eye", "Parks and Recreation", "Brooklyn Nine-Nine", "The Marvelous Mrs. Maisel", 
        "Anne with an E", "Abbott Elementary", "Trying", "Kim's Convenience", "The Durrells", 
        "All Creatures Great and Small", "Somebody Feed Phil", "The Great British Bake Off", 
        "Old Enough!", "Yeh Meri Family", "Sarabhai vs Sarabhai", "Pushpavalli", "Kota Factory", 
        "Aspirants", "Little Things", "Gilmore Girls", "Friends", "Seinfeld", "Superstore", 
        "Derry Girls", "Heartstopper", "Unbreakable Kimmy Schmidt"
    ]

    # ==========================
    # 2. HAPPY (Cheery, Comedy, Feel-Good)
    # ==========================
    happy_movies = [
        "Hera Pheri", "Home Alone", "Jab We Met", "Golmaal: Fun Unlimited", "Minions", 
        "Dhamaal", "Welcome", "Phir Hera Pheri", "Andaz Apna Apna", "Chup Chup Ke", 
        "Bhool Bhulaiyaa", "Chennai Express", "Housefull", "Singh Is Kinng", "Namastey London", 
        "Partner", "No Entry", "Masti", "Bhagam Bhag", "Malamaal Weekly", "Hungama", 
        "Hulchul", "De Dana Dan", "Garam Masala", "Khatta Meetha", "Awara Paagal Deewana", 
        "Shrek", "Kung Fu Panda", "Madagascar", "Ice Age", "Despicable Me", "The Mask", 
        "Dumb and Dumber", "Superbad", "The Hangover", "Barbie", "Free Guy"
    ]
    happy_series = [
        "Friends", "Sarabhai vs Sarabhai", "Brooklyn Nine-Nine", "The Big Bang Theory", 
        "Khichdi", "How I Met Your Mother", "Seinfeld", "The Office (US)", "Parks and Recreation", 
        "New Girl", "Community", "Arrested Development", "It's Always Sunny in Philadelphia", 
        "Curb Your Enthusiasm", "Silicon Valley", "Veep", "30 Rock", "What We Do in the Shadows", 
        "The IT Crowd", "Sex Education", "Never Have I Ever", "Emily in Paris", "The Mindy Project", 
        "2 Broke Girls", "Two and a Half Men", "Taarak Mehta Ka Ooltah Chashmah", "Bhabiji Ghar Par Hain!",
        "Workin' Moms", "Young Sheldon", "Rick and Morty", "South Park"
    ]

    # ==========================
    # 3. LOVE (Romance, Deep Connection)
    # ==========================
    love_movies = [
        "Dilwale Dulhania Le Jayenge", "Titanic", "Sita Ramam", "La La Land", "Yeh Jawaani Hai Deewani", 
        "About Time", "Veer-Zaara", "Kal Ho Naa Ho", "Kuch Kuch Hota Hai", "Kabhi Khushi Kabhie Gham", 
        "Mohabbatein", "Dil To Pagal Hai", "Hum Dil De Chuke Sanam", "Devdas", "Goliyon Ki Raasleela Ram-Leela", 
        "Bajirao Mastani", "Padmaavat", "Aashiqui 2", "The Notebook", "Pride and Prejudice", 
        "The Fault in Our Stars", "Me Before You", "500 Days of Summer", "Eternal Sunshine of the Spotless Mind", 
        "Her", "Before Sunrise", "Before Sunset", "Before Midnight", "Rockstar", "Barfi!", 
        "Laila Majnu", "Raanjhanaa", "2 States", "Crazy Rich Asians", "Past Lives"
    ]
    love_series = [
        "Little Things", "Bridgerton", "Mismatched", "Emily in Paris", "Modern Love", 
        "Flames", "Permanent Roommates", "This Is Us", "Grey's Anatomy", "Outlander", 
        "The Summer I Turned Pretty", "Heartstopper", "Sex and the City", "Gossip Girl", 
        "The Vampire Diaries", "The O.C.", "One Tree Hill", "Dawson's Creek", "Gilmore Girls", 
        "Normal People", "Fleabag", "Lovesick", "Dash & Lily", "Bandish Bandits", 
        "Made in Heaven", "Four More Shots Please!", "Broken But Beautiful", "Feels Like Ishq", 
        "College Romance", "K-Drama: Crash Landing on You", "K-Drama: Business Proposal", "K-Drama: Hometown Cha-Cha-Cha"
    ]

    # ==========================
    # 4. FUN (Excitement, Masala, Action)
    # ==========================
    fun_movies = [
        "Avengers: Endgame", "Stree 2", "Bhool Bhulaiyaa 2", "Jawan", "Guardians of the Galaxy", 
        "RRR", "Deadpool", "Pathaan", "War", "Tiger 3", "K.G.F: Chapter 1", "K.G.F: Chapter 2", 
        "Pushpa: The Rise", "Baahubali: The Beginning", "Baahubali 2: The Conclusion", "Kantara", 
        "Vikram", "Leo", "Jailer", "Dhoom", "Dhoom 2", "Don", "Kick", "Wanted", "Dabangg", 
        "Singham", "Simmba", "Sooryavanshi", "Fast & Furious 7", "Mission: Impossible - Fallout", 
        "Jurassic Park", "Spider-Man: No Way Home", "Iron Man", "Top Gun: Maverick", "Bullet Train"
    ]
    fun_series = [
        "The Family Man", "Stranger Things", "Farzi", "Money Heist", "The Boys", "Loki", 
        "Sex Education", "The Umbrella Academy", "The Mandalorian", "Andor", "WandaVision", 
        "Hawkeye", "Moon Knight", "Ms. Marvel", "Daredevil", "The Punisher", "Reacher", 
        "Jack Ryan", "The Night Agent", "Citadel", "Rana Naidu", "Guns & Gulaabs", 
        "Commando", "Special Ops", "The Freelancer", "The Night Manager (India)", "Lupin", 
        "Slow Horses", "Gen V", "One Piece (Live Action)", "Cobra Kai", "Warrior"
    ]

    # ==========================
    # 5. SURPRISE (Twists, Mind-bending, Thriller)
    # ==========================
    surprise_movies = [
        "Drishyam", "Drishyam 2", "Inception", "Andhadhun", "Kahaani", "Parasite", "Shutter Island", 
        "Talaash", "The Sixth Sense", "The Prestige", "Memento", "Se7en", "Gone Girl", 
        "Primal Fear", "The Usual Suspects", "Oldboy", "Arrival", "Get Out", "A Quiet Place", 
        "Hereditary", "The Others", "Saw", "Knives Out", "Glass Onion", "Searching", 
        "Missing", "Badla", "Ittefaq", "Race", "Don (2006)", "Tenet", "Interstellar", "Coherence"
    ]
    surprise_series = [
        "Dark", "Squid Game", "Asur", "Black Mirror", "Severance", "Westworld", "1899", 
        "Sherlock", "Lost", "The Leftovers", "Mr. Robot", "Orphan Black", "Sense8", "The OA", 
        "Yellowjackets", "The Haunting of Hill House", "Midnight Mass", "Behind Her Eyes", 
        "Archive 81", "3 Body Problem", "Manifest", "Fringe", "The X-Files", "Twin Peaks", 
        "Wayward Pines", "Alice in Borderland", "Kingdom", "Sweet Home", "The Silent Sea", 
        "Dexter", "Hannibal", "Mindhunter"
    ]

    # ==========================
    # 6. HATE (Revenge, Intense Rivalry, Dark Crime)
    # ==========================
    hate_movies = [
        "Gangs of Wasseypur", "Gangs of Wasseypur 2", "Gone Girl", "Haider", "Badlapur", 
        "Joker", "V for Vendetta", "Raman Raghav 2.0", "Nightcrawler", "American Psycho", 
        "There Will Be Blood", "No Country for Old Men", "A Clockwork Orange", "Taxi Driver", 
        "Scarface", "The Godfather", "Goodfellas", "Casino", "The Departed", "Reservoir Dogs", 
        "Pulp Fiction", "Kill Bill: Vol. 1", "Django Unchained", "Inglourious Basterds", 
        "Satya", "Company", "Sarkar", "Raajneeti", "Omkara", "Maqbool", "Ugly", "NH10"
    ]
    hate_series = [
        "Mirzapur", "Succession", "Paatal Lok", "Game of Thrones", "House of Cards", 
        "Sacred Games", "Peaky Blinders", "Breaking Bad", "Better Call Saul", "Ozark", 
        "Narcos", "Narcos: Mexico", "The Wire", "The Sopranos", "Boardwalk Empire", 
        "Sons of Anarchy", "Mad Men", "Yellowstone", "Billions", "Rome", "Spartacus", 
        "Vikings", "The Last Kingdom", "Warrior", "Banshee", "Gangs of London", "McMafia", 
        "Fargo", "True Detective", "Tabbar", "Delhi Crime"
    ]

    # ==========================
    # 7. ANGER (Rage, Action, Aggression)
    # ==========================
    anger_movies = [
        "Animal", "John Wick", "John Wick: Chapter 4", "Kabir Singh", "Mad Max: Fury Road", 
        "Gladiator", "Agneepath", "Fight Club", "Rambo", "Rocky", "The Terminator", 
        "Predator", "300", "Logan", "The Raid", "Ong Bak", "Ip Man", "Kill (2024)", 
        "Monkey Man", "Nobody", "Taken", "Man on Fire", "The Equalizer", "Wrath of Man", 
        "Shootout at Lokhandwala", "Shootout at Wadala", "Satya", "Gulaal", "Rakta Charitra", 
        "Bandit Queen", "Singham", "Gadar: Ek Prem Katha", "KGF"
    ]
    anger_series = [
        "The Boys", "Breaking Bad", "Reacher", "Daredevil", "The Punisher", "Vikings", 
        "Warrior", "Banshee", "Spartacus", "Kingdom", "Cobra Kai", "Primal", "Blue Eye Samurai", 
        "Invincible", "Gangs of London", "Tokyo Vice", "Tulsa King", "Mayor of Kingstown", 
        "Yellowstone", "1883", "1923", "Peaky Blinders", "Sons of Anarchy", "Mayans M.C.", 
        "Snowfall", "The Shield", "Southland", "Bosch", "Justified", "Terminal List"
    ]

    # ==========================
    # 8. SAD (Grief, Emotional, Cry)
    # ==========================
    sad_movies = [
        "Kal Ho Naa Ho", "The Pursuit of Happyness", "Taare Zameen Par", "Schindler's List", 
        "Grave of the Fireflies", "October", "The Sky Is Pink", "Hachi: A Dog's Tale", 
        "Marley & Me", "The Green Mile", "Manchester by the Sea", "Blue Valentine", 
        "Requiem for a Dream", "The Boy in the Striped Pyjamas", "Life Is Beautiful", 
        "Atonement", "The Pianist", "12 Years a Slave", "Moonlight", "Roma", "Lion", 
        "Capernaum", "Rang De Basanti", "Anand", "Sadma", "Masoom", "Lootera", "Guzaarish", 
        "Black", "Devdas", "Up (Opening Scene)", "Coco", "Inside Out"
    ]
    sad_series = [
        "Aspirants", "This Is Us", "Kota Factory", "After Life", "Maid", "The Crown", 
        "13 Reasons Why", "Normal People", "BoJack Horseman", "The Handmaid's Tale", 
        "When They See Us", "Unbelievable", "Dopesick", "Chernobyl", "Band of Brothers", 
        "The Pacific", "Five Days at Memorial", "Patrick Melrose", "I Know This Much Is True", 
        "Mare of Easttown", "Broadchurch", "The Leftovers", "Six Feet Under", "Shameless", 
        "Pose", "It's a Sin", "Violet Evergarden", "Clannad", "Your Lie in April", "Move to Heaven"
    ]

    # ==========================
    # 9. FEAR (Horror, Anxiety)
    # ==========================
    fear_movies = [
        "Tumbbad", "The Conjuring", "Raaz", "A Quiet Place", "Hereditary", "It", "Pari", 
        "The Exorcist", "The Shining", "Psycho", "Halloween", "The Texas Chain Saw Massacre", 
        "A Nightmare on Elm Street", "Scream", "Insidious", "Sinister", "The Ring", 
        "The Grudge", "Saw", "Hostel", "Get Out", "Us", "Nope", "Midsommar", "The Witch", 
        "1920", "Ragini MMS", "Bhool Bhulaiyaa", "Stree", "Pizza", "Train to Busan", 
        "Evil Dead Rise", "Barbarian", "Smile", "Talk to Me"
    ]
    fear_series = [
        "The Haunting of Hill House", "Ghoul", "All of Us Are Dead", "Marianne", "Betaal", 
        "Typewriter", "Midnight Mass", "The Haunting of Bly Manor", "The Fall of the House of Usher", 
        "American Horror Story", "Penny Dreadful", "The Walking Dead", "Fear the Walking Dead", 
        "The Last of Us", "Kingdom", "Sweet Home", "Hellbound", "Parasyte: The Grey", 
        "Alice in Borderland", "Stranger Things", "Black Mirror", "Guillermo del Toro's Cabinet of Curiosities", 
        "Creepshow", "Channel Zero", "Slasher", "Bates Motel", "Hannibal", "From", "Yellowjackets", "Archive 81"
    ]

    # ==========================
    # LOGIC TO SELECT CATEGORY
    # ==========================
    options = []
    
    if 'joy' in label or 'enthusiasm' in label or 'relief' in label:
        options = joy_series if pref_type == 'w' else joy_movies
    elif 'happy' in label or 'happiness' in label:
        options = happy_series if pref_type == 'w' else happy_movies
    elif 'love' in label:
        options = love_series if pref_type == 'w' else love_movies
    elif 'fun' in label:
        options = fun_series if pref_type == 'w' else fun_movies
    elif 'surprise' in label:
        options = surprise_series if pref_type == 'w' else surprise_movies
    elif 'hate' in label:
        options = hate_series if pref_type == 'w' else hate_movies
    elif 'ang' in label or 'rage' in label or 'mad' in label:
        options = anger_series if pref_type == 'w' else anger_movies
    elif 'sad' in label or 'cry' in label or 'empty' in label:
        options = sad_series if pref_type == 'w' else sad_movies
    elif 'fear' in label or 'scared' in label:
        options = fear_series if pref_type == 'w' else fear_movies
    else:
        # Default Mix for Neutral/Unknown
        default_movies = ["Interstellar", "Kantara", "The Dark Knight", "Lagaan", "Dangal", 
                          "Swades", "Top Gun: Maverick", "Avatar: The Way of Water", "Dune", 
                          "Oppenheimer", "Barbie", "Inception", "Parasite", "RRR", "Zindagi Na Milegi Dobara",
                          "3 Idiots", "Avengers: Endgame", "Jawan", "Stree 2", "Drishyam", "Titanic",
                          "The Godfather", "Pulp Fiction", "Schindler's List", "Forrest Gump", "The Matrix",
                          "Spirited Away", "Your Name", "Spider-Man: Across the Spider-Verse", "Everything Everywhere All At Once"]
        
        default_series = ["Sherlock", "Narcos", "Scam 1992", "Rocket Boys", "Band of Brothers", 
                          "Chernobyl", "True Detective", "The Wire", "Breaking Bad", "Game of Thrones", 
                          "The Crown", "Stranger Things", "The Boys", "The Mandalorian", "The Last of Us", 
                          "Succession", "The Bear", "The White Lotus", "Fargo", "Better Call Saul", 
                          "Mindhunter", "Dark", "Black Mirror", "Peaky Blinders", "Arcane", 
                          "Blue Eye Samurai", "Shōgun", "Ripley", "Baby Reindeer", "House of the Dragon"]
        
        options = default_series if pref_type == 'w' else default_movies

    # -------------------------------------------
    # RANDOM SELECTION LOGIC
    # -------------------------------------------
    # Select 4 random recommendations from the pool of 30+
    if len(options) >= 4:
        return random.sample(options, 4)
    else:
        return options
# # ==========================================
# # PART 5: THE INTERFACE
# # ==========================================
# print("\n" + "="*40)
# print("💬 PASTE CHAT HISTORY (Type 'END' to finish)")
# print("="*40)

# user_lines = []
# while True:
#     try:
#         line = input()
#         if line.strip().upper() == 'END': break
#         user_lines.append(line)
#     except EOFError:
#         break

# speakers = parse_chat_history("\n".join(user_lines))

# if not speakers:
#     print("❌ No valid chat messages found.")
#     print("   Tip: Paste lines like 'Rahul: I am happy today' or '[10/10/24] Rahul: Hello'")
# else:
#     names = list(speakers.keys())
#     for i, name in enumerate(names): print(f"{i+1}. {name}")
    
#     try:
#         idx = int(input(f"\nSelect person (1-{len(names)}): ")) - 1
#         if 0 <= idx < len(names):
#             selected = names[idx]
            
#             # Predict
#             bpe_input = bpe_tokenize(speakers[selected])
#             mood_idx = model.predict(vectorizer.transform([bpe_input]))[0]
#             mood = label_encoder.inverse_transform([mood_idx])[0]
            
#             print(f"\n✨ {selected}'s Mood: {mood.upper()}")
#             pref = input("Choice: Movie (m) or Series (w)? ").lower()
            
#             recs = get_recommendations(mood, pref)
#             print(f"🍿 Recommendations: {', '.join(recs)}")
#         else:
#             print("❌ Invalid selection.")
#     except ValueError:
#         print("❌ Please enter a number.")
def process_chat_file(filename):
    """
    Reads a WhatsApp export .txt file and extracts just the messages.
    It removes timestamps like [12/07/2024, 4:05 PM] to focus on the mood words.
    """
    if not os.path.exists(filename):
        print(f"❌ Error: File '{filename}' not found.")
        return ""

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to clean timestamps (Standard WhatsApp format)
    # Matches "dd/mm/yyyy, hh:mm - " or "[dd/mm/yyyy, hh:mm] "
    # This is optional; keeping timestamps doesn't hurt, but cleaning is nicer.
    clean_text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}.*?-\s', '', content) # Android style
    clean_text = re.sub(r'\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}:\d{2}\]', '', clean_text) # iOS style
    
    return clean_text

# ==========================================
# 3. MANUAL INPUT FUNCTION
# ==========================================
def get_manual_input():
    print("\n📝 Paste your chat or type your feelings below.")
    print("   (Type 'DONE' on a new line when you are finished)")
    print("-" * 40)
    
    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        lines.append(line)
    
    return " ".join(lines)

# ==========================================
# 4. MAIN APPLICATION LOOP
# ==========================================
print("\n" + "="*60)
print("🎬  MOOD RECOMMENDER (Chat & File Import)  🎬")
print("="*60)

while True:
    print("\nSelect Input Method:")
    print("1. Manual Entry (Type/Paste)")
    print("2. Import Text File (e.g., chat.txt)")
    print("3. Exit")
    
    choice = input("👉 Enter choice (1-3): ").strip()

    extracted_text = ""

    if choice == '1':
        extracted_text = get_manual_input()
        if not extracted_text:
            print("⚠️ No text entered. Try again.")
            continue

    elif choice == '2':
        fname = input("📂 Enter filename (e.g., 'chat.txt'): ").strip()
        extracted_text = process_chat_file(fname)
        if not extracted_text:
            continue
        print(f"✅ Loaded {len(extracted_text)} characters from file.")

    elif choice == '3' or choice.lower() == 'exit':
        print("👋 Bye!")
        break
    
    else:
        print("❌ Invalid choice.")
        continue

    # --- PREDICTION & RESULT ---
    # We pass the 'extracted_text' (which contains the whole chat) to the recommender.
    # It checks for keywords like 'happy', 'sad', 'love' inside that text.
    
    pref = input("📺 Prefer Movie (m) or Web Series (w)? ").strip().lower()
    recs = get_recommendations(extracted_text, pref)
    
    print(f"\n✨ Based on your chat, here are 4 picks:")
    for r in recs:
        print(f"  - {r}")