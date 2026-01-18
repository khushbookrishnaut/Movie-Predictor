import streamlit as st
import joblib
import random
import re
import pypdf
from tokenizers import Tokenizer

# ======================
# CONFIG & STATE
# ======================
st.set_page_config(page_title="CineMood", page_icon="🎬", layout="wide")
st.title("🎬 CineMood")
st.markdown("---")

# Initialize Session State
if 'chat_data' not in st.session_state:
    st.session_state['chat_data'] = {}
if 'users' not in st.session_state:
    st.session_state['users'] = []
if 'manual_analyzed' not in st.session_state:
    st.session_state['manual_analyzed'] = False

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_brain():
    try:
        model = joblib.load("model/model.pkl")
        vectorizer = joblib.load("model/vectorizer.pkl")
        label_encoder = joblib.load("model/label_encoder.pkl")
        tokenizer = Tokenizer.from_file("model/bpe_tokenizer.json")
        return model, vectorizer, tokenizer, label_encoder
    except:
        return None, None, None, None

model, vectorizer, tokenizer, label_encoder = load_brain()

# ======================
# CORE LOGIC
# ======================
def predict_mood(text):
    if model:
        bpe_text = " ".join(tokenizer.encode(str(text)).tokens)
        pred = model.predict(vectorizer.transform([bpe_text]))[0]
        return label_encoder.inverse_transform([pred])[0]
    # Fallback
    text = text.lower()
    if "sad" in text: return "sad"
    if "happy" in text: return "happy"
    if "angry" in text: return "anger"
    if "fear" in text: return "fear"
    return "love"

def get_recommendations(mood, media_type, count):
    MOVIE_DB = {
        "sad": [
        "The Sky Is Pink", "Kal Ho Naa Ho", "Taare Zameen Par", "Rockstar", "Lootera", 
        "Masaan", "October", "Anand", "Rang De Basanti", "Ghajini", "Tere Naam", 
        "Ae Dil Hai Mushkil", "Highway", "Udaan", "Swades", "Kapoor & Sons", "Dear Zindagi", 
        "The Shawshank Redemption", "Forrest Gump", "The Pursuit of Happyness", "Schindler's List", 
        "Good Will Hunting", "Manchester by the Sea", "The Green Mile", "Titanic", "Hachi: A Dog's Tale", 
        "Requiem for a Dream", "A Star Is Born", "Blue Valentine", "Her", "Into the Wild", 
        "Eternal Sunshine of the Spotless Mind", "The Whale", "Aftersun", "Marriage Story", 
        "12 Years a Slave", "The Pianist", "Dead Poets Society", "Brokeback Mountain", "Million Dollar Baby"
    ],
    "happy": [
        "3 Idiots", "Zindagi Na Milegi Dobara", "Dil Dhadakne Do", "Band Baaja Baaraat", 
        "Queen", "Piku", "English Vinglish", "Jab We Met", "Hera Pheri", "Munna Bhai M.B.B.S.", 
        "Welcome", "Dhamaal", "Golmaal: Fun Unlimited", "Chennai Express", "Yeh Jawaani Hai Deewani", 
        "Chhichhore", "Andaz Apna Apna", "Dream Girl", "Stree 2", "Khoobsurat", "Bareilly Ki Barfi",
        "Barbie", "The Hangover", "Superbad", "Ferris Bueller's Day Off", "Ratatouille", 
        "Paddington 2", "The Grand Budapest Hotel", "Singin' in the Rain", "Amélie", "Legally Blonde", 
        "School of Rock", "Elf", "Mrs. Doubtfire", "Shrek", "Kung Fu Panda", "Despicable Me", 
        "Mamma Mia!", "Crazy Rich Asians", "Pitch Perfect", "La La Land", "Palm Springs"
    ],
    "anger": [
        "Animal", "KGF: Chapter 1", "KGF: Chapter 2", "Pushpa: The Rise", "Kabir Singh", 
        "Gangs of Wasseypur", "Agneepath", "RRR", "Baahubali", "Satya", "Haider", 
        "Udta Punjab", "Shootout at Lokhandwala", "Vikram", "Leo", "Kaithi", "Salaar", 
        "Singham", "Dabangg", "Nayak", "Gangaajal", "NH10", "Pink", "Mardaani", "Ghajini",
        "John Wick", "Mad Max: Fury Road", "Fight Club", "Gladiator", "The Dark Knight", 
        "Django Unchained", "Kill Bill", "Joker", "Taxi Driver", "Whiplash", "V for Vendetta", 
        "Inglourious Basterds", "The Batman", "American Psycho", "Scarface", "Reservoir Dogs", 
        "No Country for Old Men", "Sicario", "Heat", "Logan", "Oldboy", "Nobody", "Taken"
    ],
    "fear": [
        "Tumbbad", "Stree", "Bhool Bhulaiyaa", "Raaz", "1920", "Pari", "NH10", "Drishyam", 
        "Andhadhun", "Ratsasan", "Kantara", "Phobia", "13B", "Phoonk", "Darna Mana Hai", 
        "Krishna Cottage", "Bhoot", "Raaz 3", "Pizza", "Kaun?", "Talaash",
        "The Conjuring", "Hereditary", "Get Out", "It", "The Shining", "Psycho", "A Quiet Place", 
        "The Exorcist", "Sinister", "Silence of the Lambs", "Se7en", "Zodiac", "Gone Girl", 
        "Shutter Island", "Black Swan", "Midsommar", "Us", "Train to Busan", "Bird Box", 
        "Nope", "Smile", "Talk to Me", "Annabelle", "Insidious", "The Ring"
    ],
    "love": [
        "Dilwale Dulhania Le Jayenge", "Jab Tak Hai Jaan", "Veer-Zaara", "Sita Ramam", 
        "Aashiqui 2", "Barfi!", "Rehnaa Hai Terre Dil Mein", "Kuch Kuch Hota Hai", 
        "Mughal-e-Azam", "Bajirao Mastani", "Ram-Leela", "2 States", "Love Aaj Kal", 
        "Fitoor", "Laila Majnu", "Rock On!!", "Mohabbatein", "Kabhi Khushi Kabhie Gham", 
        "Hum Dil De Chuke Sanam", "Raanjhanaa", "Shershaah", "Kabir Singh",
        "Titanic", "The Notebook", "Pride & Prejudice", "Before Sunrise", "About Time", 
        "Casablanca", "Crazy, Stupid, Love", "500 Days of Summer", "Notting Hill", 
        "Silver Linings Playbook", "Me Before You", "The Vow", "Dear John", 
        "A Walk to Remember", "Fault in Our Stars", "Love Actually", "10 Things I Hate About You", 
        "Pretty Woman", "Dirty Dancing", "Ghost"
    ]
        
    }
    SERIES_DB = {
        "sad": [
        # Western
        "This Is Us", "The Crown", "Normal People", "After Life", "Chernobyl", 
        "Maid", "When They See Us", "Unbelievable", "The Leftovers", "Patrick Melrose", 
        "I Know This Much Is True", "BoJack Horseman", "The Handmaid's Tale", "Six Feet Under", 
        "Broadchurch", "13 Reasons Why", "A Series of Unfortunate Events", "Violet Evergarden",
        # Indian
        "Gullak (Emotional Moments)", "Aspirants", "Kota Factory", "Tabbar", "Grahan", 
        "Mumbai Diaries 26/11", "Kaafir", "Stories by Rabindranath Tagore", "Little Things (S3/S4)", 
        "The Railway Men", "Jubilee", "Rocket Boys", "Yeh Kaali Kaali Ankhein", "Decoupled (Drama parts)"
    ],
    "happy": [
        # Western
        "Friends", "The Office", "Brooklyn Nine-Nine", "Ted Lasso", "Schitt's Creek", 
        "Modern Family", "Parks and Recreation", "The Big Bang Theory", "How I Met Your Mother", 
        "Seinfeld", "The Good Place", "New Girl", "Superstore", "Abbott Elementary", 
        "Community", "Arrested Development", "It's Always Sunny in Philadelphia", "Marvelous Mrs. Maisel",
        # Indian
        "Panchayat", "Sarabhai vs Sarabhai", "Little Things", "Tripling", "Ye Meri Family", 
        "Hostel Daze", "Chacha Vidhayak Hain Humare", "Home Shanti", "Metro Park", 
        "Permanent Roommates", "What the Folks", "Mind the Malhotras", "Comicstaan", "The Aam Aadmi Family"
    ],
    "anger": [
        # Western
        "The Boys", "Breaking Bad", "Peaky Blinders", "Game of Thrones", "Succession", 
        "Vikings", "The Punisher", "Daredevil", "Sons of Anarchy", "Reacher", 
        "Better Call Saul", "Ozark", "The Wire", "Narcos", "The Sopranos", 
        "Fargo", "True Detective", "Sherlock", "Money Heist", "Jack Ryan",
        # Indian
        "Mirzapur", "Paatal Lok", "Sacred Games", "The Family Man", "Farzi", 
        "Rana Naidu", "Asur", "Rangbaaz", "Delhi Crime", "Criminal Justice", 
        "Breathe", "Aarya", "Bambai Meri Jaan", "Scoop", "Kohrra", 
        "Jamtara", "Apharan", "Raktanchal", "Scam 1992"
    ],
    "fear": [
        # Western
        "Stranger Things", "Black Mirror", "Dark", "Mindhunter", "The Haunting of Hill House", 
        "Alice in Borderland", "Squid Game", "Hannibal", "American Horror Story", "Midnight Mass", 
        "The Walking Dead", "Penny Dreadful", "Bates Motel", "The X-Files", "Twin Peaks", 
        "Yellowjackets", "Severance", "From", "Archive 81", "The Last of Us",
        # Indian
        "Ghoul", "Betaal", "Typewriter", "Breathe: Into the Shadows", "Dahaad", 
        "Aranyak", "The Last Hour", "Bhram", "Dahan", "Adhura", 
        "Tooth Pari", "Shaitaan Haveli", "Gehraiyaan (Thriller aspects)", "School of Lies"
    ],
    "love": [
        # Western
        "Bridgerton", "Outlander", "Gossip Girl", "Emily in Paris", "Sex and the City", 
        "Grey's Anatomy", "Downton Abbey", "Heartstopper", "Modern Love", "Fleabag", 
        "The Summer I Turned Pretty", "Vampire Diaries", "Gilmore Girls", "One Day", 
        "Virgin River", "Sweet Magnolias", "Crash Landing on You",
        # Indian
        "Mismatched", "Flames", "Broken But Beautiful", "College Romance", "Bandish Bandits", 
        "Feels Like Ishq", "Made in Heaven", "Four More Shots Please!", "Indori Ishq", 
        "Baarish", "Kaisi Yeh Yaariaan", "Permanent Roommates", "Romil and Jugal", "Taj Mahal 1989"
    ]
    }
    db = SERIES_DB if media_type == "Web Series" else MOVIE_DB
    items = db.get(mood.lower(), db["happy"])
    return random.sample(items, min(count, len(items)))

def parse_chat_logic(full_text):
    """
    Parses text into {User: [Messages]}
    """
    data = {}
    lines = full_text.split('\n')
    current_speaker = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Look for Name: Message pattern
        if ':' in line:
            parts = line.split(':', 1)
            meta_chunk = parts[0].strip()
            message = parts[1].strip()
            
            # Clean Name (Remove timestamps like [10:00], 12/12/23)
            clean_name = re.sub(r"\[.*?\]", "", meta_chunk) 
            clean_name = clean_name.split(' - ')[-1] # WhatsApp dash
            clean_name = clean_name.split(', ')[-1]  # WhatsApp commas
            clean_name = clean_name.strip()
            
            # Identify if it's a valid speaker name (short length)
            if 0 < len(clean_name) < 25:
                current_speaker = clean_name
                if current_speaker not in data:
                    data[current_speaker] = []
                data[current_speaker].append(message)
                continue
        
        if current_speaker and current_speaker in data:
            data[current_speaker].append(line)
            
    return data

# ======================
# UI LAYOUT
# ======================
st.sidebar.header("⚙️ Preferences")
pref_type = st.sidebar.radio("Watch:", ["Movie", "Web Series"])
rec_count = st.sidebar.slider("Count", 3, 10, 5)

tab1, tab2 = st.tabs(["✍️ Manual / Paste Chat", "📂 Upload PDF"])

# ==========================================
# TAB 1: SMART MANUAL INPUT (Unified Logic)
# ==========================================
with tab1:
    st.caption("Paste a full chat conversation (without Time-Stamps) OR just type how you feel.")
    manual_input = st.text_area("Input Text:", height=200, placeholder="Rohan: I am happy.\n Vikram: I am sad.")

    # We use a single button to trigger detection
    if st.button("Analyze Input"):
        if not manual_input.strip():
            st.warning("Please type something.")
        else:
            # 1. Try to parse as Chat
            parsed_data = parse_chat_logic(manual_input)
            detected_users = list(parsed_data.keys())

            if len(detected_users) > 1:
                # === MULTI-USER DETECTED ===
                st.session_state['manual_chat_data'] = parsed_data
                st.session_state['manual_users'] = detected_users
                st.session_state['manual_analyzed'] = True
                st.rerun() # Refresh to show selectbox below
            else:
                # === SINGLE USER / SIMPLE TEXT ===
                # Clear complex state if it exists
                st.session_state['manual_analyzed'] = False 
                
                # Analyze directly
                mood = predict_mood(manual_input)
                st.success(f"**Detected Mood:** {mood.upper()}")
                recs = get_recommendations(mood, pref_type, rec_count)
                for r in recs: st.info(f"🍿 {r}")

    # === PERSISTENT UI FOR MULTI-USER SELECTION ===
    # This block runs if we detected multiple users in the previous step
    if st.session_state.get('manual_analyzed') and 'manual_users' in st.session_state:
        st.divider()
        st.info(f"👥 **Multi-Speaker Chat Detected!** Found: {', '.join(st.session_state['manual_users'])}")
        
        target_user = st.selectbox("Who do you want to analyze?", st.session_state['manual_users'])
        
        if st.button(f"Analyze {target_user}"):
            user_text = " ".join(st.session_state['manual_chat_data'][target_user])
            mood = predict_mood(user_text)
            
            st.subheader(f"Results for {target_user}: **{mood.upper()}**")
            recs = get_recommendations(mood, pref_type, rec_count)
            cols = st.columns(2)
            for i, r in enumerate(recs):
                cols[i%2].success(f"🎥 {r}")

# ==========================================
# TAB 2: PDF UPLOAD (Same Logic)
# ==========================================
with tab2:
    uploaded_file = st.file_uploader("Upload Chat PDF without Time-Stamps", type=["pdf"])
    
    if uploaded_file:
        full_text = ""
        try:
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
            
            chat_data = parse_chat_logic(full_text)
            users = list(chat_data.keys())
            
            if not users:
                st.error("No speakers found in PDF.")
            else:
                st.success(f"✅ Found: {', '.join(users)}")
                target_user = st.selectbox("Select Person:", users, key="pdf_user_select")
                
                if st.button("Analyze PDF User"):
                    user_text = " ".join(chat_data[target_user])
                    mood = predict_mood(user_text[:5000])
                    st.subheader(f"Mood: {mood.upper()}")
                    recs = get_recommendations(mood, pref_type, rec_count)
                    for r in recs: st.info(f"🎥 {r}")
                    
        except Exception as e:
            st.error(f"Error: {e}")