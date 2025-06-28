import streamlit as st
import pickle
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# --- Download stopwords if not available ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

# --- Preprocess function ---
def preprocess(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    return ' '.join([stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2])

# --- Load model, vectorizer, and data ---
@st.cache_resource
def load_model():
    files_needed = ['tfidf_vectorizer.pkl', 'knn_model.pkl', 'qa_dataset.pkl']
    for file in files_needed:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file '{file}' is missing in the current directory.")

    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    qa = pd.read_pickle("qa_dataset.pkl")
    qa['Answer'] = qa['Answer'].apply(lambda x: re.sub(r'<[^>]+>', '', x).strip())
    return vectorizer, knn, qa

# --- App Config ---
st.set_page_config(page_title="💬 Tech Chatbot", layout="centered")
st.title("💡 Tech Q&A Chatbot")
st.markdown("Ask your tech-related questions and get instant answers! 🤖")

# --- Try loading model/data ---
try:
    vectorizer, knn, qa = load_model()
    model_loaded = True
except FileNotFoundError as e:
    st.error(str(e))
    model_loaded = False

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Chat Form ---
st.markdown("---")
with st.form("chat_form"):
    user_input = st.text_input("Enter your tech question:", placeholder="e.g., How does Docker work?")
    col1, col2 = st.columns([1, 4])
    with col1:
        clear = st.form_submit_button("🧹 Clear Chat")
    with col2:
        send = st.form_submit_button("🚀 Send")

# --- Chat Logic ---
if clear:
    st.session_state.history.clear()

if send and user_input:
    if model_loaded:
        cleaned_input = preprocess(user_input)
        vect_input = vectorizer.transform([cleaned_input])
        dist, idx = knn.kneighbors(vect_input)
        answer = qa.iloc[idx[0][0]]['Answer']
        trimmed = '. '.join(answer.split('. ')[:3]) + '.'
        st.session_state.history.append((user_input, trimmed))
    else:
        st.warning("Model or data files not loaded. Please ensure all .pkl files are in the same folder.")

# --- Chat History ---
st.markdown("---")
st.subheader("🗨️ Chat History")
for user_q, bot_a in reversed(st.session_state.history):
    st.markdown(f"**You:** {user_q}")
    st.markdown(f"**Bot:** {bot_a}")
    st.markdown("---")

# --- Footer ---
st.markdown("""
<style>
footer {visibility: hidden;}
[data-testid="stSidebar"] > div:first-child {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

if model_loaded:
    st.success("🚀 App is running smoothly. Ask away!")
else:
    st.warning("⚠️ App is running, but waiting for required model/data files.")
