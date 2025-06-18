import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Setup stemmer
stemmer = PorterStemmer()

# ✅ Move stopword loading inside preprocessing function
def preprocess(text):
    stop_words = set(stopwords.words('english'))  # ← moved inside to prevent premature call
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered)

# Streamlit UI
st.title("🧠 Sentiment Analysis on Tweets")
st.subheader("Enter a tweet to analyze sentiment")

user_input = st.text_area("Your Tweet")

if st.button("Analyze"):
    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    label = {1: "🟢 Positive", 0: "🟡 Neutral", -1: "🔴 Negative"}
    st.success(f"Predicted Sentiment: {label[prediction]}")

