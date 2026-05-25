import pandas as pd
import numpy as np

import streamlit as st
import pickle
import re
from pathlib import Path
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Title and description
st.title("🎯 Sentiment Analysis ")
st.write("Analyze the sentiment of your text - Positive or Negative")
st.write("---")

# Base path for model files
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'sentiment_analysis_model.pkl'
VECTORIZER_PATH = BASE_DIR / 'tfidf_vectorizer.pkl'

# Load the pre-trained model and vectorizer
@st.cache_resource
def load_models():
    import joblib
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    return model, tfidf

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load models
try:
    model, tfidf = load_models()
except FileNotFoundError:
    st.error("❌ Error: Model files (sentiment_analysis_model.pkl or tfidf_vectorizer.pkl) not found!")
    st.info("Run the notebook first to generate these files.")
    st.stop()

# User input
st.subheader("Enter Your Text")
user_input = st.text_area("Type or paste your review/text here:", height=100, placeholder="Enter text to analyze...")

# Make prediction
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if user_input.strip():
        # Clean the text
        cleaned_text = clean_text(user_input)
        
        # Transform using TF-IDF
        X = tfidf.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        
        # Display results
        st.success("✅ Analysis Complete!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sentiment", prediction.upper(), delta=None)
        
        with col2:
            confidence = max(prediction_proba) * 100
            st.metric("Confidence", f"{confidence:.2f}%", delta=None)
        
        # Show probabilities
        st.subheader("Prediction Details")
        probability_df = pd.DataFrame({
            'Sentiment': model.classes_,
            'Probability': prediction_proba
        })
        
        # Color code the results
        if prediction == 'positive':
            st.success(f"**This text has a {prediction.upper()} sentiment** 😊")
        else:
            st.error(f"**This text has a {prediction.upper()} sentiment** 😞")
        
        # Display probability chart
        st.bar_chart(probability_df.set_index('Sentiment'))
        
        # Show cleaned text
        with st.expander("View Cleaned Text"):
            st.write(cleaned_text)
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.write("---")
st.write("💡 **How it works:** The app cleans your text, converts it to numerical features using TF-IDF, and predicts whether the sentiment is positive or negative.")
