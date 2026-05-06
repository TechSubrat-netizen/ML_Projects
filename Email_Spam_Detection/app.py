import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# -------------------- SETUP --------------------
st.set_page_config(
    page_title="Spam Detector AI",
    page_icon="📧",
    layout="centered"
)

# Download stopwords if needed
try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
/* Background */
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Title */
.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: white;
}

.subtitle {
    text-align: center;
    color: #e0e0e0;
    margin-bottom: 30px;
}

/* Glass Card */
.glass {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

/* Button */
.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 10px;
    background: linear-gradient(90deg,#667eea,#764ba2);
    color: white;
    font-size: 18px;
    border: none;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.05);
}

/* Result */
.result {
    text-align: center;
    font-size: 22px;
    padding: 20px;
    border-radius: 12px;
    font-weight: bold;
}

.spam {
    background: rgba(255,82,82,0.2);
    color: #ff5252;
}

.ham {
    background: rgba(76,175,80,0.2);
    color: #4caf50;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("email_spam_model.joblib")
        tfidf = joblib.load("tfidf_vectorizer.joblib")
        return model, tfidf
    except:
        st.error("❌ Model files not found")
        return None, None

model, tfidf = load_model()

# -------------------- TEXT CLEANING --------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = re.sub(r'\bsubject\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# -------------------- HEADER --------------------
st.markdown("<div class='title'>📧 Spam Detector AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect spam emails instantly using Machine Learning</div>", unsafe_allow_html=True)

# -------------------- MAIN UI --------------------
if model and tfidf:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    email_text = st.text_area(
        "Enter Email",
        placeholder="Paste your email content here...",
        height=200,
        label_visibility="collapsed"
    )

    analyze = st.button("🚀 Analyze Email")

    st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- PREDICTION --------------------
    if analyze:
        if not email_text.strip():
            st.warning("⚠️ Please enter email content")
        else:
            with st.spinner("Analyzing..."):
                cleaned = clean_text(email_text)
                vector = tfidf.transform([cleaned])

                prediction = model.predict(vector)[0]
                prob = model.predict_proba(vector)[0]

                is_spam = prediction == 1

                # ---------------- RESULT ----------------
                st.markdown("---")

                result_class = "spam" if is_spam else "ham"
                result_text = "⚠️ SPAM DETECTED" if is_spam else "✅ SAFE EMAIL"

                st.markdown(f"""
                <div class='result {result_class}'>
                    {result_text}
                </div>
                """, unsafe_allow_html=True)

                # ---------------- METRICS ----------------
                st.subheader("📊 Confidence Scores")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Legitimate", f"{prob[0]*100:.2f}%")
                    st.progress(prob[0])

                with col2:
                    st.metric("Spam", f"{prob[1]*100:.2f}%")
                    st.progress(prob[1])

                # ---------------- CLEAN TEXT ----------------
                with st.expander("🧹 Processed Text"):
                    st.write(cleaned)

                # ---------------- INFO ----------------
                st.info("""
                **How it works:**
                - Removes subject line
                - Cleans text
                - Removes stopwords
                - Converts using TF-IDF
                - Predicts using ML model
                """)

else:
    st.error("Model not loaded. Please check files.")