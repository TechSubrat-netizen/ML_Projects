# Email Spam Detection Web App

A Streamlit-based web application for detecting spam emails using machine learning.

## Features

- 🎯 **Real-time Spam Detection** - Instantly classify emails as spam or legitimate
- 📊 **Probability Scores** - See confidence percentages for both spam and ham classifications
- 🧹 **Text Processing** - Automatic email text cleaning and normalization
- 💻 **User-Friendly Interface** - Clean, modern, interactive web interface
- ⚡ **Fast Predictions** - Quick processing with pre-trained model
- 📱 **Responsive Design** - Works on desktop and mobile

## Requirements

- Python 3.8 or higher
- Flask
- scikit-learn
- pandas
- nltk
- numpy

## Files Required

Make sure you have the following joblib files in the same directory as `app.py`:
- `email_spam_model.joblib` - Trained Logistic Regression model
- `tfidf_vectorizer.joblib` - Trained TF-IDF Vectorizer

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Place the model files:**
   - Copy `email_spam_model.joblib` to the project directory
   - Copy `tfidf_vectorizer.joblib` to the project directory

## Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run app.py
```

2. **Open your browser:**
   - Streamlit will automatically open at `http://localhost:8501`
   - Or navigate to that URL manually

3. **Using the app:**
   - Paste your email content in the text area
   - Click "🔍 Check for Spam" button
   - View the prediction results with probability scores and processed text

## How It Works

1. **Input Processing** - User enters email text
2. **Text Cleaning** - Email is cleaned by:
   - Removing "Subject:" prefix
   - Removing special characters
   - Converting to lowercase
   - Removing stopwords
3. **Vectorization** - Cleaned text is converted to TF-IDF features
4. **Prediction** - Model predicts if email is spam or ham
5. **Display Results** - Shows prediction and probability scores

## Model Information

- **Algorithm**: Logistic Regression
- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classes**: 
  - Ham (Legitimate Email) - Label: 0
  - Spam - Label: 1

## App Features

### Detector Tab
- **Email Input** - Large text area for pasting email content
- **Spam Check Button** - One-click spam detection
- **Real-time Results** - Instant prediction with visual feedback
- **Probability Metrics** - See confidence scores for both categories
- **Processed Text** - View how the email was cleaned and processed
- **Informational Tips** - Learn how the app processes emails

### About Tab
- Complete overview of the application
- Technical details about the ML model
- Information about algorithms and features
- Performance metrics

### Key Indicators
- ✅ Green indicators for legitimate emails
- ⚠️ Red indicators for spam emails
- Probability bars showing confidence levels
```

## Troubleshooting

### Models not loading
- Verify that `email_spam_model.joblib` and `tfidf_vectorizer.joblib` exist in the project directory
- Check the file names match exactly (case-sensitive)

### NLTK stopwords error
- The app automatically downloads stopwords if missing
- If issues persist, run: `python -m nltk.downloader stopwords`

### Port already in use
- Change the port in `app.py`: `app.run(debug=True, port=5001)`

## Performance

- Average prediction time: < 100ms
- Supports emails of any length
- Handles special characters and multiple languages

## License

This project is for educational purposes.
