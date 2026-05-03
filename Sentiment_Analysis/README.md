# 🎯 Sentiment Analysis Web App

A machine learning-based web application that analyzes the sentiment of text reviews and classifies them as **Positive** or **Negative**. Built with Python, Scikit-learn, and Streamlit.

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup & Training](#setup--training)
- [Running the App](#running-the-app)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

✅ **Real-time Sentiment Analysis** - Analyze text sentiment instantly  
✅ **Text Preprocessing** - Automatic text cleaning and stopword removal  
✅ **Confidence Score** - Shows prediction confidence percentage  
✅ **Probability Visualization** - Bar chart showing positive/negative probabilities  
✅ **Clean UI** - User-friendly Streamlit interface with emojis and formatting  
✅ **Text Transformation** - TF-IDF vectorization for ML-ready features  
✅ **Balanced Dataset** - Trained on balanced positive/negative reviews  

---

## 📁 Project Structure

```
Sentiment_Analysis/
├── app.py                           # Streamlit web application
├── sentiment_model.ipynb            # Jupyter notebook with ML pipeline
├── dataset.csv                      # Raw review data
├── sentiment_analysis_model.pkl     # Trained logistic regression model
├── tfidf_vectorizer.pkl             # TF-IDF vectorizer
├── README.md                        # This file
└── requirements.txt                 # Python dependencies
```

---

## 🔧 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Step 1: Clone or Download the Project
```bash
cd c:\Users\subra\Desktop\ML_Projects\Sentiment_Analysis
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install pandas numpy scikit-learn nltk streamlit wordcloud matplotlib seaborn joblib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

---

## 🚀 Setup & Training

### Step 1: Run the Jupyter Notebook
The notebook (`sentiment_model.ipynb`) handles:
- Data loading and exploration
- Text cleaning and preprocessing
- Model training and evaluation
- Pickle file generation

```bash
jupyter notebook sentiment_model.ipynb
```

### Step 2: Execute All Notebook Cells
Run all cells in order. The notebook will:
1. Load the dataset
2. Clean and preprocess reviews
3. Balance positive/negative samples
4. Create TF-IDF vectors
5. Train a Logistic Regression model
6. Save models to pickle files:
   - `sentiment_analysis_model.pkl`
   - `tfidf_vectorizer.pkl`

**Important:** Make sure the last cells execute successfully to generate the pickle files!

---

## 🌐 Running the App

Once the model files are generated, run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🔍 How It Works

### Text Preprocessing Pipeline
1. **Remove Special Characters** - Eliminates punctuation and numbers
2. **Lowercase Conversion** - Standardizes text
3. **Stopword Removal** - Removes common English words (the, is, etc.)
4. **Tokenization** - Splits text into words

### ML Pipeline
1. **TF-IDF Vectorization** - Converts cleaned text to numerical features
   - Max 5000 features
   - Captures word importance
2. **Logistic Regression** - Predicts sentiment
   - Fast and interpretable
   - Provides probability scores
3. **Classification** - Outputs Positive or Negative

### Example Flow
```
User Input: "This product is amazing! I love it."
    ↓
Clean: "product amazing love"
    ↓
TF-IDF: [0.35, 0.52, 0.47, 0.28, ...]
    ↓
Model: Positive (96.8% confidence)
```

---

## 📊 Model Details

### Training Data
- **Dataset:** `dataset.csv`
- **Features:** Customer reviews
- **Labels:** Ratings (converted to positive/negative)
- **Positive Reviews:** Rating > 5
- **Negative Reviews:** Rating ≤ 5

### Model Specifications
- **Algorithm:** Logistic Regression
- **Vectorizer:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Max Features:** 5000
- **Test Size:** 20%
- **Random State:** 43

### Data Balancing
The dataset is balanced to prevent bias:
- Positive samples downsampled to match negative samples
- Ensures fair model training

---

## 📦 Requirements

```
pandas>=1.0.0
numpy>=1.18.0
scikit-learn>=0.24.0
nltk>=3.5
streamlit>=1.0.0
wordcloud>=1.8.0
matplotlib>=3.3.0
seaborn>=0.11.0
joblib>=1.0.0
```

---

## 💡 Usage Examples

### Example 1: Positive Review
```
Input: "Excellent quality and fast shipping! Highly recommended."
Output: POSITIVE (98.5% confidence)
```

### Example 2: Negative Review
```
Input: "Poor quality, broke after one day. Very disappointed."
Output: NEGATIVE (94.2% confidence)
```

### Example 3: Mixed Review
```
Input: "Good product but delivery took too long."
Output: POSITIVE (61.3% confidence)
```

---

## 🛠️ Troubleshooting

### Issue: "Model files not found" Error
**Solution:** 
- Ensure you've run all cells in `sentiment_model.ipynb`
- Check that pickle files exist in the project directory
- Run the notebook again and verify joblib.dump() executes

### Issue: NLTK Stopwords Error
**Solution:**
```python
import nltk
nltk.download('stopwords')
```

### Issue: Streamlit Won't Start
**Solution:**
```bash
# Kill any existing Streamlit process
# Reinstall Streamlit
pip install --upgrade streamlit

# Run again
streamlit run app.py
```

### Issue: Low Accuracy
**Solution:**
- Review the dataset quality
- Check text preprocessing in notebook
- Ensure balanced dataset code runs
- Try different model parameters in notebook

---

## 📈 Performance Metrics

The model's performance is evaluated using:
- **Accuracy Score** - Overall prediction correctness
- **Confusion Matrix** - TP, TN, FP, FN values
- **Classification Report** - Precision, Recall, F1-Score
- **Probability Distributions** - Confidence in predictions

Check the notebook output for detailed metrics.

---

## 🎓 Learning Points

This project demonstrates:
- Text preprocessing and NLP fundamentals
- TF-IDF vectorization technique
- Logistic Regression classification
- Dataset balancing techniques
- Model persistence with pickle/joblib
- Web app development with Streamlit
- Data visualization with matplotlib/seaborn

---

## 📝 Notes

- The model is trained on the provided dataset. Results may vary with different data.
- For production use, consider more sophisticated models (NLP, BERT, etc.)
- Regularly retrain the model with new data for better accuracy.
- The app caches model loading for faster performance.

---

## 🔗 Project Workflow

```
1. Data Collection (dataset.csv)
    ↓
2. Exploratory Data Analysis (sentiment_model.ipynb)
    ↓
3. Text Preprocessing & Cleaning
    ↓
4. Feature Engineering (TF-IDF)
    ↓
5. Dataset Balancing
    ↓
6. Model Training (Logistic Regression)
    ↓
7. Model Evaluation & Metrics
    ↓
8. Model Serialization (pickle files)
    ↓
9. Web App Deployment (Streamlit)
```

---

## 📧 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the notebook comments
3. Verify all dependencies are installed
4. Ensure pickle files are generated correctly

---

**Happy Sentiment Analysis! 🚀**
