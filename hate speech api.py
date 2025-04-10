import re
import string
import numpy as np
import pandas as pd
import joblib
import nltk
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')
stopword_set = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv("twitter.csv")

# Map class labels
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
data = data[["tweet", "labels"]]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove new lines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove numbers
    text = ' '.join([word for word in text.split() if word not in stopword_set])  # Remove stopwords
    return text

# Apply cleaning to dataset
data["tweet"] = data["tweet"].apply(clean_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["tweet"])
y = data["labels"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & vectorizer
joblib.dump(model, "hate_speech_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Load model & vectorizer for API
model = joblib.load("hate_speech_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for hate speech detection."""
    data = request.get_json()
    
    if "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = clean_text(data["text"])  # Clean input
    vectorized_text = vectorizer.transform([text])  # Transform using TF-IDF
    prediction = model.predict(vectorized_text)[0]  # Make prediction

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)