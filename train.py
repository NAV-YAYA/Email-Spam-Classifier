# ==========================================
# EMAIL SPAM CLASSIFIER
# ==========================================

# Import Libraries
import pandas as pd
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Download NLTK Resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# ==========================================
# LOAD DATASET
# ==========================================

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(
    url,
    sep="\t",
    names=["label", "message"]
)

print("========== DATASET LOADED ==========\n")
print(df.head())

# ==========================================
# DATA EXPLORATION
# ==========================================

print("\n========== DATASET INFO ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# DATA CLEANING
# ==========================================

df = df.drop_duplicates()

print("\n========== AFTER REMOVING DUPLICATES ==========")
print("Dataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())

# ==========================================
# TEXT PREPROCESSING
# ==========================================

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenize
    words = word_tokenize(text)

    cleaned_words = []

    for word in words:

        # Keep only letters and numbers
        if word.isalnum():

            # Remove stopwords
            if word not in stop_words:

                # Stem the word
                cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)

df["transformed_message"] = df["message"].apply(transform_text)

print("\n========== ORIGINAL VS CLEANED ==========")
print(df[["message", "transformed_message"]].head())

# ==========================================
# LABEL ENCODING
# ==========================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

print("\n========== LABEL ENCODING ==========")
print(df.head())

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(df["transformed_message"]).toarray()

y = df["label"]

print("\n========== FEATURE MATRIX ==========")
print("Feature Matrix Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN TEST SPLIT ==========")
print("Training Samples:", X_train.shape)
print("Testing Samples :", X_test.shape)

# ==========================================
# TRAIN MODEL
# ==========================================

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(model, "models/spam_classifier.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\n====================================")
print("Model and Vectorizer Saved Successfully!")
print("====================================")