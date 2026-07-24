import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Load model and vectorizer
model = joblib.load("models/spam_classifier.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))


def transform_text(text):
    text = text.lower()

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:
        if word.isalnum():
            if word not in stop_words:
                cleaned_words.append(ps.stem(word))

    return " ".join(cleaned_words)


# Take input from the user
message = input("Enter your message:\n")

# Preprocess
transformed_message = transform_text(message)

# Convert to TF-IDF features
vector = tfidf.transform([transformed_message])

# Predict
prediction = model.predict(vector)

print("\n==============================")

if prediction[0] == 1:
    print("Prediction: SPAM 🚨")
else:
    print("Prediction: HAM ✅")

print("==============================")