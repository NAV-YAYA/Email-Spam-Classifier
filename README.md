# Email Spam Classifier

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.

---

## Features

- Data cleaning and preprocessing
- Duplicate removal
- Text preprocessing using NLTK
- TF-IDF Vectorization
- Multinomial Naive Bayes classifier
- Model evaluation
- Save and load trained model
- Predict custom SMS messages

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Joblib

---

## Dataset

SMS Spam Collection Dataset

- 5572 SMS messages
- Ham messages: 4825
- Spam messages: 747

Dataset Source:
https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

---

## Workflow

Dataset
↓
Data Cleaning
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Train/Test Split
↓
Multinomial Naive Bayes
↓
Model Evaluation
↓
Prediction

---

## Model Performance

Accuracy: 97.68%

Precision: 100%

Recall: 81.68%

F1 Score: 89.92%

---

## Project Structure

Email-Spam-Classifier/
│
├── models/
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── train.py
├── predict.py
├── README.md
├── requirements.txt
└── .gitignore

---

## Installation

Clone the repository

```bash
git clone https://github.com/NAV-YAYA/Email-Spam-Classifier.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Predict a message

```bash
python predict.py
```

---

## Example

Input

Congratulations! You have won a FREE iPhone.

Output

Prediction: SPAM

---

## Author

Navya Awasthi