import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# 1. LOAD DATASET
# -----------------------------

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data = data[['text', 'label']]

# Reduce dataset size for faster training
data = data.sample(10000, random_state=42)

print("Dataset Loaded")


# -----------------------------
# 2. CLEAN TEXT
# -----------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text


data['text'] = data['text'].apply(clean_text)

print("Text Cleaning Done")


# -----------------------------
# 3. SPLIT DATA
# -----------------------------

X = data['text']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train Test Split Done")


# -----------------------------
# 4. TF-IDF FEATURES
# -----------------------------

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF Completed")


# -----------------------------
# 5. TRAIN FAST SVM
# -----------------------------

model = LinearSVC()

model.fit(X_train_vec, y_train)

print("Model Training Completed")


# -----------------------------
# 6. TEST MODEL
# -----------------------------

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))


# -----------------------------
# 7. PREDICT CUSTOM NEWS
# -----------------------------

def predict_news(news):

    news = clean_text(news)

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)

    if prediction[0] == 0:
        print("\nResult: Fake News")
    else:
        print("\nResult: Real News")


print("\nTest the model")

user_input = input("Enter news text: ")

predict_news(user_input)