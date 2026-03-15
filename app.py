import streamlit as st
import pickle
import pandas as pd
import re
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from collections import Counter

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# -------------------------
# Load Dataset
# -------------------------

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])
data = data[["text", "label"]]

# -------------------------
# Clean Function
# -------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# -------------------------
# Model Evaluation (Dynamic)
# -------------------------

sample = data.sample(3000)

X_sample = vectorizer.transform(sample["text"])
y_sample = sample["label"]

pred = model.predict(X_sample)

accuracy = accuracy_score(y_sample, pred)
precision = precision_score(y_sample, pred)
recall = recall_score(y_sample, pred)
f1 = f1_score(y_sample, pred)

# -------------------------
# Sidebar
# -------------------------

st.sidebar.title("Project Overview")

st.sidebar.metric("Total Articles", len(data))
st.sidebar.metric("Fake Articles", len(fake))
st.sidebar.metric("Real Articles", len(true))

st.sidebar.markdown("---")

st.sidebar.metric("Model Accuracy", f"{accuracy*100:.2f}%")
st.sidebar.metric("Precision", f"{precision*100:.2f}%")
st.sidebar.metric("Recall", f"{recall*100:.2f}%")
st.sidebar.metric("F1 Score", f"{f1*100:.2f}%")

# -------------------------
# Title
# -------------------------

st.title("📰 Fake News Detection System")

st.write(
    "This system analyzes news content and predicts whether the article is **authentic or misleading**."
)

st.markdown("---")

# -------------------------
# Example News
# -------------------------

examples = [
    "Government announces new economic reform to boost employment.",
    "Aliens landed in New York and met the president.",
    "Scientists discovered water on Mars.",
    "Secret organization controls global politics."
]

if st.button("Generate Example News"):
    st.session_state.example = random.choice(examples)

if "example" in st.session_state:
    news = st.text_area("Enter News Text", value=st.session_state.example, height=200)
else:
    news = st.text_area("Enter News Text", height=200)

# -------------------------
# Prediction
# -------------------------

if st.button("Predict"):

    cleaned = clean_text(news)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    confidence = abs(model.decision_function(vector))[0]
    confidence = min(100, round(confidence * 10, 2))

    col1, col2 = st.columns(2)

    if prediction[0] == 0:
        col1.error("🚨 Fake News Detected")
    else:
        col1.success("✅ Real News")

    col2.metric("Confidence Score", f"{confidence}%")

    st.progress(confidence / 100)

    # Save prediction history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "news": news[:120],
        "prediction": "Fake" if prediction[0] == 0 else "Real",
        "confidence": confidence
    })

# -------------------------
# Prediction History
# -------------------------

st.markdown("---")
st.subheader("Prediction History")

if "history" in st.session_state and len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

else:
    st.write("No predictions yet.")


# -------------------------
# Dataset Statistics
# -------------------------

st.markdown("---")
st.subheader("Dataset Distribution")

labels = ["Fake News", "Real News"]
values = [len(fake), len(true)]

fig, ax = plt.subplots()

ax.bar(labels, values)

ax.set_ylabel("Number of Articles")

st.pyplot(fig)


# -------------------------
# Text Statistics
# -------------------------

st.markdown("---")
st.subheader("Dataset Text Statistics")

data["length"] = data["text"].apply(lambda x: len(str(x).split()))

avg_len = int(data["length"].mean())
max_len = int(data["length"].max())
min_len = int(data["length"].min())

col1, col2, col3 = st.columns(3)

col1.metric("Average Article Length", avg_len)
col2.metric("Longest Article", max_len)
col3.metric("Shortest Article", min_len)

# -------------------------
# Dataset Preview
# -------------------------

st.markdown("---")
st.subheader("Dataset Sample")

st.dataframe(data.sample(5))