import streamlit as st
import pickle
import pandas as pd
import re
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# -------------------------
# Load Model (Cached)
# -------------------------

@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()


# -------------------------
# Load Dataset (Cached)
# -------------------------

@st.cache_data
def load_data():
    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data[["text", "label"]]

    return fake, true, data

fake, true, data = load_data()


# -------------------------
# Text Cleaning
# -------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -------------------------
# Model Evaluation
# -------------------------

@st.cache_data
def evaluate_model(data):

    sample = data.sample(3000)

    X_sample = vectorizer.transform(sample["text"])
    y_sample = sample["label"]

    pred = model.predict(X_sample)

    accuracy = accuracy_score(y_sample, pred)
    precision = precision_score(y_sample, pred)
    recall = recall_score(y_sample, pred)
    f1 = f1_score(y_sample, pred)

    return accuracy, precision, recall, f1


accuracy, precision, recall, f1 = evaluate_model(data)


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
    "This system analyzes news content and predicts whether the article is **Real or Fake**."
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

news = st.text_area(
    "Enter News Text",
    value=st.session_state.get("example", ""),
    height=200
)


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

    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "News": news[:120],
        "Prediction": "Fake" if prediction[0] == 0 else "Real",
        "Confidence": confidence
    })


# -------------------------
# Prediction History
# -------------------------

st.markdown("---")
st.subheader("Prediction History")

if "history" in st.session_state and st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

else:
    st.info("No predictions yet.")


# -------------------------
# Dataset Distribution
# -------------------------

st.markdown("---")
st.subheader("Dataset Distribution")

labels = ["Fake News", "Real News"]
values = [len(fake), len(true)]

fig, ax = plt.subplots()

ax.bar(labels, values)

ax.set_ylabel("Number of Articles")
ax.set_title("Fake vs Real News Distribution")

st.pyplot(fig, clear_figure=True)


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
# Dataset Sample
# -------------------------

st.markdown("---")
st.subheader("Dataset Sample")

st.dataframe(data.sample(5), use_container_width=True)