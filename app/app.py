import streamlit as st
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Movie Genre Classification",
    page_icon="🎬",
    layout="wide"
)

# ---------------- NLTK ----------------
nltk.download("stopwords")

# ---------------- LOAD MODEL ----------------
model = joblib.load("../models/genre_model.pkl")
tfidf = joblib.load("../models/tfidf.pkl")

stop_words = set(stopwords.words("english"))

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.block-container {
    padding-top: 1rem;
}

.title {
    text-align: center;
    color: black;
    font-size: 60px;
    font-weight: 900;
    letter-spacing: 3px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 22px;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    color: gray;
    padding-top: 20px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎥 Project Information")

st.sidebar.markdown("""
### Technologies Used

✅ NLP

✅ TF-IDF Vectorization

✅ Logistic Regression

✅ Streamlit
""")

st.sidebar.markdown("---")

st.sidebar.metric("Movies", "54,214")
st.sidebar.metric("Genres", "27")
st.sidebar.metric("Accuracy", "58.4%")
st.sidebar.metric("Features", "10,000")

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------- TITLE ----------------
st.markdown(
    '<div class="title">🎬 MOVIE GENRE CLASSIFICATION</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict movie genres using Machine Learning and NLP</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SAMPLE INPUTS ----------------
with st.expander("📖 Sample Movie Descriptions"):

    st.write("🔪 A detective investigates a mysterious murder in New York.")
    st.write("🚀 Astronauts travel through space to save humanity.")
    st.write("❤️ Two lovers struggle against family opposition.")
    st.write("👻 A haunted house terrifies a small town.")
    st.write("⚔️ Soldiers fight during a historic war.")

# ---------------- INPUT ----------------
description = st.text_area(
    "✍️ Enter Movie Description",
    height=220,
    placeholder="Type the movie plot here..."
)



# ---------------- GENRE ICONS ----------------
genre_icons = {
    "action": "🔥",
    "adventure": "🏔️",
    "animation": "🎨",
    "biography": "📚",
    "comedy": "😂",
    "crime": "🚔",
    "documentary": "🎬",
    "drama": "🎭",
    "family": "👨‍👩‍👧",
    "fantasy": "🧙",
    "history": "📜",
    "horror": "👻",
    "music": "🎵",
    "mystery": "🕵️",
    "romance": "❤️",
    "sci-fi": "🚀",
    "sport": "⚽",
    "thriller": "🔪",
    "war": "⚔️",
    "western": "🤠"
}

# ---------------- PREDICTION ----------------
if st.button("🎯 Predict Genre", use_container_width=True):

    if description.strip() == "":
        st.warning("Please enter a movie description.")

    else:

        cleaned = clean_text(description)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)

        genre = prediction[0]

        icon = genre_icons.get(
            genre.lower(),
            "🎥"
        )

        st.success(
            f"{icon} Predicted Genre: {genre.upper()}"
        )

        # Top 3 Predictions
        probabilities = model.predict_proba(vector)[0]

        classes = model.classes_

        top_indices = probabilities.argsort()[-3:][::-1]

        st.subheader("🎯 Top 3 Predictions")

        for idx in top_indices:

            confidence = probabilities[idx]

            st.write(
                f"**{classes[idx].title()}** : {confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

# ---------------- METRICS ----------------
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Movies", "54K+")
col2.metric("Genres", "27")
col3.metric("Accuracy", "58.4%")
col4.metric("Features", "10K")

# ---------------- GENRE DISTRIBUTION ----------------
st.markdown("---")

if st.checkbox("📊 Show Genre Distribution"):

    df = pd.read_csv(
        "../data/train_data.txt",
        sep=" ::: ",
        names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
        engine="python"
    )

    genre_count = df["GENRE"].value_counts().head(10)

    st.subheader("Top 10 Genres")

    st.bar_chart(genre_count)

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <div class="footer">
        Developed  with Machine Learning, NLP, and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)