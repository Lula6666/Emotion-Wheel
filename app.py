import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os

# -----------------------------------------
# Page config 
# -----------------------------------------
st.set_page_config(page_title="Emotion-Based Recommender", layout="wide")

# -----------------------------------------
# Styling
# -----------------------------------------
st.markdown("""
    <style>
        .stRadio > div { flex-direction: row !important; gap: 8px; }
        .block-container { padding-top: 1rem; }
        .stSlider { margin-bottom: 20px !important; }
        footer {visibility: hidden;}
        h3, h4 { font-size: 18px !important; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Download data from Google Drive
# -----------------------------------------
ratings_url = "https://drive.google.com/uc?id=1idUzlmzAvjtUCVRgdpFTWab5-Q6yUaD9"
movies_url = "https://drive.google.com/uc?id=1r5Iw7cVWSEZrAqQsSqClC-Meq_o8g392"

if not os.path.exists("ratings.csv"):
    gdown.download(ratings_url, "ratings.csv", quiet=False)

if not os.path.exists("movies.csv"):
    gdown.download(movies_url, "movies.csv", quiet=False)

# -----------------------------------------
# Load Data
# -----------------------------------------
ratings = pd.read_csv("ratings.csv")
ratings.columns = ratings.columns.str.strip()

movies = pd.read_csv("movies.csv")
movies.columns = movies.columns.str.strip()

# -----------------------------------------
# Prepare Stats
# -----------------------------------------
movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
    .merge(movies[["movieId", "title"]], on="movieId")
)

sample8 = movies.sample(8, random_state=42)["title"].tolist()

emotions = [
    "Joy", "Trust", "Fear", "Surprise",
    "Sadness", "Disgust", "Anger", "Anticipation"
]

COLOR_MAP = {
    "Joy": "#FFD700", "Trust": "#9ACD32", "Fear": "#00FF00", "Surprise": "#00CED1",
    "Sadness": "#1E90FF", "Disgust": "#8A2BE2", "Anger": "#FF4500", "Anticipation": "#FFA500"
}
SIZE_MAP = {"Low": 0.6, "Diverse": 0.2, "High": 1.0}

# -----------------------------------------
# Plot Emotion Wheel
# -----------------------------------------
def plot_petal_wheel(emo_choices):
    labels = list(emo_choices.keys())
    sizes = [SIZE_MAP[emo_choices[l]] for l in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(2, 2))
    for angle, lbl, size in zip(angles, labels, sizes):
        face = "white" if emo_choices[lbl] == "Diverse" else COLOR_MAP[lbl]
        edge = COLOR_MAP[lbl] if emo_choices[lbl] != "Diverse" else "gray"
        ax.bar(angle, size, width=0.7, facecolor=face, edgecolor=edge, linewidth=1.5, alpha=0.8)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_yticks([])
    fig.tight_layout()
    return fig

# -----------------------------------------
# Session State
# -----------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False

# -----------------------------------------
# Step 0 
# -----------------------------------------
if st.session_state.step == 0:
    st.title("Welcome to the Emotion‑Based Recommender")
    st.markdown("""
    1. **Rate 4+ movies (1 = dislike, 5 = like)**  
    2. **We prepare your recommendations**  
    3. **Choose your emotional tone preferences**  
    4. **Get your custom movie list + emotion wheel**
    """)
    if st.button("Get Started"):
        st.session_state.step = 1

# -----------------------------------------
# Step 1 — Rate Movies
# -----------------------------------------
elif st.session_state.step == 1:
    st.header("Step 1: Rate These Movies")
    st.write("Rate at least **4** of the following 8 randomly selected movies:")
    ratings_inputs = {}
    for i, title in enumerate(sample8):
        col = st.columns(2)[i % 2]
        with col:
            ratings_inputs[title] = st.slider(f"{title}", 1, 5, 3, key=f"slider_{i}")
    if st.button("Continue"):
        st.session_state.user_ratings = ratings_inputs
        st.session_state.step = 2

# -----------------------------------------
# Step 2 — Buffer
# -----------------------------------------
elif st.session_state.step == 2:
    st.header("Preparing your personalized recommendations…")
    if st.button("Next"):
        st.session_state.step = 3

# -----------------------------------------
# Step 3 — Configure & Show
# -----------------------------------------
elif st.session_state.step == 3:
    st.header("Step 3: Adjust Your Preferences")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Settings")
        user_id = st.number_input("User ID", value=int(ratings["userId"].min()), step=1)
        num_rec = st.slider("How many movies to recommend?", 1, 20, 7)

        st.markdown("### Select Your Emotional Intensity")
        emo_choices = {}
        emo_cols_row1 = st.columns(4)
        emo_cols_row2 = st.columns(4)

        for i, emo in enumerate(emotions[:4]):
            with emo_cols_row1[i]:
                emo_choices[emo] = st.radio(emo, ["Low", "Diverse", "High"], key=f"emo_{emo}")

        for i, emo in enumerate(emotions[4:]):
            with emo_cols_row2[i]:
                emo_choices[emo] = st.radio(emo, ["Low", "Diverse", "High"], key=f"emo_{emo}")

        if st.button("Show Recommendations"):
            st.session_state.show_recs = True

    with col2:
        st.markdown("### Emotion Wheel")
        fig = plot_petal_wheel(emo_choices)
        st.pyplot(fig)

    if st.session_state.show_recs:
        seen = set(ratings[ratings["userId"] == user_id]["movieId"])
        cands = movie_stats[~movie_stats["movieId"].isin(seen)]
        top = (
            cands.sort_values(["avg_rating", "num_ratings"], ascending=False)
                 .head(num_rec)[["title", "avg_rating", "num_ratings"]]
        )

        st.markdown("## Recommended Movies")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
