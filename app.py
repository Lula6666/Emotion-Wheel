import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------
# Page config — must be first!
# -----------------------------------------
st.set_page_config(page_title="Emotion-Based Recommender", layout="wide")

# Optional styling
st.markdown("""
    <style>
        .stRadio > div { flex-direction: row !important; gap: 10px; }
        .block-container { padding-top: 1.5rem; }
        .stSlider { margin-bottom: 20px !important; }
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Load local cleaned CSVs
# -----------------------------------------
ratings = pd.read_csv("ratings.csv")
ratings.columns = ratings.columns.str.strip()

movies = pd.read_csv("movies.csv")
movies.columns = movies.columns.str.strip()

# -----------------------------------------
# Prepare movie stats
# -----------------------------------------
movie_stats = (
    ratings.groupby("movieId")["rating"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_rating", "count": "num_ratings"})
    .merge(movies[["movieId", "title"]], on="movieId")
)

sample30 = movies.sample(30, random_state=42)["title"].tolist()

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
# Emotion Wheel Plot
# -----------------------------------------
def plot_petal_wheel(emo_choices):
    labels = list(emo_choices.keys())
    sizes = [SIZE_MAP[emo_choices[l]] for l in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(2.2, 2.2))
    for angle, lbl, size in zip(angles, labels, sizes):
        face = "white" if emo_choices[lbl] == "Diverse" else COLOR_MAP[lbl]
        edge = COLOR_MAP[lbl] if emo_choices[lbl] != "Diverse" else "gray"
        ax.bar(angle, size, width=0.7, facecolor=face, edgecolor=edge, linewidth=1.5, alpha=0.8)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7, fontweight='regular')
    ax.set_yticks([])
    fig.tight_layout()
    return fig

# -----------------------------------------
# Multi-Step Navigation State
# -----------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 0
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False

# -----------------------------------------
# Step 0 — Welcome
# -----------------------------------------
if st.session_state.step == 0:
    st.title("🎬 Welcome to the Emotion‑Based Recommender")
    st.markdown("""
    **1. Rate 10 movies (1 = dislike, 5 = like)**  
    **2. We prepare your recommendations**  
    **3. Choose # of recs & set emotional tone**  
    **4. View your list & emotion wheel**
    """)
    if st.button("Get Started"):
        st.session_state.step = 1

# -----------------------------------------
# Step 1 — Preference Elicitation
# -----------------------------------------
elif st.session_state.step == 1:
    st.header("Step 1: Preference Elicitation")
    st.write("Rate at least **10** of these 30 movies:")
    ratings_inputs = {}
    for i, title in enumerate(sample30):
        col = st.columns(3)[i % 3]
        with col:
            ratings_inputs[title] = st.slider(f"{title}", 1, 5, 3, key=f"slider_{i}")
    if st.button("Continue"):
        st.session_state.user_ratings = ratings_inputs
        st.session_state.step = 2

# -----------------------------------------
# Step 2 — Buffering Screen
# -----------------------------------------
elif st.session_state.step == 2:
    st.header("Preparing your personalized recommendations…")
    if st.button("Proceed"):
        st.session_state.step = 3

# -----------------------------------------
# Step 3 — Emotion Config + Results
# -----------------------------------------
elif st.session_state.step == 3:
    st.header("Step 3: Configure & View Recommendations")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Recommendations")
        user_id = st.number_input("User ID", value=int(ratings["userId"].min()), step=1)
        num_rec = st.slider("Recommendations", 1, 20, 7)

        st.markdown("### Select the Emotional Intensity for Each Emotion")
        emo_choices = {}
        emo_cols_row1 = st.columns(4)
        emo_cols_row2 = st.columns(4)

        for i, emo in enumerate(emotions[:4]):
            with emo_cols_row1[i]:
                emo_choices[emo] = st.radio(emo, ["Low", "Diverse", "High"], key=emo)

        for i, emo in enumerate(emotions[4:]):
            with emo_cols_row2[i]:
                emo_choices[emo] = st.radio(emo, ["Low", "Diverse", "High"], key=emo)

        st.markdown(" ")
        if st.button("🎯 Show Recommendations"):
            st.session_state.show_recs = True

    with col2:
        st.markdown("### Your Emotion Wheel")
        fig = plot_petal_wheel(emo_choices)
        st.pyplot(fig)

    if st.session_state.show_recs:
        seen = set(ratings[ratings["userId"] == user_id]["movieId"])
        cands = movie_stats[~movie_stats["movieId"].isin(seen)]
        top = (
            cands.sort_values(["avg_rating", "num_ratings"], ascending=False)
                 .head(num_rec)[["title", "avg_rating", "num_ratings"]]
        )

        st.markdown("## 🎥 Recommended Movies")
        st.dataframe(top.reset_index(drop=True), use_container_width=True)
