# ----------------------------------
# Netflix Movie Recommendation System
# ----------------------------------

import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load & Clean Data
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(__file__)
    file_path = os.path.join(BASE_DIR, "netflix_titles.csv")

    df = pd.read_csv(file_path)

    df['director'] = df['director'].fillna("Unknown")
    df['cast'] = df['cast'].fillna("Unknown")
    df['listed_in'] = df['listed_in'].fillna("Unknown")
    df['description'] = df['description'].fillna("")
    df['country'] = df['country'].fillna("Unknown")

    df['combined_features'] = (
        df['director'] + " " +
        df['cast'] + " " +
        df['listed_in'] + " " +
        df['description']
    )

    return df.reset_index(drop=True)

df = load_data()

# -----------------------------
# Build Recommendation Model
# -----------------------------
@st.cache_data
def build_model(features):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(features)
    return cosine_similarity(tfidf_matrix)

cosine_sim = build_model(df['combined_features'])

# -----------------------------
# Helper Functions
# -----------------------------
def recommend(title, n=5):
    idx_list = df.index[df['title'] == title].tolist()
    if not idx_list:
        return []

    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    return [
        (df.iloc[i]['title'], round(score, 3))
        for i, score in sim_scores[1:n+1]
    ]

def closest_titles(title):
    return get_close_matches(title, df['title'].tolist(), n=5, cutoff=0.6)

def show_movie_details(title):
    row = df[df['title'] == title].iloc[0]
    st.markdown(f"""
    ### 🎬 {row['title']}
    **Type:** {row['type']}  
    **Release Year:** {row['release_year']}  
    **Genre:** {row['listed_in']}  
    **Director:** {row['director']}  
    **Cast:** {row['cast']}  
    **Country:** {row['country']}  
    """)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("🔍 Filters")

content_type = st.sidebar.selectbox(
    "Content Type",
    ["All"] + sorted(df['type'].unique())
)

year_range = st.sidebar.slider(
    "Release Year",
    int(df['release_year'].min()),
    int(df['release_year'].max()),
    (2000, int(df['release_year'].max()))
)

filtered_df = df.copy()

if content_type != "All":
    filtered_df = filtered_df[filtered_df['type'] == content_type]

filtered_df = filtered_df[
    (filtered_df['release_year'] >= year_range[0]) &
    (filtered_df['release_year'] <= year_range[1])
]

# -----------------------------
# Main UI
# -----------------------------
st.title("🎬 Netflix Movie Recommendation & Insights Dashboard")

movie_name = st.text_input("Enter a movie or TV show name:")

if movie_name:
    if movie_name not in df['title'].values:
        suggestions = closest_titles(movie_name)
        if suggestions:
            st.warning("Movie not found. Did you mean:")
            for s in suggestions:
                st.write(f"- {s}")
        else:
            st.error("Movie not found in dataset.")
    else:
        show_movie_details(movie_name)

        st.subheader("✨ Recommended for You")
        recommendations = recommend(movie_name)

        for i, (rec, score) in enumerate(recommendations, 1):
            st.write(f"{i}. **{rec}** — Similarity Score: `{score}`")

# -----------------------------
# Data Insights Section
# -----------------------------
st.divider()
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Top Content Types**")
    st.bar_chart(df['type'].value_counts())

with col2:
    st.markdown("**Top Release Years**")
    st.line_chart(df['release_year'].value_counts().sort_index())


