# 🎬 Netflix Movie Recommendation System

A **content-based movie recommendation system** built with **Python, Streamlit, and scikit-learn**, providing personalized Netflix movie and TV show suggestions.  
Users can search for a movie, see details, and get recommended similar titles. Includes data insights and interactive filters.

---

## 🔹 Features

- Search for Netflix movies and TV shows by title
- Content-based recommendations using **TF-IDF** and **cosine similarity**
- Fuzzy search suggestions for mistyped titles
- Filter by:
  - **Content type** (Movie / TV Show)
  - **Release year**
- Interactive dashboard with:
  - Top content types (bar chart)
  - Release year trends (line chart)

---

## 🛠 Tech Stack

- **Python 3.x**
- **Streamlit** for web app UI
- **Pandas** for data handling
- **scikit-learn** for TF-IDF & cosine similarity


---

## 📂 Folder Structure

netflix-recommender/
│
├─ app.py # Main Streamlit app
├─ netflix_titles.csv # Dataset
├─ requirements.txt # Python dependencies
└─ README.md # Project documentation
