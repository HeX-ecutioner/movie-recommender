# app.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# ----------------------------
# Utility functions
# ----------------------------


def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()


def best_match(results, clean_title, year=None):
    if not results:
        return None
    best, best_score = None, 0
    for r in results:
        tmdb_title = r.get("title", "")
        score = SequenceMatcher(None, clean_title.lower(), tmdb_title.lower()).ratio()
        if year and "release_date" in r and r["release_date"]:
            if str(year) in r["release_date"]:
                score += 0.1
        if score > best_score:
            best, best_score = r, score
    return best


def tmdb_search_poster(title, year, tmdb_api_key):
    try:
        clean = clean_title(title)
        params = {"api_key": tmdb_api_key, "query": clean, "include_adult": False}
        if not pd.isna(year):
            params["year"] = int(year)

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie", params=params, timeout=5
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])

        if not results:  # retry without year
            params.pop("year", None)
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie", params=params, timeout=5
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        match = best_match(results, clean, year)
        if match and match.get("poster_path"):
            url = f"https://image.tmdb.org/t/p/w300{match['poster_path']}"
            img_data = requests.get(url, timeout=5).content
            return Image.open(BytesIO(img_data))

        return Image.open("assets/no_poster.png")

    except Exception:
        return Image.open("assets/no_poster.png")


# ----------------------------
# Load MovieLens data
# ----------------------------


@st.cache_data
def download_movielens_small(dest_path="./data"):
    os.makedirs(dest_path, exist_ok=True)
    movies_path = os.path.join(dest_path, "movies.csv")
    ratings_path = os.path.join(dest_path, "ratings.csv")
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return movies_path, ratings_path

    import zipfile
    import urllib.request

    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = os.path.join(dest_path, "ml-latest-small.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("ml-latest-small/movies.csv", dest_path)
        z.extract("ml-latest-small/ratings.csv", dest_path)
        os.replace(
            os.path.join(dest_path, "ml-latest-small", "movies.csv"), movies_path
        )
        os.replace(
            os.path.join(dest_path, "ml-latest-small", "ratings.csv"), ratings_path
        )
    os.remove(zip_path)
    return movies_path, ratings_path


@st.cache_data
def load_data():
    movies_path, ratings_path = download_movielens_small()
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")[0].astype(float)
    movies["title_clean"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )
    return movies, ratings


@st.cache_data
def build_genre_matrix(movies_df):
    def split_genres(s):
        return [] if s == "(no genres listed)" else s.split("|")

    genres_list = movies_df["genres"].apply(split_genres)
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(genres_list), columns=mlb.classes_, index=movies_df.index
    )
    return genre_matrix


@st.cache_data
def compute_similarity(genre_matrix):
    return cosine_similarity(genre_matrix.fillna(0))


@st.cache_data
def aggregate_ratings(ratings_df):
    agg = ratings_df.groupby("movieId").rating.agg(["mean", "count"]).reset_index()
    agg.rename(columns={"mean": "avg_rating", "count": "num_ratings"}, inplace=True)
    return agg


def recommend_by_genre(
    movie_title,
    movies_df,
    similarity_matrix,
    top_n=10,
    min_avg_rating=None,
    rating_agg=None,
):
    titles = movies_df["title_clean"]
    exact_matches = movies_df[titles.str.lower() == movie_title.strip().lower()]
    if not exact_matches.empty:
        idx = exact_matches.index[0]
    else:
        contains = movies_df[
            titles.str.lower().str.contains(movie_title.strip().lower(), na=False)
        ]
        if contains.empty:
            return []
        idx = contains.index[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    recommendations = []
    for i, score in sim_scores:
        if i == idx:
            continue
        movieid = movies_df.loc[i, "movieId"]
        if min_avg_rating and rating_agg is not None:
            row = rating_agg[rating_agg.movieId == movieid]
            if row.empty or row.iloc[0].avg_rating < min_avg_rating:
                continue
        recommendations.append(
            (
                movies_df.loc[i, "title"],
                movies_df.loc[i, "genres"],
                float(score),
                movies_df.loc[i, "year"],
            )
        )
        if len(recommendations) >= top_n:
            break
    return recommendations


# ----------------------------
# Main App Logic
# ----------------------------

movies, ratings = load_data()
rating_agg = aggregate_ratings(ratings)
genre_matrix = build_genre_matrix(movies)
similarity_matrix = compute_similarity(genre_matrix)

# --- Data Exploration ---
st.header("📊 Data Exploration")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Movies", movies.shape[0])
with col2:
    st.metric("Total Ratings", ratings.shape[0])
with col3:
    st.metric("Unique Users", ratings["userId"].nunique())

# Top 5 most-rated movies
st.subheader("Top 5 Most-Rated Movies")
top5 = rating_agg.merge(movies[["movieId", "title"]], on="movieId")
top5 = top5.sort_values("num_ratings", ascending=False).head(5)
top5_display = top5.rename(
    columns={"num_ratings": "Number of Ratings", "avg_rating": "Average Rating"}
)
st.table(
    top5_display[["title", "Number of Ratings", "Average Rating"]].set_index("title")
)

st.markdown("---")

# --- Recommendations ---
st.header("🔍 Find Similar Movies")
movie_name = st.text_input(
    "Enter a movie you like:", placeholder="e.g. The Dark Knight"
)
min_rating = st.slider("Filter by minimum average rating", 0.0, 5.0, 3.5, 0.1)

if movie_name:
    recs = recommend_by_genre(
        movie_name,
        movies,
        similarity_matrix,
        top_n=10,
        min_avg_rating=min_rating,
        rating_agg=rating_agg,
    )

    if not recs:
        st.warning(
            "No recommendations found. Try a different movie or lower the rating filter."
        )
    else:
        display_n = min(5, len(recs))
        st.subheader(f"Top {display_n} recommendations for **{movie_name}**")

        # Poster grid
        cols = st.columns(display_n)
        for col, rec in zip(cols, recs[:display_n]):
            title, genres, score, year = rec
            with col:
                poster = tmdb_search_poster(title, year, st.secrets["tmdb"]["api_key"])
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title}**")
                st.caption(f"{genres}\n⭐ {score:.2f}")

        # Full top 10 recommendations table
        st.subheader("All Top 10 Candidates")
        df_out = pd.DataFrame(
            [{"Title": r[0], "Genres": r[1], "Similarity": r[2]} for r in recs]
        )
        st.dataframe(df_out)
