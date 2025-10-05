import os
import re
import base64
import zipfile
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import urllib.request
from PIL import Image
from io import BytesIO
from difflib import SequenceMatcher
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommender System", page_icon="assets/icon.jpg", layout="wide"
)

# Utility functions


def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


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


def render_stars(rating: float) -> str:
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + ("⯨" if half else "") + "☆" * empty


# TMDb Poster Fetching (Cloud-safe)
@st.cache_data(show_spinner=False)
def fetch_poster_bytes(title, year, tmdb_api_key):
    """Fetch poster from TMDb and return bytes. Returns placeholder if failed."""
    try:
        clean = clean_title(title)
        params = {"api_key": tmdb_api_key, "query": clean, "include_adult": False}
        if year and not pd.isna(year):
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
            return img_data

    except Exception as e:
        st.warning(f"TMDb fetch failed for '{title}': {e}")

    placeholder_path = os.path.join("assets", "no_poster.png")  # fallback placeholder
    with open(placeholder_path, "rb") as f:
        return f.read()


def get_image_from_bytes(img_bytes):
    return Image.open(BytesIO(img_bytes))


# MovieLens Data Handling
@st.cache_data(show_spinner=False)
def download_movielens_small():
    dest_path = tempfile.gettempdir()
    movies_path = os.path.join(dest_path, "movies.csv")
    ratings_path = os.path.join(dest_path, "ratings.csv")
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return movies_path, ratings_path

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


@st.cache_data(show_spinner=False)
def load_data():
    movies_path, ratings_path = download_movielens_small()
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")[0].astype(float)
    movies["title_clean"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)$", "", regex=True
    )
    return movies, ratings


@st.cache_data(show_spinner=False)
def build_genre_matrix(movies_df):
    def split_genres(s):
        return [] if s == "(no genres listed)" else s.split("|")

    genres_list = movies_df["genres"].apply(split_genres)
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(genres_list), columns=mlb.classes_, index=movies_df.index
    )
    return genre_matrix


@st.cache_data(show_spinner=False)
def compute_similarity(genre_matrix):
    return cosine_similarity(
        genre_matrix.fillna(0).astype(np.float32)
    )  # Use float32 to reduce memory usage


@st.cache_data(show_spinner=False)
def aggregate_ratings(ratings_df):
    agg = ratings_df.groupby("movieId").rating.agg(["mean", "count"]).reset_index()
    agg.rename(columns={"mean": "avg_rating", "count": "num_ratings"}, inplace=True)
    return agg


# Recommendation Engine
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


# Header with icon
st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:space-between;">
        <h1 style="margin:0;">🎬 Movie Recommender System</h1>
        <img src="data:image/png;base64,{get_base64_of_bin_file('assets/icon.jpg')}" width="160" style="margin-left:20px;">
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
movies, ratings = load_data()
rating_agg = aggregate_ratings(ratings)
genre_matrix = build_genre_matrix(movies)
similarity_matrix = compute_similarity(genre_matrix)

st.header("📊 Data Exploration")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"<p style='font-size:18px;'>Total Movies</p><p style='font-size:28px; font-weight:bold;'>{movies.shape[0]}</p>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<p style='font-size:18px;'>Total Ratings</p><p style='font-size:28px; font-weight:bold;'>{ratings.shape[0]}</p>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<p style='font-size:18px;'>Unique Users</p><p style='font-size:28px; font-weight:bold;'>{ratings['userId'].nunique()}</p>",
        unsafe_allow_html=True,
    )

# Top 5 most-rated movies
st.subheader("🔝 Top 5 Most-Rated Movies")
top5 = rating_agg.merge(movies[["movieId", "title"]], on="movieId")
top5 = top5.sort_values("num_ratings", ascending=False).head(5)
top5["Average Rating"] = top5["avg_rating"].apply(render_stars)

table_html = "<table style='border-collapse: collapse; width: auto;'>"
table_html += (
    "<tr><th>Title of Movie</th><th>Number of Ratings</th><th>Average Rating</th></tr>"
)
for _, row in top5.iterrows():
    table_html += f"<tr><td>{row['title']}</td><td>{row['num_ratings']}</td><td>{row['Average Rating']}</td></tr>"
table_html += "</table>"
st.markdown(f"<div style='overflow-x:auto;'>{table_html}</div>", unsafe_allow_html=True)
st.markdown("---")

# Recommendations
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

        cols = st.columns(display_n)
        for col, rec in zip(cols, recs[:display_n]):
            title, genres, score, year = rec
            with col:
                poster_bytes = fetch_poster_bytes(
                    title, year, st.secrets["tmdb"]["api_key"]
                )
                st.image(get_image_from_bytes(poster_bytes), use_container_width=True)
                st.markdown(f"**{title}**")
                st.markdown(f"{' | '.join(genres.split('|'))}", unsafe_allow_html=True)
                st.markdown(
                    f"{render_stars(min_rating)} • Similarity: {score*100:.2f}%",
                    unsafe_allow_html=True,
                )

        # Full top 10 table
        df_out = pd.DataFrame(
            [
                {
                    "Sl. no.": i + 1,
                    "Title": r[0],
                    "Genres": " | ".join(r[1].split("|")),
                    "Similarity (%)": f"{r[2]*100:.2f}%",
                    "Rating": render_stars(min_rating),
                }
                for i, r in enumerate(recs)
            ]
        )
        st.dataframe(df_out.set_index("Sl. no."))

# Footer
st.markdown(
    """<hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:right; color:gray; font-size:14px;">🎬 Movie Recommender System • Built with Python using Streamlit</div>""",
    unsafe_allow_html=True,
)
