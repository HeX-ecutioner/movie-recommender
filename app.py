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

st.set_page_config(page_title="Movie Recommender System", page_icon="🎬", layout="wide")


# Utility functions
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""


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
            if str(int(year)) in r["release_date"]:
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
        pass  # Silently fail and use placeholder

    # Fallback to an empty grey placeholder if no image exists
    img = Image.new("RGB", (300, 450), color=(73, 109, 137))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


# --- HYBRID RECOMENDER CORE LOGIC ---


# 1. Content-Based Matrix (Genres)
@st.cache_data(show_spinner=False)
def compute_content_similarity(movies_df):
    def split_genres(s):
        return [] if s == "(no genres listed)" else s.split("|")

    genres_list = movies_df["genres"].apply(split_genres)
    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(genres_list), columns=mlb.classes_, index=movies_df.index
    )
    return cosine_similarity(genre_matrix.fillna(0).astype(np.float32))


# 2. Collaborative Matrix (User Ratings)
@st.cache_data(show_spinner=False)
def compute_collaborative_similarity(movies_df, ratings_df):
    item_user_matrix = ratings_df.pivot(
        index="movieId", columns="userId", values="rating"
    )

    # Align with movies_df order
    aligned_matrix = item_user_matrix.reindex(movies_df["movieId"])

    # Normalize ratings (subtract mean per movie)
    normalized = aligned_matrix.sub(aligned_matrix.mean(axis=1), axis=0)

    # Fill NaN after normalization
    normalized = normalized.fillna(0)

    return cosine_similarity(normalized.astype(np.float32))


@st.cache_data(show_spinner=False)
def aggregate_ratings(ratings_df):
    agg = ratings_df.groupby("movieId").rating.agg(["mean", "count"]).reset_index()
    agg.rename(columns={"mean": "avg_rating", "count": "num_ratings"}, inplace=True)

    # Compute global mean
    C = agg["avg_rating"].mean()
    m = agg["num_ratings"].quantile(0.6)

    # Weighted rating (IMDb formula)
    agg["weighted_rating"] = (agg["num_ratings"] / (agg["num_ratings"] + m)) * agg[
        "avg_rating"
    ] + (m / (agg["num_ratings"] + m)) * C

    return agg


# 3. Hybrid Engine Combines Both Matrices
def recommend_hybrid(
    movie_title,
    movies_df,
    content_sim_matrix,
    cf_sim_matrix,
    content_weight=0.5,
    top_n=10,
    min_avg_rating=None,
    rating_agg=None,
):
    titles = movies_df["title_clean"]
    exact_matches = movies_df[titles.str.lower() == movie_title.strip().lower()]

    if not exact_matches.empty:
        idx = exact_matches.index[0]

    else:
        # Step 2: Partial match
        contains = movies_df[
            titles.str.lower().str.contains(movie_title.strip().lower(), na=False)
        ].copy()

        if contains.shape[0] == 0:
            return []

        # Step 3: Rank by similarity
        contains["match_score"] = contains["title_clean"].apply(
            lambda x: SequenceMatcher(None, movie_title.lower(), x.lower()).ratio()
        )

        idx = contains.sort_values("match_score", ascending=False).index[0]

    # Combine scores dynamically based on the slider weight
    cf_weight = 1.0 - content_weight
    hybrid_scores = (content_weight * content_sim_matrix[idx]) + (
        cf_weight * cf_sim_matrix[idx]
    )

    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores:
        if i == idx:
            continue
        movieid = movies_df.loc[i, "movieId"]

        # Rating Filter Guard
        if min_avg_rating and rating_agg is not None:
            row = rating_agg[rating_agg.movieId == movieid]
            if row.empty or row.iloc[0].weighted_rating < min_avg_rating:
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

# --- UI LAYOUT ---

st.markdown(
    f"""
    <div style="display:flex; align-items:center; justify-content:space-between;">
        <h1 style="margin:0;">🎬 Movie Recommender System</h1>
        <img src="data:image/png;base64,{get_base64_of_bin_file('assets/icon.jpg')}" 
             width="160" style="margin-left:20px;border-radius:10px;">
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

# Load Data and Models
with st.spinner("Loading Recommendation Engine..."):
    movies, ratings = load_data()
    rating_agg = aggregate_ratings(ratings)
    content_sim = compute_content_similarity(movies)
    cf_sim = compute_collaborative_similarity(movies, ratings)

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

table_html = "<table style='border-collapse: collapse; width: 100%;'>"
table_html += "<tr style='background-color:#1e1e1e; text-align:left;'><th>Title of Movie</th><th>Number of Ratings</th><th>Average Rating</th></tr>"
for _, row in top5.iterrows():
    table_html += f"<tr><td>{row['title']}</td><td>{row['num_ratings']}</td><td>{row['Average Rating']}</td></tr>"
table_html += "</table>"
st.markdown(
    f"<div style='overflow-x:auto;'>{table_html}</div><br>", unsafe_allow_html=True
)
st.markdown("---")


# Recommendations UI
st.header("🔍 Find Similar Movies")

col_input, col_filters = st.columns([2, 1])

with col_input:
    movie_name = st.text_input(
        "Enter a movie you like:", placeholder="e.g. The Dark Knight"
    )

with col_filters:
    min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    content_weight = st.slider("Content vs Collaborative", 0.0, 1.0, 0.5, 0.05)

if movie_name:
    # Notice we don't pass content_weight anymore, so it uses the 0.5 (balanced) default
    recs = recommend_hybrid(
        movie_title=movie_name,
        movies_df=movies,
        content_sim_matrix=content_sim,
        cf_sim_matrix=cf_sim,
        content_weight=content_weight,
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
                # Add your TMDB API Key into `.streamlit/secrets.toml` as [tmdb] api_key="YOUR_KEY"
                # For safety, defaulting to "" if it's missing so the app doesn't crash
                api_key = st.secrets["tmdb"]["api_key"] if "tmdb" in st.secrets else ""

                poster_bytes = fetch_poster_bytes(title, year, api_key)
                st.image(get_image_from_bytes(poster_bytes), use_container_width=True)

                st.markdown(f"**{title}**")
                st.caption(f"{' | '.join(genres.split('|'))}")
                movieid = movies.loc[movies["title"] == title, "movieId"].values[0]
                row = rating_agg[rating_agg.movieId == movieid]
                actual_rating = row.iloc[0].avg_rating if not row.empty else 0

                st.markdown(
                    f"<small>{render_stars(actual_rating)}<br><b>Match: {score*100:.1f}%</b></small>",
                    unsafe_allow_html=True,
                )

        # Full top 10 table
        st.markdown("<br><b>Top 10 Detailed View</b>", unsafe_allow_html=True)
        df_out = pd.DataFrame(
            [
                {
                    "Rank": i + 1,
                    "Title": r[0],
                    "Genres": " | ".join(r[1].split("|")),
                    "Match (%)": f"{r[2]*100:.2f}%",
                }
                for i, r in enumerate(recs)
            ]
        )
        st.dataframe(df_out.set_index("Rank"), use_container_width=True)

# Footer
st.markdown(
    """<hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:right; color:gray; font-size:14px;">🎬 Movie Recommender System • Built with Python & Streamlit</div>""",
    unsafe_allow_html=True,
)
