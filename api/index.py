import os
import math
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.data_loader import load_data
from src.engine import (
    compute_content_features,
    compute_collaborative_features,
    aggregate_ratings,
    recommend_hybrid,
    resolve_movie_title,
)
from src.utils import fetch_poster_url, get_tmdb_api_key, fix_title_display

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files will be mounted at the end of the file to allow API routes to take precedence

# Load data at startup
movies, ratings = load_data()
rating_agg = aggregate_ratings(ratings)
# To save memory on Vercel free tier, we only take the top 5000 movies
# Or we can just compute it for all if it's small enough (movies is small).
content_features = compute_content_features(movies)
cf_features = compute_collaborative_features(movies, ratings)

tmdb_api_key = get_tmdb_api_key()

class RecommendRequest(BaseModel):
    movie: str
    content_weight: float = 0.5
    min_rating: float = 3.5
    top_n: int = 10

@app.get("/api/stats")
def get_stats():
    total_movies = int(movies.shape[0])
    total_ratings = int(ratings.shape[0])
    unique_users = int(ratings["userId"].nunique())
    
    top5_df = rating_agg.merge(movies[["movieId", "title"]], on="movieId")
    top5_df = top5_df.sort_values("num_ratings", ascending=False).head(5)
    
    top5 = []
    for _, row in top5_df.iterrows():
        top5.append({
            "title": fix_title_display(row["title"]),
            "num_ratings": int(row["num_ratings"]),
            "avg_rating": float(row["avg_rating"])
        })
        
    return {
        "total_movies": total_movies,
        "total_ratings": total_ratings,
        "unique_users": unique_users,
        "top5_movies": top5
    }

@app.get("/api/recommend")
def get_recommendations(movie: str, content_weight: float = 0.5, min_rating: float = 3.5, top_n: int = 10):
    try:
        resolution = resolve_movie_title(movie, movies)
        if resolution.is_ambiguous:
            candidates = resolution.candidates
            # Rank candidates by popularity using rating_agg if available
            if rating_agg is not None and not rating_agg.empty:
                counts = dict(zip(rating_agg["movieId"], rating_agg["num_ratings"]))
                sorted_candidates = sorted(
                    candidates,
                    key=lambda c: counts.get(c.get("movieId"), 0),
                    reverse=True,
                )
            else:
                sorted_candidates = candidates

            # Select prominent candidate for the message example
            best_cand = sorted_candidates[0] if sorted_candidates else None
            if best_cand:
                ex_title = fix_title_display(best_cand["title"])
                if best_cand.get("year") and str(best_cand["year"]) not in ex_title:
                    ex_title = f"{ex_title} ({best_cand['year']})"
            else:
                ex_title = movie

            message = (
                f"Multiple movies matched \u201c{movie}\u201d. "
                f"Please include a year, such as \u201c{ex_title}\u201d."
            )

            matches = [
                {
                    "title": c["title"],
                    "year": int(c["year"]) if c.get("year") is not None else None,
                }
                for c in candidates
            ]

            return {
                "recommendations": [],
                "status": "ambiguous",
                "query": movie,
                "message": message,
                "matches": matches,
            }

        recs = recommend_hybrid(
            movie,
            movies,
            content_features,
            cf_features,
            content_weight,
            top_n,
            min_rating,
            rating_agg,
        )
        
        result = []
        for title, genres, score, year in recs:
            poster_url = fetch_poster_url(title, year, tmdb_api_key)
            result.append({
                "title": fix_title_display(title),
                "genres": genres.split("|") if isinstance(genres, str) else genres,
                "score": float(score),
                "year": int(year) if not math.isnan(year) else None,
                "poster_url": poster_url
            })
            
        return {"recommendations": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the public directory at the root / so that index.html and styles.css are served correctly locally
public_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")
