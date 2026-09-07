import re
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import process, fuzz

    USE_RAPIDFUZZ = True
except:
    USE_RAPIDFUZZ = False

from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import fix_title_display


def compute_content_features(movies_df):
    # Initialize TF-IDF, removing common English stop words (the, and, is)
    tfidf = TfidfVectorizer(stop_words="english")
    # Fit the vectorizer on our new metadata soup
    tfidf_matrix = tfidf.fit_transform(movies_df["metadata"])
    # Return as an array for the cosine_similarity function
    return tfidf_matrix.toarray().astype(np.float32)


def compute_collaborative_features(movies_df, ratings_df):
    item_user_matrix = ratings_df.pivot(
        index="movieId", columns="userId", values="rating"
    )
    aligned_matrix = item_user_matrix.reindex(movies_df["movieId"])
    normalized = aligned_matrix.sub(aligned_matrix.mean(axis=1), axis=0)
    normalized = normalized.fillna(0)
    return normalized.astype(np.float32).values


def aggregate_ratings(ratings_df):
    agg = ratings_df.groupby("movieId").rating.agg(["mean", "count"]).reset_index()
    agg.rename(columns={"mean": "avg_rating", "count": "num_ratings"}, inplace=True)

    C = agg["avg_rating"].mean()
    m = agg["num_ratings"].quantile(0.6)

    agg["weighted_rating"] = (agg["num_ratings"] / (agg["num_ratings"] + m)) * agg[
        "avg_rating"
    ] + (m / (agg["num_ratings"] + m)) * C
    return agg


class AmbiguityResult:
    """Represents an explicit ambiguity when multiple candidates are similarly plausible."""

    def __init__(self, query, candidates, method="Ambiguous match"):
        self.query = query
        self.candidates = candidates
        self.method = method
        self.is_ambiguous = True
        self.is_resolved = False
        self.index = None
        self.movieId = None
        self.title = None

    def __bool__(self):
        return False

    def __repr__(self):
        candidate_titles = [c.get("title", "") for c in self.candidates]
        return f"<AmbiguityResult query='{self.query}' method='{self.method}' candidates={candidate_titles}>"

    def to_dict(self):
        return {
            "query": self.query,
            "status": "ambiguous",
            "method": self.method,
            "candidates": self.candidates,
        }


class TitleResolutionResult:
    """Represents the complete result of a title resolution attempt."""

    def __init__(
        self,
        query,
        status,
        index=None,
        title=None,
        movie_id=None,
        year=None,
        method=None,
        year_constraint_respected=True,
        fuzzy_score=None,
        candidates=None,
    ):
        self.query = query
        self.status = status
        self.index = index
        self.title = title
        self.movie_id = movie_id
        self.year = year
        self.method = method
        self.year_constraint_respected = year_constraint_respected
        self.fuzzy_score = fuzzy_score
        self.candidates = candidates or []
        self.is_resolved = status == "resolved"
        self.is_ambiguous = status == "ambiguous"

    def __bool__(self):
        return self.is_resolved

    def __repr__(self):
        if self.is_resolved:
            return (
                f"<TitleResolutionResult status='resolved' title='{self.title}' "
                f"movieId={self.movie_id} index={self.index} method='{self.method}'>"
            )
        elif self.is_ambiguous:
            return f"<TitleResolutionResult status='ambiguous' method='{self.method}' candidates={len(self.candidates)}>"
        return f"<TitleResolutionResult status='not_found' query='{self.query}'>"


def parse_query_title_and_year(query):
    if not isinstance(query, str):
        return "", None
    q = query.strip()
    # Match trailing (YYYY) e.g. "Batman Begins (2005)"
    m = re.search(r"^(.*?)\s*\((\d{4})\)$", q)
    if m and m.group(1).strip():
        return m.group(1).strip(), int(m.group(2))
    # Match trailing YYYY without parens e.g. "The Avengers 2012"
    m = re.search(r"^(.*?)\s+(\d{4})$", q)
    if m and m.group(1).strip():
        year = int(m.group(2))
        if 1880 <= year <= 2035:
            return m.group(1).strip(), year
    return q, None


def canonical_title_norm(title):
    if not title:
        return ""
    t = fix_title_display(str(title)).lower().strip()
    if t.startswith("the "):
        t = t[4:]
    elif t.startswith("a "):
        t = t[2:]
    elif t.startswith("an "):
        t = t[3:]
    return re.sub(r"[^a-z0-9]", "", t)


def direct_title_norm(title):
    if not title:
        return ""
    t = fix_title_display(str(title)).lower().strip()
    return re.sub(r"[^a-z0-9]", "", t)


def resolve_movie_title(movie_title, movies_df):
    if "canonical_norm" not in movies_df.columns:
        movies_df["canonical_norm"] = movies_df["title_clean"].apply(
            canonical_title_norm
        )
    if "direct_norm" not in movies_df.columns:
        movies_df["direct_norm"] = movies_df["title_clean"].apply(direct_title_norm)

    raw_query = str(movie_title).strip()
    q_title, q_year = parse_query_title_and_year(raw_query)
    q_direct = direct_title_norm(q_title)
    q_canonical = canonical_title_norm(q_title)

    # 1. Exact direct match (preserves articles, e.g. "The Dark Knight")
    exact_direct = movies_df[movies_df["direct_norm"] == q_direct]
    # 2. Exact canonical match (article-insensitive, e.g. "Dark Knight" -> "The Dark Knight")
    exact_canon = movies_df[movies_df["canonical_norm"] == q_canonical]

    exact = exact_direct if not exact_direct.empty else exact_canon

    if not exact.empty:
        if q_year is not None:
            yr_match = exact[exact["year"] == float(q_year)]
            if len(yr_match) >= 1:
                row = yr_match.iloc[0]
                return TitleResolutionResult(
                    query=raw_query,
                    status="resolved",
                    index=yr_match.index[0],
                    title=row["title"],
                    movie_id=int(row["movieId"]),
                    year=row["year"],
                    method="Exact normalized title + year match",
                    year_constraint_respected=True,
                    fuzzy_score=None,
                )
            # An explicit year was supplied, but exact title matches have a different year.
            # Fall through to fuzzy matching constrained to q_year.
        else:
            distinct_years = exact["year"].dropna().unique()
            if len(distinct_years) <= 1:
                row = exact.iloc[0]
                method_name = (
                    "Exact direct title match"
                    if not exact_direct.empty
                    else "Exact canonical title match (article-insensitive)"
                )
                return TitleResolutionResult(
                    query=raw_query,
                    status="resolved",
                    index=exact.index[0],
                    title=row["title"],
                    movie_id=int(row["movieId"]),
                    year=row["year"],
                    method=method_name,
                    year_constraint_respected=True,
                    fuzzy_score=None,
                )
            else:
                # Multiple distinct movies share this exact title across different years
                candidates = [
                    {
                        "index": int(i),
                        "movieId": int(r_row["movieId"]),
                        "title": str(r_row["title"]),
                        "year": (
                            int(r_row["year"]) if not pd.isna(r_row["year"]) else None
                        ),
                    }
                    for i, r_row in exact.drop_duplicates(subset=["title"]).iterrows()
                ]
                return TitleResolutionResult(
                    query=raw_query,
                    status="ambiguous",
                    method="Ambiguous exact title matches across multiple release years",
                    year_constraint_respected=True,
                    candidates=candidates,
                )

    # 3. Fuzzy matching (used only when exact matching fails)
    pool = (
        movies_df[movies_df["year"] == float(q_year)]
        if q_year is not None
        else movies_df
    )
    if pool.empty:
        return TitleResolutionResult(
            query=raw_query,
            status="not_found",
            method="No title found matching requested year",
            year_constraint_respected=True,
        )

    scores = []
    q_lower = q_title.lower()
    for idx, row in pool.iterrows():
        cand_search = str(row["title_search"])
        cand_clean = str(row["title_clean"]).lower()
        if USE_RAPIDFUZZ:
            s1 = fuzz.WRatio(q_lower, cand_search)
            s2 = fuzz.WRatio(q_lower, cand_clean)
            sc = max(s1, s2)
        else:
            s1 = SequenceMatcher(None, q_lower, cand_search).ratio() * 100.0
            s2 = SequenceMatcher(None, q_lower, cand_clean).ratio() * 100.0
            sc = max(s1, s2)
        if sc >= 70.0:
            scores.append((idx, sc, row))

    if not scores:
        return TitleResolutionResult(
            query=raw_query,
            status="not_found",
            method="No match above fuzzy threshold",
            year_constraint_respected=(q_year is not None),
        )

    scores.sort(key=lambda x: x[1], reverse=True)
    best_score = scores[0][1]
    top_candidates = [s for s in scores if s[1] >= best_score - 3.0]

    unique_titles = set(s[2]["title"] for s in top_candidates)
    if len(unique_titles) > 1:
        candidates = [
            {
                "index": int(s[0]),
                "movieId": int(s[2]["movieId"]),
                "title": str(s[2]["title"]),
                "year": int(s[2]["year"]) if not pd.isna(s[2]["year"]) else None,
                "fuzzy_score": float(s[1]),
            }
            for s in top_candidates
        ]
        return TitleResolutionResult(
            query=raw_query,
            status="ambiguous",
            method=f"Ambiguous fuzzy matches (score: {best_score:.1f})",
            year_constraint_respected=(q_year is not None),
            candidates=candidates,
        )

    best = scores[0]
    return TitleResolutionResult(
        query=raw_query,
        status="resolved",
        index=best[0],
        title=best[2]["title"],
        movie_id=int(best[2]["movieId"]),
        year=best[2]["year"],
        method=(
            f"Fuzzy match WRatio ({best[1]:.1f})"
            if USE_RAPIDFUZZ
            else f"SequenceMatcher ({best[1]:.1f})"
        ),
        year_constraint_respected=(q_year is not None),
        fuzzy_score=float(best[1]),
    )


def find_movie_index(movie_title, movies_df, return_details=False):
    resolution = resolve_movie_title(movie_title, movies_df)
    if return_details:
        return resolution
    if resolution.status == "resolved":
        return resolution.index
    elif resolution.status == "ambiguous":
        return AmbiguityResult(
            query=resolution.query,
            candidates=resolution.candidates,
            method=resolution.method,
        )
    return None


def recommend_hybrid(
    movie_title,
    movies_df,
    content_features,
    cf_features,
    content_weight=0.5,
    top_n=5,
    min_avg_rating=None,
    rating_agg=None,
):

    resolution = find_movie_index(movie_title, movies_df, return_details=True)
    if not resolution.is_resolved:
        return []
    idx = resolution.index

    content_scores = cosine_similarity(
        content_features[idx : idx + 1], content_features
    ).ravel()
    cf_scores = cosine_similarity(cf_features[idx : idx + 1], cf_features).ravel()

    hybrid_scores = (content_weight * content_scores) + (
        (1 - content_weight) * cf_scores
    )

    sim_scores = list(enumerate(hybrid_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in sim_scores:
        if i == idx:
            continue

        movieid = movies_df.iloc[i]["movieId"]

        if min_avg_rating is not None and rating_agg is not None:
            row = rating_agg[rating_agg.movieId == movieid]
            if row.empty or row.iloc[0].weighted_rating < min_avg_rating:
                continue

        recommendations.append(
            (
                movies_df.iloc[i]["title"],
                movies_df.iloc[i]["genres"],
                float(score),
                movies_df.iloc[i]["year"],
            )
        )

        if len(recommendations) >= top_n:
            break

    return recommendations
