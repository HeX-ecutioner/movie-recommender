# 🌐 Cine Expert REST API Specification

This document provides the complete technical specification for the Cine Expert REST API exposed by the FastAPI backend in [`api/index.py`](../api/index.py).

## 1. Global Specifications

### Base URLs
- **Local Development**: `http://localhost:8000`
- **Vercel Production**: `https://cine-expert.vercel.app`

### Global Middleware & Headers
- **CORS**: All endpoints support Cross-Origin Resource Sharing (`Access-Control-Allow-Origin: *`).
- **Cache-Control**: Static assets (`.css`, `.js`, `.html`) are served with `Cache-Control: no-cache, must-revalidate` to prevent stale client caching.
- **Content-Type**: All API endpoints return `application/json; charset=utf-8`.

## 2. Endpoints

### 2.1 Platform Statistics: `GET /api/stats`

Returns aggregated platform metrics and the top 5 most frequently rated movies from the MovieLens dataset.

#### Request
- **Method**: `GET`
- **Path**: `/api/stats`
- **Headers**: None
- **Query Parameters**: None

#### Response
- **Status Code**: `200 OK`
- **Content-Type**: `application/json`

```json
{
  "total_movies": 9742,
  "total_ratings": 100836,
  "unique_users": 610,
  "top5_movies": [
    {
      "title": "Forrest Gump (1994)",
      "num_ratings": 329,
      "avg_rating": 4.164133738601824
    },
    {
      "title": "The Shawshank Redemption (1994)",
      "num_ratings": 317,
      "avg_rating": 4.429022082018927
    },
    {
      "title": "Pulp Fiction (1994)",
      "num_ratings": 307,
      "avg_rating": 4.197068403908795
    },
    {
      "title": "The Silence of the Lambs (1991)",
      "num_ratings": 279,
      "avg_rating": 4.161290322580645
    },
    {
      "title": "The Matrix (1999)",
      "num_ratings": 278,
      "avg_rating": 4.192446043165468
    }
  ]
}
```

### 2.2 Movie Recommendations: `GET /api/recommend`

Generates hybrid recommendations for a movie query, balancing semantic content similarity and collaborative filtering patterns.

#### Request Parameters

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `movie` | `string` | Cond.* | `None` | Movie title to query (e.g. `"The Dark Knight (2008)"`). |
| `title` | `string` | Cond.* | `None` | Alternative alias for `movie`. |
| `content_weight` | `float` | No | `0.5` | Weight for content similarity ($w \in [0.0, 1.0]$). $1.0$ = content-only, $0.0$ = collaborative-only. |
| `min_rating` | `float` | No | `3.5` | Minimum Bayesian weighted rating quality threshold. |
| `top_n` | `integer` | No | `10` | Maximum number of recommendations to return. |

*\*Note: At least one of `movie` or `title` must be provided.*

#### Response Scenarios

#### Scenario A: Query Successfully Resolved (`200 OK`)
When the query is matched to a single movie, the API returns the ranked recommendations list:

```json
{
  "recommendations": [
    {
      "title": "Batman Begins (2005)",
      "genres": [
        "Action",
        "Crime",
        "IMAX"
      ],
      "score": 0.8124,
      "year": 2005,
      "poster_url": "https://image.tmdb.org/t/p/w300/dr6x4GyyegBWtinPBzipY02te2F.jpg"
    },
    {
      "title": "The Dark Knight Rises (2012)",
      "genres": [
        "Action",
        "Adventure",
        "Crime",
        "IMAX"
      ],
      "score": 0.7681,
      "year": 2012,
      "poster_url": "https://image.tmdb.org/t/p/w300/hr0L2aueqlP2BYUblTTjmtn0hw4.jpg"
    }
  ]
}
```

#### Field Specifications:
- `title` (`string`): Natural formatted title with release year (e.g., `"The Dark Knight Rises (2012)"`).
- `genres` (`list[string]`): List of genre strings.
- `score` (`float`): Normalized hybrid match score in range `[0.0, 1.0]`.
- `year` (`integer` or `null`): 4-digit release year.
- `poster_url` (`string`): Public CDN URL of the TMDB poster (width 300px), or empty string `""` if unavailable.

#### Scenario B: Ambiguous Query (`200 OK`)
When a query matches multiple distinct movies across different release years or multiple fuzzy matches within a 3.0-point score margin:

```json
{
  "recommendations": [],
  "status": "ambiguous",
  "query": "Avengers",
  "message": "Multiple movies matched \u201cAvengers\u201d. Please include a year, such as \u201cThe Avengers (2012)\u201d.",
  "matches": [
    {
      "title": "The Avengers (2012)",
      "year": 2012
    },
    {
      "title": "The Avengers (1998)",
      "year": 1998
    }
  ]
}
```

#### Field Specifications:
- `status` (`string`): Literal value `"ambiguous"`.
- `query` (`string`): The original query entered by the user.
- `message` (`string`): Helpful suggestion prompting the user to specify a year, featuring the most popular candidate.
- `matches` (`list[object]`): List of plausible candidates, each containing natural `title` and `year`.

#### Scenario C: No Match Found (`200 OK`)
If no movie in the dataset matches the query:

```json
{
  "recommendations": []
}
```

#### Scenario D: Missing Query Parameter (`422 Unprocessable Entity`)
If neither `movie` nor `title` is supplied in the request:

```json
{
  "detail": "Either 'movie' or 'title' query parameter is required."
}
```

## 3. External TMDB Poster Integration

The backend interacts with The Movie Database (TMDB) API via [`src/utils.py`](file:///c:/Users/Sagnik/Documents/GitHub%20repos/cine-expert/src/utils.py):
- **Endpoint**: `https://api.themoviedb.org/3/search/movie`
- **Authentication**: `TMDB_API_KEY` loaded securely from `.env` via `python-dotenv`.
- **In-Memory Caching**: Responses are cached using Python's `@lru_cache(maxsize=500)` to optimize latency and prevent TMDB rate-limiting.
- **Graceful Fallback**: If an image is not found, the network times out (5-second timeout), or no API key is configured, the endpoint safely returns an empty string `""`, allowing the frontend to render a fallback placeholder card.
