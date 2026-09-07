# Changelog

All notable changes to the **Cine Expert** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-09-08

### Added
- **Tiered Title Resolution Pipeline**:
  - Implemented multi-stage matching in `src/engine.py` covering exact direct matches, canonical article-insensitive matches, and year-constrained matching (`parse_query_title_and_year`).
  - Added fuzzy matching fallback using `rapidfuzz` / `difflib` with a threshold of $\ge 70.0$ and a $3.0$-point candidate evaluation window.
- **Structured Ambiguity Handling**:
  - Added `status: "ambiguous"` JSON response contract for multi-year title collisions (e.g. "Avengers" matching 1998 and 2012) and close fuzzy ties.
  - Implemented popularity-based sorting for ambiguous candidates to surface the most relevant movie in user suggestions.
- **Interactive Ambiguity UI**:
  - Added ambiguity banner (`.ambiguity-banner`) with dynamic clickable suggestion chips (`.suggestion-chip`) that automatically execute targeted searches.
- **Hybrid Score Normalization**:
  - Implemented `_normalize_component_scores()` in `src/engine.py` with a stable reference ceiling ($CF_{\text{ref}} = 0.50$).
  - Added zero-clipping for negative collaborative correlations to prevent inverse rating penalization.
  - Added explicit NaN, infinity, and query self-similarity suppression.
- **Natural Title Display Formatting**:
  - Implemented `fix_title_display()` in `src/utils.py` to convert inverted dataset titles (e.g. "Avengers, The (2012)" $\to$ "The Avengers (2012)") across all API responses and UI components while preserving dataset indexing integrity.
- **Static Asset Cache Management**:
  - Added FastAPI HTTP middleware sending `Cache-Control: no-cache, must-revalidate` for `.css`, `.js`, and `.html` assets.
  - Added cache-busting query versioning (`styles.css?v=1.1.0`) in `public/index.html`.
- **Comprehensive Documentation Suite**:
  - Created modular documentation inside `docs/` covering System Architecture, Recommendation Engine, Title Resolution, REST API, and Frontend Design Tokens.
- **Automated Test Suites**:
  - Added 29 unit and integration tests across `tests/test_title_resolution.py`, `tests/test_ambiguity.py`, and `tests/test_hybrid_scoring.py`.

### Changed
- Refactored `recommend_hybrid()` to use normalized component scores and ensure strict $[0.0, 1.0]$ ranking stability across content-only, collaborative-only, and blended modes.
- Updated root `README.md` with links to documentation and project specifications.

## [1.0.0] - 2025-05-15

### Added
- Initial release of **Cine Expert** hybrid movie recommender system.
- FastAPI backend serving REST API and static frontend assets.
- Content-based filtering using TF-IDF vectorization across movie genres and user tags ("metadata soup").
- Mean-centered item-user collaborative filtering matrix based on the MovieLens Small dataset.
- IMDb-style Bayesian weighted rating filter with dynamic 60th-percentile minimum rating threshold.
- TMDB API integration for dynamic movie poster fetching with LRU caching.
- Vanilla HTML5/CSS3/JavaScript single-page application featuring glassmorphism, responsive grid, dynamic rating & weight sliders, and dark/light mode toggle.
- Vercel Serverless deployment configuration via `@vercel/python` and `@vercel/static`.
