# 🎬 Cine Expert

A high-performance, decoupled **hybrid movie recommender system** featuring a **FastAPI backend** and a visually stunning **Vanilla JS + CSS frontend**. Enter a movie you like, and the engine recommends similar movies along with their dynamic posters fetched via the **TMDB API**.

## ✨ Features

- ⚡ **FastAPI Backend** — Lightweight, high-performance API perfectly suited for Vercel Serverless Functions.
- 🎨 **Glorious UI** — Beautifully crafted, responsive interface featuring intense glassmorphism, fluid micro-animations, and a sleek Light/Dark mode slider toggle.
- 🎥 **MovieLens Small Dataset** integration (movies, ratings, and tags).
- 🤖 **Hybrid Recommendation System** (Content-based + Collaborative Filtering).
- 🧠 **Metadata Soup (TF-IDF)** — Advanced content-filtering utilizing TF-IDF vectorization across movie genres, title tokens, and deep thematic tags.
- ⭐ **Weighted Rating System (IMDb-style)** for high-quality ranking control.
- 🎚️ **Adjustable Hybrid Slider** to dynamically balance content vs. collaborative influences on the fly.
- 🖼️ **Poster Displays** dynamically requested and optimized through the TMDB API.

## 🧠 Recommendation Approach

This application utilizes a powerful **hybrid recommendation engine**:

- **Content-based filtering (TF-IDF)** → Generates recommendations based on the semantic metadata of the movie (genres, tags, and title semantics).
- **Collaborative filtering** → Recommends movies liked by users with similar historical rating patterns.
- **Hybrid scoring** → Intelligently merges both paradigms using a dynamic weighted formula:

`Final Score = (w × Content Similarity) + ((1 - w) × Collaborative Similarity)`

Users can seamlessly adjust this balance (`w`) using the slider in the UI.

## ⚙️ How It Works

1. Loads the complete MovieLens dataset (automatically downloading it if necessary).
2. Builds a **TF-IDF content matrix** from the rich metadata soup.
3. Builds a **collaborative filtering matrix** based on user interactions.
4. Computes semantic proximity using **cosine similarity**.
5. Fuses the content and collaborative data points using a **weighted hybrid score**.
6. Applies an **IMDb-style weighted rating filter** for stringent quality control.
7. Fetches TMDB movie posters securely via backend environment variables.
8. Frontend flawlessly renders:
    - 🎬 Top 5 recommendations (poster grid)
    - 📊 Top 10 recommendations (detailed analytical table)

## 🚀 Getting Started

### 🔧 Prerequisites

- Python 3.8+  
- TMDB API Key ([Get one here](https://www.themoviedb.org/))  

### 🛠️ Installation

1. Clone this repository:

```bash
git clone https://github.com/HeX-ecutioner/movie-recommender.git
cd movie-recommender
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Create your environment file by duplicating `.env.example`:

```bash
cp .env.example .env
```
Add your TMDB API key inside `.env`:

```bash
TMDB_API_KEY="YOUR_API_KEY_HERE"
```

### ▶️ Running the App Locally

Start the highly optimized FastAPI development server (which also statically serves the frontend UI):

```bash
uvicorn api.index:app --reload
```

Then, seamlessly open `http://localhost:8000` in your browser!

## ℹ️ Additional Information

### 📂 Dataset

**MovieLens Small Dataset**

Includes:
- `movies.csv` — Movie titles & genres
- `ratings.csv` — User ratings
- `tags.csv` — Descriptive movie tags (metadata)

The app intelligently handles the dataset structure, automatically downloading it securely to your local directory or temp environments if missing.

### 📦 Dependencies

- numpy
- pandas
- pillow
- fastapi
- uvicorn
- requests
- rapidfuzz
- scikit-learn
- python-dotenv

### ⚖️ License

This app uses the [MIT License](LICENSE)