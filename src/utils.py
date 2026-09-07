import os
import re
import base64
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from functools import lru_cache
from dotenv import load_dotenv
from difflib import SequenceMatcher


def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""


def clean_title(title):
    return re.sub(r"\(\d{4}\)", "", title).strip()


def fix_title_display(title):
    if not title or not isinstance(title, str):
        return "" if title is None else str(title)

    # Separate the year if it exists at the end
    match = re.search(r"^(.*?)(\s*\(\d{4}\))?$", title)
    if not match:
        return title

    name = match.group(1).strip()
    year = match.group(2) or ""

    # Check for trailing article, with optional non-year parenthetical (e.g. alternate foreign title)
    art_match = re.search(r"^(.*?),\s*(The|A|An)(\s+\([^)]*?\))?$", name, re.IGNORECASE)
    if art_match:
        base = art_match.group(1).strip()
        article = art_match.group(2).capitalize()
        extra = art_match.group(3) or ""
        return f"{article} {base}{extra}{year}"

    return name + year


def best_match(results, clean_title, year=None):
    if not results:
        return None
    best, best_score = None, 0
    for r in results:
        tmdb_title = r.get("title", "")
        score = SequenceMatcher(None, clean_title.lower(), tmdb_title.lower()).ratio()
        if pd.notna(year) and "release_date" in r and r["release_date"]:
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


@lru_cache(maxsize=500)
def fetch_poster_url(title, year, tmdb_api_key):
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

        if not results:
            params.pop("year", None)
            resp = requests.get(
                "https://api.themoviedb.org/3/search/movie", params=params, timeout=5
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])

        match = best_match(results, clean, year)
        if match and match.get("poster_path"):
            return f"https://image.tmdb.org/t/p/w300{match['poster_path']}"

    except Exception:
        pass

    return ""


def get_image_from_bytes(img_bytes):
    return Image.open(BytesIO(img_bytes))


def get_tmdb_api_key():
    # Load environment variables from .env file if it exists
    load_dotenv()

    # Return environment variable
    return os.environ.get("TMDB_API_KEY", "")
