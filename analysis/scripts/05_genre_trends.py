import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

TOP_N_GENRES = 5
MIDDLE_CATEGORIES = ["mcu", "m", "ml", "l"]
CATEGORY_LABELS = {"mcu": "Medium close-up", "m": "Medium", "ml": "Medium long", "l": "Long"}
SCALE_COLS = ["xcu", "cu", "mcu", "m", "ml", "l", "xls"]


def load_genres(path):
    genres = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["id"]:
                genres[row["id"]] = row["genres"]
    return genres


def load_film_level(path, genre_by_id):
    movies = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            counts = [float(row[c]) for c in SCALE_COLS]
            total = sum(counts)
            if total <= 0:
                continue
            shares = {c: v / total for c, v in zip(SCALE_COLS, counts)}
            movies.append({
                "id": row["id"],
                "year": float(row["year"]),
                "is_popular": row["is_popular"] == "1",
                "genres": genre_by_id.get(row["id"], ""),
                "shares": shares,
            })
    return movies


def movies_by_genre(movies):
    by_genre = defaultdict(list)
    for movie in movies:
        if not movie["is_popular"]:
            continue
        for genre in (movie["genres"] or "").split(","):
            genre = genre.strip()
            if genre:
                by_genre[genre].append(movie)
    return by_genre


def top_genres(by_genre, n=TOP_N_GENRES):
    return sorted(by_genre, key=lambda g: -len(by_genre[g]))[:n]


def bootstrap_by_year(movies_in_genre, category, n_boot=1000, seed=0):
    years = np.array([m["year"] for m in movies_in_genre])
    shares = np.array([m["shares"][category] for m in movies_in_genre])
    rng = np.random.default_rng(seed)
    out = []
    for year in sorted(set(years)):
        vals = shares[years == year]
        n_movies = len(vals)
        resamples = vals[rng.integers(0, n_movies, size=(n_boot, n_movies))]
        boot_means = resamples.mean(axis=1)
        out.append({
            "year": year,
            "mean": vals.mean(),
            "lo": np.percentile(boot_means, 2.5),
            "hi": np.percentile(boot_means, 97.5),
            "n_movies": n_movies,
        })
    return out


def build_genre_trends(movies):
    by_genre = movies_by_genre(movies)
    genres = top_genres(by_genre)

    rows = []
    for genre in genres:
        for cat in MIDDLE_CATEGORIES:
            for row in bootstrap_by_year(by_genre[genre], cat):
                row["category"] = CATEGORY_LABELS[cat]
                row["genre"] = genre
                rows.append(row)
    return rows, genres


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])


def render_figures(r_script_path):
    if shutil.which("Rscript") is None:
        sys.exit(f"Rscript not found on PATH -- the table is written, but the figure "
                 f"needs R. Install R (with ggplot2) and run: Rscript {r_script_path}")
    subprocess.run(["Rscript", str(r_script_path)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default=str(SCRIPT_DIR / ".." / "data" / "metadata" / "movie_metadata - Aggregate.tsv"))
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"))
    parser.add_argument("--intermediate_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    genre_by_id = load_genres(args.aggregate)
    movies = load_film_level(args.film_level, genre_by_id)
    rows, genres = build_genre_trends(movies)

    intermediate_dir = pathlib.Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(intermediate_dir / "genre_trends_by_year.csv", rows,
              columns=["year", "mean", "lo", "hi", "n_movies", "category", "genre"])
    print(f"Top {TOP_N_GENRES} genres among popular movies: {', '.join(genres)}")

    render_figures(SCRIPT_DIR / "05_genre_trends.R")


if __name__ == "__main__":
    main()
