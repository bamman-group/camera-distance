import argparse
import csv
import pathlib
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

TOP_N_ASPECT_RATIOS = 5


def load_imdb_by_id(path):
    imdb_by_id = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["id"]:
                imdb_by_id[row["id"]] = row["imdb"]
    return imdb_by_id


def load_aspect_ratios(path, imdb_by_id):
    by_imdb = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            raw = (row.get("all_aspect_ratios") or "").strip()
            by_imdb[row["imdb_id"]] = [t.strip() for t in raw.split(";") if t.strip()]

    ratios = {}
    for movie_id, imdb in imdb_by_id.items():
        if imdb in by_imdb:
            ratios[movie_id] = by_imdb[imdb]
    return ratios


def load_movies(path, ratios_by_id):
    movies = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["is_popular"] != "1":
                continue
            ratios = ratios_by_id.get(row["id"])
            if not ratios:
                continue
            movies.append({"id": row["id"], "year": float(row["year"]), "aspect_ratios": ratios})
    return movies


def top_aspect_ratios(movies, n=TOP_N_ASPECT_RATIOS):
    counts = defaultdict(int)
    for movie in movies:
        for ratio in movie["aspect_ratios"]:
            counts[ratio] += 1
    return sorted(counts, key=lambda r: -counts[r])[:n]


def bootstrap_indicator_by_year(years, indicator, n_boot=1000, seed=0):
    years = np.asarray(years)
    indicator = np.asarray(indicator, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = []
    for year in sorted(set(years)):
        vals = indicator[years == year]
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


def build_aspect_ratio_prevalence(movies):
    ratios = top_aspect_ratios(movies)
    years = [m["year"] for m in movies]

    rows = []
    for ratio in ratios:
        indicator = [1.0 if ratio in m["aspect_ratios"] else 0.0 for m in movies]
        for row in bootstrap_indicator_by_year(years, indicator):
            row["aspect_ratio"] = ratio
            rows.append(row)
    return rows, ratios


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
    parser.add_argument("--aspect_ratios", default=str(
        SCRIPT_DIR / ".." / "data" / "aspect_ratios_extracted.tsv"))
    parser.add_argument("--aggregate", default=str(SCRIPT_DIR / ".." / "data" / "metadata" / "movie_metadata - Aggregate.tsv"))
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"))
    parser.add_argument("--intermediate_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    imdb_by_id = load_imdb_by_id(args.aggregate)
    ratios_by_id = load_aspect_ratios(args.aspect_ratios, imdb_by_id)
    movies = load_movies(args.film_level, ratios_by_id)
    rows, ratios = build_aspect_ratio_prevalence(movies)

    intermediate_dir = pathlib.Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(intermediate_dir / "aspect_ratio_prevalence_by_year.csv", rows,
              columns=["year", "mean", "lo", "hi", "n_movies", "aspect_ratio"])
    print(f"{len(movies)} popular movies with a known aspect ratio")
    print(f"Top {TOP_N_ASPECT_RATIOS} aspect ratios: {', '.join(ratios)}")

    render_figures(SCRIPT_DIR / "08_aspect_ratio_prevalence.R")


if __name__ == "__main__":
    main()
