import argparse
import csv
import pathlib
import shutil
import subprocess
import sys

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

MIDDLE_CATEGORIES = ["mcu", "m", "ml", "l"]
CATEGORY_LABELS = {"mcu": "Medium close-up", "m": "Medium", "ml": "Medium long", "l": "Long"}
SCALE_COLS = ["xcu", "cu", "mcu", "m", "ml", "l", "xls"]


def load_film_level(path):
    movies = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["is_popular"] != "1":
                continue
            counts = [float(row[c]) for c in SCALE_COLS]
            total = sum(counts)
            if total <= 0:
                continue
            shares = {c: v / total for c, v in zip(SCALE_COLS, counts)}
            movies.append({"id": row["id"], "year": float(row["year"]), "shares": shares})
    return movies


def bootstrap_by_year(movies, category, n_boot=1000, seed=0):
    years = np.array([m["year"] for m in movies])
    shares = np.array([m["shares"][category] for m in movies])
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


def build_middle_distances(movies):
    rows = []
    for cat in MIDDLE_CATEGORIES:
        for row in bootstrap_by_year(movies, cat):
            row["category"] = CATEGORY_LABELS[cat]
            rows.append(row)
    return rows


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
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"))
    parser.add_argument("--intermediate_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    movies = load_film_level(args.film_level)
    rows = build_middle_distances(movies)

    intermediate_dir = pathlib.Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    write_csv(intermediate_dir / "middle_distances_popular_by_year.csv", rows,
              columns=["year", "mean", "lo", "hi", "n_movies", "category"])
    print(f"{len(movies)} popular movies with a computed category share")

    render_figures(SCRIPT_DIR / "07_middle_distances_popular.R")


if __name__ == "__main__":
    main()
