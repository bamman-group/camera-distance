import argparse
import csv
import math
import pathlib
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

CATEGORY_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls", "na"]
CATEGORY_LABELS = {
    "xcu": "Extreme close-up", "cu": "Close-up", "mcu": "Medium close-up", "m": "Medium",
    "ml": "Medium long", "l": "Long", "xls": "Extreme long", "na": "N/A",
}



def load_aggregate(path):
    from movie_collections import load_collection_ids
    collections = load_collection_ids(path)
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if not row["id"]:
                continue
            try:
                year = float(row["year"])
            except (TypeError, ValueError):
                year = float("nan")
            rows.append({
                "id": row["id"],
                "genres": row["genres"],
                "year": year,
                "is_popular": row["id"] in collections["popular"],
                "is_prestige": row["id"] in collections["prestige"],
                "is_animated": "Animat" in (row["genres"] or ""),
            })
    return rows


def load_film_level(path):
    shares_by_id = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = float(row["n_frames"])
            if n <= 0:
                continue
            shares_by_id[row["id"]] = {c: float(row[c]) / n for c in CATEGORY_ORDER}
    return shares_by_id



def primary_genre(genres):
    for genre in (genres or "").split(","):
        genre = genre.strip()
        if genre and "Animat" not in genre:
            return genre
    return None


def paired_animated(agg, movie_ids, seed=0):
    eligible = set(movie_ids)
    duplicated = {movie_id for movie_id, n in Counter(row["id"] for row in agg).items() if n > 1}

    by_cell = defaultdict(lambda: {"animated": [], "live_action": []})
    for row in agg:
        if row["id"] not in eligible or row["id"] in duplicated or math.isnan(row["year"]):
            continue
        stratum = "popular" if row["is_popular"] else "prestige" if row["is_prestige"] else None
        if stratum is None:
            continue
        genre = primary_genre(row["genres"])
        if genre is None:
            continue
        side = "animated" if row["is_animated"] else "live_action"
        by_cell[(stratum, row["year"], genre)][side].append(row["id"])

    rng = np.random.default_rng(seed)
    pairs = []
    for stratum in ("popular", "prestige"):
        cells = sorted(key for key in by_cell if key[0] == stratum)
        for _, year, genre in cells:
            sides = by_cell[(stratum, year, genre)]
            animated, live_action = sorted(sides["animated"]), sorted(sides["live_action"])
            if not animated or not live_action:
                continue
            drawn = rng.permutation(len(live_action))[:len(animated)]
            for movie_id, i in zip(animated, drawn):
                pairs.append({
                    "stratum": stratum,
                    "year": year,
                    "genre": genre,
                    "animated": movie_id,
                    "live_action": live_action[i],
                })
    return pairs



def bootstrap_by_movie(buckets, values, group_name, n_boot=1000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return []
    if np.ndim(buckets) == 0:
        bucket_keys, bucket_codes = np.array([buckets]), np.zeros(len(values), dtype=np.int64)
    else:
        buckets = np.asarray(buckets)
        bucket_keys, bucket_codes = np.unique(buckets, return_inverse=True)

    rng = np.random.default_rng(seed)
    out = []
    for code, key in enumerate(bucket_keys):
        vals = values[bucket_codes == code]
        n_movies = len(vals)
        if n_movies == 0:
            continue
        resamples = vals[rng.integers(0, n_movies, size=(n_boot, n_movies))]
        boot_means = resamples.mean(axis=1)
        out.append({
            group_name: key.item() if isinstance(key, np.generic) else key,
            "mean": float(vals.mean()),
            "lo": float(np.percentile(boot_means, 2.5)),
            "hi": float(np.percentile(boot_means, 97.5)),
            "n_movies": n_movies,
        })
    return out



def build_whole_movie_bootstrap(shares_by_id, pairs):
    popular = [pair for pair in pairs if pair["stratum"] == "popular"]
    animated_ids = {pair["animated"] for pair in popular}
    live_action_ids = {pair["live_action"] for pair in popular}
    selected_ids = sorted(animated_ids | live_action_ids)

    agg_anim_bootstrap = []
    for cat in CATEGORY_ORDER:
        is_animated = np.array([mid in animated_ids for mid in selected_ids])
        shares = np.array([shares_by_id[mid][cat] for mid in selected_ids])
        for row in bootstrap_by_movie(is_animated, shares, group_name="is_animated"):
            row["category"] = CATEGORY_LABELS.get(cat, cat)
            row["collection"] = "Popular"
            row["status"] = "Animated" if row["is_animated"] else "Live-action"
            agg_anim_bootstrap.append(row)
    return agg_anim_bootstrap


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def render_figures(r_script_path):
    if shutil.which("Rscript") is None:
        sys.exit(f"Rscript not found on PATH -- the tables are written, but the figures "
                 f"need R. Install R (with ggplot2) and run: Rscript {r_script_path}")
    subprocess.run(["Rscript", str(r_script_path)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", default=str(
        SCRIPT_DIR / ".." / "data" / "metadata" / "movie_metadata - Aggregate.tsv"))
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"))
    parser.add_argument("--intermediate_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    agg = load_aggregate(args.aggregate)
    shares_by_id = load_film_level(args.film_level)

    intermediate_dir = pathlib.Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    pairs = paired_animated(agg, shares_by_id.keys())
    write_csv(intermediate_dir / "paired_animated.csv", pairs)

    agg_anim_bootstrap = build_whole_movie_bootstrap(shares_by_id, pairs)
    write_csv(intermediate_dir / "animated_whole_movie_bootstrap_intervals.csv", agg_anim_bootstrap)

    render_figures(SCRIPT_DIR / "02_animated_vs_live_action.R")


if __name__ == "__main__":
    main()
