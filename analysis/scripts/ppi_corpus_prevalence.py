import argparse
import csv
import pathlib
import sys

import numpy as np
from scipy.stats import norm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "benchmark" / "shared"))

from evaluate_distance import load_gold, match_predictions

WINDOW = 12



ID_ALIASES = {
    "philadelphia.st_tt0032904": "philadelphia.story_tt0032904",
}

YEAR_OVERRIDES = {
    "ia.HisGirlFriday_tt0032599": 1940,
}


def resolve_ids(movies):
    return np.array([ID_ALIASES.get(m, m) for m in movies])


def load_movie_years(aggregate_path):
    years = dict(YEAR_OVERRIDES)
    with open(aggregate_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if not row["id"]:
                continue
            try:
                years[row["id"]] = int(row["year"])
            except (TypeError, ValueError):
                years[row["id"]] = None
    return years

CATEGORY_LABELS = {
    "xcu": "Extreme close-up",
    "cu": "Close-up",
    "mcu": "Medium close-up",
    "m": "Medium",
    "ml": "Medium long",
    "l": "Long",
    "xls": "Extreme long",
    "na": "N/A",
}

COLLECTION_FLAG = {"popular": "is_popular", "prestige": "is_prestige", "indie": "is_indie"}


def load_labeled_shares(gold_path, dev_pred, test_pred, aggregate_path, category):
    gold_regions, clip_anim = load_gold(gold_path)
    dev_true, dev_pred_y, dev_movies, _, _ = match_predictions(dev_pred, gold_regions, clip_anim)
    test_true, test_pred_y, test_movies, _, _ = match_predictions(test_pred, gold_regions, clip_anim)
    y_true = np.concatenate([dev_true, test_true])
    y_pred = np.concatenate([dev_pred_y, test_pred_y])
    movies = resolve_ids(np.concatenate([dev_movies, test_movies]))

    year_by_movie = load_movie_years(aggregate_path)
    keep = np.array([year_by_movie.get(m) is not None for m in movies])
    y_true, y_pred, movies = y_true[keep], y_pred[keep], movies[keep]

    shares = {}
    for movie in np.unique(movies):
        mask = movies == movie
        shares[movie] = (
            year_by_movie[movie],
            float(np.mean(y_true[mask] == category)),
            float(np.mean(y_pred[mask] == category)),
        )
    return shares


def window_shares(center_year, shares):
    lo_year, hi_year = center_year - WINDOW, center_year + WINDOW
    window = [(t, p) for _, (y, t, p) in shares.items() if lo_year <= y <= hi_year]
    if not window:
        return None, None
    true_shares = np.array([t for t, _ in window])
    pred_shares = np.array([p for _, p in window])
    return true_shares, pred_shares


def load_film_level(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n = float(row["n_frames"])
            if n <= 0:
                continue
            rows.append({
                "id": row["id"],
                "year": int(float(row["year"])),
                "is_popular": row["is_popular"] == "1",
                "is_prestige": row["is_prestige"] == "1",
                "is_indie": row["is_indie"] == "1",
                "shares": {c: float(row[c]) / n for c in CATEGORY_LABELS},
            })
    return rows


def corpus_movie_shares(film_level, movies_in_collection, category, year):
    shares = [
        row["shares"][category] for row in film_level
        if row["year"] == year and row["id"] in movies_in_collection
    ]
    return np.array(shares) if shares else None


def ppi_mean_estimate(unlabeled, true_labeled, pred_labeled, z):
    n_capital = len(unlabeled)
    n = len(true_labeled)

    theta_hat = unlabeled.mean()
    diffs = pred_labeled - true_labeled
    delta = diffs.mean()
    theta_pp = theta_hat - delta

    var_unlabeled = np.mean((unlabeled - theta_hat) ** 2)
    var_correction = np.mean((diffs - delta) ** 2)

    w = z * np.sqrt(var_correction / n + var_unlabeled / n_capital)

    return theta_hat, delta, theta_pp, w, var_unlabeled, n_capital


def main():
    benchmark_dir = REPO_ROOT / "benchmark"
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(benchmark_dir / "data" / "distance_region.jsonl"))
    parser.add_argument("--dev_pred", default=str(
        benchmark_dir / "checkpoints" / "siglip2_base_384" / "dev_predictions.smoothed.jsonl"))
    parser.add_argument("--test_pred", default=str(
        benchmark_dir / "predictions" / "siglip2_base_384" / "predictions.smoothed.jsonl"))
    parser.add_argument("--aggregate", default=str(SCRIPT_DIR / ".." / "data" / "metadata" / "movie_metadata - Aggregate.tsv"),
                         help="this project's movie reference list (id, year, genres columns)")
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"),
                         help="export_film_level_categories.py's precomputed per-film argmax counts")
    parser.add_argument("--category", default="mcu", choices=list(CATEGORY_LABELS))
    parser.add_argument("--collection", default="popular", choices=list(COLLECTION_FLAG))
    parser.add_argument("--alpha", type=float, default=0.05, help="error level -- output is a (1-alpha) confidence set")
    parser.add_argument("--report_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    z = norm.ppf(1 - args.alpha / 2)

    shares = load_labeled_shares(args.gold, args.dev_pred, args.test_pred, args.aggregate, args.category)
    print(f"Pooled dev+test labeled movies: {len(shares)}")

    film_level = load_film_level(args.film_level)
    flag_col = COLLECTION_FLAG[args.collection]
    movies_in_collection = {row["id"] for row in film_level if row[flag_col]}
    years = sorted({row["year"] for row in film_level if row["id"] in movies_in_collection})
    print(f"Corpus movies in {args.collection}: {len(movies_in_collection)}, years {years[0]}-{years[-1]}")

    rows = []
    for year in years:
        corpus_shares = corpus_movie_shares(film_level, movies_in_collection, args.category, year)
        true_shares, pred_shares = window_shares(year, shares)
        if corpus_shares is None or true_shares is None:
            print(f"  {year}: missing corpus or labeled data, skipping")
            continue

        theta_hat, delta, theta_pp, w, var_unlabeled, n_capital = ppi_mean_estimate(
            corpus_shares, true_shares, pred_shares, z
        )
        raw_w = z * np.sqrt(var_unlabeled / n_capital)

        rows.append({
            "year": year, "n_movies_corpus": n_capital, "n_labeled_movies": len(true_shares),
            "raw_mean": theta_hat, "raw_lo": theta_hat - raw_w, "raw_hi": theta_hat + raw_w,
            "correction_mean": delta,
            "ppi_mean": theta_pp, "ppi_lo": theta_pp - w, "ppi_hi": theta_pp + w,
        })
    rows.sort(key=lambda r: r["year"])
    category_label = CATEGORY_LABELS[args.category]
    print(f"Corpus years covered ({args.collection}, {category_label}): "
          f"{rows[0]['year']}-{rows[-1]['year']} ({len(rows)} years)")

    report_dir = pathlib.Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{args.category}_ppi_corpus_prevalence_{args.collection}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
