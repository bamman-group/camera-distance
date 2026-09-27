import argparse
import csv
import math
import pathlib

import numpy as np
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

CATEGORY_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls"]
SALT_COLUMNS = {
    "xcu": [], "cu": ["BCU"], "mcu": ["CU"], "m": ["MCU", "MS"],
    "ml": ["MLS"], "l": ["LS"], "xls": ["VLS"],
}


def load_salt(path):
    movies = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            movie_id = row["match_id"].strip()
            if not movie_id:
                continue
            counts = [sum(float(row[col] or 0) for col in SALT_COLUMNS[cat]) for cat in CATEGORY_ORDER]
            movies.append({
                "id": movie_id,
                "title": row["match_title"].strip() or row["Title"].strip(),
                "year": row["match_year"].strip() or row["Year"].strip(),
                "counts": counts,
            })
    return movies


def load_film_level(path):
    by_id = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_id[row["id"]] = [float(row[cat]) for cat in CATEGORY_ORDER]
    return by_id


def normalize(counts):
    total = sum(counts)
    return [c / total for c in counts] if total > 0 else [0.0] * len(counts)


def spearman_rho(p, q):
    p, q = np.array(p), np.array(q)
    if p.std() == 0 or q.std() == 0:
        return float("nan")
    rho = stats.spearmanr(p, q).correlation
    return float(rho) if not math.isnan(rho) else float("nan")


def compare(salt_movies, film_by_id):
    rows = []
    for movie in salt_movies:
        model_counts = film_by_id.get(movie["id"])
        if model_counts is None:
            continue
        gold, model = normalize(movie["counts"]), normalize(model_counts)
        rows.append({
            "id": movie["id"], "title": movie["title"], "year": movie["year"],
            "rho": spearman_rho(gold, model),
        })
    rows.sort(key=lambda r: (math.isnan(r["rho"]), -r["rho"] if not math.isnan(r["rho"]) else 0))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--salt", default=str(SCRIPT_DIR / ".." / "data" / "salt.tsv"))
    parser.add_argument("--film_level", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "film_level_categories.csv"))
    parser.add_argument("--report_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    args = parser.parse_args()

    salt_movies = load_salt(args.salt)
    film_by_id = load_film_level(args.film_level)
    rows = compare(salt_movies, film_by_id)

    valid_rho = [r["rho"] for r in rows if not math.isnan(r["rho"])]
    mean_rho = sum(valid_rho) / len(valid_rho) if valid_rho else float("nan")
    median_rho = sorted(valid_rho)[len(valid_rho) // 2] if valid_rho else float("nan")

    print(f"SALT vs. model shot-distance rank agreement "
          f"({len(rows)} of {len(salt_movies)} SALT movies matched)")
    print(f"{'title':<45} {'year':>5} {'rho':>7}")
    for r in rows:
        rho_str = "n/a" if math.isnan(r["rho"]) else f"{r['rho']:.3f}"
        print(f"{r['title']:<45} {r['year']:>5} {rho_str:>7}")
    print(f"\nmean rho = {mean_rho:.3f}, median rho = {median_rho:.3f}")

    report_dir = pathlib.Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "salt_spearman.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "title", "year", "rho"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
