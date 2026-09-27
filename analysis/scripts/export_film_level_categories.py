import argparse
import csv
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import INTERMEDIATE_DIR, load_aggregate, load_all_preds

CATEGORY_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls", "na"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "film_level_categories.csv"))
    args = parser.parse_args()

    agg = load_aggregate()
    agg_by_id = {row["id"]: row for row in agg}
    all_preds = load_all_preds(agg)

    code_by_label = {label: i for i, label in enumerate(all_preds.y_labels)}
    category_codes = {c: code_by_label[c] for c in CATEGORY_ORDER if c in code_by_label}

    rows = []
    for movie_id, idx in all_preds.by_movie():
        agg_row = agg_by_id.get(movie_id)
        if agg_row is None:
            continue
        y = all_preds.y[idx]
        row = {
            "id": movie_id,
            "year": int(all_preds.year[idx[0]]),
            "is_popular": int(bool(agg_row["is_popular"])),
            "is_prestige": int(bool(agg_row["is_prestige"])),
            "is_indie": int(bool(agg_row["is_indie"])),
            "is_animated": int(bool(agg_row["is_animated"])),
            "n_frames": len(idx),
        }
        for cat in CATEGORY_ORDER:
            row[cat] = int((y == category_codes[cat]).sum()) if cat in category_codes else 0
        rows.append(row)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "year", "is_popular", "is_prestige", "is_indie", "is_animated", "n_frames"] + CATEGORY_ORDER
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} movies to {out_path}")


if __name__ == "__main__":
    main()
