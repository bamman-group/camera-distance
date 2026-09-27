import csv
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from movie_collections import (
    COLLECTION_FILES,
    COLLECTION_ORDER,
    DEFAULT_AGGREGATE_PATH,
    METADATA_DIR,
    load_valid_ids,
    read_tsv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INTERMEDIATE_DIR = SCRIPT_DIR.parent / "data" / "analysis_outputs"


def digitized_counts_by_year(files, valid_ids):
    ids_by_year = defaultdict(set)
    for filename, min_rank, max_rank in files:
        for row in read_tsv(METADATA_DIR / filename):
            movie_id = row.get("id")
            if not movie_id or movie_id not in valid_ids:
                continue
            if min_rank is not None or max_rank is not None:
                try:
                    rank = int(row.get("rank"))
                except (TypeError, ValueError):
                    continue
                if (min_rank is not None and rank < min_rank) or (max_rank is not None and rank > max_rank):
                    continue
            try:
                year = int(row.get("year"))
            except (TypeError, ValueError):
                continue
            ids_by_year[year].add(movie_id)
    return Counter({year: len(ids) for year, ids in ids_by_year.items()})


def main():
    valid_ids = load_valid_ids(DEFAULT_AGGREGATE_PATH)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    for collection in COLLECTION_ORDER:
        counts = digitized_counts_by_year(COLLECTION_FILES[collection], valid_ids)
        path = INTERMEDIATE_DIR / f"digitization_by_year_{collection}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["year", "digitized"])
            for year in sorted(counts):
                writer.writerow([year, counts[year]])

    if shutil.which("Rscript") is None:
        sys.exit(
            f"Rscript not found on PATH -- the tables are written, but the figures "
            f"need R. Install R (with ggplot2) and run: "
            f"Rscript {SCRIPT_DIR / '04_digitization_by_year.R'}"
        )
    subprocess.run(["Rscript", str(SCRIPT_DIR / "04_digitization_by_year.R")], check=True)


if __name__ == "__main__":
    main()
