import csv
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DATA = SCRIPT_DIR.parent / "data"
METADATA_DIR = LOCAL_DATA / "metadata"
DEFAULT_AGGREGATE_PATH = METADATA_DIR / "movie_metadata - Aggregate.tsv"

NA_STRINGS = frozenset([
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a",
    "nan", "null",
])

POPULAR_FILES = [
    ("movie_metadata - BOM 1980-2025.tsv", 1, 50),
    ("movie_metadata - Variety 1922-1979.tsv", 1, 50),
]
PRESTIGE_FILES = [
    ("movie_metadata - AFI 100.tsv", None, None),
    ("movie_metadata - BFI all time.tsv", None, None),
    ("movie_metadata - BFI directors.tsv", None, None),
    ("movie_metadata - NYT 21c.tsv", None, None),
    ("movie_metadata - Prestige.tsv", None, None),
]
INDIE_FILES = [
    ("movie_metadata - American Independent.tsv", None, None),
]
COLLECTION_FILES = {"popular": POPULAR_FILES, "prestige": PRESTIGE_FILES, "indie": INDIE_FILES}
COLLECTION_ORDER = ["popular", "prestige", "indie"]


def read_tsv(path):
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for parts in reader:
            row = dict.fromkeys(header)
            row.update(
                (name, None if value in NA_STRINGS else value)
                for name, value in zip(header, parts)
            )
            yield row


def load_valid_ids(aggregate_path=DEFAULT_AGGREGATE_PATH):
    return {row["id"] for row in read_tsv(aggregate_path) if row["id"]}


def load_file_ids(path, valid_ids, min_rank, max_rank):
    ids = set()
    stats = defaultdict(int)
    for row in read_tsv(path):
        stats["total_rows"] += 1
        movie_id = row.get("id")
        if not movie_id:
            stats["no_id"] += 1
            continue
        if min_rank is not None or max_rank is not None:
            try:
                rank = int(row.get("rank"))
            except (TypeError, ValueError):
                stats["unparseable_rank"] += 1
                continue
            if (min_rank is not None and rank < min_rank) or (max_rank is not None and rank > max_rank):
                stats["out_of_rank_range"] += 1
                continue
        if movie_id not in valid_ids:
            stats["not_in_aggregate"] += 1
            continue
        ids.add(movie_id)
    stats["counted"] = len(ids)
    return ids, stats


def load_collection_ids(aggregate_path=DEFAULT_AGGREGATE_PATH, with_stats=False):
    valid_ids = load_valid_ids(aggregate_path)
    collections = {}
    per_file_stats = {}
    for name in COLLECTION_ORDER:
        all_ids = set()
        file_stats = []
        for filename, min_rank, max_rank in COLLECTION_FILES[name]:
            ids, stats = load_file_ids(METADATA_DIR / filename, valid_ids, min_rank, max_rank)
            all_ids |= ids
            file_stats.append((filename, min_rank, max_rank, stats))
        collections[name] = all_ids
        per_file_stats[name] = file_stats
    if with_stats:
        return collections, valid_ids, per_file_stats
    return collections
