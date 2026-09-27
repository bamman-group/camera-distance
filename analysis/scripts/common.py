import csv
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from movie_collections import NA_STRINGS, read_tsv, load_collection_ids

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = SCRIPT_DIR.parent
LOCAL_DATA = ANALYSIS_DIR / "data"
FIGURES_DIR = ANALYSIS_DIR / "figures" / "outputs"
ARTICLE_DIR = ANALYSIS_DIR / "figures" / "article"
INTERMEDIATE_DIR = LOCAL_DATA / "analysis_outputs"

SHOT_DISTANCE_DIR = LOCAL_DATA / "shot_distance"
RECOG_OUTPUT_DIR = LOCAL_DATA / "recognition_output"

_verified_dirs = set()


def require_data_dir(path):
    """Check for one of the large corpus directories at the point a loader
    actually needs it, rather than at import time -- importing this module
    must not require the download, so that anything reading only the
    precomputed CSVs in analysis_outputs/ can run without it."""
    if path in _verified_dirs:
        return
    if not path.is_dir():
        raise SystemExit(
            f"missing data directory: {path}\n"
            f"shot_distance/ and recognition_output/ are not checked into this "
            f"repo; download and extract them into {LOCAL_DATA} as described in "
            f"the top-level README's Setup section. Only export_film_level_"
            f"categories.py and export_person_frame_counts.py need them -- every "
            f"other analysis script reads the CSVs already in {INTERMEDIATE_DIR}."
        )
    _verified_dirs.add(path)


FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTICLE_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

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
CATEGORY_DISPLAY_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls", "na"]




def _csv_field(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return "" if value != value else repr(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return value


def write_csv(path, rows, columns=None):
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_field(row.get(c)) for c in columns])


def render_figures(r_script):
    if shutil.which("Rscript") is None:
        sys.exit(
            f"Rscript not found on PATH -- the tables are written, but the figures "
            f"need R.  Install R (with ggplot2) and run: Rscript {SCRIPT_DIR / r_script}"
        )
    subprocess.run(["Rscript", str(SCRIPT_DIR / r_script)], check=True)



def load_aggregate():
    def parse_year(y):
        try:
            return float(y)
        except (TypeError, ValueError):
            return float("nan")

    aggregate_path = LOCAL_DATA / "metadata" / "movie_metadata - Aggregate.tsv"
    collections = load_collection_ids(aggregate_path)

    rows = []
    for row in read_tsv(aggregate_path):
        rows.append({
            "imdb": row["imdb"],
            "id": row["id"],
            "title": row["title"],
            "genres": row["genres"],
            "year": parse_year(row["year"]),
            "is_popular": row["id"] in collections["popular"],
            "is_prestige": row["id"] in collections["prestige"],
            "is_indie": row["id"] in collections["indie"],
            "is_animated": "Animat" in (row["genres"] or ""),
        })

    require_data_dir(RECOG_OUTPUT_DIR)
    digitized_ids = {p.name[: -len(".recog.txt")] for p in (RECOG_OUTPUT_DIR / "recog").glob("*.recog.txt")}
    for row in rows:
        row["has_recognition_output"] = row["id"] in digitized_ids

    return [row for row in rows if row["id"] is not None]



def load_shot_distance(movie_id, smoothed=True):
    require_data_dir(SHOT_DISTANCE_DIR)
    suffix = "predictions.smoothed.jsonl" if smoothed else "predictions.jsonl"
    path = SHOT_DISTANCE_DIR / f"{movie_id}.{suffix}"
    if os.path.isfile(path):
        return [json.loads(line) for line in path.open()]
    return []


def load_recog(movie_id):
    require_data_dir(RECOG_OUTPUT_DIR)
    path = RECOG_OUTPUT_DIR / "recog" / f"{movie_id}.recog.txt"
    rows = []
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            track, rep_fno, rep_face_num, matches = parts[:4]
            kind = parts[4] if len(parts) > 4 else None
            if not matches.strip():
                continue
            first = matches.split()[0]
            if ":" not in first:
                continue
            top_actor, top_score = first.rsplit(":", 1)
            try:
                top_score = float(top_score)
            except ValueError:
                continue
            rows.append({
                "track": int(track),
                "rep_fno": int(rep_fno),
                "rep_face_num": int(rep_face_num),
                "top_actor": top_actor,
                "top_score": top_score,
                "kind": kind,
            })
    return rows


def load_fps(movie_id):
    require_data_dir(RECOG_OUTPUT_DIR)
    path = RECOG_OUTPUT_DIR / "fps" / f"{movie_id}.fps.txt"
    movie, n_frames, width, height, fps = path.read_text().strip().split("\t")
    return {"n_frames": int(n_frames), "width": int(width), "height": int(height), "fps": float(fps)}


def load_scenes(movie_id):
    require_data_dir(RECOG_OUTPUT_DIR)
    path = RECOG_OUTPUT_DIR / "shots" / f"{movie_id}.scenes.txt"
    with path.open() as f:
        return [
            {"start_fno": int(start), "end_fno": int(end)}
            for start, end in (line.split() for line in f if line.strip())
        ]


def load_tracks(movie_id):
    require_data_dir(RECOG_OUTPUT_DIR)
    path = RECOG_OUTPUT_DIR / "tracks" / f"{movie_id}.tracks.txt"
    fields = ["track", "fno", "face_num", "x0", "y0", "x1", "y1"]
    with path.open() as f:
        return [
            dict(zip(fields, (int(v) for v in line.rstrip("\n").split("\t"))))
            for line in f if line.strip()
        ]



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



def group_means(codes, values, n_groups):
    codes = np.asarray(codes)
    values = np.asarray(values, dtype=np.float64)

    known = ~np.isnan(values)
    if not known.all():
        codes, values = codes[known], values[known]

    order = np.argsort(codes, kind="stable")
    codes, values = codes[order], values[order]

    counts = np.bincount(codes, minlength=n_groups)
    starts = np.zeros(n_groups, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    by_size = np.argsort(-counts, kind="stable")
    sorted_starts, sorted_counts = starts[by_size], counts[by_size]

    total = np.zeros(n_groups)
    compensation = np.zeros(n_groups)
    for i in range(int(sorted_counts[0]) if n_groups else 0):
        active = int(np.searchsorted(-sorted_counts, -i, side="left"))
        value = values[sorted_starts[:active] + i]
        y = value - compensation[:active]
        t = total[:active] + y
        c = (t - total[:active]) - y
        c[np.isnan(c)] = 0.0
        compensation[:active] = c
        total[:active] = t

    means = np.full(n_groups, np.nan)
    nonempty = sorted_counts > 0
    means[by_size[nonempty]] = total[nonempty] / sorted_counts[nonempty]
    return means


def _mean(values, axis=None):
    if np.isnan(values).any():
        return np.nanmean(values, axis=axis)
    return values.mean(axis=axis)


def bootstrap_ci_by_movie(groups, movies, values, group_name, n_boot=1000, seed=0):
    movies = np.asarray(movies)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return []

    if np.ndim(groups) == 0:
        group_keys, group_codes = np.array([groups]), np.zeros(len(values), dtype=np.int64)
    else:
        groups = np.asarray(groups)
        known = groups == groups if groups.dtype.kind == "f" else groups != None
        if not known.all():
            groups, movies, values = groups[known], movies[known], values[known]
        group_keys, group_codes = np.unique(groups, return_inverse=True)

    movie_keys, movie_codes = np.unique(movies, return_inverse=True)

    pairs = group_codes.astype(np.int64) * len(movie_keys) + movie_codes
    pair_keys, pair_codes = np.unique(pairs, return_inverse=True)
    pair_means = group_means(pair_codes, values, len(pair_keys))
    pair_groups = pair_keys // len(movie_keys)

    rng = np.random.default_rng(seed)
    out = []
    for code, key in enumerate(group_keys):
        lo, hi = np.searchsorted(pair_groups, [code, code + 1])
        movie_means = pair_means[lo:hi]
        n_movies = len(movie_means)
        if n_movies == 0:
            continue
        resamples = movie_means[rng.integers(0, n_movies, size=(n_boot, n_movies))]
        boot_means = _mean(resamples, axis=1)
        out.append({
            group_name: key.item() if isinstance(key, np.generic) else key,
            "mean": _mean(movie_means),
            "lo": np.percentile(boot_means, 2.5),
            "hi": np.percentile(boot_means, 97.5),
            "n_movies": n_movies,
        })
    return out



@dataclass
class Predictions:

    movie_ids: list
    movie_code: np.ndarray
    year: np.ndarray
    fno: np.ndarray
    y_labels: list
    y: np.ndarray
    probs: dict

    def __len__(self):
        return len(self.movie_code)

    def movie_mask(self, movie_ids):
        movie_ids = set(movie_ids)
        wanted = np.array([m in movie_ids for m in self.movie_ids], dtype=bool)
        return wanted[self.movie_code]

    def by_movie(self):
        order = np.argsort(self.movie_code, kind="stable")
        bounds = np.searchsorted(self.movie_code[order], np.arange(len(self.movie_ids) + 1))
        for code, movie_id in enumerate(self.movie_ids):
            rows = order[bounds[code]:bounds[code + 1]]
            if len(rows):
                yield movie_id, rows


def load_all_preds(agg):
    movie_index, y_index = {}, {}
    prob_keys = []
    blocks = []

    for row in tqdm(agg, desc="loading predictions"):
        frames = load_shot_distance(row["id"])
        if not frames:
            continue
        n = len(frames)
        for key in frames[0]:
            if key not in ("id", "fno", "y") and key not in prob_keys:
                prob_keys.append(key)
        block = {
            "movie_code": np.fromiter(
                (movie_index.setdefault(f["id"].replace(".mp4", ""), len(movie_index)) for f in frames),
                dtype=np.int32, count=n),
            "fno": np.fromiter((f["fno"] for f in frames), dtype=np.int64, count=n),
            "y": np.fromiter(
                (y_index.setdefault(f["y"], len(y_index)) for f in frames),
                dtype=np.int8, count=n),
            "year": np.full(n, row["year"]),
        }
        for key in frames[0]:
            if key in prob_keys:
                block[key] = np.fromiter((f[key] for f in frames), dtype=np.float64, count=n)
        blocks.append(block)

    if not blocks:
        raise SystemExit(
            f"no predictions were read for any of the {len(agg)} movies in aggregate.tsv.\n"
            f"Looked for <id>.predictions.smoothed.jsonl in {SHOT_DISTANCE_DIR}\n"
            f"e.g. {SHOT_DISTANCE_DIR / (agg[0]['id'] + '.predictions.smoothed.jsonl')}"
        )

    def column(key):
        return np.concatenate([
            b[key] if key in b else np.full(len(b["fno"]), np.nan) for b in blocks
        ])

    movie_ids = list(movie_index)
    ranked = sorted(range(len(movie_ids)), key=movie_ids.__getitem__)
    renumber = np.empty(len(movie_ids), dtype=np.int32)
    renumber[ranked] = np.arange(len(movie_ids), dtype=np.int32)

    return Predictions(
        movie_ids=[movie_ids[i] for i in ranked],
        movie_code=renumber[column("movie_code")],
        year=column("year"),
        fno=column("fno"),
        y_labels=list(y_index),
        y=column("y").astype(np.int8),
        probs={key: column(key) for key in prob_keys},
    )
