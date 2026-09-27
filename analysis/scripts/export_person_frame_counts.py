import argparse
import csv
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (
    LOCAL_DATA,
    INTERMEDIATE_DIR,
    load_aggregate,
    load_all_preds,
    load_tracks,
    load_recog,
    read_tsv,
)

THRESHOLDS = {True: 0.42, False: 0.18}

UNKNOWN = -1
MALE, FEMALE = 0, 1

GENDER_BY_LABEL = {
    "female": FEMALE,
    "trans woman": FEMALE,
    "male": MALE,
    "trans man": MALE,
}

HUMAN, ANIMAL, OTHER = 0, 1, 2
CHARACTER_KINDS = ["human", "animal", "other"]



def load_actor_gender_history(path):
    history = {}
    with open(path, newline="") as f:
        for line in f:
            nm_id, _name, gender_json = line.rstrip("\n").split("\t")
            history[nm_id] = {int(y): g for y, g in json.loads(gender_json).items()}
    return history


def gender_for_year(gender_json, year):
    years = sorted(gender_json)
    if not years:
        return None
    in_force = [y for y in years if y <= year]
    return gender_json[in_force[-1] if in_force else years[0]]


def resolve_gender(label):
    if label is None:
        return UNKNOWN
    resolved = {GENDER_BY_LABEL.get(claim.strip(), UNKNOWN) for claim in label.split("#")}
    return resolved.pop() if len(resolved) == 1 else UNKNOWN


def load_animated_gender():
    return {
        (row["movie id"], row["character_name"]): (row["gender"], row["category"])
        for row in read_tsv(LOCAL_DATA / "animated_character_gender.tsv")
    }



def frame_person_table(movie_id, fnos, categories, threshold):
    try:
        tracks = load_tracks(movie_id)
        recog = load_recog(movie_id)
    except FileNotFoundError:
        return []

    if not tracks or not recog or len(fnos) == 0:
        return []

    recognized = {}
    for row in recog:
        if row["top_score"] >= threshold:
            recognized.setdefault(row["track"], []).append(row["top_actor"])
    if not recognized:
        return []

    first_actor, n_actors = {}, {}
    for track in tracks:
        for actor in recognized.get(track["track"], ()):
            fno = track["fno"]
            if fno in n_actors:
                n_actors[fno].add(actor)
            else:
                first_actor[fno], n_actors[fno] = actor, {actor}

    single_person = {fno: actor for fno, actor in first_actor.items() if len(n_actors[fno]) == 1}

    return [
        (fno, category, single_person[fno])
        for fno, category in zip(fnos, categories)
        if category != "na" and fno in single_person
    ]


def build_counts(agg, all_preds, actor_gender_history, animated_gender):
    def live_action_is_female(nm_id, year):
        hist = actor_gender_history.get(nm_id)
        if not hist or year is None:
            return UNKNOWN
        return resolve_gender(gender_for_year(hist, year))

    def animated_is_female(movie_id, character_name):
        entry = animated_gender.get((movie_id, character_name))
        return UNKNOWN if entry is None else resolve_gender(entry[0])

    def animated_kind(movie_id, character_name):
        entry = animated_gender.get((movie_id, character_name))
        if entry is None or entry[1] not in CHARACTER_KINDS:
            return UNKNOWN
        return CHARACTER_KINDS.index(entry[1])

    duplicated = {mid for mid, n in Counter(row["id"] for row in agg).items() if n > 1}
    agg_lookup = {row["id"]: row for row in agg if row["id"] not in duplicated}

    y_labels = np.array(all_preds.y_labels)
    counts = Counter()
    for movie_id, rows in tqdm(all_preds.by_movie(), total=len(all_preds.movie_ids),
                               desc="counting person frames"):
        info = agg_lookup.get(movie_id)
        if info is None:
            continue
        animated = info["is_animated"]
        frames = frame_person_table(
            movie_id, all_preds.fno[rows], y_labels[all_preds.y[rows]], THRESHOLDS[animated]
        )
        if not frames:
            continue

        identities = {identity for _, _, identity in frames}
        if animated:
            gender_of = {i: animated_is_female(movie_id, i) for i in identities}
            kind_of = {i: animated_kind(movie_id, i) for i in identities}
        else:
            gender_of = {i: live_action_is_female(i, info["year"]) for i in identities}
            kind_of = dict.fromkeys(identities, UNKNOWN)

        for _, category, identity in frames:
            counts[(movie_id, category, gender_of[identity], kind_of[identity])] += 1

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(INTERMEDIATE_DIR / "person_frame_counts.csv"))
    args = parser.parse_args()

    agg = load_aggregate()
    agg_by_id = {row["id"]: row for row in agg}
    all_preds = load_all_preds(agg)

    actor_gender_history = load_actor_gender_history(LOCAL_DATA / "wikidata.actor.historical.gender.080526.tsv")
    animated_gender = load_animated_gender()

    counts = build_counts(agg, all_preds, actor_gender_history, animated_gender)
    print(f"{len(counts)} (movie, category, gender, kind) groups, "
          f"{sum(counts.values())} total frames, "
          f"{len({k[0] for k in counts})} movies")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "year", "is_popular", "is_prestige", "is_indie",
                          "category", "is_female", "kind", "n_frames"])
        for (movie_id, category, is_female, kind), n in sorted(counts.items()):
            info = agg_by_id[movie_id]
            writer.writerow([
                movie_id, info["year"],
                int(info["is_popular"]), int(info["is_prestige"]), int(info["is_indie"]),
                category, is_female, kind, n,
            ])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
