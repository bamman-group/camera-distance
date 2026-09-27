import argparse
import csv
import math
import pathlib
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

CATEGORY_DISPLAY_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls", "na"]
CATEGORY_LABELS = {
    "xcu": "Extreme close-up", "cu": "Close-up", "mcu": "Medium close-up", "m": "Medium",
    "ml": "Medium long", "l": "Long", "xls": "Extreme long", "na": "N/A",
}
CATEGORY_INDEX = {c: i for i, c in enumerate(CATEGORY_DISPLAY_ORDER)}

UNKNOWN = -1
MALE, FEMALE = 0, 1
HUMAN, ANIMAL, OTHER = 0, 1, 2



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


def load_predicted_ids(path):
    with open(path, newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def load_person_frame_counts(path):
    ids = []
    years, is_popular, is_prestige, is_indie = [], [], [], []
    categories, is_female, kind, n_frames = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            years.append(float(row["year"]))
            is_popular.append(row["is_popular"] == "1")
            is_prestige.append(row["is_prestige"] == "1")
            is_indie.append(row["is_indie"] == "1")
            categories.append(CATEGORY_INDEX[row["category"]])
            is_female.append(int(row["is_female"]))
            kind.append(int(row["kind"]))
            n_frames.append(int(row["n_frames"]))

    movie_ids, movie_code = np.unique(ids, return_inverse=True)
    pf = {
        "movie_code": movie_code.astype(np.int32),
        "year": np.array(years),
        "is_popular": np.array(is_popular),
        "is_prestige": np.array(is_prestige),
        "is_indie": np.array(is_indie),
        "category": np.array(categories, dtype=np.int8),
        "is_female": np.array(is_female, dtype=np.int8),
        "kind": np.array(kind, dtype=np.int8),
        "n_frames": np.array(n_frames, dtype=np.float64),
    }
    return pf, movie_ids



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



def paired_gender_rates(pf, selected, groups, group_name, n_boot=1000, seed=0):
    rows = np.flatnonzero(selected & (pf["is_female"] != UNKNOWN))
    if len(rows) == 0:
        return []

    if np.ndim(groups) == 0:
        group_keys, group_codes = [groups], np.zeros(len(rows), dtype=np.int64)
    else:
        keys = np.asarray(groups)[rows]
        known = keys == keys if keys.dtype.kind == "f" else keys != None
        rows, keys = rows[known], keys[known]
        group_keys, group_codes = np.unique(keys, return_inverse=True)

    movie_code = pf["movie_code"][rows]
    category = pf["category"][rows]
    is_female = pf["is_female"][rows] == FEMALE
    weight = pf["n_frames"][rows]

    n_categories = len(CATEGORY_DISPLAY_ORDER)
    rng = np.random.default_rng(seed)
    out = []

    for code, key in enumerate(group_keys):
        in_group = group_codes == code
        movies, movie_index = np.unique(movie_code[in_group], return_inverse=True)
        categories, female, w = category[in_group], is_female[in_group], weight[in_group]

        per_movie = []
        for wanted in (~female, female):
            flat = np.bincount(movie_index[wanted] * n_categories + categories[wanted],
                               weights=w[wanted], minlength=len(movies) * n_categories)
            per_movie.append(flat.reshape(len(movies), n_categories))
        male_frames, female_frames = per_movie
        male_total, female_total = male_frames.sum(axis=1), female_frames.sum(axis=1)

        both = (male_total > 0) & (female_total > 0)
        n_movies = int(both.sum())
        if n_movies == 0:
            continue
        male_rate = male_frames[both] / male_total[both, None]
        female_rate = female_frames[both] / female_total[both, None]

        sample = rng.integers(0, n_movies, size=(n_boot, n_movies))
        boot_male = np.stack([male_rate[s].mean(axis=0) for s in sample])
        boot_female = np.stack([female_rate[s].mean(axis=0) for s in sample])

        for i, cat in enumerate(CATEGORY_DISPLAY_ORDER):
            if cat == "na":
                continue
            boot_diff = boot_female[:, i] - boot_male[:, i]
            male_mean, female_mean = male_rate[:, i].mean(), female_rate[:, i].mean()
            out.append({
                group_name: key.item() if isinstance(key, np.generic) else key,
                "category": cat,
                "n_movies": n_movies,
                "m_mean": male_mean,
                "m_lo": np.percentile(boot_male[:, i], 2.5),
                "m_hi": np.percentile(boot_male[:, i], 97.5),
                "w_mean": female_mean,
                "w_lo": np.percentile(boot_female[:, i], 2.5),
                "w_hi": np.percentile(boot_female[:, i], 97.5),
                "diff_mean": female_mean - male_mean,
                "diff_lo": np.percentile(boot_diff, 2.5),
                "diff_hi": np.percentile(boot_diff, 97.5),
            })
    return out



ALPHA = 0.05
MCU_TEST_N_PERMUTE = 10_000
MCU_TEST_SEED = 1


def mcu_gender_gap_test(pf, selected, n_permute=MCU_TEST_N_PERMUTE, seed=MCU_TEST_SEED):
    rows = np.flatnonzero(selected & (pf["is_female"] != UNKNOWN))
    movie_code = pf["movie_code"][rows]
    is_female = pf["is_female"][rows] == FEMALE
    is_mcu = pf["category"][rows] == CATEGORY_INDEX["mcu"]
    weight = pf["n_frames"][rows]

    movies, movie_index = np.unique(movie_code, return_inverse=True)
    n_movies_total = len(movies)

    male_mcu = np.bincount(movie_index[~is_female & is_mcu],
                            weights=weight[~is_female & is_mcu], minlength=n_movies_total)
    male_total = np.bincount(movie_index[~is_female],
                              weights=weight[~is_female], minlength=n_movies_total)
    female_mcu = np.bincount(movie_index[is_female & is_mcu],
                              weights=weight[is_female & is_mcu], minlength=n_movies_total)
    female_total = np.bincount(movie_index[is_female],
                                weights=weight[is_female], minlength=n_movies_total)

    both = (male_total > 0) & (female_total > 0)
    male_rate = male_mcu[both] / male_total[both]
    female_rate = female_mcu[both] / female_total[both]
    diff = female_rate - male_rate
    n_movies = len(diff)

    t = stats.ttest_rel(female_rate, male_rate)
    ci = t.confidence_interval(1 - ALPHA)

    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    count = 0
    for _ in range(n_permute):
        signs = rng.choice([-1.0, 1.0], size=n_movies)
        if abs((diff * signs).mean()) >= obs - 1e-12:
            count += 1
    perm_p = (count + 1) / (n_permute + 1)

    return dict(
        n_movies=n_movies, male_mean=male_rate.mean(), female_mean=female_rate.mean(),
        diff_mean=diff.mean(), lo=ci.low, hi=ci.high, p=t.pvalue, perm_p=perm_p,
    )


def in_movies(pf, movie_ids, wanted_ids):
    wanted_ids = set(wanted_ids)
    wanted = np.array([m in wanted_ids for m in movie_ids], dtype=bool)
    return wanted[pf["movie_code"]]


def write_gender_rate_table(path, rows, description, label):
    def number(x):
        value = 100 * x
        return f"$-${abs(value):.1f}" if value < 0 else f"{value:.1f}"

    def cell(row, prefix):
        return f"{number(row[prefix + '_mean'])} ({number(row[prefix + '_lo'])}, {number(row[prefix + '_hi'])})"

    body = "\n".join(
        f"{CATEGORY_LABELS[row['category']]} & {cell(row, 'm')} & {cell(row, 'w')} & {cell(row, 'diff')} \\\\"
        for row in rows
    )
    n_movies = rows[0]["n_movies"] if rows else 0
    path.write_text(
        "% Generated by 03_character_attention.py -- do not edit by hand.\n"
        "% Requires \\usepackage{booktabs}.\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Shot distance & M (\\%) & W (\\%) & W $-$ M (\\%) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{Share of a character's on-screen frames at each shot distance, for men (M) "
        f"and women (W) among {description}, with 95\\% bootstrap confidence intervals. Each "
        f"movie contributes one observation and resampling is at the level of movies. All three "
        f"columns are computed from the same resamples: the W $-$ M point estimate is exactly the "
        f"difference of the M and W point estimates, but its interval is the percentile interval "
        f"of the paired per-resample difference, not the difference of the M and W intervals. "
        f"Because M and W move together from resample to resample, it is the narrower of the two "
        f"and is the one to read for whether the gap differs from zero. "
        f"Restricted to the {n_movies} movies with recognized characters of both genders.}}\n"
        f"\\label{{tab:{label}}}\n"
        "\\end{table}\n"
    )


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
    parser.add_argument("--person_frame_counts", default=str(
        SCRIPT_DIR / ".." / "data" / "analysis_outputs" / "person_frame_counts.csv"))
    parser.add_argument("--intermediate_dir", default=str(SCRIPT_DIR / ".." / "data" / "analysis_outputs"))
    parser.add_argument("--article_dir", default=str(SCRIPT_DIR / ".." / "figures" / "article"))
    args = parser.parse_args()

    agg = load_aggregate(args.aggregate)
    predicted_ids = load_predicted_ids(args.film_level)
    pf, movie_ids = load_person_frame_counts(args.person_frame_counts)

    intermediate_dir = pathlib.Path(args.intermediate_dir)
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    article_dir = pathlib.Path(args.article_dir)
    article_dir.mkdir(parents=True, exist_ok=True)

    mcu_test = mcu_gender_gap_test(pf, pf["is_popular"])
    print(
        f"MCU rate, women - men (popular, {mcu_test['n_movies']} movies with both genders): "
        f"{mcu_test['male_mean']*100:.1f}% (M) vs {mcu_test['female_mean']*100:.1f}% (W), "
        f"diff {mcu_test['diff_mean']*100:+.2f}pp [{mcu_test['lo']*100:+.2f}, {mcu_test['hi']*100:+.2f}], "
        f"paired t p={mcu_test['p']:.3g}, permutation p={mcu_test['perm_p']:.4f}"
    )
    write_csv(intermediate_dir / "mcu_gender_gap_test.csv", [mcu_test])

    popular_rates = paired_gender_rates(pf, pf["is_popular"], "all years", "period")
    write_gender_rate_table(
        article_dir / "gender_rate_table_popular.tex",
        popular_rates,
        description="characters in popular films, across all years",
        label="gender-rate-popular",
    )
    write_csv(intermediate_dir / "gender_rate_popular.csv", popular_rates)

    pairs = paired_animated(agg, predicted_ids)
    animated_movies = in_movies(pf, movie_ids, {pair["animated"] for pair in pairs})
    live_action_movies = in_movies(pf, movie_ids, {pair["live_action"] for pair in pairs})
    kind = pf["kind"]

    paired_rows = []
    for name, group, selected, description in (
        ("animated_human", "Animated human", animated_movies & (kind == HUMAN),
         "animated human characters in the year-matched animated/live-action sample"),
        ("animated_non_human", "Animated non-human",
         animated_movies & ((kind == ANIMAL) | (kind == OTHER)),
         "animated non-human (animal or other) characters in the year-matched sample"),
        ("live_action", "Live-action", live_action_movies,
         "characters in the live-action half of the year-matched sample"),
    ):
        rows = paired_gender_rates(pf, selected, "all years", "period")
        write_gender_rate_table(
            article_dir / f"gender_rate_table_paired_{name}.tex",
            rows,
            description=description,
            label=f"gender-rate-paired-{name.replace('_', '-')}",
        )
        for row in rows:
            row["group"] = group
            paired_rows.append(row)

    write_csv(intermediate_dir / "gender_rate_paired_groups.csv", paired_rows)

    by_year = []
    for collection in ("popular", "prestige", "indie"):
        for row in paired_gender_rates(pf, pf[f"is_{collection}"], pf["year"], "year"):
            row["collection"] = collection
            by_year.append(row)
    write_csv(intermediate_dir / "gender_rate_diff_by_year.csv", by_year)

    by_decade = []
    decade = np.floor(pf["year"] / 10) * 10
    for collection in ("popular", "prestige", "indie"):
        for row in paired_gender_rates(pf, pf[f"is_{collection}"], decade, "decade"):
            row["collection"] = collection
            by_decade.append(row)
    write_csv(intermediate_dir / "gender_rate_diff_by_decade.csv", by_decade)

    render_figures(SCRIPT_DIR / "03_character_attention.R")


if __name__ == "__main__":
    main()
