# shot-distance

This repository contains code and data to support the following paper:

David Bamman, Allison Cooper, Dan Hickey and Madison Mar (2026), "Measuring the evolution of camera distance across a century of film" ([preprint](https://people.ischool.berkeley.edu/~dbamman/pubs/pdf/camera_distance.pdf))



## Layout

- `benchmark/` — trains and evaluates the shot-distance classifier. See `benchmark/README.md` for documentation of the training/prediction scripts.
- `analysis/` — applies the trained classifier's predictions across the full digitized corpus to produce other figures, tables, and statistics (temporal trends, gender, animation, genre, aspect ratio, PPI correction, the Salt comparison, the ten-year transition tests).
- `demo/` -- illlustrates the use of the trained model to predict shot distance for an input mp4.  See `demo/README.md` to run on your own data.


## Setup

```bash
cd benchmark
conda env create -f environment.yml
conda activate shot-distance
```

### Provenance of `analysis/data/analysis_outputs/`

`film_level_categories.csv` and `person_frame_counts.csv` are derived from the full corpus's per-frame predictions and face-recognition output, via `export_film_level_categories.py` and `export_person_frame_counts.py` (step 2). That underlying data is too large to check into this repo; it's available separately:

```bash
cd analysis/data
curl -O http://yosemite.ischool.berkeley.edu/david/distance/shot_distance.tar.gz
curl -O http://yosemite.ischool.berkeley.edu/david/distance/recognition_output.tar.gz
tar xzf shot_distance.tar.gz
tar xzf recognition_output.tar.gz
```

Extracting produces `analysis/data/shot_distance/` and `analysis/data/recognition_output/` (with `recog/`, `tracks/`, `fps/`, `shots/` subdirectories), where `analysis/scripts/common.py` looks for them. This is documentation of where the already-included CSVs came from, not a required step: you only need to download this data and rerun step 2 if you want to regenerate those CSVs yourself.

All commands below are given relative to the directory noted in each section header (`cd` there first).

## 1. Classifier training/evaluation

Checkpoints and predictions for all 8 architectures are already in `benchmark/checkpoints/` and `benchmark/predictions/`. Training and prediction both need the underlying movie clips digitized from purchased DVDs/Blu-rays under the US 37 CFR 201.40(b) text-and-data-mining exemption; these are not redistributable and hence not included in this repository. The commands below are documentation of how the included checkpoints and predictions were produced.

Each architecture was trained by its own `train_<arch>.sh` wrapper. The wrappers are the authoritative record of the
hyperparameters.

```bash
cd benchmark
# CLIPS points at the digitized clips; OUTPUT_DIR at this repo's checkpoints/. 
for script in train_siglip2_base224 train_siglip2_base384 train_siglip2_so400m384 \
              train_convnext_large train_swin_b train_resnet50 \
              train_vit_b16 train_vit_l16; do
    CLIPS=clips/ OUTPUT_DIR=checkpoints/ ./models/classifiers/$script.sh
done
```

Every flag each wrapper passes is echoed into that arch's `train_results.json`.

Then, to assess accuracy, within-one accuracy, per-class precision/recall/F1, and training time:

```bash
python3 shared/evaluate_all_models.py \
    --checkpoints_dir checkpoints --jsonl data/distance_region.jsonl \
    --clips clips/ --predictions_dir predictions
python3 shared/report_to_latex.py \
    --report predictions/report/report.md --output predictions/report/tables.tex
```

The best model (`siglip2_base_384`) is the one each downstream `analysis/` script uses.


## 2. Corpus-wide exports

`film_level_categories.csv` and `person_frame_counts.csv` are already included (see Setup). The commands below are documentation of how they were produced, from `analysis/scripts/` — you'd only run them yourself if you've downloaded the underlying data (Setup) and want to regenerate those CSVs.

```bash
python3 export_film_level_categories.py
python3 export_person_frame_counts.py
```

- `export_film_level_categories.py` writes `analysis/data/analysis_outputs/film_level_categories.csv` (per-film category frame counts + year + collection flags), read by steps 3–6 below.
- `export_person_frame_counts.py` writes `analysis/data/analysis_outputs/person_frame_counts.csv` (per-movie category × gender × kind frame counts), read by `03_character_attention.py` in step 3.


## 3. Temporal trends, gender, animation

Run from `analysis/scripts/`:

```bash
python3 01_temporal_trends.py
python3 02_animated_vs_live_action.py
python3 03_character_attention.py
```

| Script | Produces |
|---|---|
| `01_temporal_trends.py` | Prevalence of every shot-distance category over time, for popular and prestige films; a comparison of that trend across the popular/prestige/indie collections; a table of medium-close-up prevalence by genre; the underlying bootstrapped year/decade intermediate tables |
| `02_animated_vs_live_action.py` | The shot-distance distribution compared between paired live-action and animated films. |
| `03_character_attention.py` | Shot distance by character gender for popular films; the same gender difference for the paired live-action/animated groups; gender difference over time for popular and prestige films; the tables of gender rate by shot distance (for popular films and for each paired group)|

## 4. Genre, aspect ratio, Salt comparison, ten-year tests

Run from `analysis/scripts/`:

```bash
python3 05_genre_trends.py          # shot-distance prevalence over time within the top genres
python3 06_aspect_ratio_trends.py   # shot-distance prevalence over time within each aspect ratio
python3 08_aspect_ratio_prevalence.py  # prevalence of the most common aspect ratios over time
python3 07_middle_distances_popular.py # shot-distance-over-time chart
python3 salt_spearman.py            # correlation between predicted and Barry Salt's hand-coded shot-distance distributions
python3 camera_distance_tests.py    # before/after comparison of shot-distance prevalence around the introduction of sound, television, and smartphones
```


## 5. Digitization counts

```bash
cd analysis/scripts
python3 04_digitization_by_year.py
```

Produces a chart of the number of digitized movies per year, one per collection, from `movie_collections.py`'s metadata.

`count_category_movies.py` (same directory) prints the popular/prestige/indie counts and per-file join stats to stdout, using the same `movie_collections.py` logic.

## 6. PPI-corrected prevalence

`ppi_corpus_prevalence.py` provides corrections for prediction-powered inference; run once per category, from `analysis/scripts/`:

```bash
for cat in xcu cu mcu m ml l xls na; do
    python3 ppi_corpus_prevalence.py --category $cat
done
```

Then, once all 8 CSVs exist in `analysis/data/analysis_outputs/`:

```bash
Rscript plot_all_categories_ppi_corpus_prevalence.R
```

Writes two charts (raw vs. prediction-powered-corrected prevalence, split into the well-attested middle categories and the rarer edge categories) directly to `analysis/figures/article/`.

## 7. TV penetration

Produces a chart of the percentage of US households with a television set, by year.

```bash
cd analysis/scripts
Rscript plot_tv_penetration.R
```

