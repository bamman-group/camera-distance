# Shot Distance Classification

## Overview

This project trains models to predict the **shot distance** (e.g., close-up, medium shot, long shot) of individual frames in movie clips. Models are trained on human annotations that label contiguous regions of each clip with a distance category.


**Inputs:**
- `data/distance_region.jsonl` — human-annotated clips with labeled frame regions and train/dev/test split assignments
- `clips/` — directory of mp4 movie clip files (path supplied via `--clips`, not part of this repo)


## Installation

```bash
conda env create -f environment.yml
conda activate shot-distance
```


## Data format

**Input:** `data/distance_region.jsonl`; one clip per line, with contiguous labeled regions:

```json
{"id": "house.party_tt0099800__148704__4961.75680___1.mp4", "regions": [{"start": 0, "end": 106, "distance": "mcu"}, {"start": 107, "end": 205, "distance": "m"}], "anim": "live", "split": "train"}
```

- `start` and `end` are inclusive frame numbers
- `split` must be one of `train`, `dev`, `test`

## Models

### `models/classifiers/` — single-frame vision classifiers

One shared pipeline covering every architecture, selected with `--arch`:

| `--arch` | HF checkpoint | Input |
|---|---|---|
| `siglip2_base_224` | `google/siglip2-base-patch16-224` | 224px |
| `siglip2_base_384` | `google/siglip2-base-patch16-384` | 384px |
| `siglip2_so400m_384` | `google/siglip2-so400m-patch14-384` | 384px |
| `convnext_large` | `facebook/convnext-large-224` | 224px |
| `swin_b` | `microsoft/swin-base-patch4-window7-224` | 224px |
| `resnet50` | `microsoft/resnet-50` | 224px |
| `vit_b_16` | `google/vit-base-patch16-224-in21k` | 224px |
| `vit_l_16` | `google/vit-large-patch16-224-in21k` | 224px |


#### `finetune.py` — Training

Fine-tunes the chosen `--arch` with a linear classification head on the `train` split. The encoder and the randomly initialized head get separate learning rates (`--lr`, `--head_lr`).  Training frames are augmented with random horizontal flips and mild color jitter (both preserve shot distance; crops/scaling would corrupt it), and the loss uses label smoothing (`--label_smoothing`, default 0.1). `--freeze_epochs N` trains only the head for the first N epochs before unfreezing the encoder (LP-FT). Evaluates on `dev` after each epoch and saves the best checkpoint.

**Frame cache.** Frames are decoded once into a flat uint8 file under `--cache_dir` (default `<output_dir>/frame_cache`) and read back through a memory map. Because the key is the *input size* and not the arch, any two arches at the same resolution share one cache automatically: the five 224px architectures share one with each other and with `siglip2_base_224`, while `siglip2_base_384` and `siglip2_so400m_384` share a separate 384px one — the cached frames are raw uint8 RGB, with normalization applied later on the GPU.


```bash
# run from repo root
python3 models/classifiers/finetune.py \
    --arch resnet50 \
    --jsonl data/distance_region.jsonl \
    --clips clips/ \
    --output_dir checkpoints/ \
    [--fps 4] \
    [--epochs 50] \
    [--lr_schedule plateau] \
    [--batch_size 32] \
    [--lr 2e-5] \
    [--head_lr 1e-3] \
    [--freeze_epochs 0] \
    [--plateau_factor 0.5] \
    [--plateau_patience 4] \
    [--plateau_threshold 1e-3] \
    [--early_stop_patience 0] \
    [--weight_decay_scope all] \
    [--weight_decay 0.01] \
    [--max_grad_norm 1.0] \
    [--num_workers 4] \
    [--cache_dir checkpoints/frame_cache] \
    [--rebuild_cache] \
    [--resume]
```

- `--fps`: target sampling rate in frames per second (default: 4). Stride is computed per-clip as `round(clip_fps / target_fps)`
- Frames are resized to the model's input size when the cache is built, then mmapped — RAM use is bounded by the page cache, not by split size
- Outputs go to a per-arch subdirectory of `--output_dir` (e.g. `checkpoints/resnet50/`, `checkpoints/siglip2_so400m_384/`): `best_model.pt`, `last_checkpoint.pt`, `train_results.json`, `train_metrics.jsonl`, `train.log`, and `dev_predictions.jsonl` (per-frame dev predictions from the best epoch — always corresponds to `best_model.pt`; same format as `predictions.jsonl`)
- `best_model.pt` is a dict containing `state_dict`, `classes`, `arch`, and `model_name` (the resolved HF checkpoint).

**`train_results.json`** format:
```json
{"arch": "siglip2_base_384", "model_name": "google/siglip2-base-patch16-384", "max_epochs": 10, "epochs_run": 10, "best_epoch": 9, "batch_size": 32, "lr": 2e-05, "head_lr": 0.001, "weight_decay": 0.01, "weight_decay_scope": "weights_only", "lr_schedule": "cosine", "warmup_frac": 0.05, "plateau_factor": 0.5, "plateau_patience": 4, "plateau_threshold": 0.001, "plateau_cooldown": 0, "min_lr": 0.0, "early_stop_patience": 0, "label_smoothing": 0.1, "max_grad_norm": 1.0, "freeze_epochs": 2, "resize_filter": "area", "rewarmup_on_unfreeze": true, "num_workers": 4, "fps": 4.0, "seed": 42, "train_frames": 27644, "dev_frames": 27866, "train_time_s": 6145.9, "best_dev_acc": 0.6888, "completed": true, "stop_reason": "epoch budget reached"}
```

**`train_metrics.jsonl`** (one line per epoch, appended as training runs).
```json
{"epoch": 9, "train_loss": 0.46905, "dev_acc": 0.68876, "encoder_lr": 5.418275829936537e-07, "head_lr": 2.709137914968268e-05, "encoder_frozen": false, "best_dev_acc": 0.68876, "best_epoch": 9, "epochs_since_improvement": 0, "epoch_time_s": 520.7, "elapsed_s": 5641.3}
```

---

#### `predict.py` — Prediction

Loads a trained checkpoint and writes per-frame predictions to JSONL. Only writes predictions (no evaluation). The class list and arch are read from the checkpoint. Frames are sampled at `--fps` frames per second per clip. Frames are streamed clip by clip. 

Frame sampling, resizing and normalization all come from `common.py`, the same code training uses. The downsampling filter is `--resize_filter`: `linear` (the argparse default) always uses `cv2.INTER_LINEAR`, while `area` uses `cv2.INTER_AREA` when the source frame is at least as large as the target in both dimensions and `cv2.INTER_LINEAR` otherwise. `predict.py` does not take the default — it reads the filter recorded in the checkpoint, so it matches whatever training used (`area`, for every checkpoint shipped here). 


```bash
# run from repo root
python3 models/classifiers/predict.py \
    --jsonl data/distance_region.jsonl \
    --clips clips/ \
    --model_dir checkpoints/siglip2_so400m_384/ \
    --split test \
    --output predictions/siglip2_so400m_384/predictions.jsonl \
    [--fps 4] \
    [--batch_size 32]
```

**Output format** (`predictions.jsonl`):
```json
{"id": "house.party_..._1.mp4", "fno": 10, "y": "mcu", "cu": 0.01200, "l": 0.00300, "m": 0.02100, "mcu": 0.84000, "ml": 0.01000, "na": 0.00050, "xcu": 0.00100, "xls": 0.00240}
```

- `fno`: frame number
- `y`: predicted class (argmax)
- One key per class with probability rounded to 5 decimal places
- Prediction is sampled at `--fps` (default 4)
- Also writes `predictions.timing.json` alongside the output file:

```json
{"model_dir": "checkpoints/siglip2_so400m_384/", "arch": "siglip2_so400m_384", "split": "test", "output": "predictions/siglip2_so400m_384/predictions.jsonl", "fps": 4.0, "n_frames": 84321, "predict_time_s": 412.3}
```

---

#### `train_<arch>.sh`

One convenience script per architecture, all wrapping `finetune.py` with the same hyperparameters as editable variables at the top.

```bash
./models/classifiers/train_siglip2_so400m384.sh
./models/classifiers/train_siglip2_base384.sh
./models/classifiers/train_siglip2_base224.sh
./models/classifiers/train_convnext_large.sh
./models/classifiers/train_swin_b.sh
./models/classifiers/train_resnet50.sh
./models/classifiers/train_vit_b16.sh
./models/classifiers/train_vit_l16.sh
```

---

## Shared scripts

These are model-agnostic — they work on the `predictions.jsonl` output of any model under `models/` as long as it follows the output format documented above.

### `shared/smooth_predictions.py` 

Reads a predictions file, replaces each frame's class probabilities with the average over neighboring frames of the same clip (within `--window` frames on either side, default 12 ≈ 0.5s at 24 fps), re-argmaxes, and writes a new predictions file in the same format. 

```bash
python3 shared/smooth_predictions.py \
    --pred predictions/siglip2_so400m_384/predictions.jsonl \
    --output predictions/siglip2_so400m_384/predictions.smoothed.jsonl \
    [--window 12]
```

---

### `shared/evaluate_distance.py`

Takes a predictions file and a gold file, matches each predicted frame to its gold region label, and reports accuracy, within-one accuracy, and per-class precision, recall, F1 with 95% bootstrap confidence intervals. 

Within-one accuracy counts a prediction one step away on the ordinal distance scale (`xcu < cu < mcu < m < ml < l < xls`) as correct; labels off the scale (`na`) must match exactly.

The bootstrap resamples movies (not clips or frames), since frames within a clip and clips within a movie are correlated. The movie id is the clip id up to the first `__` (e.g. `robots_tt0358082` for `robots_tt0358082__98190__4095.34124___15.mp4`).

```bash
python3 shared/evaluate_distance.py \
    --gold data/distance_region.jsonl \
    --pred predictions/siglip2_so400m_384/predictions.jsonl
```

---

### `shared/evaluate_all_models.py`

Runs the whole pipeline over every model in a `checkpoints_dir` (any subdirectory containing a `best_model.pt`, e.g. one written by `models/classifiers/finetune.py`) and writes one combined report, rather than doing each model by hand with the scripts above. It imports its statistics from `evaluate_distance.py` directly (`load_gold`/`match_predictions`/`compute_metrics`).

```bash
python3 shared/evaluate_all_models.py \
    --checkpoints_dir checkpoints_cosine \
    --jsonl data/distance_region.jsonl \
    --clips clips/ \
    --predictions_dir predictions_cosine \
    [--report_dir predictions_cosine/report] \
    [--split test] [--fps 4] [--batch_size 32] [--window 12] \
    [--n_bootstrap 1000] [--seed 42] [--rebuild]
```

Writes, under `--report_dir` (default `<predictions_dir>/report`):

- **`report.md`** — one summary table (model, accuracy `[95% CI]`, within-one accuracy `[95% CI]`, total training time in minutes, read from that model's `train_results.json`), followed by three per-class precision/recall/F1 tables per model (total / animated / live-action), each captioned with that breakdown's own accuracy `[95% CI]`. 
- **`<arch>.confusion.csv`** — one per model, the "total" (all movies) confusion matrix in long format (`true,pred,count`, every class pair including zeros).



