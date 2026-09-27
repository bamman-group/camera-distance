# Demo

`predict_movie.py` runs the shot-distance classifier (`siglip2_base_384`, the best
architecture per the paper) on a single input movie, using the same frame-sampling
(4fps), resizing and normalization.

### Setup

Uses the same conda environment as the rest of the repo (see the top-level `README.md`):

```bash
cd benchmark
conda env create -f environment.yml
conda activate shot-distance
cd ..
```

### Download the weights

```bash
cd demo
mkdir -p weights
curl -o weights/siglip2_base_384_distance.pt \
    https://yosemite.ischool.berkeley.edu/david/distance/siglip2_base_384_distance.pt
cd ..
```

### Run

```bash
python3 demo/predict_movie.py \
    --video "demo/data/ia.bulldog-drummond#1929_tt0019735__107522__4484.56342___5.mp4" \
    --model demo/weights/siglip2_base_384_distance.pt \
    --output demo/predictions.jsonl \
    [--fps 4] [--batch_size 32]
```

Writes `demo/predictions.jsonl` (one line per sampled frame, in the same format as
`benchmark/predictions/*/predictions.jsonl`) and `demo/predictions.timing.json`.
