
import argparse
import json
import sys
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pred",   required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--window", type=int, default=12,
                    help="frames on each side of the target to average over")
args = parser.parse_args()

by_clip = defaultdict(list)
classes = None

with open(args.pred) as f:
    for line in f:
        rec = json.loads(line)
        if classes is None:
            classes = sorted(k for k in rec if k not in ("id", "fno", "y"))
        by_clip[rec["id"]].append((rec["fno"], [rec[c] for c in classes]))

if not by_clip:
    sys.exit("No predictions found in input file.")

n_frames = sum(len(v) for v in by_clip.values())
print(f"{n_frames} frames in {len(by_clip)} clips | classes: {classes}")

changed = 0

with open(args.output, "w") as out_f:
    for clip_id, frames in by_clip.items():
        frames.sort()
        fnos  = np.array([fno for fno, _ in frames])
        probs = np.array([p for _, p in frames])

        lo  = np.searchsorted(fnos, fnos - args.window, side="left")
        hi  = np.searchsorted(fnos, fnos + args.window, side="right")
        cum = np.vstack([np.zeros(len(classes)), np.cumsum(probs, axis=0)])
        smoothed = (cum[hi] - cum[lo]) / (hi - lo)[:, None]

        changed += int((probs.argmax(axis=1) != smoothed.argmax(axis=1)).sum())

        for i, fno in enumerate(fnos):
            rec = {"id": clip_id, "fno": int(fno), "y": classes[smoothed[i].argmax()]}
            for j, cls in enumerate(classes):
                rec[cls] = round(float(smoothed[i, j]), 5)
            out_f.write(json.dumps(rec) + "\n")

print(f"Smoothed predictions written to {args.output}")
print(f"Labels changed by smoothing: {changed} / {n_frames} ({changed / n_frames:.2%})")
