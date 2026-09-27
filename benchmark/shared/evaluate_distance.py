
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DISTANCE_ORDER = ["xcu", "cu", "mcu", "m", "ml", "l", "xls"]
RANK = {c: i for i, c in enumerate(DISTANCE_ORDER)}


def within_one(true, pred):
    if true in RANK and pred in RANK:
        return abs(RANK[true] - RANK[pred]) <= 1
    return true == pred


def ci(arr):
    return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)



def load_gold(gold_path):
    gold_regions = {}
    clip_anim = {}
    with open(gold_path) as f:
        for line in f:
            rec = json.loads(line)
            gold_regions[rec["id"]] = rec.get("regions", [])
            clip_anim[rec["id"]] = rec.get("anim")
    return gold_regions, clip_anim


def gold_label(gold_regions, clip_id, fno):
    for region in gold_regions.get(clip_id, []):
        if region["start"] <= fno <= region["end"]:
            return region["distance"]
    return None



def match_predictions(pred_path, gold_regions, clip_anim):
    y_true, y_pred, movies, anims = [], [], [], []
    skipped = 0

    with open(pred_path) as f:
        for line in f:
            rec = json.loads(line)
            gold = gold_label(gold_regions, rec["id"], rec["fno"])
            if gold is None:
                skipped += 1
                continue
            y_true.append(gold)
            y_pred.append(rec["y"])
            movies.append(rec["id"].split("__")[0])
            anims.append(clip_anim.get(rec["id"]))

    return (
        np.array(y_true), np.array(y_pred), np.array(movies), np.array(anims),
        skipped,
    )



def compute_metrics(y_true, y_pred, movies, near, mask, classes, n_bootstrap, rng):
    yt, yp, mv, nr = y_true[mask], y_pred[mask], movies[mask], near[mask]

    if len(yt) == 0:
        return {"n_frames": 0, "n_movies": 0, "acc": None, "acc_ci": None,
                "acc1": None, "acc1_ci": None, "classes": classes, "per_class": {}}

    frames_by_movie = defaultdict(list)
    for i, movie in enumerate(mv):
        frames_by_movie[movie].append(i)
    frames_by_movie = [np.array(v) for v in frames_by_movie.values()]
    n_movies = len(frames_by_movie)

    boot_acc = np.zeros(n_bootstrap)
    boot_acc1 = np.zeros(n_bootstrap)
    boot_p = np.zeros((n_bootstrap, len(classes)))
    boot_r = np.zeros((n_bootstrap, len(classes)))
    boot_f = np.zeros((n_bootstrap, len(classes)))

    for i in range(n_bootstrap):
        chosen = rng.integers(0, n_movies, size=n_movies)
        idx = np.concatenate([frames_by_movie[c] for c in chosen])
        boot_acc[i] = accuracy_score(yt[idx], yp[idx])
        boot_acc1[i] = nr[idx].mean()
        p, r, f, _ = precision_recall_fscore_support(
            yt[idx], yp[idx], labels=classes, zero_division=0, average=None
        )
        boot_p[i] = p
        boot_r[i] = r
        boot_f[i] = f

    acc = accuracy_score(yt, yp)
    acc1 = nr.mean()
    p_pts, r_pts, f_pts, support = precision_recall_fscore_support(
        yt, yp, labels=classes, zero_division=0, average=None
    )
    p_lo, p_hi = ci(boot_p)
    r_lo, r_hi = ci(boot_r)
    f_lo, f_hi = ci(boot_f)

    per_class = {}
    for j, cls in enumerate(classes):
        per_class[cls] = {
            "p": p_pts[j], "p_ci": (p_lo[j], p_hi[j]),
            "r": r_pts[j], "r_ci": (r_lo[j], r_hi[j]),
            "f": f_pts[j], "f_ci": (f_lo[j], f_hi[j]),
            "support": int(support[j]),
        }

    return {
        "n_frames": len(yt), "n_movies": n_movies,
        "acc": acc, "acc_ci": ci(boot_acc),
        "acc1": acc1, "acc1_ci": ci(boot_acc1),
        "classes": classes, "per_class": per_class,
    }


def print_report(title, metrics):
    print(f"\n{'=' * 120}")
    print(title)
    print("=" * 120)
    if metrics["n_frames"] == 0:
        print("No frames in this subset.")
        return

    print(f"Frames evaluated: {metrics['n_frames']} in {metrics['n_movies']} movies")

    acc, (acc_lo, acc_hi) = metrics["acc"], metrics["acc_ci"]
    acc1, (acc1_lo, acc1_hi) = metrics["acc1"], metrics["acc1_ci"]
    print(f"Accuracy:            {acc:.4f}  95% CI [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"Within-one accuracy: {acc1:.4f}  95% CI [{acc1_lo:.4f}, {acc1_hi:.4f}]"
          "  (adjacent distance classes count as correct)\n")

    col = "{:<30}  {:>7}  {:>20}  {:>7}  {:>20}  {:>7}  {:>20}  {:>7}"
    print(col.format("class", "P", "95% CI", "R", "95% CI", "F1", "95% CI", "support"))
    print("-" * 120)
    for cls in metrics["classes"]:
        pc = metrics["per_class"][cls]
        print(col.format(
            cls,
            f"{pc['p']:.4f}", f"[{pc['p_ci'][0]:.4f}, {pc['p_ci'][1]:.4f}]",
            f"{pc['r']:.4f}", f"[{pc['r_ci'][0]:.4f}, {pc['r_ci'][1]:.4f}]",
            f"{pc['f']:.4f}", f"[{pc['f_ci'][0]:.4f}, {pc['f_ci'][1]:.4f}]",
            pc["support"],
        ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold",        required=True)
    parser.add_argument("--pred",        required=True)
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    gold_regions, clip_anim = load_gold(args.gold)
    y_true, y_pred, movies, anims, skipped = match_predictions(
        args.pred, gold_regions, clip_anim)
    classes = sorted(set(y_true) | set(y_pred))
    near = np.array([within_one(t, p) for t, p in zip(y_true, y_pred)])

    print(f"Frames matched: {len(y_true)}  (skipped {skipped} with no gold label)")
    off_scale = [str(c) for c in classes if c not in RANK]
    if off_scale:
        print(f"Labels off the ordinal scale (must match exactly for within-one accuracy): {off_scale}")

    all_mask = np.ones(len(y_true), dtype=bool)
    print_report("ALL MOVIES", compute_metrics(
        y_true, y_pred, movies, near, all_mask, classes, args.n_bootstrap, rng))
    print_report("ANIMATION ONLY", compute_metrics(
        y_true, y_pred, movies, near, anims == "animation", classes, args.n_bootstrap, rng))
    print_report("LIVE ACTION ONLY", compute_metrics(
        y_true, y_pred, movies, near, anims == "live", classes, args.n_bootstrap, rng))


if __name__ == "__main__":
    main()
