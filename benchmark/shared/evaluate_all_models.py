
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
from sklearn.metrics import confusion_matrix

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "models" / "classifiers"))

from evaluate_distance import (
    compute_metrics, load_gold, match_predictions, within_one,
)

try:
    from common import ARCH_CHECKPOINTS
except ImportError:
    ARCH_CHECKPOINTS = {}


def discover_archs(checkpoints_dir):
    found = sorted(
        p.name for p in checkpoints_dir.iterdir()
        if p.is_dir() and (p / "best_model.pt").exists()
    )
    known_order = {arch: i for i, arch in enumerate(ARCH_CHECKPOINTS)}
    return sorted(found, key=lambda a: (known_order.get(a, len(known_order)), a))


def run_step(name, cmd):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited {result.returncode}")


def ensure_predictions(arch, model_dir, pred_dir, args):
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "predictions.jsonl"
    smoothed_file = pred_dir / "predictions.smoothed.jsonl"

    if args.rebuild or not (pred_file.exists() and pred_file.stat().st_size > 0):
        run_step("predict.py", [
            sys.executable, str(REPO_ROOT / "models" / "classifiers" / "predict.py"),
            "--jsonl", args.jsonl,
            "--clips", args.clips,
            "--model_dir", str(model_dir),
            "--split", args.split,
            "--output", str(pred_file),
            "--fps", str(args.fps),
            "--batch_size", str(args.batch_size),
        ])
    else:
        print(f"  {pred_file} exists, skipping predict.py")

    if args.rebuild or not (smoothed_file.exists() and smoothed_file.stat().st_size > 0):
        run_step("smooth_predictions.py", [
            sys.executable, str(REPO_ROOT / "shared" / "smooth_predictions.py"),
            "--pred", str(pred_file),
            "--output", str(smoothed_file),
            "--window", str(args.window),
        ])
    else:
        print(f"  {smoothed_file} exists, skipping smooth_predictions.py")

    return smoothed_file


def read_train_time_minutes(model_dir):
    results_path = model_dir / "train_results.json"
    if not results_path.exists():
        return None
    try:
        train_time_s = json.loads(results_path.read_text())["train_time_s"]
    except (json.JSONDecodeError, KeyError):
        return None
    return train_time_s / 60.0


def fmt_ci(lo, hi):
    return f"[{lo:.4f}, {hi:.4f}]"


def write_confusion_csv(path, y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    with open(path, "w") as f:
        f.write("true,pred,count\n")
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                f.write(f"{true_cls},{pred_cls},{cm[i, j]}\n")


def per_class_table_md(metrics):
    lines = [
        "| class | P | 95% CI | R | 95% CI | F1 | 95% CI |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls in metrics["classes"]:
        pc = metrics["per_class"][cls]
        lines.append(
            f"| {cls} | {pc['p']:.4f} | {fmt_ci(*pc['p_ci'])} "
            f"| {pc['r']:.4f} | {fmt_ci(*pc['r_ci'])} "
            f"| {pc['f']:.4f} | {fmt_ci(*pc['f_ci'])} |"
        )
    return "\n".join(lines)


def breakdown_section(heading, metrics):
    if metrics["n_frames"] == 0:
        return f"**{heading}** — no frames in this subset.\n"
    acc, (lo, hi) = metrics["acc"], metrics["acc_ci"]
    caption = (f"**{heading}** — accuracy {acc:.4f} 95% CI {fmt_ci(lo, hi)} "
               f"({metrics['n_frames']} frames, {metrics['n_movies']} movies)")
    return f"{caption}\n\n{per_class_table_md(metrics)}\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", required=True)
    parser.add_argument("--jsonl",           required=True)
    parser.add_argument("--clips",           required=True)
    parser.add_argument("--predictions_dir", required=True)
    parser.add_argument("--report_dir",      default=None,
                        help="default: <predictions_dir>/report")
    parser.add_argument("--split",           default="test")
    parser.add_argument("--fps",             type=float, default=4.0)
    parser.add_argument("--batch_size",      type=int, default=32)
    parser.add_argument("--window",          type=int, default=12,
                        help="smooth_predictions.py's --window")
    parser.add_argument("--n_bootstrap",     type=int, default=1000)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--rebuild",         action="store_true",
                        help="re-run predict.py/smooth_predictions.py even if their output already exists")
    args = parser.parse_args()

    checkpoints_dir = pathlib.Path(args.checkpoints_dir)
    predictions_dir = pathlib.Path(args.predictions_dir)
    report_dir = pathlib.Path(args.report_dir) if args.report_dir else predictions_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    archs = discover_archs(checkpoints_dir)
    if not archs:
        raise SystemExit(f"No model directories with best_model.pt found under {checkpoints_dir}")
    print(f"Models found in {checkpoints_dir}: {archs}")

    gold_regions, clip_anim = load_gold(args.jsonl)
    rng = np.random.default_rng(args.seed)

    summary_rows = []
    per_model_sections = []
    failed = []

    for arch in archs:
        print(f"\n=== {arch} ===")
        model_dir = checkpoints_dir / arch
        try:
            smoothed_file = ensure_predictions(arch, model_dir, predictions_dir / arch, args)

            y_true, y_pred, movies, anims, skipped = match_predictions(
                str(smoothed_file), gold_regions, clip_anim)
            if len(y_true) == 0:
                raise RuntimeError(f"0 frames matched gold labels (skipped {skipped})")
            classes = sorted(set(y_true) | set(y_pred))
            near = np.array([within_one(t, p) for t, p in zip(y_true, y_pred)])
            all_mask = np.ones(len(y_true), dtype=bool)

            metrics_total = compute_metrics(y_true, y_pred, movies, near, all_mask,
                                            classes, args.n_bootstrap, rng)
            metrics_anim = compute_metrics(y_true, y_pred, movies, near, anims == "animation",
                                           classes, args.n_bootstrap, rng)
            metrics_live = compute_metrics(y_true, y_pred, movies, near, anims == "live",
                                           classes, args.n_bootstrap, rng)

            train_time_min = read_train_time_minutes(model_dir)

            write_confusion_csv(report_dir / f"{arch}.confusion.csv",
                                y_true, y_pred, classes)

            acc, (acc_lo, acc_hi) = metrics_total["acc"], metrics_total["acc_ci"]
            acc1, (acc1_lo, acc1_hi) = metrics_total["acc1"], metrics_total["acc1_ci"]
            summary_rows.append({
                "arch": arch,
                "acc": f"{acc:.4f} {fmt_ci(acc_lo, acc_hi)}",
                "acc1": f"{acc1:.4f} {fmt_ci(acc1_lo, acc1_hi)}",
                "train_min": f"{train_time_min:.1f}" if train_time_min is not None else "N/A",
            })

            per_model_sections.append(
                f"### {arch}\n\n"
                + breakdown_section("Total", metrics_total) + "\n"
                + breakdown_section("Animated", metrics_anim) + "\n"
                + breakdown_section("Live-action", metrics_live)
            )
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failed.append((arch, str(exc)))

    lines = ["# Model evaluation report", ""]
    lines.append(f"Split: `{args.split}` | bootstrap: {args.n_bootstrap} resamples, seed {args.seed} "
                 f"| smoothing window: {args.window}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| model | accuracy [95% CI] | within-one accuracy [95% CI] | training time (min) |")
    lines.append("|---|---|---|---|")
    for row in summary_rows:
        lines.append(f"| {row['arch']} | {row['acc']} | {row['acc1']} | {row['train_min']} |")
    if failed:
        lines.append("")
        lines.append("**Failed (excluded above):** " + ", ".join(f"`{a}` ({e})" for a, e in failed))
    lines.append("")
    lines.append("## Per-model breakdowns")
    lines.append("")
    lines.extend(per_model_sections)

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    print(f"Confusion-matrix CSVs written to {report_dir}/<arch>.confusion.csv")

    if failed:
        raise SystemExit(f"{len(failed)} of {len(archs)} model(s) failed: "
                         f"{', '.join(a for a, _ in failed)}")


if __name__ == "__main__":
    main()
