import argparse
import json
import pathlib
import time

import numpy as np
import torch
from transformers import AutoImageProcessor
from tqdm import tqdm

from common import (
    ARCH_CHECKPOINTS, RESIZE_FILTERS, BackboneClassifier, count_sampled_frames,
    iter_clip_frames, load_split_records, make_normalizer,
    normalization_stats, open_clip, prediction_record, processor_frame_size,
    region_frame_labels, select_device,
)


def iter_region_frames(records, clips_dir, target_fps, frame_size, resize_filter):
    clips_dir = pathlib.Path(clips_dir)
    missing = unreadable = empty = 0

    for rec in records:
        frame_labels = region_frame_labels(rec)
        if not frame_labels:
            continue
        clip_path = clips_dir / rec["id"]
        if not clip_path.exists():
            missing += 1
            continue
        cap = open_clip(clip_path)
        if cap is None:
            print(f"WARNING: {rec['id']}: unreadable video, skipping")
            unreadable += 1
            continue

        yielded = 0
        for fno, rgb in iter_clip_frames(cap, frame_labels, target_fps,
                                         frame_size, resize_filter):
            yield rec["id"], fno, rgb
            yielded += 1
        cap.release()
        if yielded == 0:
            print(f"WARNING: {rec['id']}: opened but decoded 0 frames in its regions")
            empty += 1

    if missing or unreadable or empty:
        print(f"Skipped {missing} missing, {unreadable} unreadable, {empty} empty clips")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",        required=True)
    parser.add_argument("--clips",        required=True)
    parser.add_argument("--model_dir",    required=True)
    parser.add_argument("--split",        default="test")
    parser.add_argument("--output",       required=True)
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--fps",          type=float, default=4.0)
    parser.add_argument("--resize_filter", choices=RESIZE_FILTERS, default=None,
                        help="override the downsampling filter; defaults to the one recorded "
                             "in the checkpoint, which is almost always what you want")
    args = parser.parse_args()

    device = select_device()
    print(f"Device: {device} | fp32")

    ckpt = torch.load(pathlib.Path(args.model_dir) / "best_model.pt",
                      map_location=device, weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt["arch"]
    model_name = ckpt.get("model_name") or ARCH_CHECKPOINTS[arch]
    resize_filter = args.resize_filter or ckpt.get("resize_filter", "linear")
    if args.resize_filter and args.resize_filter != ckpt.get("resize_filter", "linear"):
        print(f"WARNING: overriding the checkpoint's resize_filter "
              f"({ckpt.get('resize_filter', 'linear')}) with {args.resize_filter}; "
              f"the model will see frames filtered differently from training")
    print(f"Arch: {arch} ({model_name}) | Classes ({len(classes)}): {classes} "
          f"| resize_filter: {resize_filter}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    frame_size = processor_frame_size(processor)
    rescale, mean, std = normalization_stats(processor)
    normalize = make_normalizer(rescale, mean, std, device)
    model = BackboneClassifier(arch, num_classes=len(classes),
                               frame_size=frame_size).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    records = load_split_records(args.jsonl, args.split)
    n_clips, n_target_frames = count_sampled_frames(records, args.clips, args.fps)
    print(f"[{args.split}] {n_clips} clips, {n_target_frames} frames to predict at {args.fps} fps")

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    start = time.time()

    with open(output_path, "w") as out_f, torch.no_grad(), \
         tqdm(total=n_target_frames, desc=f"Predicting {args.split}", unit="frame") as pbar:
        batch = []

        def flush():
            nonlocal n_frames
            if not batch:
                return
            frames = np.stack([b[2] for b in batch])
            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2)
            pixel_values = normalize(pixel_values.contiguous().to(device))
            logits = model(pixel_values)
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for (clip_id, fno, _), prob_row in zip(batch, probs):
                out_f.write(json.dumps(
                    prediction_record(clip_id, fno, classes, prob_row)) + "\n")
            pbar.update(len(batch))
            n_frames += len(batch)
            batch.clear()

        for clip_id, fno, frame in iter_region_frames(records, args.clips, args.fps,
                                                      frame_size, resize_filter):
            batch.append((clip_id, fno, frame))
            if len(batch) == args.batch_size:
                flush()
        flush()

    elapsed = time.time() - start
    timing_path = output_path.with_suffix(".timing.json")
    with open(timing_path, "w") as f:
        json.dump({
            "model_dir":       args.model_dir,
            "arch":            arch,
            "split":           args.split,
            "output":          args.output,
            "fps":             args.fps,
            "resize_filter":   resize_filter,
            "n_frames":        n_frames,
            "predict_time_s":  round(elapsed, 1),
        }, f, indent=2)
    print(f"Prediction time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Predictions written to {args.output}")
    print(f"Timing written to {timing_path}")

    if n_frames == 0:
        raise SystemExit(
            f"ERROR: predicted 0 frames. No clip in split '{args.split}' of "
            f"{args.jsonl} was found under {args.clips} with a usable region."
        )


if __name__ == "__main__":
    main()
