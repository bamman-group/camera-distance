#!/usr/bin/env python3
"""
Run the shot-distance classifier on an arbitrary input movie.

Samples the video at a target fps (4 by default), resizes/normalizes each
frame the way the model was trained, and writes per-frame class probabilities
to a jsonl file.

Requires a trained checkpoint (a .pt file, e.g. demo/weights/siglip2_base_384_distance.pt).

Usage (run from repo root):
    python3 demo/predict_movie.py \
        --video demo/data/ia.bulldog.drummond.clip.mp4 \
        --model demo/weights/siglip2_base_384_distance.pt \
        --output demo/predictions.jsonl \
        [--fps 4] [--batch_size 32]
"""

import argparse
import json
import pathlib
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = REPO_ROOT / "demo" / "data" / \
    "ia.bulldog.drummond.clip.mp4"
DEFAULT_MODEL = REPO_ROOT / "demo" / "weights" / "siglip2_base_384_distance.pt"

DEFAULT_FPS = 30.0
RESIZE_FILTERS = ("linear", "area")

ARCH_CHECKPOINTS = {
    "siglip2_base_224":   "google/siglip2-base-patch16-224",
    "siglip2_base_384":   "google/siglip2-base-patch16-384",
    "siglip2_so400m_384": "google/siglip2-so400m-patch14-384",
    "convnext_large":     "facebook/convnext-large-224",
    "swin_b":             "microsoft/swin-base-patch4-window7-224",
    "resnet50":           "microsoft/resnet-50",
    "vit_b_16":           "google/vit-base-patch16-224-in21k",
    "vit_l_16":           "google/vit-large-patch16-224-in21k",
}
_SIGLIP2_ARCHES = {"siglip2_base_224", "siglip2_base_384", "siglip2_so400m_384"}


class BackboneClassifier(nn.Module):

    def __init__(self, arch, num_classes, frame_size):
        super().__init__()
        checkpoint = ARCH_CHECKPOINTS[arch]
        encoder_cls = SiglipVisionModel if arch in _SIGLIP2_ARCHES else AutoModel
        self.encoder = encoder_cls.from_pretrained(checkpoint)
        width, height = frame_size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            hidden_size = self.encoder(pixel_values=dummy).pooler_output.flatten(1).shape[-1]
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        return self.head(outputs.pooler_output.flatten(1))


def select_device():
    return (
        torch.device("cuda") if torch.cuda.is_available()  else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )


def resolve_checkpoint_path(model):
    path = pathlib.Path(model)
    if path.is_file():
        return path
    return path / "best_model.pt"


def open_clip(clip_path):
    clip_path = pathlib.Path(clip_path)
    if not clip_path.exists():
        return None
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def clip_stride(cap, target_fps):
    clip_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (clip_fps and clip_fps > 0):
        clip_fps = DEFAULT_FPS
    return max(1, round(clip_fps / target_fps))


def resize_frame(frame_bgr, frame_size, resize_filter="linear"):
    height, width = frame_bgr.shape[:2]
    if resize_filter == "area" and width >= frame_size[0] and height >= frame_size[1]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, frame_size, interpolation=interpolation)


def iter_movie_frames(video_path, target_fps, frame_size, resize_filter):
    cap = open_clip(video_path)
    if cap is None:
        raise SystemExit(f"ERROR: could not open video {video_path}")
    stride = clip_stride(cap, target_fps)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            yield frame_idx, resize_frame(frame, frame_size, resize_filter)
        frame_idx += 1
    cap.release()


def processor_frame_size(processor):
    size = getattr(processor, "crop_size", None) or processor.size
    if "width" in size and "height" in size:
        return int(size["width"]), int(size["height"])
    if "shortest_edge" in size:
        edge = int(size["shortest_edge"])
        return edge, edge
    raise ValueError(f"Cannot determine input size from processor size={size}")


def normalization_stats(processor):
    rescale = float(getattr(processor, "rescale_factor", 1.0 / 255.0)) if getattr(processor, "do_rescale", True) else 1.0
    if getattr(processor, "do_normalize", True):
        mean = [float(m) for m in processor.image_mean]
        std = [float(s) for s in processor.image_std]
    else:
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    return rescale, mean, std


def make_normalizer(rescale, mean, std, device):
    mean_t = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def normalize(batch_uint8):
        x = batch_uint8.to(dtype=torch.float32)
        if rescale != 1.0:
            x = x.mul_(rescale)
        return x.sub_(mean_t).div_(std_t)

    return normalize


def prediction_record(clip_id, fno, classes, prob_row):
    rec = {"id": clip_id, "fno": int(fno), "y": classes[int(prob_row.argmax())]}
    for j, cls in enumerate(classes):
        rec[cls] = round(float(prob_row[j]), 5)
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="a checkpoint .pt file, or a directory containing best_model.pt")
    parser.add_argument("--output", default=str(REPO_ROOT / "demo" / "predictions.jsonl"))
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resize_filter", choices=RESIZE_FILTERS, default=None,
                        help="override the downsampling filter; defaults to the one "
                             "recorded in the checkpoint")
    args = parser.parse_args()

    video_path = pathlib.Path(args.video)
    checkpoint_path = resolve_checkpoint_path(args.model)
    if not checkpoint_path.exists():
        raise SystemExit(
            f"ERROR: no checkpoint at {checkpoint_path}\n"
            f"Train one with benchmark/models/classifiers/finetune.py, or point "
            f"--model at a checkpoint .pt file or a directory containing best_model.pt."
        )

    device = select_device()
    print(f"Device: {device} | fp32")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    arch = ckpt["arch"]
    model_name = ckpt.get("model_name") or ARCH_CHECKPOINTS[arch]
    resize_filter = args.resize_filter or ckpt.get("resize_filter", "linear")
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

    clip_id = video_path.name
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    start = time.time()

    with open(output_path, "w") as out_f, torch.no_grad(), \
         tqdm(desc=f"Predicting {clip_id}", unit="frame") as pbar:
        batch = []

        def flush():
            nonlocal n_frames
            if not batch:
                return
            frames = np.stack([b[1] for b in batch])
            pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2)
            pixel_values = normalize(pixel_values.contiguous().to(device))
            logits = model(pixel_values)
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            for (fno, _), prob_row in zip(batch, probs):
                out_f.write(json.dumps(
                    prediction_record(clip_id, fno, classes, prob_row)) + "\n")
            pbar.update(len(batch))
            n_frames += len(batch)
            batch.clear()

        for fno, frame in iter_movie_frames(video_path, args.fps, frame_size, resize_filter):
            batch.append((fno, frame))
            if len(batch) == args.batch_size:
                flush()
        flush()

    elapsed = time.time() - start
    timing_path = output_path.with_suffix(".timing.json")
    with open(timing_path, "w") as f:
        json.dump({
            "video":          str(video_path),
            "model":          args.model,
            "arch":           arch,
            "output":         str(output_path),
            "fps":            args.fps,
            "resize_filter":  resize_filter,
            "n_frames":       n_frames,
            "predict_time_s": round(elapsed, 1),
        }, f, indent=2)

    print(f"Prediction time: {elapsed:.1f}s")
    print(f"Predictions written to {output_path}")
    print(f"Timing written to {timing_path}")

    if n_frames == 0:
        raise SystemExit(f"ERROR: predicted 0 frames from {video_path}")


if __name__ == "__main__":
    main()
