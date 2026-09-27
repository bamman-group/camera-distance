import hashlib
import json
import os
import pathlib
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, SiglipVisionModel

CLASS_SPLITS = ("train", "dev")

DEFAULT_FPS = 30.0

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


def select_device():
    return (
        torch.device("cuda") if torch.cuda.is_available()  else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_classes(jsonl_path, splits=CLASS_SPLITS):
    labels = set()
    for rec in iter_records(jsonl_path):
        if rec.get("split") not in splits:
            continue
        for region in rec.get("regions", []):
            labels.add(region["distance"])
    classes = sorted(labels)
    print(f"Classes ({len(classes)}) from splits {list(splits)}: {classes}")
    return classes


def iter_records(jsonl_path):
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_split_records(jsonl_path, split, max_clips=None):
    records = [r for r in iter_records(jsonl_path) if r.get("split") == split]
    if max_clips:
        records = records[:max_clips]
    return records


def region_frame_labels(rec):
    frame_labels = {}
    for region in rec.get("regions", []):
        if region["end"] < region["start"]:
            print(f"WARNING: {rec['id']}: region {region} has end < start; skipping it")
            continue
        label = region.get("distance")
        for fno in range(region["start"], region["end"] + 1):
            frame_labels[fno] = label
    return frame_labels


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


RESIZE_FILTERS = ("linear", "area")


def resize_frame(frame_bgr, frame_size, resize_filter="linear"):
    height, width = frame_bgr.shape[:2]
    if resize_filter == "area" and width >= frame_size[0] and height >= frame_size[1]:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, frame_size, interpolation=interpolation)


def iter_clip_frames(cap, wanted_fnos, target_fps, frame_size, resize_filter="linear"):
    stride = clip_stride(cap, target_fps)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0 and frame_idx in wanted_fnos:
            yield frame_idx, resize_frame(frame, frame_size, resize_filter)
        frame_idx += 1


def count_sampled_frames(records, clips_dir, target_fps):
    clips_dir = pathlib.Path(clips_dir)
    n_clips = 0
    total = 0
    for rec in records:
        cap = open_clip(clips_dir / rec["id"])
        if cap is None:
            continue
        n_clips += 1
        stride = clip_stride(cap, target_fps)
        cap.release()
        total += sum(1 for fno in region_frame_labels(rec) if fno % stride == 0)
    return n_clips, total


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


def param_groups(model, lr, head_lr, weight_decay, scope="all"):
    groups = []
    for name, module in (("encoder", model.encoder), ("head", model.head)):
        group_lr = lr if name == "encoder" else head_lr
        if scope == "all":
            groups.append({"params": list(module.parameters()), "lr": group_lr,
                           "weight_decay": weight_decay, "name": f"{name}_decay"})
            continue
        groups.append({"params": [p for p in module.parameters() if p.ndim > 1],
                       "lr": group_lr, "weight_decay": weight_decay,
                       "name": f"{name}_decay"})
        groups.append({"params": [p for p in module.parameters() if p.ndim <= 1],
                       "lr": group_lr, "weight_decay": 0.0,
                       "name": f"{name}_no_decay"})
    return groups


def group_lr(optimizer, role):
    for group in optimizer.param_groups:
        if group["name"].startswith(role):
            return group["lr"]
    raise KeyError(role)


def prediction_record(clip_id, fno, classes, prob_row):
    rec = {"id": clip_id, "fno": int(fno), "y": classes[int(prob_row.argmax())]}
    for j, cls in enumerate(classes):
        rec[cls] = round(float(prob_row[j]), 5)
    return rec


class FrameCache:

    def __init__(self, frames_path, shape, samples):
        self.frames_path = frames_path
        self.shape = shape
        self.samples = samples

    def __len__(self):
        return self.shape[0]

    def open_memmap(self):
        return np.memmap(self.frames_path, dtype=np.uint8, mode="r", shape=self.shape)

    @property
    def nbytes(self):
        return int(np.prod(self.shape))


def _cache_fingerprint(records):
    payload = json.dumps([[r["id"], r.get("regions", [])] for r in records],
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_frame_cache(jsonl_path, clips_dir, split, target_fps, frame_size,
                      cache_dir, max_clips=None, rebuild=False,
                      resize_filter="linear"):
    from tqdm import tqdm

    records = load_split_records(jsonl_path, split, max_clips)
    width, height = frame_size
    meta = {
        "split": split,
        "fps": target_fps,
        "width": width,
        "height": height,
        "max_clips": max_clips,
        "resize_filter": resize_filter,
        "fingerprint": _cache_fingerprint(records),
    }

    slug = f"{split}__{width}x{height}__fps{target_fps:g}__{resize_filter}"
    if max_clips:
        slug += f"__max{max_clips}"
    cache_path = pathlib.Path(cache_dir) / slug
    frames_path = cache_path / "frames.u8"
    index_path = cache_path / "index.json"

    if not rebuild and index_path.exists() and frames_path.exists():
        try:
            index = json.loads(index_path.read_text())
            expected_bytes = index["n_frames"] * height * width * 3
            if index.get("meta") == meta and frames_path.stat().st_size == expected_bytes:
                samples = [(index["clip_ids"][c], fno, label)
                           for c, fno, label in index["samples"]]
                print(f"[{split}] reusing frame cache: {index['n_frames']} frames "
                      f"({expected_bytes / 1e9:.1f} GB) at {cache_path}")
                return FrameCache(frames_path, (index["n_frames"], height, width, 3), samples)
            reason = "does not match this run"
        except (ValueError, KeyError, TypeError) as exc:
            reason = f"is unreadable ({type(exc).__name__})"
        print(f"[{split}] frame cache at {cache_path} {reason}; rebuilding")

    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"[{split}] {len(records)} clips -> building frame cache at {cache_path}")

    clips_dir = pathlib.Path(clips_dir)
    clip_ids, clip_idx = [], {}
    samples = []
    missing = unreadable = empty = 0

    tmp_frames = cache_path / f"frames.{os.getpid()}.partial"
    try:
        with open(tmp_frames, "wb") as out:
            for rec in tqdm(records, desc=f"Caching {split}"):
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

                before = len(samples)
                for fno, rgb in iter_clip_frames(cap, frame_labels, target_fps,
                                                 frame_size, resize_filter):
                    if frame_labels[fno] is None:
                        print(f"WARNING: {rec['id']}: frame {fno} has no distance label; skipping")
                        continue
                    out.write(np.ascontiguousarray(rgb).tobytes())
                    if rec["id"] not in clip_idx:
                        clip_idx[rec["id"]] = len(clip_ids)
                        clip_ids.append(rec["id"])
                    samples.append((clip_idx[rec["id"]], fno, frame_labels[fno]))
                cap.release()
                if len(samples) == before:
                    print(f"WARNING: {rec['id']}: opened but decoded 0 labeled frames")
                    empty += 1

        tmp_frames.replace(frames_path)
    except BaseException:
        tmp_frames.unlink(missing_ok=True)
        raise

    n_frames = len(samples)
    tmp_index = cache_path / f"index.{os.getpid()}.json.tmp"
    tmp_index.write_text(json.dumps({
        "meta": meta,
        "n_frames": n_frames,
        "height": height,
        "width": width,
        "clip_ids": clip_ids,
        "samples": samples,
    }))
    tmp_index.replace(index_path)

    print(f"[{split}] {n_frames} frames cached "
          f"({n_frames * height * width * 3 / 1e9:.1f} GB); "
          f"skipped {missing} missing, {unreadable} unreadable, {empty} empty clips")
    return FrameCache(frames_path, (n_frames, height, width, 3),
                      [(clip_ids[c], fno, label) for c, fno, label in samples])
