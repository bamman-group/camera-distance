import argparse
import json
import math
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from common import (
    ARCH_CHECKPOINTS, CLASS_SPLITS, RESIZE_FILTERS, BackboneClassifier,
    build_classes, build_frame_cache, group_lr, make_normalizer,
    normalization_stats, param_groups, prediction_record,
    processor_frame_size, seed_everything, seed_worker, select_device,
)

RESUME_CRITICAL = (
    "arch", "epochs", "lr_schedule", "batch_size", "warmup_frac", "lr",
    "head_lr", "min_lr", "freeze_epochs", "rewarmup_on_unfreeze",
    "weight_decay", "weight_decay_scope", "label_smoothing", "max_grad_norm",
    "fps", "resize_filter", "max_clips", "seed",
)


class FrameDataset(Dataset):

    def __init__(self, cache, class_to_idx, augment=False):
        self.frames_path = cache.frames_path
        self.shape = cache.shape
        self.samples = [(row, clip_id, fno, class_to_idx[label])
                        for row, (clip_id, fno, label) in enumerate(cache.samples)]
        self.augment = augment
        self.jitter_pil = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
        ) if augment else None
        self._frames = None

    def __len__(self):
        return len(self.samples)

    def _memmap(self):
        if self._frames is None:
            self._frames = np.memmap(self.frames_path, dtype=np.uint8,
                                     mode="r", shape=self.shape)
        return self._frames

    def __getitem__(self, idx):
        row, clip_id, fno, label = self.samples[idx]
        frame = self._memmap()[row]
        if self.augment and random.random() < 0.5:
            frame = frame[:, ::-1]
        if self.jitter_pil is not None:
            frame = np.asarray(self.jitter_pil(Image.fromarray(np.ascontiguousarray(frame))))
        pixel_values = torch.from_numpy(frame.transpose(2, 0, 1).copy())
        return pixel_values.contiguous(), label, clip_id, fno


def evaluate(model, loader, device, classes, normalize,
             predictions_path=None):
    model.eval()
    all_preds, all_labels = [], []
    out_f = open(predictions_path, "w") if predictions_path else None

    with torch.no_grad():
        for batch, labels, clip_ids, fnos in tqdm(loader, desc="Eval", leave=False):
            pixel_values = normalize(batch.to(device, non_blocking=True))
            logits = model(pixel_values)
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            all_preds.extend(probs.argmax(dim=-1).tolist())
            all_labels.extend(labels.tolist())

            if out_f:
                for clip_id, fno, prob_row in zip(clip_ids, fnos.tolist(), probs):
                    out_f.write(json.dumps(
                        prediction_record(clip_id, fno, classes, prob_row)) + "\n")

    if out_f:
        out_f.close()

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        zero_division=0,
    )
    return acc, report


def truncate_metrics(metrics_path, last_epoch):
    if not metrics_path.exists():
        return
    lines = [l for l in metrics_path.read_text().splitlines() if l.strip()]
    kept = [l for l in lines if json.loads(l)["epoch"] <= last_epoch]
    if len(kept) == len(lines):
        return
    metrics_path.write_text("".join(l + "\n" for l in kept))
    print(f"Dropped {len(lines) - len(kept)} train_metrics.jsonl line(s) for epoch(s) "
          f"past {last_epoch}, which the checkpoint being resumed does not cover")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",         required=True, choices=sorted(ARCH_CHECKPOINTS),
                        help="model architecture; resolves to a fixed HF checkpoint (see ARCH_CHECKPOINTS)")
    parser.add_argument("--jsonl",        required=True)
    parser.add_argument("--clips",        required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--fps",          type=float, default=4.0)
    parser.add_argument("--epochs",       type=int,   default=50, help="under cosine this IS the schedule length; under plateau it is a max budget")
    parser.add_argument("--lr_schedule",  choices=["cosine", "plateau"], default="plateau",
                        help="plateau: warmup then ReduceLROnPlateau on dev accuracy. "
                             "cosine: linear warmup then cosine decay to --min_lr across all --epochs.")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=2e-5, help="encoder learning rate")
    parser.add_argument("--head_lr",      type=float, default=1e-3, help="classification head learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--weight_decay_scope", choices=["all", "weights_only"], default="all",
                        help="all = decay every parameter; "
                             "weights_only = exclude biases and normalization gains")
    parser.add_argument("--warmup_frac",  type=float, default=0.05, help="fraction of --epochs' worth of steps used for linear warmup")
    parser.add_argument("--plateau_factor",   type=float, default=0.5, help="multiply both LRs by this when dev accuracy plateaus")
    parser.add_argument("--plateau_patience", type=int,   default=4, help="epochs with no dev accuracy improvement before reducing LR")
    parser.add_argument("--plateau_threshold", type=float, default=1e-3, help="minimum dev accuracy improvement (absolute) to reset plateau patience")
    parser.add_argument("--plateau_cooldown", type=int,   default=0, help="epochs to wait after an LR reduction before resuming plateau tracking")
    parser.add_argument("--min_lr",       type=float, default=0.0, help="floor for both LRs under ReduceLROnPlateau")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="stop after this many epochs with no new best dev accuracy; "
                             "0 disables it, which is what a cosine budget wants -- stopping "
                             "early leaves the model un-annealed mid-decay")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient-norm clipping; 0 disables")
    parser.add_argument("--freeze_epochs", type=int, default=0,
                        help="train only the head for the first N epochs, then unfreeze the encoder (LP-FT)")
    parser.add_argument("--resize_filter", choices=RESIZE_FILTERS, default="linear",
                        help="downsampling filter; part of the frame-cache key, so switching re-decodes")
    parser.add_argument("--rewarmup_on_unfreeze", action="store_true",
                        help="restart LR warmup when the encoder unfreezes (LP-FT); off means "
                             "the encoder jumps straight to --lr")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--cache_dir",    default=None,
                        help="where to keep decoded frames (default: <output_dir>/frame_cache)")
    parser.add_argument("--rebuild_cache", action="store_true", help="re-decode frames even if a cache exists")
    parser.add_argument("--resume",       action="store_true", help="continue from last_checkpoint.pt if present")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_clips",    type=int,   default=None, help="limit clips per split (for smoke-testing)")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    model_name = ARCH_CHECKPOINTS[args.arch]

    output_dir = pathlib.Path(args.output_dir) / args.arch
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else pathlib.Path(args.output_dir) / "frame_cache"
    print(f"Output directory: {output_dir}")

    device = select_device()
    print(f"Device: {device} | fp32")
    print(f"Arch: {args.arch} ({model_name})")
    print(f"LR schedule: {args.lr_schedule} over {args.epochs} epochs "
          f"(warmup_frac={args.warmup_frac})")

    classes = build_classes(args.jsonl, CLASS_SPLITS)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print("Loading processor and model...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    frame_size = processor_frame_size(processor)
    rescale, mean, std = normalization_stats(processor)
    print(f"Input size: {frame_size} | rescale={rescale:g} mean={mean} std={std}")
    normalize = make_normalizer(rescale, mean, std, device)
    model = BackboneClassifier(args.arch, num_classes=len(classes),
                               frame_size=frame_size).to(device)

    train_cache = build_frame_cache(args.jsonl, args.clips, "train", args.fps,
                                    frame_size, cache_dir, args.max_clips,
                                    args.rebuild_cache, args.resize_filter)
    dev_cache = build_frame_cache(args.jsonl, args.clips, "dev", args.fps,
                                  frame_size, cache_dir, args.max_clips,
                                  args.rebuild_cache, args.resize_filter)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        FrameDataset(train_cache, class_to_idx, augment=True), shuffle=True,
        generator=train_generator, **loader_kwargs)
    dev_loader = DataLoader(
        FrameDataset(dev_cache, class_to_idx), shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        param_groups(model, args.lr, args.head_lr, args.weight_decay,
                     args.weight_decay_scope))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    warmup_start_step = 0

    def in_warmup(step):
        return step - warmup_start_step <= warmup_steps

    def cosine_scale(step):
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    def set_lr(step):
        if args.lr_schedule == "cosine":
            scale = cosine_scale(step)
            if in_warmup(step):
                scale *= (step - warmup_start_step) / warmup_steps
        elif in_warmup(step):
            scale = (step - warmup_start_step) / warmup_steps
        else:
            return
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = max(args.min_lr, base_lr * scale)

    def new_plateau_scheduler():
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=args.plateau_factor,
            patience=args.plateau_patience, threshold=args.plateau_threshold,
            threshold_mode="abs", cooldown=args.plateau_cooldown, min_lr=args.min_lr,
        )

    plateau_scheduler = new_plateau_scheduler()

    def set_encoder_frozen(frozen):
        for p in model.encoder.parameters():
            p.requires_grad = not frozen
        if frozen:
            model.encoder.eval()
        else:
            model.encoder.train()

    best_acc = -1.0
    best_epoch = 0
    epoch = 0
    epochs_since_improvement = 0
    global_step = 0
    start_epoch = 1
    elapsed_before = 0.0
    encoder_frozen = args.freeze_epochs > 0
    if encoder_frozen:
        set_encoder_frozen(True)
        print(f"Encoder frozen for the first {args.freeze_epochs} epoch(s) (head only)")

    last_ckpt_path = output_dir / "last_checkpoint.pt"
    metrics_path = output_dir / "train_metrics.jsonl"

    if args.resume and last_ckpt_path.exists():
        state = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        if state["classes"] != classes or state["arch"] != args.arch:
            raise SystemExit(
                f"Cannot resume: {last_ckpt_path} was written for "
                f"arch={state['arch']} classes={state['classes']}, "
                f"but this run has arch={args.arch} classes={classes}."
            )
        drift = {k: (v, getattr(args, k)) for k, v in state.get("run_config", {}).items()
                 if getattr(args, k, None) != v}
        if drift:
            detail = "\n".join(f"    --{k}: checkpoint={was!r} but this run has {now!r}"
                               for k, (was, now) in sorted(drift.items()))
            raise SystemExit(
                f"Cannot resume: {last_ckpt_path} was written under a different "
                f"configuration.\n{detail}\n"
                f"  Either restore those values, or start a fresh run without --resume."
            )
        encoder_frozen = state["encoder_frozen"]
        set_encoder_frozen(encoder_frozen)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        plateau_scheduler.load_state_dict(state["plateau"])
        best_acc = state["best_acc"]
        best_epoch = state["best_epoch"]
        epochs_since_improvement = state["epochs_since_improvement"]
        global_step = state["global_step"]
        warmup_start_step = state["warmup_start_step"]
        elapsed_before = state["train_time_s"]
        start_epoch = state["epoch"] + 1
        epoch = state["epoch"]
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
        if state["rng"].get("train_generator") is not None:
            train_generator.set_state(state["rng"]["train_generator"])
        truncate_metrics(metrics_path, state["epoch"])
        print(f"Resumed from epoch {state['epoch']} "
              f"(best_dev_acc={best_acc:.4f} at epoch {best_epoch}); "
              f"continuing at epoch {start_epoch}")
    elif args.resume:
        print(f"--resume given but {last_ckpt_path} does not exist; starting fresh")

    def write_results(completed, stop_reason=None):
        with open(output_dir / "train_results.json", "w") as f:
            json.dump({
                "arch":                args.arch,
                "model_name":          model_name,
                "max_epochs":          args.epochs,
                "epochs_run":          epoch,
                "best_epoch":          best_epoch,
                "batch_size":          args.batch_size,
                "lr":                  args.lr,
                "head_lr":             args.head_lr,
                "weight_decay":        args.weight_decay,
                "weight_decay_scope":  args.weight_decay_scope,
                "lr_schedule":         args.lr_schedule,
                "warmup_frac":         args.warmup_frac,
                "plateau_factor":      args.plateau_factor,
                "plateau_patience":    args.plateau_patience,
                "plateau_threshold":   args.plateau_threshold,
                "plateau_cooldown":    args.plateau_cooldown,
                "min_lr":              args.min_lr,
                "early_stop_patience": args.early_stop_patience,
                "label_smoothing":     args.label_smoothing,
                "max_grad_norm":       args.max_grad_norm,
                "freeze_epochs":       args.freeze_epochs,
                "resize_filter":       args.resize_filter,
                "rewarmup_on_unfreeze": args.rewarmup_on_unfreeze,
                "num_workers":         args.num_workers,
                "fps":                 args.fps,
                "seed":                args.seed,
                "train_frames":        len(train_cache),
                "dev_frames":          len(dev_cache),
                "train_time_s":        round(elapsed_before + time.time() - train_start, 1),
                "best_dev_acc":        round(best_acc, 4),
                "completed":           completed,
                "stop_reason":         stop_reason,
            }, f, indent=2)

    train_start = time.time()
    stop_reason = "epoch budget reached"
    trainable = [p for p in model.parameters() if p.requires_grad]

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        if encoder_frozen and epoch > args.freeze_epochs:
            set_encoder_frozen(False)
            encoder_frozen = False
            trainable = [p for p in model.parameters() if p.requires_grad]
            print("Encoder unfrozen")
            plateau_scheduler = new_plateau_scheduler()
            epochs_since_improvement = 0
            if args.rewarmup_on_unfreeze:
                warmup_start_step = global_step
                print("Warmup restarted for the unfrozen encoder")

        model.train()
        if encoder_frozen:
            model.encoder.eval()
        total_loss = 0.0
        for batch, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            pixel_values = normalize(batch.to(device, non_blocking=True))
            labels = labels.to(device, non_blocking=True)
            global_step += 1
            set_lr(global_step)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(pixel_values), labels)
            loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        dev_pred_tmp = output_dir / "dev_predictions.jsonl.tmp"
        dev_acc, dev_report = evaluate(model, dev_loader, device, classes,
                                       normalize, dev_pred_tmp)
        encoder_lr, head_lr = group_lr(optimizer, "encoder"), group_lr(optimizer, "head")
        print(f"\nEpoch {epoch} | loss={avg_loss:.4f} | dev_acc={dev_acc:.4f} "
              f"| lr={encoder_lr:.2e}/{head_lr:.2e}")
        print(dev_report)

        if dev_acc > best_acc:
            best_acc = dev_acc
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "classes":    classes,
                "arch":       args.arch,
                "model_name": model_name,
                "resize_filter": args.resize_filter,
            }, output_dir / "best_model.pt")
            dev_pred_tmp.replace(output_dir / "dev_predictions.jsonl")
            print(f"  → Saved best model (acc={best_acc:.4f})")
            epochs_since_improvement = 0
        else:
            dev_pred_tmp.unlink()
            epochs_since_improvement += 1

        if args.lr_schedule == "plateau" and not in_warmup(global_step):
            plateau_scheduler.step(dev_acc)

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "train_loss": round(avg_loss, 5),
                "dev_acc": round(dev_acc, 5),
                "encoder_lr": encoder_lr,
                "head_lr": head_lr,
                "encoder_frozen": encoder_frozen,
                "best_dev_acc": round(best_acc, 5),
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement,
                "epoch_time_s": round(time.time() - epoch_start, 1),
                "elapsed_s": round(elapsed_before + time.time() - train_start, 1),
            }) + "\n")

        tmp_ckpt = last_ckpt_path.with_suffix(".pt.tmp")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "plateau": plateau_scheduler.state_dict(),
            "classes": classes,
            "arch": args.arch,
            "model_name": model_name,
            "run_config": {k: getattr(args, k) for k in RESUME_CRITICAL},
            "best_acc": best_acc,
            "best_epoch": best_epoch,
            "epochs_since_improvement": epochs_since_improvement,
            "global_step": global_step,
            "warmup_start_step": warmup_start_step,
            "encoder_frozen": encoder_frozen,
            "train_time_s": elapsed_before + time.time() - train_start,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "train_generator": train_generator.get_state(),
            },
        }, tmp_ckpt)
        tmp_ckpt.replace(last_ckpt_path)
        write_results(completed=False)

        if args.early_stop_patience > 0 and epochs_since_improvement >= args.early_stop_patience:
            stop_reason = f"no dev improvement for {args.early_stop_patience} epochs"
            print(f"No dev improvement for {args.early_stop_patience} epochs, "
                  f"stopping early (best epoch {best_epoch}, best_dev_acc={best_acc:.4f})")
            break

    train_time = elapsed_before + time.time() - train_start
    print(f"\nTotal training time: {train_time:.1f}s ({train_time/60:.1f}m)")
    write_results(completed=True, stop_reason=stop_reason)
    print(f"Done. Best model → {output_dir}/best_model.pt")


if __name__ == "__main__":
    main()
