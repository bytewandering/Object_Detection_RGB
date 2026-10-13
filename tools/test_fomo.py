"""Evaluate a FOMO Lightning checkpoint on the test split of its saved config.

Run directly: python Object_Detection_RGB/tools/test_fomo.py --checkpoint PATH.
The test split reads dataloader.val of the checkpoint config (test uses val.json).
"""

import argparse
import copy
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, help="Default: checkpoint val batch_size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-samples", type=int, help="Debug subset; default: all test images")
    parser.add_argument("--image-every", type=int, default=50,
                        help="Save images for every Nth test sample, starting with the first (default: 50)")
    parser.add_argument("--score-threshold", type=float,
                        help="Face probability threshold; default: model.score_threshold of the checkpoint")
    args = parser.parse_args()
    if not args.checkpoint.expanduser().is_file():
        parser.error("--checkpoint must be an existing Lightning checkpoint")
    for key in ("batch_size", "threads", "limit_samples", "image_every"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.score_threshold is not None and not 0 < args.score_threshold < 1:
        parser.error("--score-threshold must be in (0, 1)")
    return args


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def restore_task(checkpoint):
    from Object_Detection_RGB.tasks.training_tasks.TrainingTask_FOMO_RGB import TrainingTask_FOMO_RGB
    from Object_Detection_RGB.utils import CfgNode
    import torch.ao.quantization as aq

    if "state_dict" not in checkpoint or "cfg" not in checkpoint.get("hyper_parameters", {}):
        raise ValueError("Expected a complete Lightning checkpoint with hyper_parameters.cfg")
    task = TrainingTask_FOMO_RGB(CfgNode(copy.deepcopy(checkpoint["hyper_parameters"]["cfg"])))
    # Restore Python quantization ranges/dtypes before loading scale/zero_point buffers.
    task.on_load_checkpoint(checkpoint)
    task.load_state_dict(checkpoint["state_dict"], strict=True)
    task.eval()
    task.model.apply(aq.disable_observer)
    return task


def save_detection_image(output, image_id, original, heatmap, truths, predictions):
    """Overlay the face probability heatmap, GT boxes and predicted centroids."""
    import numpy as np
    from PIL import Image, ImageDraw

    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(original.size, Image.Resampling.NEAREST)
    red = Image.new("RGB", original.size, (255, 0, 0))
    canvas = Image.composite(red, original, heat.point(lambda value: value * 0.5))
    draw = ImageDraw.Draw(canvas)
    for box in truths:
        draw.rectangle(box["bbox_xyxy"], outline="orange" if box["ignored"] else "lime", width=2)
    for item in predictions:
        x, y = item["centroid"]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow", outline="black")
        draw.text((x + 4, max(0, y - 12)), f"{item['score']:.2f}", fill="yellow")
    draw.text((4, 4), "GT: green | ignored: orange | centroid: yellow | red: face prob",
              fill="white", stroke_width=1, stroke_fill="black")
    relative = Path("images") / f"{image_id}.png"
    (output / relative).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output / relative)
    return relative.as_posix()


def evaluate(task, loader, output, device, image_every=50):
    import torch
    from PIL import Image
    from Object_Detection_RGB.tasks.training_tasks.utils.centroid_metrics import (
        centroid_metrics, load_ground_truth, match_image)

    model = task.model
    records, samples, losses = {}, [], []
    dataset = loader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    truth = load_ground_truth(dataset.ann_path)
    task.to(device)
    task.eval()
    with torch.inference_mode():
        for batch_index, cpu_batch in enumerate(loader):
            batch = task.transfer_batch_to_device(cpu_batch, device, 0)
            logits = model.train_forward(batch["img"], batch)
            if not torch.isfinite(logits).all().item():
                raise ValueError(f"Nonfinite logits in test batch {batch_index}")
            loss, loss_states = task.compute_losses(logits, batch)
            size = batch["img"].shape[0]
            losses.append((size, float(loss), {k: float(v) for k, v in loss_states.items()}))
            detections = model.post_process(logits, batch)
            heatmaps = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
            if records.keys() & detections.keys():
                raise ValueError("Duplicate test image IDs")
            for index, image_id in enumerate(batch["img_info"]["id"]):
                image_id = int(image_id)
                predictions = detections[image_id][0]
                records[image_id] = predictions
                file_name = batch["img_info"]["file_name"][index]
                width, height = (int(batch["img_info"][key][index]) for key in ("width", "height"))
                gt = truth[image_id]
                truths = ([{"bbox_xyxy": box, "ignored": False} for box in gt["boxes"]]
                          + [{"bbox_xyxy": box, "ignored": True} for box in gt["ignored"]])
                tp, fp, fn = match_image(predictions, gt["boxes"], gt["ignored"])
                selected = len(samples) % image_every == 0
                path = None
                if selected:
                    with Image.open(dataset.img_path / file_name) as image:
                        path = save_detection_image(output, image_id, image.convert("RGB"),
                                                    heatmaps[index], truths, predictions)
                samples.append({
                    "image_id": image_id, "file_name": file_name, "width": width, "height": height,
                    "recording": Path(file_name).parts[0], "has_face": bool(gt["boxes"]),
                    "tp": tp, "fp": fp, "fn": fn, "max_probability": float(heatmaps[index].max()),
                    "ground_truth_boxes": truths, "predictions": predictions, "image": path,
                })
            if batch_index % 10 == 0 or len(samples) == len(loader.dataset):
                print(f"Evaluated {len(samples)}/{len(loader.dataset)} images", flush=True)
    if not samples:
        raise ValueError("Test set is empty")

    summary = {key: value for key, value in
               centroid_metrics(records, truth, model.score_threshold).items()}
    total = sum(size for size, _, _ in losses)
    summary["loss"] = sum(size * value for size, value, _ in losses) / total
    for key in losses[0][2]:
        summary[key] = sum(size * states[key] for size, _, states in losses) / total
    per_recording = {}
    for sample in samples:
        entry = per_recording.setdefault(sample["recording"], {"images": 0, "faces": 0, "tp": 0, "fp": 0})
        entry["images"] += 1
        entry["faces"] += len([box for box in sample["ground_truth_boxes"] if not box["ignored"]])
        entry["tp"] += sample["tp"]
        entry["fp"] += sample["fp"]
    summary["per_recording"] = per_recording
    return summary, samples


def main():
    args = parse_args()
    import torch
    from Object_Detection_RGB.dataloaders import build_dataloader

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    data = checkpoint_path.read_bytes()
    task = restore_task(torch.load(io.BytesIO(data), map_location="cpu", weights_only=True))
    if args.score_threshold is not None:
        task.model.score_threshold = args.score_threshold
    cfg = task.cfg
    loader_cfg = copy.deepcopy(cfg.dataloader.val)
    for key in ("img_path", "ann_path"):
        if loader_cfg.get(key):
            loader_cfg[key] = str(project_path(loader_cfg[key]))
    loader_cfg.update(workers=args.workers, num_workers=args.workers, shuffle=False, drop_last=False)
    if args.workers == 0:
        loader_cfg.update(persistent_workers=False, prefetch_factor=None)
    if args.batch_size is not None:
        loader_cfg.batch_size = args.batch_size
    gpu_ids = cfg.get("device", {}).get("gpu_ids", [0])
    device = torch.device(f"cuda:{gpu_ids[0]}" if not args.cpu and torch.cuda.is_available() and gpu_ids else "cpu")
    loader = build_dataloader(loader_cfg, mode="test", pin_memory=device.type == "cuda")
    total = len(loader.dataset)
    if args.limit_samples is not None:
        subset = torch.utils.data.Subset(loader.dataset, range(min(args.limit_samples, total)))
        loader = torch.utils.data.DataLoader(subset, batch_size=loader.batch_size,
                                             collate_fn=loader.collate_fn, num_workers=args.workers,
                                             pin_memory=device.type == "cuda")
    # Save beside the run that holds the checkpoint, even if its directory was renamed.
    run_dir = (checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints"
               else project_path(cfg.save_dir))
    output = run_dir / "test results" / (
        f"{checkpoint_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    output.mkdir(parents=True, exist_ok=False)
    print(f"Checkpoint stage: {task.quantization_mode}; device: {device}; output: {output}", flush=True)
    summary, samples = evaluate(task, loader, output, device, args.image_every)
    report = {
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        "saved_config": cfg, "quantization_mode": task.quantization_mode,
        "quantization": "checkpoint fake-quant parameters preserved; observers frozen",
        "model": {"width_mult": cfg.model.get("width_mult", 0.35), "cut_block": cfg.model.get("cut_block", 6),
                  "output_stride": task.model.stride, "score_threshold": task.model.score_threshold},
        "input": "RGB (pixel - 128) / 128, native resolution",
        "annotation_file": str(loader_cfg.ann_path), "device": str(device),
        "batch_size": loader.batch_size, "test_dataset_size": total, "is_subset": len(samples) < total,
        "image_sampling": {"every": args.image_every, "start_index": 0,
                           "saved_samples": sum(sample["image"] is not None for sample in samples)},
        "coordinate_system": "original image pixels; centroids [x, y], boxes xyxy enclose object cells",
        "metric_notes": {
            "objects": "8-connected cells with face probability >= score_threshold form one object; centroid is "
                       "the probability-weighted mean of cell centres, score the maximum probability.",
            "matching": "Predictions sorted by score; a centroid inside an unmatched GT box is a TP, inside an "
                        "ignored box is skipped, otherwise FP; unmatched GT boxes are FN.",
            "image_level": "An image is predicted to contain a face if it has any object.",
            "loss": "Weighted cross entropy (face cells x object_weight) averaged per image, as in validation.",
        },
        "summary": summary, "samples": samples,
    }
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "results.json")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Results saved: {output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
