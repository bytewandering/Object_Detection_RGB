"""Model-independent helpers shared by the testing tasks: ground truth, images and the report."""

import hashlib
import json
import shutil
from pathlib import Path


def annotation_path(cfg):
    """COCO file of the test split (dataloader.val; test uses val.json)."""
    loader_cfg = cfg.dataloader.val
    return Path(loader_cfg.get("ann_path") or Path(loader_cfg.img_path) / "annotations/val.json").expanduser()


def output_dir(cfg, checkpoint):
    """Empty <save_dir>/test results/<checkpoint stem>/ directory; a previous test of it is replaced."""
    path = Path(cfg.save_dir) / "test results" / Path(checkpoint).stem
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def image_info(batch, index):
    info = batch["img_info"]
    return (int(info["id"][index]), info["file_name"][index],
            int(info["width"][index]), int(info["height"][index]))


def ground_truth_boxes(gt, width, height):
    """load_ground_truth entry -> [{"bbox_xyxy", "ignored"}] clipped to the image."""
    clip = lambda box: [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
    return ([{"bbox_xyxy": clip(box), "ignored": False} for box in gt["boxes"]]
            + [{"bbox_xyxy": clip(box), "ignored": True} for box in gt["ignored"]])


def draw_truths(draw, truths):
    for box in truths:
        draw.rectangle(box["bbox_xyxy"], outline="orange" if box["ignored"] else "lime", width=2)


def save_image(output, image_root, image_id, file_name, render):
    """render(RGB image) -> canvas, saved as images/<image_id>.png; returns the relative path."""
    from PIL import Image

    with Image.open(Path(image_root) / file_name) as image:
        canvas = render(image.convert("RGB"))
    relative = Path("images") / f"{image_id}.png"
    (output / relative).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output / relative)
    return relative.as_posix()


def loss_summary(losses):
    """losses: [(batch size, loss, {name: value})] -> per-image averages."""
    total = sum(size for size, _, _ in losses)
    summary = {"loss": sum(size * value for size, value, _ in losses) / total}
    for key in losses[0][2]:
        summary[key] = sum(size * states[key] for size, _, states in losses) / total
    return summary


def write_report(output, report):
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output / "results.json")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    print(f"Results saved: {output / 'results.json'}", flush=True)


def report_header(task, samples, truth, annotation, image_every):
    """Report fields common to every testing task."""
    checkpoint = Path(task.checkpoint_path)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "saved_config": task.hparams["cfg"], "quantization_mode": task.quantization_mode,
        "quantization": "checkpoint fake-quant parameters preserved; observers frozen",
        "input": "RGB (pixel - 128) / 128, native resolution",
        "annotation_file": str(annotation), "device": str(task.device),
        "batch_size": task.cfg.dataloader.val.batch_size, "test_dataset_size": len(truth),
        "is_subset": len(samples) < len(truth),
        "image_sampling": {"every": image_every, "start_index": 0,
                           "saved_samples": sum(sample["image"] is not None for sample in samples)},
    }
