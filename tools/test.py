"""Test the checkpoints of a training run with the test task of its config.

Run directly: python Object_Detection_RGB/tools/test.py WORKSPACE [--checkpoint NAME ...].
WORKSPACE is the save_dir of a run (holding config.yml and the checkpoint dir).
By default every model_best_<stage>.ckpt is tested. The test split reads
dataloader.val of config.yml (test uses val.json); results go to
WORKSPACE/test results/<checkpoint stem>/ (replacing an earlier test of it).
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

import pytorch_lightning as pl
import torch

from Object_Detection_RGB.dataloaders import build_dataloader
from Object_Detection_RGB.tasks import build_test_task
from Object_Detection_RGB.tasks.training_tasks.TrainingTask_NanoDet_RGB import STAGES
from Object_Detection_RGB.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--checkpoint", nargs="+", type=Path,
                        help="Checkpoint names in the checkpoint dir, or paths; default: model_best_*.ckpt")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, help="Default: dataloader.val.batch_size")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit-samples", type=int, help="Debug subset; default: all test images")
    parser.add_argument("--image-every", type=int, default=50,
                        help="Save images for every Nth test sample, starting with the first (default: 50)")
    parser.add_argument("--score-threshold", type=float,
                        help="Default: model.score_threshold for FOMO, 0.3 for NanoDet")
    args = parser.parse_args()
    if not (args.workspace / "config.yml").is_file():
        parser.error("workspace must contain the config.yml written by train.py")
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
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def find_checkpoints(cfg, names):
    directory = Path(cfg.get("checkpoint", {}).get("dir", ".")).expanduser()
    directory = directory if directory.is_absolute() else Path(cfg.save_dir) / directory
    if names:
        paths = [name.expanduser() if name.expanduser().is_file() else directory / name for name in names]
    else:
        paths = [directory / f"model_best_{stage}.ckpt" for stage in STAGES]
        paths = [path for path in paths if path.is_file()]
        if not paths:
            raise FileNotFoundError(f"No model_best_*.ckpt in {directory}")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    return [path.resolve() for path in paths]


def main():
    args = parse_args()
    workspace = args.workspace.expanduser().resolve()
    cfg = load_config(workspace / "config.yml")

    # Save beside the run's checkpoints, even if its directory was renamed.
    cfg.save_dir = str(workspace)
    checkpoints = find_checkpoints(cfg, args.checkpoint)

    loader_cfg = cfg.dataloader.val
    for key in ("img_path", "ann_path"):
        if loader_cfg.get(key):
            loader_cfg[key] = project_path(loader_cfg[key])
    loader_cfg.update(workers=args.workers, num_workers=args.workers, shuffle=False, drop_last=False)
    if args.workers == 0:
        loader_cfg.update(persistent_workers=False, prefetch_factor=None)
    if args.batch_size is not None:
        loader_cfg.batch_size = args.batch_size

    torch.set_num_threads(args.threads)
    pl.seed_everything(42)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    gpu_ids = list(cfg.get("device", {}).get("gpu_ids", [0]))
    use_gpu = torch.cuda.is_available() and not args.cpu and bool(gpu_ids)
    loader = build_dataloader(loader_cfg, mode="test", pin_memory=use_gpu)
    if args.limit_samples is not None:
        subset = torch.utils.data.Subset(loader.dataset, range(min(args.limit_samples, len(loader.dataset))))
        loader = torch.utils.data.DataLoader(subset, batch_size=loader.batch_size,
                                             collate_fn=loader.collate_fn, num_workers=args.workers,
                                             pin_memory=use_gpu)

    for checkpoint in checkpoints:
        task = build_test_task(cfg, checkpoint,
                               score_threshold=args.score_threshold, image_every=args.image_every)
        # One device: the task keeps per-sample results locally.
        trainer = pl.Trainer(
            accelerator="gpu" if use_gpu else "cpu", devices=gpu_ids[:1] if use_gpu else 1,
            precision="32-true", logger=False, enable_checkpointing=False,
            enable_progress_bar=sys.stdout.isatty(), default_root_dir=cfg.save_dir,
        )
        trainer.test(task, loader, ckpt_path=str(checkpoint), verbose=False, weights_only=True)


if __name__ == "__main__":
    main()
