"""Train the RGB detector with a YAML config and progressive FP32/INT24/INT16/INT8 QAT."""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger

from Object_Detection_RGB.dataloaders import build_dataloader
from Object_Detection_RGB.tasks import build_task
from Object_Detection_RGB.tasks.training_tasks.utils.logger import StepWandbLogger
from Object_Detection_RGB.utils import load_config


# These dataloader options define the model input and data split, so a resume must keep them.
INPUT_KEYS = ("ann_path", "augmentation")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="Resume a progressive Lightning checkpoint")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 0:
        parser.error("--workers must be >= 0")
    for name in ("batch_size", "limit_train_batches", "limit_val_batches"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def project_path(value):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def output_path(save_dir, value):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(save_dir) / path).resolve())


def build_logger(cfg):
    """W&B when log.wandb is configured; otherwise CSV metrics under log.dir."""
    wandb_cfg = cfg.log.get("wandb")
    if not wandb_cfg:
        return CSVLogger(save_dir=cfg.log.dir, name="metrics")
    return StepWandbLogger(
        save_dir=cfg.log.dir,
        project=wandb_cfg.get("project", "object-detection-rgb"),
        name=wandb_cfg.get("name"),
        entity=wandb_cfg.get("entity"),
        offline=wandb_cfg.get("offline", False),
        id=wandb_cfg.get("id"),
        resume="allow" if wandb_cfg.get("id") else None,
        log_model=False,  # TrainingTask manages local checkpoints itself.
    )


def main():

    # 导入配置
    args = parse_args()
    cfg = load_config(args.config)

    # 设置保存路径和log路径
    cfg.save_dir = project_path(cfg.save_dir)
    cfg.log.dir = output_path(cfg.save_dir, cfg.log.get("dir", "."))

    #resume中存放的是要导入的ckpt文件地址
    configured_resume = cfg.get("checkpoint", {}).get("resume")
    #如果命令中提供了--resume参数，则使用命令中的，否则使用config.yml文件中的resume地址，如果都没有则为None
    resume_path = (str(args.resume.expanduser().resolve()) if args.resume else
                   output_path(cfg.save_dir, configured_resume) if configured_resume else None)

    #把数据和归一化统计的地址全都换成绝对路径
    for split in ("train", "val"):
        data_cfg = cfg.dataloader[split]
        data_cfg.img_path = project_path(data_cfg.img_path)
        for key in ("ann_path",):
            if data_cfg.get(key) is not None:
                data_cfg[key] = project_path(data_cfg[key])
        if data_cfg.get("mode", split) != split:
            raise ValueError(f"dataloader.{split}.mode must be {split!r}")

    #设置随机种子
    pl.seed_everything(args.seed, workers=True)

    #设置设备
    use_gpu = torch.cuda.is_available() and not args.cpu

    #设置dataloader的参数
    for split in ("train", "val"):
        loader_cfg = cfg.dataloader[split]
        if args.workers is not None:
            loader_cfg.workers = args.workers
        if args.batch_size is not None:
            loader_cfg.batch_size = args.batch_size

    #用于检查当前配置和checkpoint的保存的配置是否相同
    if resume_path:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        saved_cfg = checkpoint["hyper_parameters"]["cfg"]
        # A changed phase boundary can silently reset observers at the wrong epoch.
        for key in ("model", "schedule", "quantization"):
            if saved_cfg.get(key) != cfg.get(key):
                raise ValueError(f"Resume config differs from checkpoint at {key!r}")
        for split in ("train", "val"):
            for key in INPUT_KEYS:
                if saved_cfg["dataloader"][split].get(key) != cfg.dataloader[split].get(key):
                    raise ValueError(f"Resume config differs from checkpoint at "
                                     f"'dataloader.{split}.{key}'")
        del checkpoint

    #创建dataloader
    train_loader = build_dataloader(
        cfg.dataloader.train, mode="train", pin_memory=use_gpu,
    )
    val_loader = build_dataloader(
        cfg.dataloader.val, mode="val", pin_memory=use_gpu,
    )

    #设置logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    logger = build_logger(cfg)

    #创建task，本质上是一个LightningModule
    task = build_task(cfg)

    #配置pl.Trainer需要的参数
    kwargs = {
        "default_root_dir": cfg.save_dir,
        "max_epochs": cfg.schedule.total_epochs,
        "check_val_every_n_epoch": cfg.schedule.get("val_intervals", 1),
        "gradient_clip_val": cfg.get("grad_clip", 0.0),
        "precision": "32-true",
        "logger": logger,
        "log_every_n_steps": cfg.log.interval,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": sys.stdout.isatty(),
        # TrainingTask saves model_last and per-stage best checkpoints itself.
        "enable_checkpointing": False,
    }
    if args.limit_train_batches is not None:
        kwargs["limit_train_batches"] = args.limit_train_batches
    if args.limit_val_batches is not None:
        kwargs["limit_val_batches"] = args.limit_val_batches
    if use_gpu:
        gpu_ids = list(cfg.device.gpu_ids)
        if not gpu_ids:
            raise ValueError("device.gpu_ids must not be empty for GPU training")
        kwargs.update(accelerator="gpu", devices=gpu_ids)
        if len(gpu_ids) > 1:
            kwargs["strategy"] = "ddp"
    else:
        kwargs.update(accelerator="cpu", devices=1)
    trainer = pl.Trainer(**kwargs)

    #让主进程创建保存的路径，并把文字日志同时写入 log.dir/train.log
    if trainer.is_global_zero:
        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log.dir).mkdir(parents=True, exist_ok=True)
        (Path(cfg.save_dir) / "config.yml").write_text(cfg.dump(), encoding="utf-8")
        handler = logging.FileHandler(Path(cfg.log.dir) / "train.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logging.getLogger().addHandler(handler)

    #训练
    trainer.fit(task, train_loader, val_loader,
                ckpt_path=resume_path)


if __name__ == "__main__":
    main()
