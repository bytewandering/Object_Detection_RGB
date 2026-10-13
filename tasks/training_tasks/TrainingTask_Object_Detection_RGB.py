"""NanoDet-style Lightning task training only the RGB detector, with progressive
FP32 -> INT24 -> INT16 -> INT8 QAT."""

import contextlib
import copy
import io
import logging
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist
import torch.ao.quantization as aq
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from torch.ao.nn.intrinsic.qat import freeze_bn_stats, update_bn_stats

from .utils.quantization import prepare_model_for_qat, set_quantization_mode
from ...utils import CfgNode
from ...models import build_detector


STAGES = ("fp32", "int24", "int16", "int8")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class TrainingTask_Object_Detection_RGB(LightningModule):
    """cfg.model/schedule/log follow the NanoDet task layout.

    The batch comes from DareLabFaceDataLoader: img is the native RGB image and
    boxes are in pixel coordinates, so the backbone downsamples by 8/16/32 and
    the head strides are 8/16/32 pixels. Detections are scored with COCO mAP
    against the split's annotation file.

    Prepare the module structure once. FP32 disables all fake quantization;
    subsequent stages only change observer ranges and flags in place.
    """

    def __init__(self, cfg):
        super().__init__()

        config = _plain(cfg)
        self._validate_config(config)
        self.save_hyperparameters({"cfg": config})
        self.cfg = CfgNode(copy.deepcopy(config))

        self.best_maps = {stage: -math.inf for stage in STAGES}
        self._validation_loss = None
        self._validation_records = {}
        self._test_records = {}
        self._observer_flags = None

        #生成不同训练阶段的开始 epoch
        self._stage_starts = []
        epoch = 0
        for stage in STAGES:
            self._stage_starts.append(epoch)
            epoch += self.cfg.quantization.stage_epochs[stage]

        self.model = build_detector(self.cfg.model)
        if self.model.head.num_classes != 1:
            raise ValueError("TrainingTask expects one foreground class")
        # A fixed structure is essential for optimizer/DDP/checkpoint continuity.
        prepare_model_for_qat(self.model, "int24")
        self.register_buffer("_stage_index", torch.tensor(0, dtype=torch.long))
        set_quantization_mode(self.model, "fp32")

    @classmethod
    def _validate_config(cls, cfg):
        epochs = cfg["quantization"]["stage_epochs"]
        if set(epochs) != set(STAGES) or any(type(epochs[s]) is not int or epochs[s] < 1 for s in STAGES):
            raise ValueError("stage_epochs must give a positive integer for fp32/int24/int16/int8")
        if sum(epochs.values()) != cfg["schedule"]["total_epochs"]:
            raise ValueError("schedule.total_epochs must equal the sum of stage_epochs")
        for key in ("observer_freeze_after", "bn_freeze_after"):
            value = cfg["quantization"].get(key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be None or >= 1 epochs within each QAT stage")
        if cfg["log"]["interval"] < 1:
            raise ValueError("log.interval must be positive")
        cls._validate_model(cfg)

    @staticmethod
    def _validate_model(cfg):
        arch = cfg["model"]["arch"]
        if list(arch["head"]["strides"]) != [8, 16, 32]:
            raise ValueError("RGB boxes use pixel coordinates; head.strides must be [8, 16, 32]")
        if list(arch["backbone"].get("stage_strides", [2, 2, 2, 2, 2])) != [2, 2, 2, 2, 2]:
            raise ValueError("backbone.stage_strides must be [2, 2, 2, 2, 2] for strides 8/16/32")
        if arch["backbone"].get("input_channels", 3) != 3:
            raise ValueError("backbone.input_channels must be 3 for RGB input")

    @property
    def quantization_mode(self):
        return STAGES[int(self._stage_index.item())]

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Only img is a model input; NumPy annotations and warp_matrix stay on CPU.
        result = dict(batch)
        result["img"] = batch["img"].to(device, non_blocking=True)
        return result

    def forward(self, batch):
        return self.model(batch["img"], batch)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.model.infer_forward(batch["img"], batch)

    #######training_step#############
    def training_step(self, batch, batch_idx):
        preds = self.model.train_forward(batch["img"], batch)
        loss, loss_states = self.compute_losses(preds, batch)
        self._log_losses("Train", loss, loss_states, batch, batch_idx)
        return loss

    def compute_losses(self, preds, batch):
        result = self.model.head.loss(preds, batch)
        if result is None:
            raise RuntimeError("Detector target assignment failed")
        return result

    #######validation_step#############
    def validation_step(self, batch, batch_idx):
        # Eval-mode forward without post-processing gives both loss and detections.
        preds = self.model.train_forward(batch["img"], batch)
        loss, loss_states = self.compute_losses(preds, batch)
        self._log_losses("Val", loss, loss_states, batch, batch_idx)
        size = batch["img"].shape[0]
        self._validation_loss += torch.stack((loss.detach() * size, loss.new_tensor(size)))
        self._validation_records.update(self.model.head.post_process(preds, batch))
        return loss

    def test_step(self, batch, batch_idx):
        detections = self.model.infer_forward(batch["img"], batch)
        self._test_records.update(detections)
        return detections

    def _log_losses(self, phase, loss, states, batch, batch_idx):
        stage = phase.lower()
        self.log(f"{stage}/loss", loss, on_step=phase == "Train", on_epoch=True,
                 batch_size=batch["img"].shape[0], sync_dist=True, prog_bar=True)
        for name, value in states.items():
            if phase == "Train":
                self.log(f"train/{name}", value, on_step=True, on_epoch=False,
                         batch_size=batch["img"].shape[0], sync_dist=True)
                name = f"{name}_epoch"
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True,
                     batch_size=batch["img"].shape[0], sync_dist=True)
        if phase == "Train" and self.trainer.optimizers:
            self.log("Train_loss/lr", self.trainer.optimizers[0].param_groups[0]["lr"],
                     on_step=True, on_epoch=False, sync_dist=False)
        if batch_idx % self.cfg.log.interval == 0:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"] if self.trainer.optimizers else 0
            memory = torch.cuda.memory_reserved(self.device) / 1e9 if self.device.type == "cuda" else 0
            msg = (f"{phase}|Epoch{self.current_epoch + 1}/{self.cfg.schedule.total_epochs}|"
                   f"Iter{self.global_step}({batch_idx + 1})|{self.quantization_mode}|"
                   f"mem:{memory:.3g}G|lr:{lr:.2e}|")
            msg += " ".join(f"{key}:{value.detach().item():.4f}" for key, value in states.items())
            self.info(msg)

    def configure_optimizers(self):
        optimizer_cfg = dict(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop("name")
        if name not in ("Adam", "AdamW", "SGD"):
            raise ValueError("Supported optimizers: Adam, AdamW, SGD")
        optimizer = getattr(torch.optim, name)(self.model.parameters(), **optimizer_cfg)
        scheduler_cfg = dict(self.cfg.schedule.get("lr_schedule", {}))
        if not scheduler_cfg:
            return optimizer
        name = scheduler_cfg.pop("name")
        scheduler = getattr(torch.optim.lr_scheduler, name)(optimizer, **scheduler_cfg)
        result = {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            result["monitor"] = "val/loss"
        return {"optimizer": optimizer, "lr_scheduler": result}

    @rank_zero_only
    def info(self, message):
        logger = self.logger if self._trainer is not None else None
        if logger and hasattr(logger, "info"):
            logger.info(message)
        else:
            logging.getLogger(__name__).info(message)

    @rank_zero_only
    def save_model_state(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "quantization_mode": self.quantization_mode,
                    "cfg": _plain(self.cfg)}, path)

    # ------------ Hooks -----------------

    def on_fit_start(self):
        if str(self.trainer.precision) not in ("32", "32-true"):
            raise ValueError("Progressive QAT requires Trainer(precision='32-true')")
        if self.trainer.max_epochs != self.cfg.schedule.total_epochs:
            raise ValueError("Trainer.max_epochs must match cfg.schedule.total_epochs")
        Path(self.cfg.save_dir).mkdir(parents=True, exist_ok=True)
        if self.cfg.get("checkpoint", {}).get("save_last", True) or self.cfg.get("checkpoint", {}).get("save_best", True):
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self):
        directory = Path(self.cfg.get("checkpoint", {}).get("dir", ".")).expanduser()
        return directory if directory.is_absolute() else Path(self.cfg.save_dir).expanduser() / directory

    def on_train_epoch_start(self):
        # A task may have been evaluated explicitly before fit/resume.
        self.train()
        self.model.set_epoch(self.current_epoch)
        self._activate_stage(self.current_epoch)

    def _activate_stage(self, epoch):
        index = max(i for i, start in enumerate(self._stage_starts) if epoch >= start)
        if index != int(self._stage_index.item()):
            set_quantization_mode(self.model, STAGES[index], reset_observers=True)
            self._stage_index.fill_(index)
            self.info(f"Quantization stage -> {STAGES[index]} at epoch {epoch}")
        self._apply_stage_flags(epoch)

    def _apply_stage_flags(self, epoch):
        if self.quantization_mode == "fp32":
            self.model.apply(aq.disable_fake_quant)
            self.model.apply(aq.disable_observer)
            return
        self.model.apply(aq.enable_fake_quant)
        self.model.apply(aq.enable_observer)
        self.model.apply(update_bn_stats)
        age = epoch - self._stage_starts[int(self._stage_index.item())]
        observer_age = self.cfg.quantization.get("observer_freeze_after")
        bn_age = self.cfg.quantization.get("bn_freeze_after")
        if observer_age is not None and age >= observer_age:
            self.model.apply(aq.disable_observer)
        if bn_age is not None and age >= bn_age:
            self.model.apply(freeze_bn_stats)

    def on_train_epoch_end(self):
        # All ranks must enter Lightning checkpoint saves for distributed strategies.
        if self.cfg.get("checkpoint", {}).get("save_last", True):
            self.trainer.save_checkpoint(str(self.checkpoint_dir / "model_last.ckpt"))

    def on_validation_epoch_start(self):
        self._validation_records.clear()
        self._validation_loss = torch.zeros(2, device=self.device)

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._validation_records.clear()
            return
        records = self._gather(dict(self._validation_records))
        self._validation_records.clear()
        if not records:
            return
        # Every rank evaluates the same gathered records, so all ranks agree on saving.
        metrics = self._coco_metrics(records, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.info("Validation metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        # Each bit width keeps its own best model, as accuracy drops with the width.
        mode = self.quantization_mode
        if self.cfg.get("checkpoint", {}).get("save_best", True) and metrics["val/map_50_95"] > self.best_maps[mode]:
            self.best_maps[mode] = metrics["val/map_50_95"]
            self.trainer.save_checkpoint(str(self.checkpoint_dir / f"model_best_{mode}.ckpt"))

    def on_test_epoch_start(self):
        self._test_records.clear()

    def on_test_epoch_end(self):
        records = self._gather(dict(self._test_records))
        self._test_records.clear()
        if records:
            metrics = self._coco_metrics(records, "test")
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
            self.info("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    @staticmethod
    def _gather(records):
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, records)
            records = {key: value for part in gathered for key, value in part.items()}
        return records

    def _coco_metrics(self, records, phase):
        """Single-class COCO bbox mAP in original-image pixels on the seen image IDs."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        loader_cfg = self.cfg.dataloader.val
        annotation_path = Path(loader_cfg.get("ann_path") or
                               Path(loader_cfg.img_path) / "label.json").expanduser()
        # COCO prints a large table; expose its values through the Lightning logger.
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(str(annotation_path))
            categories = gt.getCatIds()
            if len(categories) != 1:
                raise ValueError("Evaluation requires exactly one COCO foreground category")
            image_ids = sorted(int(image_id) for image_id in records)
            if not set(image_ids).issubset(gt.imgs):
                raise ValueError("Evaluated image IDs are absent from the COCO annotations")
            predictions = []
            for image_id, detection in records.items():
                for x1, y1, x2, y2, score in detection.get(0, []):
                    if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
                        raise ValueError("Nonfinite detection encountered during evaluation")
                    if x2 > x1 and y2 > y1:
                        predictions.append({"image_id": int(image_id), "category_id": categories[0],
                                            "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            gt.dataset.setdefault("info", {})
            if predictions:
                dt = gt.loadRes(predictions)
            else:
                dt = COCO()
                dt.dataset = {"images": copy.deepcopy(gt.dataset["images"]),
                              "categories": copy.deepcopy(gt.dataset["categories"]), "annotations": []}
                dt.createIndex()
            evaluation = COCOeval(gt, dt, "bbox")
            evaluation.params.imgIds = image_ids
            evaluation.evaluate()
            evaluation.accumulate()
            evaluation.summarize()
        # COCO reports -1 when AP is undefined (e.g. no ground-truth boxes).
        return {f"{phase}/map_50_95": float(evaluation.stats[0]),
                f"{phase}/map_50": float(evaluation.stats[1]),
                f"{phase}/map_75": float(evaluation.stats[2]),
                f"{phase}/recall_100": float(evaluation.stats[8]),
                f"{phase}/num_images": float(len(image_ids))}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["quantization_mode"] = self.quantization_mode
        checkpoint["best_maps"] = self.best_maps

    def on_load_checkpoint(self, checkpoint):
        # Range/dtype are Python attributes, not state_dict buffers.
        index = int(checkpoint["state_dict"]["_stage_index"].item())
        if not 0 <= index < len(STAGES):
            raise ValueError("Invalid quantization stage in checkpoint")
        set_quantization_mode(self.model, STAGES[index], reset_observers=False)
        self._stage_index.fill_(index)
        self.best_maps.update(checkpoint.get("best_maps", {}))

    # Evaluation must not move observer ranges; restore the stage flags afterwards.
    def _eval_quantization(self):
        self.eval()
        self._observer_flags = [(m, m.observer_enabled.clone()) for m in self.model.modules()
                                if isinstance(m, aq.FakeQuantizeBase)]
        self.model.apply(aq.disable_observer)

    def _restore_quantization(self):
        self.train()
        for module, enabled in self._observer_flags or []:
            module.observer_enabled.copy_(enabled)
        self._observer_flags = None
        self._apply_stage_flags(self.current_epoch)

    def on_validation_model_eval(self):
        self._eval_quantization()

    def on_validation_model_train(self):
        self._restore_quantization()

    def on_test_model_eval(self):
        self._eval_quantization()

    def on_test_model_train(self):
        self._restore_quantization()

    def on_predict_model_eval(self):
        self._eval_quantization()

    def on_predict_end(self):
        for module, enabled in self._observer_flags or []:
            module.observer_enabled.copy_(enabled)
        self._observer_flags = None
