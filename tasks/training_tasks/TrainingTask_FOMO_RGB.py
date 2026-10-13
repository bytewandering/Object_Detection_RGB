"""FOMO on RGB images, sharing the progressive QAT task of the NanoDet detector.

Only the loss, post-processing and metrics differ: FOMO predicts per-cell
classes, so validation reports centroid precision/recall/F1 and each stage
keeps the checkpoint with the best validation F1 (stored in best_maps).
"""

from pathlib import Path

import torch

from .utils.centroid_metrics import centroid_metrics, load_ground_truth
from .TrainingTask_Object_Detection_RGB import TrainingTask_Object_Detection_RGB


class TrainingTask_FOMO_RGB(TrainingTask_Object_Detection_RGB):

    @staticmethod
    def _validate_model(cfg):
        model = cfg["model"]
        if model.get("name") != "FOMO":
            raise ValueError("TrainingTask_FOMO_RGB requires model.name: FOMO")
        if model.get("num_classes", 1) != 1:
            raise ValueError("DareLab face detection has one foreground class")

    def compute_losses(self, preds, batch):
        return self.model.loss(preds, batch)

    def validation_step(self, batch, batch_idx):
        preds = self.model.train_forward(batch["img"], batch)
        loss, loss_states = self.compute_losses(preds, batch)
        self._log_losses("Val", loss, loss_states, batch, batch_idx)
        size = batch["img"].shape[0]
        self._validation_loss += torch.stack((loss.detach() * size, loss.new_tensor(size)))
        self._validation_records.update(self.model.post_process(preds, batch))
        return loss

    def _metrics(self, records, phase):
        loader_cfg = self.cfg.dataloader.val
        annotation_path = Path(loader_cfg.get("ann_path") or
                               Path(loader_cfg.img_path) / "annotations/val.json").expanduser()
        # post_process already dropped cells below the model's score threshold.
        predictions = {image_id: detections[0] for image_id, detections in records.items()}
        return centroid_metrics(predictions, load_ground_truth(annotation_path),
                                self.model.score_threshold, prefix=f"{phase}/")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self._validation_records.clear()
            return
        records = self._gather(dict(self._validation_records))
        self._validation_records.clear()
        if not records:
            return
        # Every rank evaluates the same gathered records, so all ranks agree on saving.
        metrics = self._metrics(records, "val")
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
        self.info("Validation metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        mode = self.quantization_mode
        if self.cfg.get("checkpoint", {}).get("save_best", True) and metrics["val/f1"] > self.best_maps[mode]:
            self.best_maps[mode] = metrics["val/f1"]
            self.trainer.save_checkpoint(str(self.checkpoint_dir / f"model_best_{mode}.ckpt"))

    def on_test_epoch_end(self):
        records = self._gather(dict(self._test_records))
        self._test_records.clear()
        if records:
            metrics = self._metrics(records, "test")
            self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False, rank_zero_only=True)
            self.info("Test metrics: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
