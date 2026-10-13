"""Lightning test task for FOMO on RGB images.

trainer.test(task, loader, ckpt_path=...) restores a checkpoint written by
TrainingTask_FOMO_RGB with observers frozen, reports centroid precision/recall/F1
at model.score_threshold and draws the face probability heatmap. Results go to
<save_dir>/test results/<checkpoint stem>_<time>/.
"""

from pathlib import Path

import torch

from ..training_tasks.TrainingTask_FOMO_RGB import TrainingTask_FOMO_RGB
from ..training_tasks.TrainingTask_Object_Detection_RGB import _plain
from ..training_tasks.utils.centroid_metrics import centroid_metrics, load_ground_truth, match_image
from .utils import test_utils


class TestingTask_FOMO_RGB(TrainingTask_FOMO_RGB):
    """Evaluate one checkpoint on dataloader.val of its run (test uses val.json).

    score_threshold defaults to model.score_threshold of the config and replaces it.
    Images are saved for every image_every-th sample.
    """

    def __init__(self, cfg, checkpoint, score_threshold=None, image_every=50):
        super().__init__(cfg)
        if score_threshold is not None:
            self.model.score_threshold = score_threshold
        self.checkpoint_path = Path(checkpoint)
        self.image_every = image_every
        self.annotation_path = test_utils.annotation_path(self.cfg)
        self.output_dir = None
        self._truth = {}
        self._samples = []
        self._losses = []

    def on_load_checkpoint(self, checkpoint):
        # config.yml must describe the checkpoint's model, or the weights would be misread.
        if checkpoint["hyper_parameters"]["cfg"]["model"] != _plain(self.cfg.model):
            raise ValueError(f"config.yml model differs from checkpoint {self.checkpoint_path}")
        super().on_load_checkpoint(checkpoint)

    def on_test_epoch_start(self):
        self._test_records.clear()
        self._samples.clear()
        self._losses.clear()
        self._truth = load_ground_truth(self.annotation_path)
        self.output_dir = test_utils.output_dir(self.cfg, self.checkpoint_path)
        self.info(f"Checkpoint stage: {self.quantization_mode}; device: {self.device}; output: {self.output_dir}")

    def test_step(self, batch, batch_idx):
        logits = self.model.train_forward(batch["img"], batch)
        if not torch.isfinite(logits).all().item():
            raise ValueError(f"Nonfinite logits in test batch {batch_idx}")
        loss, loss_states = self.compute_losses(logits, batch)
        self._losses.append((batch["img"].shape[0], float(loss),
                             {key: float(value) for key, value in loss_states.items()}))
        detections = self.model.post_process(logits, batch)
        if self._test_records.keys() & detections.keys():
            raise ValueError("Duplicate test image IDs")
        self._test_records.update(detections)
        heatmaps = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
        for index in range(len(batch["img_info"]["id"])):
            self._samples.append(self._sample(batch, index, detections, heatmaps[index]))
        if batch_idx % 10 == 0:
            self.info(f"Evaluated {len(self._samples)}/{len(self._truth)} images")

    def _sample(self, batch, index, detections, heatmap):
        image_id, file_name, width, height = test_utils.image_info(batch, index)
        predictions = detections[image_id][0]
        gt = self._truth[image_id]
        truths = test_utils.ground_truth_boxes(gt, width, height)
        tp, fp, fn = match_image(predictions, gt["boxes"], gt["ignored"])
        path = None
        if len(self._samples) % self.image_every == 0:
            path = test_utils.save_image(self.output_dir, self.cfg.dataloader.val.img_path, image_id, file_name,
                                         lambda image: self._draw(image, heatmap, truths, predictions))
        return {
            "image_id": image_id, "file_name": file_name, "width": width, "height": height,
            "recording": Path(file_name).parts[0], "has_face": bool(gt["boxes"]),
            "tp": tp, "fp": fp, "fn": fn, "max_probability": float(heatmap.max()),
            "ground_truth_boxes": truths, "predictions": predictions, "image": path,
        }

    @staticmethod
    def _draw(image, heatmap, truths, predictions):
        """Overlay the face probability heatmap, GT boxes and predicted centroids."""
        import numpy as np
        from PIL import Image, ImageDraw

        heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
        red = Image.new("RGB", image.size, (255, 0, 0))
        canvas = Image.composite(red, image, heat.point(lambda value: value * 0.5))
        draw = ImageDraw.Draw(canvas)
        test_utils.draw_truths(draw, truths)
        for item in predictions:
            x, y = item["centroid"]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="yellow", outline="black")
            draw.text((x + 4, max(0, y - 12)), f"{item['score']:.2f}", fill="yellow")
        draw.text((4, 4), "GT: green | ignored: orange | centroid: yellow | red: face prob",
                  fill="white", stroke_width=1, stroke_fill="black")
        return canvas

    def on_test_epoch_end(self):
        records = dict(self._test_records)
        self._test_records.clear()
        samples = self._samples
        if not samples:
            raise ValueError("Test set is empty")

        # post_process already dropped cells below the model's score threshold.
        predictions = {image_id: detections[0] for image_id, detections in records.items()}
        summary = centroid_metrics(predictions, self._truth, self.model.score_threshold)
        summary.update(test_utils.loss_summary(self._losses))
        per_recording = {}
        for sample in samples:
            entry = per_recording.setdefault(sample["recording"], {"images": 0, "faces": 0, "tp": 0, "fp": 0})
            entry["images"] += 1
            entry["faces"] += len([box for box in sample["ground_truth_boxes"] if not box["ignored"]])
            entry["tp"] += sample["tp"]
            entry["fp"] += sample["fp"]
        summary["per_recording"] = per_recording
        self.log_dict({f"test/{key}": float(value) for key, value in summary.items()
                       if isinstance(value, (int, float))}, on_step=False, on_epoch=True, sync_dist=False)

        model_cfg = self.cfg.model
        report = test_utils.report_header(self, samples, self._truth, self.annotation_path, self.image_every)
        report.update({
            "model": {"width_mult": model_cfg.get("width_mult", 0.35), "cut_block": model_cfg.get("cut_block", 6),
                      "output_stride": self.model.stride, "score_threshold": self.model.score_threshold},
            "coordinate_system": "original image pixels; centroids [x, y], boxes xyxy enclose object cells",
            "metric_notes": {
                "objects": "8-connected cells with face probability >= score_threshold form one object; centroid "
                           "is the probability-weighted mean of cell centres, score the maximum probability.",
                "matching": "Predictions sorted by score; a centroid inside an unmatched GT box is a TP, inside "
                            "an ignored box is skipped, otherwise FP; unmatched GT boxes are FN.",
                "image_level": "An image is predicted to contain a face if it has any object.",
                "loss": "Weighted cross entropy (face cells x object_weight) averaged per image, as in validation.",
            },
            "summary": summary, "samples": samples,
        })
        test_utils.write_report(self.output_dir, report)
