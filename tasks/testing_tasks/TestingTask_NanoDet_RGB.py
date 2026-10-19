"""Lightning test task for the RGB NanoDet detector.

trainer.test(task, loader, ckpt_path=...) restores a checkpoint written by the
training task: on_load_checkpoint restores the quantization stage before the
fake-quant buffers load, and observers stay frozen during evaluation. The task
writes results.json and sampled images to
<save_dir>/test results/<checkpoint stem>/ (replacing an earlier test of it).
"""

from pathlib import Path

import torch

from ..training_tasks.TrainingTask_NanoDet_RGB import TrainingTask_NanoDet_RGB, _plain
from ..training_tasks.utils.centroid_metrics import box_centroids, centroid_metrics, load_ground_truth
from .utils import test_utils


class TestingTask_NanoDet_RGB(TrainingTask_NanoDet_RGB):
    """Evaluate one checkpoint on dataloader.val of its run (test uses val.json).

    score_threshold only affects drawn boxes, image-level and centroid metrics;
    COCO AP uses every detection. Images are saved for every image_every-th sample.
    """

    def __init__(self, cfg, checkpoint, score_threshold=None, image_every=50):
        super().__init__(cfg)
        self.checkpoint_path = Path(checkpoint)
        self.score_threshold = 0.3 if score_threshold is None else score_threshold
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
        # Same forward as validation: raw predictions give the loss, then NMS post-processing.
        preds = self.model.train_forward(batch["img"], batch)
        if not torch.isfinite(preds).all().item():
            raise ValueError(f"Nonfinite predictions in test batch {batch_idx}")
        loss, loss_states = self.compute_losses(preds, batch)
        self._losses.append((batch["img"].shape[0], float(loss),
                             {key: float(value) for key, value in loss_states.items()}))
        detections = self.model.head.post_process(preds, batch)
        if self._test_records.keys() & detections.keys():
            raise ValueError("Duplicate test image IDs")
        self._test_records.update(detections)
        for index in range(len(batch["img_info"]["id"])):
            self._samples.append(self._sample(batch, index, detections))
        if batch_idx % 10 == 0:
            self.info(f"Evaluated {len(self._samples)}/{len(self._truth)} images")

    def _sample(self, batch, index, detections):
        image_id, file_name, width, height = test_utils.image_info(batch, index)
        truths = test_utils.ground_truth_boxes(self._truth[image_id], width, height)
        boxes = sorted(({"bbox_xyxy": [float(v) for v in box[:4]], "score": float(box[4])}
                        for box in detections[image_id].get(0, [])),
                       key=lambda box: -box["score"])
        path = None
        if len(self._samples) % self.image_every == 0:
            path = test_utils.save_image(self.output_dir, self.cfg.dataloader.val.img_path, image_id, file_name,
                                         lambda image: self._draw(image, truths, boxes))
        return {
            "image_id": image_id, "file_name": file_name, "width": width, "height": height,
            "recording": Path(file_name).parts[0],
            "has_face": any(not box["ignored"] for box in truths),
            "detected_at_threshold": any(box["score"] >= self.score_threshold for box in boxes),
            "max_score": boxes[0]["score"] if boxes else 0.0,
            "ground_truth_boxes": truths, "predicted_boxes": boxes, "image": path,
        }

    def _draw(self, image, truths, boxes):
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        test_utils.draw_truths(draw, truths)
        for box in boxes:
            if box["score"] >= self.score_threshold:
                xy = box["bbox_xyxy"]
                draw.rectangle(xy, outline="red", width=2)
                draw.text((xy[0], max(0, xy[1] - 12)), f"face {box['score']:.3f}", fill="red")
        draw.text((4, 4), "GT: green | ignored: orange | prediction: red", fill="white",
                  stroke_width=1, stroke_fill="black")
        return image

    def on_test_epoch_end(self):
        records = dict(self._test_records)
        self._test_records.clear()
        samples = self._samples
        if not samples:
            raise ValueError("Test set is empty")

        summary = {key.removeprefix("test/"): value for key, value in self._coco_metrics(records, "test").items()}
        summary.update(test_utils.loss_summary(self._losses))
        # Image-level face presence at the display threshold.
        tp = sum(s["has_face"] and s["detected_at_threshold"] for s in samples)
        fp = sum(not s["has_face"] and s["detected_at_threshold"] for s in samples)
        fn = sum(s["has_face"] and not s["detected_at_threshold"] for s in samples)
        tn = len(samples) - tp - fp - fn
        summary["image_level"] = {
            "threshold": self.score_threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / len(samples), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        }
        per_recording = {}
        for sample in samples:
            entry = per_recording.setdefault(sample["recording"], {"images": 0, "with_face": 0, "detected": 0})
            entry["images"] += 1
            entry["with_face"] += sample["has_face"]
            entry["detected"] += sample["detected_at_threshold"]
        summary["per_recording"] = per_recording
        # Same centroid matching as FOMO, using box centres, for a direct comparison.
        summary["centroid"] = centroid_metrics(
            {image_id: box_centroids(detections) for image_id, detections in records.items()},
            self._truth, self.score_threshold)
        self.log_dict({f"test/{key}": float(value) for key, value in summary.items()
                       if isinstance(value, (int, float))}, on_step=False, on_epoch=True, sync_dist=False)

        report = test_utils.report_header(self, samples, self._truth, self.annotation_path, self.image_every)
        report.update({
            "score_threshold": self.score_threshold,
            "coordinate_system": "original image pixels, xyxy; class 0 = face",
            "metric_notes": {
                "detection": "COCO bbox AP over all evaluated images (max 100 detections, score >= 0.05 after "
                             "NMS); no score-threshold filtering; -1 means undefined.",
                "loss": "Detection loss (qfl + bbox + dfl) averaged per image, as in validation.",
                "image_level": "An image is predicted to contain a face if any box scores >= score_threshold; "
                               "ground truth from non-ignored boxes.",
                "centroid": "FOMO-style metric at score_threshold: a box centre inside an unmatched GT box is a "
                            "TP, otherwise FP; unmatched GT boxes are FN.",
                "images": "Paths relative to results.json; boxes below score_threshold are not drawn.",
            },
            "summary": summary, "samples": samples,
        })
        test_utils.write_report(self.output_dir, report)
