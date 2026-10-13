"""Batch DareLab RGB samples while retaining variable-length boxes on CPU."""

import torch


def darelab_face_collate(samples):
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    tensor_keys = ("img", "reconstruction_target", "valid_mask",
                   "classification_target", "classification_weight")
    if len({tuple(sample["img"].shape) for sample in samples}) != 1:
        raise ValueError("All samples must have the same image shape")
    batch = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
    for key in ("gt_bboxes", "gt_labels", "gt_bboxes_ignore", "warp_matrix"):
        batch[key] = [sample[key] for sample in samples]
    batch["img_info"] = {
        key: [sample["img_info"][key] for sample in samples]
        for key in samples[0]["img_info"]
    }
    return batch
