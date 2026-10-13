# Copyright 2021 RangiLyu.
# Licensed under the Apache License, Version 2.0.

"""COCO dataset loader for precomputed DCT feature arrays."""

import os

import numpy as np
import torch

from .coco import CocoDataset


def _normalize_channels_to_01(feature, channel_lower, channel_upper, epsilon=1e-6):
    """Normalize and clip CHW features using training-set percentile bounds."""
    channel_lower = channel_lower.reshape(-1, 1, 1)
    channel_upper = channel_upper.reshape(-1, 1, 1)
    channel_range = channel_upper - channel_lower
    normalized = np.zeros_like(feature, dtype=np.float32)
    np.divide(
        feature - channel_lower,
        channel_range,
        out=normalized,
        where=channel_range > epsilon,
    )
    return np.clip(normalized, 0.0, 1.0, out=normalized)


class DCTCocoDataset(CocoDataset):
    """Load CHW ``.npy`` DCT features while reusing COCO annotations.

    The JSON stores bounding boxes in the original image coordinate system
    (for example 320x320), whereas a DCT feature has a smaller spatial grid
    (for example 40x40). Bounding boxes are scaled to the feature grid for
    training. ``warp_matrix`` records this scale so NanoDet can map predicted
    boxes back to the original coordinates during evaluation.

    RGB color augmentation and the standard OpenCV image pipeline are not
    applied because operations such as saturation have no valid meaning for
    DCT coefficient channels.
    """

    def __init__(
        self,
        expected_channels=192,
        num_dct_coefficients=8,
        normalize_to_01=False,
        normalization_stats=None,
        normalization_epsilon=1e-6,
        check_finite=True,
        **kwargs
    ):
        self.expected_channels = expected_channels
        if not isinstance(num_dct_coefficients, int) or num_dct_coefficients < 1:
            raise ValueError("num_dct_coefficients must be a positive integer")
        self.num_dct_coefficients = num_dct_coefficients
        self.normalize_to_01 = normalize_to_01
        self.normalization_stats = normalization_stats
        self.normalization_epsilon = normalization_epsilon
        self.check_finite = check_finite
        self.channel_lower = None
        self.channel_upper = None
        if self.normalize_to_01:
            if not self.normalization_stats:
                raise ValueError(
                    "normalization_stats is required when normalize_to_01=True"
                )
            stats = np.load(self.normalization_stats, allow_pickle=False)
            if "channel_lower" not in stats or "channel_upper" not in stats:
                raise ValueError(
                    "Normalization stats must contain channel_lower and channel_upper"
                )
            self.channel_lower = np.asarray(stats["channel_lower"], dtype=np.float32)
            self.channel_upper = np.asarray(stats["channel_upper"], dtype=np.float32)
            if self.channel_lower.shape != self.channel_upper.shape:
                raise ValueError("channel_lower and channel_upper shapes do not match")
        super().__init__(**kwargs)

        if self.use_instance_mask or self.use_keypoint or self.use_seg_mask:
            raise NotImplementedError(
                "DCTCocoDataset does not support masks, keypoints, or segmentation"
            )
        if self.load_mosaic:
            raise NotImplementedError("DCTCocoDataset does not support mosaic")
        if self.multi_scale:
            raise NotImplementedError("DCTCocoDataset does not support multi_scale")

    def _load_feature(self, image_path):
        if not image_path.lower().endswith(".npy"):
            raise ValueError("DCTCocoDataset only supports .npy files: " + image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError("DCT feature does not exist: " + image_path)

        feature = np.load(image_path, allow_pickle=False)
        if feature.ndim != 3:
            raise ValueError(
                "Expected a [C, H, W] DCT feature, got {}: {}".format(
                    feature.shape, image_path
                )
            )
        if (
            self.expected_channels is not None
            and feature.shape[0] != self.expected_channels
        ):
            raise ValueError(
                "Expected {} DCT channels, got {}: {}".format(
                    self.expected_channels, feature.shape[0], image_path
                )
            )

        feature = np.asarray(feature, dtype=np.float32)
        if self.check_finite and not np.isfinite(feature).all():
            raise ValueError("DCT feature contains NaN or Inf: " + image_path)
        return np.ascontiguousarray(feature)

    @staticmethod
    def _scale_boxes(boxes, scale_x, scale_y):
        boxes = boxes.copy()
        if boxes.size:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        return boxes

    def get_train_data(self, idx):
        img_info = self.get_per_img_info(idx)
        image_path = os.path.join(self.img_path, img_info["file_name"])
        feature = self._load_feature(image_path)
        if feature.shape[0] % 3:
            raise ValueError(
                "Stored DCT channels must contain equal Y/Cb/Cr groups, got {}".format(
                    feature.shape[0]
                )
            )
        coefficients_per_component = feature.shape[0] // 3
        count = self.num_dct_coefficients
        if count > coefficients_per_component:
            raise ValueError(
                "num_dct_coefficients={} exceeds the {} stored coefficients "
                "per component".format(count, coefficients_per_component)
            )
        y = feature[0:count]
        cb = feature[
            coefficients_per_component : coefficients_per_component + count
        ]
        cr = feature[
            2 * coefficients_per_component : 2 * coefficients_per_component + count
        ]
        feature = np.ascontiguousarray(np.concatenate([y, cb, cr], axis=0))
        if self.normalize_to_01:
            expected_stats_channels = coefficients_per_component * 3
            if self.channel_lower.shape != (expected_stats_channels,):
                raise ValueError(
                    "Expected {} normalization channels, got {}".format(
                        expected_stats_channels, self.channel_lower.shape
                    )
                )
            selected_indices = np.concatenate(
                [
                    np.arange(count),
                    np.arange(coefficients_per_component, coefficients_per_component + count),
                    np.arange(
                        2 * coefficients_per_component,
                        2 * coefficients_per_component + count,
                    ),
                ]
            )
            feature = _normalize_channels_to_01(
                feature,
                self.channel_lower[selected_indices],
                self.channel_upper[selected_indices],
                epsilon=self.normalization_epsilon,
            )

        _, feature_height, feature_width = feature.shape
        original_width = img_info["width"]
        original_height = img_info["height"]
        if original_width <= 0 or original_height <= 0:
            raise ValueError("COCO image width and height must be positive")

        scale_x = feature_width / float(original_width)
        scale_y = feature_height / float(original_height)
        ann = self.get_img_annotation(idx)
        gt_bboxes = self._scale_boxes(ann["bboxes"], scale_x, scale_y)
        gt_bboxes_ignore = self._scale_boxes(
            ann["bboxes_ignore"], scale_x, scale_y
        )

        # Maps original COCO coordinates to the DCT feature-grid coordinates.
        warp_matrix = np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return {
            "img": torch.from_numpy(feature),
            "img_info": img_info,
            "gt_bboxes": gt_bboxes,
            "gt_labels": ann["labels"],
            "gt_bboxes_ignore": gt_bboxes_ignore,
            "warp_matrix": warp_matrix,
        }

    def get_val_data(self, idx):
        return self.get_train_data(idx)
