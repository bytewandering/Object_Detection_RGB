"""COCO loader for spatial-domain 8x8 Zigzag block features."""

import os

import numpy as np
import torch
from PIL import Image

from .coco import CocoDataset


def _normalize_spatial_to_01(feature):
    """Map fixed level-shifted YCbCr samples from [-128, 127] to [0, 1]."""
    normalized = (feature + 128.0) / 255.0
    return np.clip(normalized, 0.0, 1.0, out=normalized)


def _downsample_420(channel):
    """Average each non-overlapping 2x2 area."""
    height = channel.shape[0] // 2 * 2
    width = channel.shape[1] // 2 * 2
    channel = channel[:height, :width]
    return (
        channel[0::2, 0::2]
        + channel[0::2, 1::2]
        + channel[1::2, 0::2]
        + channel[1::2, 1::2]
    ) / 4.0


def _split_blocks(channel, block_size=8):
    """Convert [H, W] to [block_rows, block_cols, N, N]."""
    height = channel.shape[0] // block_size * block_size
    width = channel.shape[1] // block_size * block_size
    channel = channel[:height, :width]
    return channel.reshape(
        height // block_size,
        block_size,
        width // block_size,
        block_size,
    ).transpose(0, 2, 1, 3)


def _zigzag_indices(size=8):
    indices = []
    for diagonal in range(2 * size - 1):
        coordinates = []
        row_start = max(0, diagonal - size + 1)
        row_end = min(size - 1, diagonal)
        for row in range(row_start, row_end + 1):
            coordinates.append((row, diagonal - row))
        if diagonal % 2 == 0:
            coordinates.reverse()
        indices.extend(row * size + col for row, col in coordinates)
    return np.asarray(indices, dtype=np.int64)


def _blocks_to_channels(blocks, indices):
    """Zigzag each block and return [N*N, block_rows, block_cols]."""
    blocks = blocks.reshape(*blocks.shape[:2], -1)[..., indices]
    return blocks.transpose(2, 0, 1).astype(np.float32, copy=False)


def image_to_spatial_zigzag(image_path, image_size=320, block_size=8):
    """Create [192, H/8, W/8] spatial Zigzag features without DCT."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)

    r, g, b = rgb
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    cb = _downsample_420(cb)
    cr = _downsample_420(cr)

    indices = _zigzag_indices(block_size)
    # Match Image_DCT's level shift so that the only removed operation is DCT.
    y_features = _blocks_to_channels(
        _split_blocks(y, block_size) - 128.0, indices
    )
    cb_features = _blocks_to_channels(
        _split_blocks(cb, block_size) - 128.0, indices
    )
    cr_features = _blocks_to_channels(
        _split_blocks(cr, block_size) - 128.0, indices
    )
    cb_features = np.repeat(np.repeat(cb_features, 2, axis=1), 2, axis=2)
    cr_features = np.repeat(np.repeat(cr_features, 2, axis=1), 2, axis=2)
    return np.ascontiguousarray(
        np.concatenate([y_features, cb_features, cr_features], axis=0),
        dtype=np.float32,
    )


class ZigzagCocoDataset(CocoDataset):
    """Read RGB images and return spatial Zigzag channels without applying DCT."""

    def __init__(
        self,
        num_dct_coefficients=8,
        image_size=320,
        block_size=8,
        normalize_to_01=False,
        check_finite=True,
        **kwargs
    ):
        if not isinstance(num_dct_coefficients, int) or not 1 <= num_dct_coefficients <= 64:
            raise ValueError("num_dct_coefficients must be an integer in [1, 64]")
        if image_size % (2 * block_size):
            raise ValueError("image_size must be divisible by 2 * block_size")
        self.num_dct_coefficients = num_dct_coefficients
        self.image_size = image_size
        self.block_size = block_size
        self.normalize_to_01 = normalize_to_01
        self.check_finite = check_finite
        super().__init__(**kwargs)

        if self.use_instance_mask or self.use_keypoint or self.use_seg_mask:
            raise NotImplementedError(
                "ZigzagCocoDataset does not support masks, keypoints, or segmentation"
            )
        if self.load_mosaic or self.multi_scale:
            raise NotImplementedError(
                "ZigzagCocoDataset does not support mosaic or multi_scale"
            )

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
        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image does not exist: " + image_path)

        feature = image_to_spatial_zigzag(
            image_path, image_size=self.image_size, block_size=self.block_size
        )
        count = self.num_dct_coefficients
        component_channels = self.block_size ** 2
        feature = np.ascontiguousarray(
            np.concatenate(
                [
                    feature[:count],
                    feature[component_channels : component_channels + count],
                    feature[2 * component_channels : 2 * component_channels + count],
                ],
                axis=0,
            )
        )
        if self.normalize_to_01:
            feature = _normalize_spatial_to_01(feature)
        if self.check_finite and not np.isfinite(feature).all():
            raise ValueError("Feature contains NaN or Inf: " + image_path)

        _, feature_height, feature_width = feature.shape
        original_width = img_info["width"]
        original_height = img_info["height"]
        scale_x = feature_width / float(original_width)
        scale_y = feature_height / float(original_height)
        ann = self.get_img_annotation(idx)
        warp_matrix = np.array(
            [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return {
            "img": torch.from_numpy(feature),
            "img_info": img_info,
            "gt_bboxes": self._scale_boxes(ann["bboxes"], scale_x, scale_y),
            "gt_labels": ann["labels"],
            "gt_bboxes_ignore": self._scale_boxes(
                ann["bboxes_ignore"], scale_x, scale_y
            ),
            "warp_matrix": warp_matrix,
        }

    def get_val_data(self, idx):
        return self.get_train_data(idx)
