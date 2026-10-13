"""RGB pixel, box and valid-region transforms with optional training augmentation."""

import random

import cv2
import numpy as np
import torch

# Pixels lie in [0, 255]; centering on 128 and dividing by 128 mirrors the DCT
# input (quantized coefficient / 128) and fits a symmetric INT8 input grid exactly.
PIXEL_CENTER = 128.0
PIXEL_SCALE = 128.0
AUGMENTATION_KEYS = {"flip", "scale", "translate", "brightness", "contrast", "saturation"}


class DareLabFaceTransform:
    """Return a [3, H, W] image at native resolution, optionally augmented.

    augmentation (train only) follows the NanoDet pipeline of the DCT_Object_Detection
    RGB baseline: flip probability, scale range about the centre, translate as a
    fraction of the size, then brightness offset, contrast and saturation ranges.
    Uncovered pixels are filled with 128, i.e. 0 after normalization. warp_matrix
    maps original to model-input coordinates; boxes follow it and are clipped,
    dropping those narrower or shorter than 2 pixels after clipping.
    """

    def __init__(self, augmentation=None):
        augmentation = dict(augmentation or {})
        unknown = set(augmentation) - AUGMENTATION_KEYS
        if unknown:
            raise ValueError(f"Unknown augmentation keys: {sorted(unknown)}")
        self.flip = float(augmentation.get("flip", 0.0))
        self.scale = tuple(augmentation.get("scale", (1.0, 1.0)))
        self.translate = float(augmentation.get("translate", 0.0))
        self.brightness = float(augmentation.get("brightness", 0.0))
        self.contrast = tuple(augmentation.get("contrast", (1.0, 1.0)))
        self.saturation = tuple(augmentation.get("saturation", (1.0, 1.0)))
        if not 0 <= self.flip <= 1 or not 0 <= self.translate < 1 or self.brightness < 0:
            raise ValueError("flip must be in [0, 1], translate in [0, 1) and brightness >= 0")
        for name in ("scale", "contrast", "saturation"):
            low, high = getattr(self, name)
            if not 0 < low <= high:
                raise ValueError(f"{name} must be [low, high] with 0 < low <= high")
        self.geometric = self.flip > 0 or self.scale != (1.0, 1.0) or self.translate > 0
        self.photometric = (self.brightness > 0 or self.contrast != (1.0, 1.0)
                            or self.saturation != (1.0, 1.0))

    def _geometric_matrix(self, width, height):
        """Continuous-coordinate affine: flip, scale about the centre, then translate."""
        matrix = np.eye(3, dtype=np.float64)
        if random.random() < self.flip:
            # Boxes use continuous edge coordinates, hence W rather than W-1.
            matrix = np.array([[-1, 0, width], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ matrix
        scale = random.uniform(*self.scale)
        centre = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]], dtype=np.float64)
        resize = np.diag([scale, scale, 1.0])
        matrix = centre @ resize @ np.linalg.inv(centre) @ matrix
        shift = np.array([[1, 0, random.uniform(-self.translate, self.translate) * width],
                          [0, 1, random.uniform(-self.translate, self.translate) * height],
                          [0, 0, 1]], dtype=np.float64)
        return shift @ matrix

    def _photometric(self, image):
        pixels = image.astype(np.float32) / 255.0
        if self.brightness > 0:
            pixels += random.uniform(-self.brightness, self.brightness)
        pixels *= random.uniform(*self.contrast)
        pixels = np.clip(pixels, 0.0, 1.0)
        if self.saturation != (1.0, 1.0):
            hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(*self.saturation), 0.0, 1.0)
            pixels = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return np.clip(np.round(pixels * 255.0), 0, 255).astype(np.uint8)

    def __call__(self, image, boxes, ignored_boxes):
        height, width = image.shape[:2]
        matrix = np.eye(3, dtype=np.float64)
        if self.geometric:
            matrix = self._geometric_matrix(width, height)
            # cv2 samples at pixel centres (index i covers [i, i + 1)), so shift by half a pixel.
            half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], dtype=np.float64)
            pixel_matrix = np.linalg.inv(half) @ matrix @ half
            image = cv2.warpAffine(image, pixel_matrix[:2], (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
        if self.photometric:
            image = self._photometric(image)
        pixels = (image.astype(np.float32) - PIXEL_CENTER) / PIXEL_SCALE

        def transform_boxes(values):
            values = np.asarray(values, dtype=np.float32).reshape(-1, 4)
            corners = values[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
            xy = corners @ matrix[:2, :2].T + matrix[:2, 2]
            result = np.concatenate((xy.min(axis=1), xy.max(axis=1)), axis=1)
            result[:, [0, 2]] = result[:, [0, 2]].clip(0, width)
            result[:, [1, 3]] = result[:, [1, 3]].clip(0, height)
            keep = (result[:, 2] - result[:, 0] >= 2) & (result[:, 3] - result[:, 1] >= 2)
            return np.ascontiguousarray(result[keep], dtype=np.float32)

        return {
            "img": torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1))),
            "valid_mask": torch.ones(1, height, width),
            "gt_bboxes": transform_boxes(boxes),
            "gt_bboxes_ignore": transform_boxes(ignored_boxes),
            "warp_matrix": matrix.astype(np.float32),
        }
