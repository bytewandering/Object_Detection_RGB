"""FOMO (Faster Objects, More Objects) after Edge Impulse's constrained object detection.

MobileNetV2 is cut at the ReLU6 after the expansion layer of Keras block
``block_{cut_block}`` (block_6_expand_relu by default, 1/8 resolution). A 1x1
conv with 32 filters and ReLU, then a 1x1 conv to background + classes, gives
per-cell logits. Each object is labelled only in the cell holding its box
centre, and object cells are weighted by object_weight in the cross entropy.

block_grid_input=True is for inputs already on an 8x8 block grid (DCT
coefficients): every stride-2 convolution before the cut runs at stride 1, so
each output cell is one 8x8 block, matching the RGB model's 1/8 grid.
"""

import numpy as np
import torch
import torch.ao.quantization as aq
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision.models import mobilenet_v2


class FOMOHead(nn.Module):
    def __init__(self, in_channels, num_classes=1, head_channels=32):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels, head_channels, 1)
        self.relu = nn.ReLU()
        # Logits for background (index 0) and each foreground class.
        self.logits = nn.Conv2d(head_channels, num_classes + 1, 1)

    def forward(self, x):
        return self.logits(self.relu(self.conv(x)))


class FOMO(nn.Module):
    """Per-cell classifier returning [B, num_classes + 1, H/stride, W/stride] logits."""

    def __init__(self, width_mult=0.35, cut_block=6, num_classes=1, head_channels=32,
                 object_weight=100.0, score_threshold=0.5, input_channels=3,
                 block_grid_input=False):
        super().__init__()
        if width_mult <= 0:
            raise ValueError("width_mult must be positive")
        if type(cut_block) is not int or not 1 <= cut_block <= 16:
            raise ValueError("cut_block must be a Keras block number in [1, 16]")
        if object_weight <= 0 or not 0 < score_threshold < 1:
            raise ValueError("object_weight must be positive and score_threshold in (0, 1)")
        if type(input_channels) is not int or input_channels < 1:
            raise ValueError("input_channels must be a positive integer")
        features = mobilenet_v2(weights=None, width_mult=width_mult).features
        stem = features[0][0]
        if input_channels != stem.in_channels:
            features[0][0] = nn.Conv2d(input_channels, stem.out_channels, stem.kernel_size,
                                       stem.stride, stem.padding, bias=False)
        # Keras expanded_conv is features[1], so block_k is features[k + 1].
        block = features[cut_block + 1]
        self.backbone = nn.Sequential(*features[:cut_block + 1], block.conv[0])
        self.stride = 2 * int(np.prod([features[i].stride for i in range(1, cut_block + 1)]))
        if block_grid_input:
            for module in self.backbone.modules():
                if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
                    module.stride = (1, 1)
            self.stride = 1
        self.out_channels = block.conv[0][0].out_channels
        self.head = FOMOHead(self.out_channels, num_classes, head_channels)
        self.num_classes = num_classes
        self.object_weight = float(object_weight)
        self.score_threshold = float(score_threshold)
        self.quant = aq.QuantStub()
        self.dequant = aq.DeQuantStub()
        self.epoch = 0

    def fuse_model(self, qat=False):
        """Fuse Conv+BN (ReLU6 cannot be fused) and the head Conv+ReLU."""
        fuse = aq.fuse_modules_qat if qat else aq.fuse_modules
        # Snapshot before fusion changes the module hierarchy.
        for module in list(self.backbone.modules()):
            children = list(module.named_children())
            for (name, child), (next_name, next_child) in zip(children, children[1:]):
                if isinstance(child, nn.Conv2d) and isinstance(next_child, nn.BatchNorm2d):
                    fuse(module, [name, next_name], inplace=True)
        fuse(self.head, ["conv", "relu"], inplace=True)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, x, meta):
        return self.train_forward(x, meta) if self.training else self.infer_forward(x, meta)

    def train_forward(self, x, meta=None):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % self.stride, 0, (-h) % self.stride), mode="constant", value=0)
        return self.dequant(self.head(self.backbone(self.quant(x))))

    def infer_forward(self, x, meta):
        return self.post_process(self.train_forward(x, meta), meta)

    def targets(self, logits, meta):
        """Centre-cell class labels [B, H', W'] and per-cell loss weights."""
        batch, _, grid_h, grid_w = logits.shape
        labels = torch.zeros(batch, grid_h, grid_w, dtype=torch.long)
        weights = torch.ones(batch, grid_h, grid_w)

        def cells(boxes):
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            cx = ((boxes[:, 0] + boxes[:, 2]) / 2 / self.stride).astype(np.int64).clip(0, grid_w - 1)
            cy = ((boxes[:, 1] + boxes[:, 3]) / 2 / self.stride).astype(np.int64).clip(0, grid_h - 1)
            return cy, cx

        for index in range(batch):
            cy, cx = cells(meta["gt_bboxes"][index])
            labels[index, cy, cx] = torch.as_tensor(meta["gt_labels"][index], dtype=torch.long) + 1
            weights[index, cy, cx] = self.object_weight
            # A crowd/ignored centre is neither object nor background.
            cy, cx = cells(meta["gt_bboxes_ignore"][index])
            labels[index, cy, cx] = 0
            weights[index, cy, cx] = 0
        return labels.to(logits.device), weights.to(logits.device)

    def loss(self, logits, meta):
        labels, weights = self.targets(logits, meta)
        loss = F.cross_entropy(logits, labels, reduction="none")
        # Weighted mean over all cells, as a Keras weighted cross entropy.
        loss_ce = (loss * weights).sum() / weights.numel()
        positive = labels > 0
        states = {"loss_ce": loss_ce,
                  "loss_object": loss[positive].mean() if positive.any() else loss.new_zeros(()),
                  "loss_background": loss[(labels == 0) & (weights > 0)].mean()}
        return loss_ce, states

    @torch.no_grad()
    def post_process(self, logits, meta):
        """Group above-threshold cells (8-connected) into objects per class.

        Returns {image_id: {class: [{"centroid", "bbox_xyxy", "score"}]}} in
        original image coordinates; the box only encloses the object's cells.
        """
        probabilities = logits.float().softmax(dim=1).cpu().numpy()
        results = {}
        for index, image_id in enumerate(meta["img_info"]["id"]):
            inverse = np.linalg.inv(meta["warp_matrix"][index])
            width = float(meta["img_info"]["width"][index])
            height = float(meta["img_info"]["height"][index])
            detections = {}
            for label in range(self.num_classes):
                probability = probabilities[index, label + 1]
                components, count = ndimage.label(probability >= self.score_threshold,
                                                  structure=np.ones((3, 3)))
                objects = []
                for component in range(1, count + 1):
                    ys, xs = np.nonzero(components == component)
                    scores = probability[ys, xs]
                    centre = np.array([(np.average(xs, weights=scores) + 0.5) * self.stride,
                                       (np.average(ys, weights=scores) + 0.5) * self.stride, 1.0])
                    corners = np.array([[xs.min() * self.stride, ys.min() * self.stride, 1.0],
                                        [(xs.max() + 1) * self.stride, (ys.max() + 1) * self.stride, 1.0]])
                    centre = inverse @ centre
                    corners = (inverse @ corners.T).T
                    x1, x2 = sorted(np.clip(corners[:, 0], 0, width))
                    y1, y2 = sorted(np.clip(corners[:, 1], 0, height))
                    objects.append({"centroid": [float(np.clip(centre[0], 0, width)),
                                                 float(np.clip(centre[1], 0, height))],
                                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                                    "score": float(scores.max())})
                detections[label] = sorted(objects, key=lambda item: -item["score"])
            results[int(image_id)] = detections
        return results
