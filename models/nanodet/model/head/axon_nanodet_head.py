import torch
import torch.nn as nn

from ..module.conv import DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead


class DepthwisePrediction(nn.Module):
    """Depthwise 3x3 followed by an unfused pointwise prediction projection."""

    def __init__(self, in_channels, out_channels):
        super(DepthwisePrediction, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AxonNanoDetHead(GFLHead):
    """NanoDet/GFL head with raw-logit export and only separable conv blocks.

    Normal PyTorch execution returns [B, N, C] for compatibility with the
    existing loss and post-processing code. ONNX export returns one raw BCHW
    tensor per feature level, avoiding output transpose, reshape and concat on
    the target. No sigmoid is part of either path; post-processing is
    responsible for converting logits to scores.
    """

    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        stacked_convs=2,
        octave_base_scale=5,
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="ReLU",
        feat_channels=64,
        strides=(8, 16, 32),
        **kwargs
    ):
        self.activation = activation
        super(AxonNanoDetHead, self).__init__(
            num_classes=num_classes,
            loss=loss,
            input_channel=input_channel,
            feat_channels=feat_channels,
            stacked_convs=stacked_convs,
            octave_base_scale=octave_base_scale,
            strides=strides,
            norm_cfg=norm_cfg,
            reg_max=reg_max,
            **kwargs
        )

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.strides:
            self.cls_convs.append(self._make_tower())
            self.reg_convs.append(self._make_tower())

        cls_channels = self.cls_out_channels
        reg_channels = 4 * (self.reg_max + 1)
        self.gfl_cls = nn.ModuleList(
            [DepthwisePrediction(self.feat_channels, cls_channels) for _ in self.strides]
        )
        self.gfl_reg = nn.ModuleList(
            [DepthwisePrediction(self.feat_channels, reg_channels) for _ in self.strides]
        )

    def _make_tower(self):
        layers = []
        for index in range(self.stacked_convs):
            in_channels = self.in_channels if index == 0 else self.feat_channels
            layers.append(
                DepthwiseConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        return nn.Sequential(*layers)

    def init_weights(self):
        for towers in (self.cls_convs, self.reg_convs):
            for module in towers.modules():
                if isinstance(module, nn.Conv2d):
                    normal_init(module, std=0.01)
        for cls_pred, reg_pred in zip(self.gfl_cls, self.gfl_reg):
            normal_init(cls_pred.depthwise, std=0.01)
            normal_init(cls_pred.pointwise, std=0.01, bias=-4.595)
            normal_init(reg_pred.depthwise, std=0.01)
            normal_init(reg_pred.pointwise, std=0.01)

    def _forward_levels(self, feats):
        outputs = []
        for x, cls_tower, reg_tower, cls_pred, reg_pred in zip(
            feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_logits = cls_pred(cls_tower(x))
            bbox_logits = reg_pred(reg_tower(x))
            outputs.append(torch.cat([cls_logits, bbox_logits], dim=1))
        return outputs

    def forward(self, feats):
        level_outputs = self._forward_levels(feats)
        if torch.onnx.is_in_onnx_export():
            return tuple(level_outputs)
        outputs = torch.cat([output.flatten(2) for output in level_outputs], dim=2)
        return outputs.permute(0, 2, 1)
