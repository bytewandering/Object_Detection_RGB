import torch.nn as nn
import torch.nn.functional as F

from ..module.conv import DepthwiseConvModule


class AxonFPN(nn.Module):
    """Three-level FPN using depthwise-separable convolutions and nearest resize."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs=3,
        activation="ReLU",
        norm_cfg=dict(type="BN"),
    ):
        super(AxonFPN, self).__init__()
        if len(in_channels) != 3 or num_outs != 3:
            raise ValueError("AxonFPN currently requires exactly three feature levels")

        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList(
            [
                DepthwiseConvModule(
                    channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
                for channels in in_channels
            ]
        )
        self.output_convs = nn.ModuleList(
            [
                DepthwiseConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
                for _ in in_channels
            ]
        )

    def forward(self, inputs):
        if len(inputs) != len(self.in_channels):
            raise ValueError("AxonFPN input count does not match in_channels")

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]
        for level in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[level], scale_factor=2, mode="nearest")
            laterals[level - 1] = laterals[level - 1] + upsampled
        return tuple(conv(x) for conv, x in zip(self.output_convs, laterals))
