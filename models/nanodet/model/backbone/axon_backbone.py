import torch.nn as nn

from ..module.conv import DepthwiseConvModule


class AxonBackbone(nn.Module):
    """Small CNN backbone restricted to Axon-friendly depthwise-separable blocks.

    stage_strides sets the five stage strides. The default (2, 2, 2, 2, 2) takes
    an RGB image and returns feature maps with strides 8, 16 and 32, matching the
    RGB baseline of DCT_Object_Detection; (1, 1, 1, 2, 2) gives strides 1, 2 and 4
    for inputs already on an 8x8 block grid. size_divisor is the total stride.
    There are no channel-shuffle, reshape or transpose operations in the backbone.
    """

    def __init__(
        self,
        input_channels=3,
        num_dct_coefficients=None,
        channels=(24, 32, 64, 96),
        stage_blocks=(1, 1, 2, 2, 2),
        stage_strides=(2, 2, 2, 2, 2),
        activation="ReLU",
        norm_cfg=dict(type="BN"),
    ):
        super(AxonBackbone, self).__init__()
        if len(channels) != 4 or len(stage_blocks) != 5:
            raise ValueError("channels needs four and stage_blocks five values")
        if len(stage_strides) != 5 or any(s not in (1, 2) for s in stage_strides):
            raise ValueError("stage_strides needs five values of 1 or 2")
        if num_dct_coefficients is not None:
            if not isinstance(num_dct_coefficients, int) or num_dct_coefficients < 1:
                raise ValueError("num_dct_coefficients must be a positive integer")
            input_channels = 3 * num_dct_coefficients
        if not isinstance(input_channels, int) or input_channels < 1:
            raise ValueError("input_channels must be a positive integer")

        self.channels = tuple(channels)
        self.input_channels = input_channels
        self.size_divisor = 1
        for stride in stage_strides:
            self.size_divisor *= stride

        self.stem = self._make_stage(
            self.input_channels, self.channels[0], stage_blocks[0], stride=stage_strides[0],
            activation=activation, norm_cfg=norm_cfg
        )
        self.stage2 = self._make_stage(
            self.channels[0], self.channels[1], stage_blocks[1], stride=stage_strides[1],
            activation=activation, norm_cfg=norm_cfg
        )
        self.stage3 = self._make_stage(
            self.channels[1], self.channels[1], stage_blocks[2], stride=stage_strides[2],
            activation=activation, norm_cfg=norm_cfg
        )
        self.stage4 = self._make_stage(
            self.channels[1], self.channels[2], stage_blocks[3], stride=stage_strides[3],
            activation=activation, norm_cfg=norm_cfg
        )
        self.stage5 = self._make_stage(
            self.channels[2], self.channels[3], stage_blocks[4], stride=stage_strides[4],
            activation=activation, norm_cfg=norm_cfg
        )

    @staticmethod
    def _make_stage(
        in_channels, out_channels, blocks, stride, activation, norm_cfg
    ):
        if blocks < 1:
            raise ValueError("each AxonBackbone stage needs at least one block")
        layers = [
            DepthwiseConvModule(
                in_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                norm_cfg=norm_cfg,
                activation=activation,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                DepthwiseConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=norm_cfg,
                    activation=activation,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5
