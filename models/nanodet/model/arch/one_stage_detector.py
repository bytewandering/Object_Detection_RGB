# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as aq

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head
from ..module.conv import ConvModule, DepthwiseConvModule


class OneStageDetector(nn.Module):
    def __init__(
        self,
        backbone_cfg,
        fpn_cfg=None,
        head_cfg=None,
    ):
        super(OneStageDetector, self).__init__()

        # backbone
        self.backbone = build_backbone(backbone_cfg)

        # fpn
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)

        # head
        if head_cfg is not None:
            self.head = build_head(head_cfg)
        self.epoch = 0
        # Without an encoder in front, the detector itself enters the quantized domain.
        self.quant = aq.QuantStub()
        self.dequant = aq.DeQuantStub()

    def fuse_model(self, qat=False):
        # Snapshot before fusion changes the module hierarchy.
        for module in list(self.modules()):
            if isinstance(module, (ConvModule, DepthwiseConvModule)):
                module.fuse_model(qat=qat)

    def forward(self, x, meta):

        if self.training:
            output = self.train_forward(x,meta)
        else:
            output = self.infer_forward(x,meta)

        return output

    def infer_forward(self,x, meta):

        # 仅在右侧和底部补零，使尺寸能被 backbone 的总下采样倍数整除。
        h, w = x.shape[-2:]
        pad_h = (-h) % self.backbone.size_divisor
        pad_w = (-w) % self.backbone.size_divisor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        x = self.quant(x)

        #先经过backbone，得到三个尺度的特征图
        x = self.backbone(x)
        x = self.fpn(x)
        preds = self.dequant(self.head(x))

        #对模型推理结果进行后处理
        output = self.head.post_process(preds, meta)

        return output

    def train_forward(self, x, meta):

        # 仅在右侧和底部补零，使尺寸能被 backbone 的总下采样倍数整除。
        h, w = x.shape[-2:]
        pad_h = (-h) % self.backbone.size_divisor
        pad_w = (-w) % self.backbone.size_divisor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
        x = self.quant(x)

        #先经过backbone，得到三个尺度的特征图
        x = self.backbone(x)
        x = self.fpn(x)
        preds = self.dequant(self.head(x))

        # loss, loss_states = self.head.loss(preds, meta)

        return preds

    def set_epoch(self, epoch):
        self.epoch = epoch
