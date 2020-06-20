#!/usr/bin/python3
# -*- coding:utf-8 -*-

import torch.nn as nn
import torchvision.models.resnet as resnet

from .backbone import Backbone

_resnet_mapper = {
    18: resnet.resnet18,
    50: resnet.resnet50,
    101: resnet.resnet101,
}


class ResnetFPNBackbone(Backbone):

    def __init__(self, cfg, input_shape=None, pretrained=True):
        super().__init__()
        depth = cfg.MODEL.RESNETS.DEPTH
        backbone = _resnet_mapper[depth](pretrained=pretrained)
        self.stage0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        p2= self.stage2(x)
        p3 = self.stage3(p2)
        p4 = self.stage4(p3)
        return [p2, p3, p4]
