#!/usr/bin/python3
# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.ops import DeformConv, ModulatedDeformConv

class CenterReppHead(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.torch.randn(8, 80, 128, 128)
        background_label (int | None): Label ID of background, set as 0 for
            RPN and num_classes for other heads. It will automatically set as
            num_classes if None is given.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 cfg,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 conv_cfg=None,
                 norm_cfg=None):
        super(CenterReppHead, self).__init__()
        self.in_channels = cfg.MODEL.CENTERNET.MBiFPN_CHANNELS
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.cls_out_channels = self.num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        pts_out_dim = 2 * self.num_points
        cls_out_dim = (2+1) * self.num_points
        self.reppoints_cls_conv = ModulatedDeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                cls_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)

    def _init_weights(self):
        print('==>init repp weights')
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def forward(self, x):
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        o1, o2, mask = torch.chunk(pts_out_init, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        dcn_offset = offset + points_init
        # refine and classify reppoints
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset, mask)))
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        pts_out_refine = pts_out_refine + dcn_offset.detach()
        wh_refine, reg_refine = self.pts2wh_offsets(pts_out_refine)
        pred = {
            'cls': cls_out,
            'wh': wh_refine,
            'reg': reg_refine
        }
        return cls_out, wh_refine,reg_refine
    
    def pts2wh_offsets(self, pts):
        """
        Converting the points set into wh and offsets.
        :param pts: pts map [num_points*2, map_size, map_size]
        :return: each points set is converting to wh and offsets [2, map_size, map_size], [2, map_size, map_size].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        # use 9 points' center as offsets
        pts_mean = pts_reshape.mean(dim=1, keepdim=True)
        # add the offsets
        refine_pts = pts_reshape + pts_mean
        # compute w and h, using abs and max
        pts_wh = refine_pts.abs().max(dim=1, keepdim=True)[0]
        wh = pts_wh.squeeze(dim=1)
        offsets = pts_mean.squeeze(dim=1)
        return wh, offsets




if __name__ == "__main__":
    mask = (torch.rand(1, 1, 128, 128) > 0.5) * 1.0
    head = CenterRepPHead(80, 256).cuda()
    outputs = head.forward(torch.randn(4, 256,128,128).cuda())
    print(outputs[0].size())
    print(outputs[1][1].size())