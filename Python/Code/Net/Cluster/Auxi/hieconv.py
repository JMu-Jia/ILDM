#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/8 9:31
# @Author : JMu
# @ProjectName : 1D的层级卷积 Hierarchical convolution

import math

import torch.nn



class HieConv(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, l=2, with_bn=False, bn_kwargs=None):
        super().__init__()

        # check arguments
        # assert 0.0 <= p <= 1.0
        # mid_channels = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        # if bn_kwargs is None:
        #     bn_kwargs = {}

        # 渐进层级 l表示层数
        assert in_channels > out_channels
        assert isinstance(l,int)
        mid_channels = int(out_channels + (in_channels-out_channels)/l)

        # pointwise 1
        self.add_module("pw1", torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn1", torch.nn.BatchNorm1d(num_features=mid_channels, **bn_kwargs))

        # pointwise 2
        self.add_module("pw2", torch.nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn2", torch.nn.BatchNorm1d(num_features=out_channels, **bn_kwargs))

    def _reg_loss(self):
        W = self[0].weight[:, :, 0]
        WWt = torch.mm(W, torch.transpose(W, 0, 1))
        I = torch.eye(WWt.shape[0], device=WWt.device)
        return torch.norm(WWt - I, p="fro")

