# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/modules/conv.py

from __future__ import annotations

import torch
import torch.nn as nn
from shape_extensions import assert_shape


class Conv1dBlock(nn.Conv1d):
    def __init__(self, channels: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__(
            channels,
            channels,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


class Conv2dBlock(nn.Conv2d):
    def __init__(self, channels: int, stride: int, padding: int, dilation: int) -> None:
        super().__init__(
            channels,
            channels,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )


def test_conv_subclass_scalar_controls() -> None:
    conv1d = Conv1dBlock(3, stride=2, padding=1, dilation=1)
    assert_shape(
        conv1d(torch.randn((2, 3, 10))).shape, (2, int, int), runtime=(2, 3, 5)
    )
    conv2d = Conv2dBlock(3, stride=2, padding=1, dilation=1)
    assert_shape(
        conv2d(torch.randn((2, 3, 8, 8))).shape,
        (2, int, int, int),
        runtime=(2, 3, 4, 4),
    )


def test_conv_omitted_controls_stay_precise() -> None:
    assert_shape(nn.Conv1d(3, 4, 3)(torch.randn((2, 3, 10))).shape, (2, 4, 8))
    assert_shape(nn.Conv2d(3, 4, 3)(torch.randn((2, 3, 8, 8))).shape, (2, 4, 6, 6))
