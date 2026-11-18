# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/modules/conv.py

from __future__ import annotations

import torch
import torch.nn as nn
from shape_extensions import assert_shape


def test_conv1d_same_padding() -> None:
    conv = nn.Conv1d(3, 4, kernel_size=3, padding="same")
    # TODO: BUG: Model string padding.
    assert_shape(conv(torch.randn((2, 3, 10))).shape, (2, 4, int), runtime=(2, 4, 10))
