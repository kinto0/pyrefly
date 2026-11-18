# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/modules/batchnorm.py

from __future__ import annotations

import torch
import torch.nn as nn
from shape_extensions import assert_shape
from torch.nn.modules.batchnorm import _BatchNorm


def test_batch_norm_base_class() -> None:
    norm1d: _BatchNorm = nn.BatchNorm1d(3)
    norm2d: _BatchNorm = nn.BatchNorm2d(3)
    norm3d: _BatchNorm = nn.BatchNorm3d(3)
    assert all(isinstance(norm, _BatchNorm) for norm in (norm1d, norm2d, norm3d))
    assert_shape(norm1d(torch.randn((2, 3, 4))).shape, (2, 3, 4))
