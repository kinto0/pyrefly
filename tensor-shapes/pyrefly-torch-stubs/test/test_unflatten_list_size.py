# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/modules/flatten.py

from __future__ import annotations

import torch
import torch.nn as nn
from shape_extensions import assert_shape, IntTuple


def test_unflatten_list_size() -> None:
    unflatten = nn.Unflatten(1, [2, 3])
    # TODO: BUG: Infer the unflattened shape.
    assert_shape(unflatten(torch.randn((4, 6))).shape, IntTuple, runtime=(4, 2, 3))
