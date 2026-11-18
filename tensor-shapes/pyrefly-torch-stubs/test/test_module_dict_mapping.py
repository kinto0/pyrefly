# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Upstream: https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/modules/container.py

from __future__ import annotations

from collections.abc import Mapping
from typing import assert_type

import torch
import torch.nn as nn
from shape_extensions import assert_shape


def test_module_dict_mapping_items() -> None:
    modules: Mapping[str, nn.Linear[3, 4]] = {"projection": nn.Linear(3, 4)}
    module_dict = nn.ModuleDict(modules)
    for name, module in module_dict.items():
        assert name == "projection"
        assert_type(module, nn.Linear[3, 4])
        projected = module(torch.ones((2, 3)))
        assert_shape(projected.shape, (2, 4))
