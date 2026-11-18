# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torch.nn import (
    _BatchNorm as _BatchNorm,
    BatchNorm1d as BatchNorm1d,
    BatchNorm2d as BatchNorm2d,
    BatchNorm3d as BatchNorm3d,
)

def __getattr__(name: str) -> Any: ...
