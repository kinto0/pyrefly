# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import assert_shape, IntTuple


def test_qint32() -> None:
    # https://docs.pytorch.org/docs/stable/quantization-support.html
    dtype = torch.qint32
    assert_type(dtype, torch.dtype)
    assert isinstance(dtype, torch.dtype)
    quantized = torch.quantize_per_tensor(torch.ones((2, 3)), 0.1, 0, dtype)
    assert quantized.dtype == dtype
    assert_shape(quantized.shape, IntTuple, runtime=(2, 3))
