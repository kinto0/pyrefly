# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

import torch
from shape_extensions import assert_shape, IntTuple
from torch.ao.quantization.observer import MinMaxObserver


def test_per_tensor_symmetric() -> None:
    # https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MinMaxObserver.html
    scheme = torch.per_tensor_symmetric
    assert_type(scheme, torch.qscheme)
    assert isinstance(scheme, torch.qscheme)
    observer = MinMaxObserver(qscheme=scheme)
    observer(torch.tensor((-1.0, 1.0)))
    scale, zero_point = observer.calculate_qparams()
    assert_shape(scale.shape, IntTuple, runtime=(1,))
    assert_shape(zero_point.shape, IntTuple, runtime=(1,))
    assert zero_point.item() == 128
