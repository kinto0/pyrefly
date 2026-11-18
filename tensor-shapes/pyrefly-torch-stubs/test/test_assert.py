# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from shape_extensions import assert_raises, assert_shape


def test_assert() -> None:
    # https://docs.pytorch.org/docs/stable/generated/torch._assert.html
    tensor = torch.ones((2, 3))
    torch._assert(tensor.shape == (2, 3), "shape")
    torch._assert(torch.tensor(True), "scalar")
    assert_shape(tensor.shape, (2, 3))
    with assert_raises(AssertionError):
        torch._assert(False, "false")
    with assert_raises(AssertionError):
        torch._assert(torch.tensor(False), "false tensor")
