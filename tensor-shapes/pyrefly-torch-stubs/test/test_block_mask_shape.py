# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import assert_type

from shape_extensions import assert_shape, IntTuple
from torch.nn.attention.flex_attention import create_block_mask, noop_mask


def test_block_mask_shape() -> None:
    # https://github.com/pytorch/pytorch/blob/v2.12.1/torch/nn/attention/flex_attention.py
    mask = create_block_mask(noop_mask, 2, 3, 128, 256, device="cpu")
    shape = mask.shape
    assert_type(shape, tuple[int, ...])
    assert_shape(shape, IntTuple, runtime=(2, 3, 128, 256))
    assert shape[-2:] == mask.seq_lengths
