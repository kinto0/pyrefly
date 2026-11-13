# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import numpy  # noqa: F401  # E: Cannot find module `numpy`
import torch  # noqa: F401  # E: Cannot find module `torch`
from einops import einsum, rearrange, reduce, repeat
from jax import Array


def test_shapes(
    image: Array[[2, 3, 5, 7]],
    matrix: Array[[2, 3]],
    left: Array[[2, 3, 5]],
    right: Array[[2, 5, 7]],
) -> None:
    assert_type(rearrange(image, "b c h w -> b h w c"), Array[[2, 5, 7, 3]])
    assert_type(reduce(matrix, "row col -> row", "mean"), Array[[2]])
    assert_type(repeat(matrix, "row col -> row col copies", copies=4), Array[[2, 3, 4]])
    assert_type(
        einsum(left, right, "batch row inner, batch inner col -> batch row col"),
        Array[[2, 3, 7]],
    )
