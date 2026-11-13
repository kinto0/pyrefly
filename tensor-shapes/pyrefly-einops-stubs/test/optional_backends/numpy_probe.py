# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import assert_type

import jax  # noqa: F401  # E: Cannot find module `jax`
import numpy as np
import torch  # noqa: F401  # E: Cannot find module `torch`
from einops import _NumpyResult, einsum, rearrange, reduce, repeat


def test_shapes(
    image: np.ndarray[[2, 3, 5, 7], np.dtype[np.float64]],
    matrix: np.ndarray[[2, 3], np.dtype[np.float64]],
    left: np.ndarray[[2, 3, 5], np.dtype[np.float64]],
    right: np.ndarray[[2, 5, 7], np.dtype[np.float64]],
) -> None:
    assert_type(
        rearrange(image, "b c h w -> b h w c"),
        np.ndarray[[2, 5, 7, 3], np.dtype[np.float64]],
    )
    assert_type(reduce(matrix, "row col -> row", "mean"), _NumpyResult[[2]])
    assert_type(
        repeat(matrix, "row col -> row col copies", copies=4),
        np.ndarray[[2, 3, 4], np.dtype[np.float64]],
    )
    assert_type(
        einsum(left, right, "batch row inner, batch inner col -> batch row col"),
        _NumpyResult[[2, 3, 7]],
    )
