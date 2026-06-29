# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import numpy as np
from shape_extensions import assert_shape


def test_row_vector_broadcasts_over_matrix() -> None:
    matrix = np.ones((3, 4))
    row = np.full(4, 2.0)

    assert_shape(matrix + row, (3, 4))
    assert_shape(row + matrix, (3, 4))
    assert_shape(matrix * row, (3, 4))
    assert_shape(row * matrix, (3, 4))
